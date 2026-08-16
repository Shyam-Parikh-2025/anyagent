"""gui.py - Phase 8's GUI mode: an interactive node-graph company editor.

The roadmap called this "meaningfully bigger than everything built so far
combined" and explicitly not a Python function. It is a small local web app:
`http.server` serving one self-contained page (gui_assets.HTML_PAGE) plus a
tiny JSON API, opened with `webbrowser.open()`. Vanilla JS and SVG on the front
end - no framework, no CDN, no build step, matching llmadapt's zero-dependency
rule on both sides of the wire.

The division of labour is the point: the browser owns *interaction* (dragging,
selection, the right-click bulk actions), and Python owns *truth* (what preset
names exist, whether a design is valid, what a template expands to). The page
never hardcodes a rank list or a skill name - it fetches them from Phase 5's
registries at load. So adding a skill in `presets.py` makes it appear in the
editor with no front-end change at all.

Security, such as it is for a local tool: the server binds to 127.0.0.1 only,
and every request must carry a random per-session token issued at launch. That
stops another process on the machine - or a page open in another tab - from
driving the editor just by guessing the port. It is not an authentication
system and does not pretend to be; it is the minimum that makes an
unauthenticated local port not a foot-gun.

Scope, stated honestly:
- The editor edits a `CompanySpec` - the same serializable format text mode
  produces - so the two halves of Phase 8 interoperate: build in the GUI, save
  the JSON, load it in code, or vice versa.
- It does **not** run tasks, show live activity, or stream logs. It is a
  builder, not a console. The same reasoning as text mode: building is cheap
  and reversible, running spends money, so starting work stays an explicit
  step the user takes in their own code afterwards.
- API keys are deliberately not editable here and never enter a saved design.
  They come from the `model_map` passed to `build_company()`, so a design file
  can be shared or committed safely.
- Layout positions live in `CompanySpec.layout` and are ignored by
  `build_company()` - purely so reopening a design doesn't scramble the
  picture.
"""

import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from .builder import BuildResult, CompanySpec, build_company
from .company import POLICY_MODES, REVIEW_MODES
from .gui_assets import HTML_PAGE
from .policy import EFFORT_LEVELS
from .presets import TASK_SIZES, default_bundle
from .router import RoleRank

# How long launch_gui() waits for the user before giving up, when block=True.
# Long enough to actually design an org, short enough that a forgotten server
# doesn't outlive the session that started it.
DEFAULT_TIMEOUT_SECONDS = 3600


def _options_payload(bundle: Any) -> Dict[str, Any]:
    """Everything the page needs to populate its controls - all of it read from
    the live preset registries rather than duplicated in JS."""
    options: Dict[str, Any] = bundle.names()
    options["ranks"] = list(RoleRank.ORDER)
    options["sizes"] = list(TASK_SIZES)
    options["review_modes"] = list(REVIEW_MODES)
    options["policy_modes"] = list(POLICY_MODES)
    options["effort_levels"] = list(EFFORT_LEVELS)
    # So the editor can say what a skill or personality is for, instead of
    # showing a checkbox list of bare names and hoping they are self-evident.
    from .builder import preset_descriptions

    options["descriptions"] = preset_descriptions(bundle)
    options["templates_detail"] = {
        name: {"description": bundle.org_templates.get(name).description}
        for name in bundle.org_templates.names()
    }
    options["palette_colors"] = {
        name: {
            "ranks_light": list(bundle.palettes.get(name).ranks_light),
            "ranks_dark": list(bundle.palettes.get(name).ranks_dark),
        }
        for name in bundle.palettes.names()
    }
    return options


def _template_employees(bundle: Any, template_name: str, size: str) -> List[Dict[str, Any]]:
    """Expand an org template into concrete employee dicts the editor can then
    edit freely.

    Expanding rather than keeping a template *reference* is deliberate: once
    the user starts dragging nodes around, "this is the small-coding-team
    template" stops being true, and a spec that claims a template it no longer
    matches is worse than one that just lists its employees.
    """
    template = bundle.org_templates.get(template_name)
    template.validate()
    roles = template.roles_for(size)
    role_keys = {r.key for r in roles}

    def effective_manager(role: Any) -> Optional[str]:
        cursor = role.reports_to
        while cursor is not None and cursor not in role_keys:
            cursor = template.role(cursor).reports_to
        return cursor

    first_name_for_key: Dict[str, str] = {}
    employees: List[Dict[str, Any]] = []
    for role in roles:
        count = role.count_for(size)
        for index in range(count):
            suffix = f" {index + 1}" if count > 1 else ""
            name = f"{role.display_title()}{suffix}"
            if index == 0:
                first_name_for_key[role.key] = name
            employees.append({
                "name": name, "rank": role.rank, "reports_to": None,
                "skills": list(role.skills), "personality": role.personality,
                "effort": role.effort, "specialty": role.specialty,
                "importance": role.importance, "provider": None, "model": None, "mode": None,
            })
    # Second pass, once every role's first employee name is known.
    cursor = 0
    for role in roles:
        manager_key = effective_manager(role)
        manager_name = first_name_for_key.get(manager_key) if manager_key else None
        for _ in range(role.count_for(size)):
            employees[cursor]["reports_to"] = manager_name
            cursor += 1
    return employees


class _Handler(BaseHTTPRequestHandler):
    """The JSON API. One handler class; the server instance carries the state."""

    server_version = "llmadapt-gui"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log - this is a desktop tool, and
        a line of HTTP log per drag would bury whatever the user's own program
        is printing."""
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _authorized(self) -> bool:
        return self.headers.get("X-Token") == self.server.token

    def _send(self, payload: Any, status: int = 200, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as e:
            return {"__parse_error__": str(e)}

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = HTML_PAGE.replace("%%TOKEN%%", self.server.token)
            self._send(page.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if not self._authorized():
            self._send({"error": "bad or missing session token"}, status=403)
            return
        if path == "/api/options":
            self._send(_options_payload(self.server.bundle))
            return
        if path == "/api/spec":
            self._send(self.server.spec.to_dict())
            return
        self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send({"error": "bad or missing session token"}, status=403)
            return
        path = self.path.split("?")[0]
        data = self._read_json()
        if isinstance(data, dict) and "__parse_error__" in data:
            self._send({"error": f"invalid JSON: {data['__parse_error__']}"}, status=400)
            return

        if path == "/api/template":
            try:
                employees = _template_employees(
                    self.server.bundle, data.get("template", ""), data.get("size", "small")
                )
            except (KeyError, ValueError) as e:
                self._send({"error": str(e)})
                return
            self._send({"employees": employees})
            return

        if path in ("/api/validate", "/api/save", "/api/build"):
            try:
                spec = CompanySpec.from_dict(data)
            except (ValueError, TypeError) as e:
                self._send({"problems": [str(e)], "warnings": []})
                return
            problems = spec.validate(self.server.bundle)
            if problems:
                self._send({"problems": problems, "warnings": []})
                return

            self.server.spec = spec
            if path == "/api/validate":
                self._send({"problems": [], "warnings": self._dry_run_warnings(spec)})
                return

            if path == "/api/save":
                saved_to = None
                if self.server.save_path:
                    with open(self.server.save_path, "w", encoding="utf-8") as fh:
                        fh.write(spec.to_json())
                    saved_to = self.server.save_path
                self._send({"problems": [], "warnings": self._dry_run_warnings(spec),
                            "path": saved_to})
                return

            # /api/build - the user is done. Build it, stash the result, and
            # let launch_gui() return. The server is shut down from a separate
            # thread because shutdown() blocks until serve_forever() exits and
            # would deadlock if called from inside a handler.
            result = build_company(spec, **self.server.runtime)
            self.server.result = result
            if self.server.save_path:
                with open(self.server.save_path, "w", encoding="utf-8") as fh:
                    fh.write(spec.to_json())
            payload = {
                "problems": result.problems,
                "warnings": result.warnings,
                "headcount": len(result.company.employees) if result.company else 0,
                "org_chart": result.company.render_org_chart(fmt="ascii") if result.company else "",
            }
            self._send(payload)
            if result.ok:
                self.server.done.set()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        self._send({"error": "not found"}, status=404)

    @staticmethod
    def _dry_run_warnings(spec: CompanySpec) -> List[str]:
        """The warnings a build would produce, without building - so the
        editor can show "you have no spend ceiling" before the user commits,
        not after."""
        warnings = []
        if spec.total_token_budget is None:
            warnings.append("No token budget set - this company will have no spend ceiling.")
        if not spec.employees:
            warnings.append("No employees yet.")
        if not any(e.reports_to is None for e in spec.employees) and spec.employees:
            warnings.append("Every employee reports to someone - nobody is at the top of the org.")
        return warnings


class CompanyBuilderServer(ThreadingHTTPServer):
    """The local server behind the editor. Bound to 127.0.0.1 only."""

    daemon_threads = True

    def __init__(self, spec: CompanySpec, bundle: Any, runtime: Dict[str, Any],
                 port: int = 0, save_path: Optional[str] = None, verbose: bool = False):
        super().__init__(("127.0.0.1", port), _Handler)
        self.spec = spec
        self.bundle = bundle
        self.runtime = runtime
        self.save_path = save_path
        self.verbose = verbose
        self.token = secrets.token_urlsafe(24)
        self.result: Optional[BuildResult] = None
        self.done = threading.Event()

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}/"


def launch_gui(
    spec: Optional[Any] = None,
    port: int = 0,
    save_path: Optional[str] = None,
    open_browser: bool = True,
    block: bool = True,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    verbose: bool = False,
    **runtime: Any,
) -> Any:
    """Open the node-graph company editor and (when block=True) wait for it.

    spec: a CompanySpec, a dict, or None to start from an empty canvas. Pass
        an existing one to reopen a design for editing.
    port: 0 (the default) asks the OS for a free port, so two editors can be
        open at once without colliding.
    save_path: if given, "Save JSON" in the editor writes the design here, and
        so does a successful build.
    block: True waits for the user to press "Build" and returns the
        BuildResult. False returns the running server immediately - use
        `server.done.wait()` and `server.result` to collect it yourself, or
        `server.shutdown()` to abandon it.
    timeout: seconds to wait when blocking. On timeout the server is shut down
        and a BuildResult carrying that as a problem is returned, rather than
        hanging a script forever.
    **runtime: model_map / on_escalation / model_policy / presets /
        log_compaction, forwarded to build_company() exactly as in text mode.
        The editor never asks for these - they are runtime concerns and API
        keys must not end up in a saved design.
    """
    if spec is None:
        spec = CompanySpec()
    elif not isinstance(spec, CompanySpec):
        spec = CompanySpec.from_dict(spec)

    bundle = runtime.get("presets") or default_bundle()
    server = CompanyBuilderServer(spec=spec, bundle=bundle, runtime=runtime,
                                  port=port, save_path=save_path, verbose=verbose)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if open_browser:
        try:
            webbrowser.open(server.url)
        except Exception:
            # A headless machine, or no browser configured. Not fatal - the URL
            # is printed and returned, and the server is already running.
            pass
    print(f"[llmadapt] company builder open at {server.url}")

    if not block:
        return server

    finished = server.done.wait(timeout)
    if not finished:
        server.shutdown()
        return BuildResult(spec=server.spec, problems=[
            f"the company builder timed out after {timeout} seconds with nothing built"
        ])
    thread.join(timeout=5)
    return server.result


__all__ = ["launch_gui", "CompanyBuilderServer", "DEFAULT_TIMEOUT_SECONDS"]
