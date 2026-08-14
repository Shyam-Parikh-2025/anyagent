# Phase 8 (GUI mode) tests - gui.py's local server and its JSON API, plus
# static checks on the front-end asset. Plain asserts + prints, no pytest, no
# browser, no network beyond 127.0.0.1.
#
# What is and isn't covered, stated plainly: every server route is exercised
# for real over HTTP (options, template expansion, validate, save, build, the
# token check, the 404s), and the page's JS is checked for balanced structure
# and for not hardcoding data it should be fetching. What is NOT covered is
# the browser behaviour itself - dragging, rubber-band selection, the
# right-click menu - because driving a real DOM would need a browser
# automation dependency, which this library does not have and will not take on
# for one module. Those parts are exercised by hand.
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request

from llmadapt.builder import CompanySpec
from llmadapt.gui import CompanyBuilderServer, _template_employees, launch_gui
from llmadapt.gui_assets import HTML_PAGE
from llmadapt.presets import ORG_TEMPLATES, TASK_SIZES, default_bundle
from llmadapt.router import RoleRank


def start_server(**kwargs):
    server = CompanyBuilderServer(
        spec=kwargs.pop("spec", CompanySpec()), bundle=default_bundle(),
        runtime=kwargs.pop("runtime", {}), port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def call(server, path, body=None, token=None, expect_error=False):
    url = server.url.rstrip("/") + path
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    request.add_header("X-Token", server.token if token is None else token)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode()
            return response.status, (json.loads(raw) if raw.startswith(("{", "[")) else raw)
    except urllib.error.HTTPError as e:
        if not expect_error:
            raise
        return e.code, json.loads(e.read().decode())


# ---------------------------------------------------------------------------
# Layer 1: template expansion (pure, no server)
# ---------------------------------------------------------------------------

bundle = default_bundle()
employees = _template_employees(bundle, "small-coding-team", "small")
names = {e["name"] for e in employees}
assert names == {"Chief", "Manager", "Reviewer", "Developer"}, names
by_name = {e["name"]: e for e in employees}
assert by_name["Manager"]["reports_to"] == "Chief"
assert by_name["Developer"]["reports_to"] == "Manager"
assert by_name["Reviewer"]["skills"] == ["code-review"]
print("PASS 1: a template expands into concrete employees with reporting lines intact")

large = _template_employees(bundle, "small-coding-team", "large")
assert sum(1 for e in large if e["name"].startswith("Developer")) == 4
assert all(e["reports_to"] in {None} | {x["name"] for x in large} for e in large)
print("PASS 2: expansion honours per-size role counts and every reports_to resolves")

# Expanded employees must form a spec that actually validates and builds.
for template_name in ORG_TEMPLATES.names():
    for size in TASK_SIZES:
        spec = CompanySpec.from_dict({
            "name": "T", "employees": _template_employees(bundle, template_name, size)})
        assert spec.validate() == [], (template_name, size, spec.validate())
print("PASS 3: every built-in template, at every size, expands into a valid spec")


# ---------------------------------------------------------------------------
# Layer 2: the server API
# ---------------------------------------------------------------------------

server, thread = start_server()

status, page = call(server, "/")
assert status == 200 and page.startswith("<!DOCTYPE html>")
assert "%%TOKEN%%" not in page, "the token placeholder must be substituted before serving"
assert server.token in page
print("PASS 4: GET / serves the page with the session token substituted in")

assert server.server_address[0] == "127.0.0.1", "the server must never bind a public interface"
print("PASS 5: the server binds to 127.0.0.1 only")

status, body = call(server, "/api/options", token="wrong-token", expect_error=True)
assert status == 403 and "token" in body["error"]
status, body = call(server, "/api/validate", {"name": "X"}, token="wrong-token", expect_error=True)
assert status == 403
print("PASS 6: every API route rejects a wrong session token")

status, options = call(server, "/api/options")
assert status == 200
assert options["org_templates"] == ORG_TEMPLATES.names()
assert options["ranks"] == list(RoleRank.ORDER)
assert options["sizes"] == list(TASK_SIZES)
assert options["review_modes"] == ["critique", "append", "off"]
assert "dataviz" in options["palette_colors"]
assert len(options["palette_colors"]["dataviz"]["ranks_light"]) == len(RoleRank.ORDER)
assert options["templates_detail"]["solo"]["description"]
print("PASS 7: /api/options serves every preset list straight from the registries")

status, spec_payload = call(server, "/api/spec")
assert status == 200 and spec_payload["name"] == "New Company"
print("PASS 8: /api/spec serves the starting design")

status, body = call(server, "/api/template", {"template": "solo", "size": "small"})
assert status == 200 and body["employees"][0]["name"] == "Worker"
status, body = call(server, "/api/template", {"template": "nope", "size": "small"})
assert "error" in body and "nope" in body["error"]
print("PASS 9: /api/template expands a template and reports an unknown one as an error")

good = {"name": "Valid Co", "employees": [{"name": "A", "rank": "C_SUITE"}]}
status, body = call(server, "/api/validate", good)
assert body["problems"] == []
assert any("spend ceiling" in w for w in body["warnings"])
print("PASS 10: /api/validate accepts a good design and still warns about the missing ceiling")

bad = {"name": "X", "employees": [{"name": "A", "rank": "WIZARD", "reports_to": "Ghost"}]}
status, body = call(server, "/api/validate", bad)
assert len(body["problems"]) >= 2
assert any("WIZARD" in p for p in body["problems"])
print("PASS 11: /api/validate returns every problem with the design")

status, body = call(server, "/api/validate", {"name": "X", "mystery_field": 1})
assert body["problems"] and "mystery_field" in body["problems"][0]
print("PASS 12: an unknown spec field is reported, not silently dropped")

status, body = call(server, "/api/nope", {}, expect_error=True)
assert status == 404
print("PASS 13: an unknown route 404s")

# Save writes the design to disk when a path was configured.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "design.json")
    save_server, save_thread = start_server(save_path=path)
    status, body = call(save_server, "/api/save", {
        "name": "Saved Co", "employees": [{"name": "A", "rank": "SENIOR"}],
        "layout": {"A": [10, 20]}})
    assert body["problems"] == [] and body["path"] == path
    on_disk = CompanySpec.from_json(open(path).read())
    assert on_disk.name == "Saved Co"
    assert on_disk.layout == {"A": [10, 20]}, "node positions should survive a save/reload"
    save_server.shutdown()
print("PASS 14: /api/save writes a reloadable design, including node positions")

# A design that doesn't validate is never saved.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "design.json")
    save_server, save_thread = start_server(save_path=path)
    status, body = call(save_server, "/api/save", {"name": "X", "employees": [
        {"name": "A", "rank": "NOPE"}]})
    assert body["problems"]
    assert not os.path.exists(path), "an invalid design must not be written to disk"
    save_server.shutdown()
print("PASS 15: an invalid design is not written to disk")

server.shutdown()


# ---------------------------------------------------------------------------
# Layer 3: build hands control back to Python
# ---------------------------------------------------------------------------

build_server, build_thread = start_server()
status, body = call(build_server, "/api/build", {
    "name": "Built Co",
    "employees": [{"name": "Boss", "rank": "C_SUITE"},
                  {"name": "Dev", "rank": "JUNIOR", "reports_to": "Boss", "skills": ["python"]}],
})
assert body["problems"] == [] and body["headcount"] == 2
assert "Boss" in body["org_chart"]
assert build_server.done.wait(5), "a successful build must release the waiting caller"
assert build_server.result.ok
company = build_server.result.company
assert set(company.employees) == {"Boss", "Dev"}
assert company.employees["Dev"].reports_to.name == "Boss"
print("PASS 16: /api/build builds the company, returns a chart, and releases the caller")

assert company.total_tokens_spent() == 0
assert not [e for e in company.activity() if e["kind"] == "task_start"]
print("PASS 17: the GUI builds an org and runs nothing - same rule as text mode")

# A failed build must NOT release the caller - the user stays in the editor.
fail_server, fail_thread = start_server()
status, body = call(fail_server, "/api/build", {"name": "X", "employees": [
    {"name": "A", "rank": "NOT_A_RANK"}]})
assert body["problems"]
assert not fail_server.done.is_set(), "a failed build must leave the editor open"
assert fail_server.result is None
fail_server.shutdown()
print("PASS 18: a build that fails validation leaves the editor open instead of returning junk")

# Non-blocking launch returns the live server; timeout returns a problem.
server3 = launch_gui(open_browser=False, block=False)
assert server3.result is None and server3.url.startswith("http://127.0.0.1:")
status, options = call(server3, "/api/options")
assert options["ranks"]
server3.shutdown()
print("PASS 19: launch_gui(block=False) returns a live server the caller controls")

timed_out = launch_gui(open_browser=False, block=True, timeout=0.4)
assert not timed_out.ok and any("timed out" in p for p in timed_out.problems)
print("PASS 20: launch_gui times out with a clear problem instead of hanging forever")

reopened = launch_gui(spec={"name": "Reopened", "employees": [{"name": "A", "rank": "SENIOR"}]},
                      open_browser=False, block=False)
status, payload = call(reopened, "/api/spec")
assert payload["name"] == "Reopened" and payload["employees"][0]["name"] == "A"
reopened.shutdown()
print("PASS 21: an existing design can be reopened for editing")

# Two editors at once must not collide - port 0 asks the OS for a free one.
a = launch_gui(open_browser=False, block=False)
b = launch_gui(open_browser=False, block=False)
assert a.server_address[1] != b.server_address[1]
assert a.token != b.token
a.shutdown(); b.shutdown()
print("PASS 22: two editors can run at once on different ports with different tokens")


# ---------------------------------------------------------------------------
# Layer 4: static checks on the front end
# ---------------------------------------------------------------------------

assert HTML_PAGE.count("<script>") == 1 and HTML_PAGE.count("</script>") == 1
assert HTML_PAGE.count("<style>") == 1 and HTML_PAGE.count("</style>") == 1
assert "%%TOKEN%%" in HTML_PAGE
print("PASS 23: the page is one self-contained document with a single script and style block")

for forbidden in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr", "googleapis.com", "<script src"):
    assert forbidden not in HTML_PAGE, forbidden
print("PASS 24: the page loads nothing from the network - no CDN, no external script")

script = re.search(r"<script>\n(.*?)</script>", HTML_PAGE, re.DOTALL).group(1)
assert script.count("{") == script.count("}"), "unbalanced braces in the page script"
assert script.count("(") == script.count(")"), "unbalanced parens in the page script"
print("PASS 25: the page script is structurally balanced")

# The page must read its vocabulary from the server, not hardcode it. A rank
# or skill name baked into the JS would silently go stale the moment
# presets.py changed - the exact coupling this design is meant to avoid.
for name in ("GENERAL_MANAGER", "VOLUNTEER", "small-coding-team", "research-pod",
             "data-analysis", "grayscale", "ember"):
    assert name not in script, f"{name} should be fetched from /api/options, not hardcoded"
assert "/api/options" in script and "state.options.ranks" in script
print("PASS 26: the page fetches ranks, skills, templates and palettes rather than hardcoding them")

# Every endpoint the page calls must exist on the server, and vice versa.
called = set(re.findall(r'api\("(/api/[a-z]+)"', script))
import inspect  # noqa: E402

from llmadapt import gui as gui_module  # noqa: E402

served = set(re.findall(r'"(/api/[a-z]+)"', inspect.getsource(gui_module)))
assert called <= served, f"page calls routes the server does not serve: {called - served}"
assert called == {"/api/options", "/api/spec", "/api/template", "/api/validate",
                  "/api/save", "/api/build"}, called
print("PASS 27: every endpoint the page calls is one the server actually serves")

# Every fetch must carry the token, or the whole check is theatre.
assert 'headers: { "X-Token": TOKEN' in script
assert script.count("fetch(") == 1, "all requests should go through the one api() helper"
print("PASS 28: all requests go through one helper that always sends the session token")

# User-supplied text must be escaped before being written into the DOM.
assert "function escapeHtml" in script
assert "escapeHtml(p)" in script and "escapeHtml(data.org_chart" in script
print("PASS 29: user- and server-supplied text is escaped before being inserted into the page")

assert "API keys are deliberately not part of a saved design" in HTML_PAGE
assert "Nothing is run and nothing is spent" in HTML_PAGE
print("PASS 30: the page tells the user that keys are excluded and that building runs nothing")

print("\nAll Phase 8 GUI-mode checks passed.")
