"""observability.py - Phase 2: queryable activity/tool-call logs and org-chart
tree rendering (ASCII, Mermaid, and a self-contained SVG diagram) for
company.py.

Kept as its own module rather than folded into company.py, for the same
reason hardware.py/router.py/compressor.py/benchmark.py/selector.py are
already separate files - this is a coherent, sizeable concern on its own, not
a premature package split (that's still reserved for when budget.py/skills/
model_policy/templates land, per the phased plan in the architecture doc).

The SVG/Mermaid renderers' colors come from Cowork's own dataviz skill
(references/palette.md): the fixed-order categorical hues, one per rank tier
in RoleRank.ORDER, validated for colorblind-safe adjacent contrast. Per that
skill's non-negotiable rule ("text wears text tokens, never the series
color"), node text is always drawn in ink tokens - the rank color only ever
appears on the node's border, never behind or as the text itself.

(Phase 5) Those colors now live in presets.py as the "dataviz" Palette in the
shared PALETTES registry, and every renderer takes palette=<name or Palette>.
That was an explicit design correction: palettes are picked by name through
the *same* mechanism as skills, personalities, and org templates - not a
bespoke colour system bolted onto this module. This file keeps only the
default-resolution helper; the colours themselves are presets.
"""

import time
from typing import Any, Callable, Dict, List, Optional


class EventLog:
    """A thin, queryable read layer over a plain list of event dicts. Company
    keeps owning the underlying list (so existing code that iterates
    company.activity_log directly keeps working unchanged) - this wraps that
    same list live, returned by Company.activity()/Company.tool_calls()."""

    def __init__(self, events: List[Dict[str, Any]]):
        self._events = events  # shared reference, not a copy - stays live

    def __iter__(self):
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, i):
        return self._events[i]

    def by_employee(self, name: str) -> List[Dict[str, Any]]:
        return [e for e in self._events if e.get("employee") == name]

    def by_kind(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self._events if e.get("kind") == kind]

    def since(self, timestamp: float) -> List[Dict[str, Any]]:
        return [e for e in self._events if e.get("time", 0) >= timestamp]

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._events)


def record_tool_call(
    log: List[Dict[str, Any]],
    employee: str,
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    duration_s: float,
    error: Optional[str] = None,
) -> None:
    """Appends one entry to a tool-call log (e.g. Company.tool_call_log).
    Kept as a free function rather than a class - the log itself is just a
    list, callers append their own shaped entries too if they want; this is
    the shape delegation calls use."""
    result_str = result if isinstance(result, str) else str(result)
    log.append({
        "time": time.time(),
        "employee": employee,
        "tool_name": tool_name,
        "args": args,
        "result_preview": result_str[:200],
        "duration_s": round(duration_s, 4),
        "error": error,
    })


# ---- org chart rendering ---------------------------------------------------

DEFAULT_PALETTE = "dataviz"

_NODE_W = 160
_NODE_H = 50
_H_GAP = 24
_V_GAP = 60
_TOP_MARGIN = 40


def resolve_palette(palette=None, registry=None):
    """Turns a palette name (or a Palette object, or None) into a Palette.

    None means the "dataviz" default, which is the palette this module used
    exclusively before Phase 5 - so every existing render_org_chart() call
    produces byte-identical output to what it produced before.
    """
    from .presets import PALETTES, Palette

    if isinstance(palette, Palette):
        return palette
    registry = registry or PALETTES
    return registry.get(palette or DEFAULT_PALETTE)


def _rank_color(rank: str, theme: str, palette=None) -> str:
    return resolve_palette(palette).color_for_rank(rank, theme)


def _escape(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_org_chart_ascii(chart: Dict[str, Any]) -> str:
    """A `tree`-command-style text rendering of Company.org_chart()'s output.
    Zero dependencies, works in any terminal or plain-text log."""
    lines = [chart.get("company", "Company")]

    def walk(nodes: List[Dict[str, Any]], prefix: str) -> None:
        for i, node in enumerate(nodes):
            is_last = i == len(nodes) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{node['name']} ({node['rank']})")
            extension = "    " if is_last else "│   "
            walk(node.get("reports", []), prefix + extension)

    walk(chart.get("org", []), "")
    return "\n".join(lines)


def _class_name(rank: str) -> str:
    return "rank_" + "".join(c if c.isalnum() else "_" for c in rank.lower())


def _safe_mermaid_id(name: str) -> str:
    return "n_" + "".join(c if c.isalnum() else "_" for c in name)


def render_org_chart_mermaid(chart: Dict[str, Any], theme: str = "light", palette=None) -> str:
    """Mermaid `graph TD` text - pastes directly into a GitHub/GitLab markdown
    file or any Mermaid-aware renderer, no image generation required.

    palette: a name from the shared PALETTES registry (presets.py) or a
    Palette object; None means the "dataviz" default."""
    pal = resolve_palette(palette)
    lines = ["graph TD"]
    seen_ranks: List[str] = []

    def walk(node: Dict[str, Any]) -> None:
        node_id = _safe_mermaid_id(node["name"])
        lines.append(f'    {node_id}["{_escape(node["name"])}<br/>{_escape(node["rank"])}"]')
        rank = node["rank"]
        if rank not in seen_ranks:
            seen_ranks.append(rank)
        lines.append(f"    class {node_id} {_class_name(rank)}")
        for child in node.get("reports", []):
            child_id = _safe_mermaid_id(child["name"])
            lines.append(f"    {node_id} --> {child_id}")
            walk(child)

    for root in chart.get("org", []):
        walk(root)

    for rank in seen_ranks:
        color = pal.color_for_rank(rank, theme)
        lines.append(
            f"    classDef {_class_name(rank)} stroke:{color},stroke-width:2px,"
            f"fill:{pal.surface[theme]},color:{pal.ink_primary[theme]}"
        )
    return "\n".join(lines)


def _leaf_count(node: Dict[str, Any]) -> int:
    reports = node.get("reports", [])
    if not reports:
        return 1
    return sum(_leaf_count(r) for r in reports)


def _layout(nodes: List[Dict[str, Any]], depth: int, x_offset: float, positions: Dict[int, Dict[str, Any]]) -> float:
    """Assigns {x_center, y, node} to each node (keyed by id(node)) into
    `positions`. Returns the total horizontal width consumed by `nodes`. A
    parent centers over the average x of its own children; a leaf just takes
    one slot."""
    cursor = x_offset
    for node in nodes:
        children = node.get("reports", [])
        slot_width = _leaf_count(node) * (_NODE_W + _H_GAP)
        if children:
            _layout(children, depth + 1, cursor, positions)
            child_xs = [positions[id(c)]["x"] for c in children]
            x_center = sum(child_xs) / len(child_xs)
        else:
            x_center = cursor + slot_width / 2
        positions[id(node)] = {"x": x_center, "y": depth * (_NODE_H + _V_GAP), "node": node}
        cursor += slot_width
    return cursor - x_offset


def render_org_chart_svg(chart: Dict[str, Any], theme: str = "light", palette=None) -> str:
    """A self-contained SVG tree diagram - no JS, no external renderer, opens
    directly in any browser or image viewer. Pure-Python tree layout (no
    graphviz/matplotlib dependency), matching llmadapt's zero-dependency
    design.

    palette: a name from the shared PALETTES registry (presets.py) or a
    Palette object; None means the "dataviz" default."""
    roots = chart.get("org", [])
    pal = resolve_palette(palette)
    surface = pal.surface[theme]
    ink_primary = pal.ink_primary[theme]
    ink_secondary = pal.ink_secondary[theme]
    connector = pal.connector[theme]
    company_name = _escape(chart.get("company", "Company"))

    if not roots:
        width, height = 320, 90
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
            f'<rect width="{width}" height="{height}" fill="{surface}"/>'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="13" '
            f'fill="{ink_secondary}">{company_name} has no employees yet</text></svg>'
        )

    positions: Dict[int, Dict[str, Any]] = {}
    total_width = _layout(roots, depth=0, x_offset=0, positions=positions)
    width = max(total_width, _NODE_W + _H_GAP) + _H_GAP
    max_y = max(p["y"] for p in positions.values())
    height = max_y + _NODE_H + _V_GAP + _TOP_MARGIN

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="{surface}"/>',
        f'<text x="{width / 2:.0f}" y="22" text-anchor="middle" font-size="14" font-weight="600" '
        f'fill="{ink_primary}">{company_name}</text>',
    ]

    def draw_connectors(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            p = positions[id(node)]
            for child in node.get("reports", []):
                cp = positions[id(child)]
                x1, y1 = p["x"], p["y"] + _NODE_H + _TOP_MARGIN
                x2, y2 = cp["x"], cp["y"] + _TOP_MARGIN
                mid_y = (y1 + y2) / 2
                parts.append(
                    f'<path d="M {x1:.1f} {y1:.1f} C {x1:.1f} {mid_y:.1f}, {x2:.1f} {mid_y:.1f}, {x2:.1f} {y2:.1f}" '
                    f'stroke="{connector}" stroke-width="1.5" fill="none"/>'
                )
            draw_connectors(node.get("reports", []))

    def draw_nodes(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            p = positions[id(node)]
            x, y = p["x"] - _NODE_W / 2, p["y"] + _TOP_MARGIN
            color = pal.color_for_rank(node["rank"], theme)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{_NODE_W}" height="{_NODE_H}" rx="8" '
                f'fill="{surface}" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{p["x"]:.1f}" y="{y + 20:.1f}" text-anchor="middle" font-size="13" '
                f'font-weight="600" fill="{ink_primary}">{_escape(node["name"])}</text>'
            )
            parts.append(
                f'<text x="{p["x"]:.1f}" y="{y + 36:.1f}" text-anchor="middle" font-size="11" '
                f'fill="{ink_secondary}">{_escape(node["rank"])}</text>'
            )
            draw_nodes(node.get("reports", []))

    draw_connectors(roots)  # connectors first so node boxes render on top
    draw_nodes(roots)
    parts.append("</svg>")
    return "\n".join(parts)
