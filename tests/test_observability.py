# Test made with AI to speed up validation, same style as the other test_*.py
# files: plain asserts + prints, no pytest. Exercises observability.py's
# renderers directly against hand-built org_chart()-shaped dicts, so none of
# this needs a real Company/Agent.
import re

from llmadapt.observability import (
    EventLog,
    record_tool_call,
    render_org_chart_ascii,
    render_org_chart_mermaid,
    render_org_chart_svg,
    _rank_color,
)
from llmadapt.router import RoleRank

SAMPLE_CHART = {
    "company": "Acme & Co",
    "org": [
        {
            "name": "CEO",
            "rank": RoleRank.C_SUITE,
            "reports": [
                {
                    "name": "Manager1",
                    "rank": RoleRank.MANAGER,
                    "reports": [
                        {"name": "Junior1", "rank": RoleRank.JUNIOR, "reports": []},
                        {"name": "Junior2", "rank": RoleRank.JUNIOR, "reports": []},
                    ],
                },
            ],
        },
    ],
}

EMPTY_CHART = {"company": "Empty Co", "org": []}


# Test 1: EventLog is a live, queryable view over a shared list
events = []
log = EventLog(events)
assert len(log) == 0
events.append({"time": 1.0, "kind": "hire", "employee": "A"})
events.append({"time": 2.0, "kind": "task_start", "employee": "A"})
events.append({"time": 3.0, "kind": "hire", "employee": "B"})
assert len(log) == 3, "EventLog should see appends to the underlying list live, not a snapshot"
assert [e["employee"] for e in log.by_employee("A")] == ["A", "A"]
assert len(log.by_kind("hire")) == 2
assert len(log.since(2.5)) == 1
assert log[0]["employee"] == "A"
assert log.to_list() == events and log.to_list() is not events, "to_list() should copy, not alias"
print("PASS: EventLog stays live over its underlying list and supports by_employee/by_kind/since")

# Test 2: record_tool_call appends a well-shaped entry, truncates long results
tool_log = []
record_tool_call(tool_log, employee="Manager1", tool_name="delegate_to_junior1",
                  args={"task": "x"}, result="ok", duration_s=0.123, error=None)
assert tool_log[0]["employee"] == "Manager1"
assert tool_log[0]["result_preview"] == "ok"
assert tool_log[0]["duration_s"] == 0.123
long_result = "x" * 500
record_tool_call(tool_log, employee="Manager1", tool_name="t", args={}, result=long_result, duration_s=0.0)
assert len(tool_log[1]["result_preview"]) == 200
print("PASS: record_tool_call() builds a well-shaped entry and truncates long result previews")

# Test 3: ascii renderer produces the right nesting and connector characters
ascii_tree = render_org_chart_ascii(SAMPLE_CHART)
lines = ascii_tree.splitlines()
assert lines[0] == "Acme & Co"
assert any("CEO" in l and "C_SUITE" in l for l in lines)
assert any("Junior1" in l for l in lines) and any("Junior2" in l for l in lines)
# Junior2 is the last child, so its line should use the "last" connector
junior2_line = next(l for l in lines if "Junior2" in l)
assert junior2_line.strip().startswith("└──")
junior1_line = next(l for l in lines if "Junior1" in l)
assert junior1_line.strip().startswith("├──")
print("PASS: ascii tree renders correct nesting with tree-style connectors")

# Test 4: ascii renderer handles an empty org without crashing
empty_ascii = render_org_chart_ascii(EMPTY_CHART)
assert empty_ascii.strip() == "Empty Co"
print("PASS: ascii renderer handles a company with no employees")

# Test 5: mermaid renderer is well-formed and includes every node + edge
mermaid = render_org_chart_mermaid(SAMPLE_CHART)
assert mermaid.startswith("graph TD")
for name in ("CEO", "Manager1", "Junior1", "Junior2"):
    assert name in mermaid
assert mermaid.count("-->") == 3  # CEO->Manager1, Manager1->Junior1, Manager1->Junior2
assert "classDef rank_c_suite" in mermaid
assert "classDef rank_manager" in mermaid
assert "classDef rank_junior" in mermaid
print("PASS: mermaid renderer includes every node, every edge, and one classDef per rank seen")

# Test 6: svg renderer produces well-formed, non-empty SVG containing every node
svg = render_org_chart_svg(SAMPLE_CHART)
assert svg.startswith("<svg") and svg.strip().endswith("</svg>")
assert svg.count("<rect") >= 4 + 1  # 4 node boxes + the background rect
for name in ("CEO", "Manager1", "Junior1", "Junior2"):
    assert name in svg
# every declared color is a real hex code, not something that broke mid-string
for hexcode in re.findall(r'stroke="(#[0-9a-fA-F]{6})"', svg):
    assert len(hexcode) == 7
print("PASS: svg renderer produces well-formed markup with a node box per employee")

# Test 7: svg renderer handles an empty org without crashing (small placeholder svg)
empty_svg = render_org_chart_svg(EMPTY_CHART)
assert empty_svg.startswith("<svg") and "Empty Co" in empty_svg and "no employees" in empty_svg
print("PASS: svg renderer handles a company with no employees")

# Test 8: rank colors are assigned in RoleRank.ORDER, fixed and distinct, light vs dark differ
light_colors = {rank: _rank_color(rank, "light") for rank in RoleRank.ORDER}
dark_colors = {rank: _rank_color(rank, "dark") for rank in RoleRank.ORDER}
assert len(set(light_colors.values())) == len(RoleRank.ORDER), "every rank should get a distinct light color"
assert len(set(dark_colors.values())) == len(RoleRank.ORDER), "every rank should get a distinct dark color"
assert light_colors[RoleRank.C_SUITE] != dark_colors[RoleRank.C_SUITE], "dark theme should use its own step, not the light hex"
assert _rank_color("SOME_UNKNOWN_RANK", "light") in light_colors.values(), "unknown ranks fall back gracefully, not crash"
print("PASS: rank colors are fixed-order, distinct per rank, and theme-aware")

# Test 9: svg for the dark theme actually uses the dark surface/ink, not the light one
light_svg = render_org_chart_svg(SAMPLE_CHART, theme="light")
dark_svg = render_org_chart_svg(SAMPLE_CHART, theme="dark")
assert 'fill="#1a1a19"' in dark_svg and "#fcfcfb" not in dark_svg, "dark theme should use the dark surface only"
assert 'fill="#fcfcfb"' in light_svg and "#1a1a19" not in light_svg, "light theme should use the light surface only"
assert 'fill="#ffffff"' in dark_svg, "dark theme should use white primary ink for node text"
print("PASS: theme='dark' actually swaps in the dark surface/ink colors instead of the light ones")

print("\nAll checks passed.")
