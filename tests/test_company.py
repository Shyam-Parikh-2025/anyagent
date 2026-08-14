# Test made with AI to speed up validation, same style as test_agent.py: plain
# asserts + prints, no pytest, fully offline against a FakeResponder standing
# in for each Agent's HTTP layer (same pattern test_agent.py already uses).
import json

from llmadapt.company import Company, EscalationDecision, EscalationUnresolved
from llmadapt.router import RoleRank


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)


def text_response(text):
    return {"content": [{"type": "text", "text": text}]}


def tool_call_response(name, args, call_id="c1"):
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": args}]}


MODEL_MAP = {
    RoleRank.C_SUITE: {"provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
    RoleRank.MANAGER: {"provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
    RoleRank.JUNIOR: {"provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
}


def make_company(emergency_iteration_reserve=0, on_escalation=None, **kwargs):
    if on_escalation is None:
        def on_escalation(event):
            return EscalationDecision(approve=False)
    return Company(
        name="Test Co",
        model_map=MODEL_MAP,
        on_escalation=on_escalation,
        emergency_iteration_reserve=emergency_iteration_reserve,
        **kwargs,
    )


# Test 1: hire() builds a working Agent per rank and rejects unknown ranks / duplicate names
company = make_company()
manager = company.hire("Manager1", RoleRank.MANAGER)
assert manager.agent.provider == "anthropic" and manager.agent.model == "claude-x"
try:
    company.hire("Manager1", RoleRank.MANAGER)
    assert False, "should have rejected a duplicate name"
except ValueError as e:
    assert "already hired" in str(e)
try:
    company.hire("Ghost", "NOT_A_REAL_RANK")
    assert False, "should have rejected a rank with no model_map entry"
except ValueError as e:
    assert "No model_map entry" in str(e)
print("PASS: hire() builds a working Agent per rank, rejects duplicates and unknown ranks")

# Test 2: hiring a subordinate wires a delegate_to_<name> tool onto the manager's agent
company = make_company()
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)
assert "delegate_to_junior1" in manager.agent.tool_registry.schemas
assert junior.reports_to is manager
assert junior in manager.subordinates
print("PASS: hiring a subordinate wires a delegate_to_<name> tool onto the manager's agent")

# Test 3: delegation actually round-trips through the tool-calling loop
company = make_company()
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)

junior_responder = FakeResponder([text_response("junior finished the subtask")])
junior.agent._send_request = junior_responder

manager_responder = FakeResponder([
    tool_call_response("delegate_to_junior1", {"task": "do the subtask"}),
    text_response("manager reports: junior finished the subtask"),
])
manager.agent._send_request = manager_responder

result = company.run("do the big task", entry_point=manager)
assert result == "manager reports: junior finished the subtask"
assert len(junior_responder.calls) == 1, "delegation should have actually invoked the junior's agent"
print("PASS: delegation round-trips through the normal tool-calling loop")

# Test 3b (Phase 2): that same delegation call was recorded to tool_call_log
tool_calls = company.tool_calls().by_employee("Manager1")
assert len(tool_calls) == 1
assert tool_calls[0]["tool_name"] == "delegate_to_Junior1"
assert tool_calls[0]["args"]["delegated_to"] == "Junior1"
assert "junior finished the subtask" in tool_calls[0]["result_preview"]
assert tool_calls[0]["error"] is None
assert tool_calls[0]["duration_s"] >= 0
print("PASS: delegation calls are recorded to company.tool_call_log with a result preview")

# Test 4: escalation propagates up the reporting chain to the company when unresolved
company = make_company(emergency_iteration_reserve=0)
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)
junior.agent.set_max_tool_iterations(1)

infinite_tool_call = tool_call_response("some_unregistered_tool", {})
junior.agent._send_request = FakeResponder([infinite_tool_call])

escalation_seen = []
def capturing_on_escalation(event):
    escalation_seen.append(event)
    return EscalationDecision(approve=False)
company.on_escalation = capturing_on_escalation

try:
    junior.run("an impossible task", company=company)
    assert False, "should have raised EscalationUnresolved"
except EscalationUnresolved as e:
    assert e.event.employee_name == "Junior1"
    assert e.event.kind == "tool_iteration_limit"
assert len(escalation_seen) == 1
print("PASS: an unresolved iteration-limit failure escalates up to the company and raises EscalationUnresolved")

# Test 5: emergency_token_budget > 0 auto-recovers WITHOUT ever calling on_escalation
company = make_company(emergency_iteration_reserve=10)
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)
junior.agent.set_max_tool_iterations(1)

# First attempt overflows (1 iteration is too few), the retry with granted
# extra iterations succeeds.
junior.agent._send_request = FakeResponder([infinite_tool_call, text_response("done on retry")])

escalation_calls = []
company.on_escalation = lambda event: escalation_calls.append(event) or EscalationDecision(approve=False)

result = junior.run("a task that needs a bit more room", company=company)
assert result == "done on retry"
assert len(escalation_calls) == 0, "on_escalation should NOT be called while the emergency reserve can cover it"
assert company._emergency_iteration_remaining < 10, "emergency reserve should have been drawn down"
print("PASS: a positive emergency_token_budget auto-recovers without ever bothering the human")

# Test 6: emergency_token_budget = 0 always asks the human immediately (the cost-control default)
company = make_company(emergency_iteration_reserve=0)
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)
junior.agent.set_max_tool_iterations(1)
junior.agent._send_request = FakeResponder([infinite_tool_call])

asked = []
company.on_escalation = lambda event: asked.append(event) or EscalationDecision(approve=False)
try:
    junior.run("task", company=company)
except EscalationUnresolved:
    pass
assert len(asked) == 1, "with 0 emergency budget, on_escalation must be called immediately, not skipped"
print("PASS: emergency_token_budget=0 always asks the human immediately, per the cost-control default")

# Test 7: on_escalation approving with extra_tool_iterations lets the employee retry and succeed
company = make_company(emergency_iteration_reserve=0)
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)
junior.agent.set_max_tool_iterations(1)
junior.agent._send_request = FakeResponder([infinite_tool_call, text_response("done after human approval")])
company.on_escalation = lambda event: EscalationDecision(approve=True, extra_tool_iterations=5)

result = junior.run("task", company=company)
assert result == "done after human approval"
print("PASS: on_escalation approving with extra_tool_iterations lets the employee retry and succeed")

# Test 8: org_chart() reflects the reporting structure
company = make_company()
csuite = company.hire("CEO", RoleRank.C_SUITE)
manager = company.hire("Manager1", RoleRank.MANAGER, reports_to=csuite)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)
chart = company.org_chart()
assert chart["org"][0]["name"] == "CEO"
assert chart["org"][0]["reports"][0]["name"] == "Manager1"
assert chart["org"][0]["reports"][0]["reports"][0]["name"] == "Junior1"
print("PASS: org_chart() reflects the actual reporting structure")

# Test 8b (Phase 2): render_org_chart() dispatches to all three formats and
# reflects the same structure as org_chart()
ascii_tree = company.render_org_chart(fmt="ascii")
assert "CEO" in ascii_tree and "Manager1" in ascii_tree and "Junior1" in ascii_tree
assert "└──" in ascii_tree or "├──" in ascii_tree
mermaid = company.render_org_chart(fmt="mermaid")
assert mermaid.startswith("graph TD")
assert "-->" in mermaid
svg = company.render_org_chart(fmt="svg")
assert svg.startswith("<svg") and svg.strip().endswith("</svg>")
assert "CEO" in svg and "Manager1" in svg and "Junior1" in svg
try:
    company.render_org_chart(fmt="not_a_real_format")
    assert False, "should have rejected an unknown fmt"
except ValueError:
    pass
print("PASS: render_org_chart() produces ascii/mermaid/svg output, rejects unknown formats")

# Test 9: activity_log records hire/task lifecycle events, and activity()/tool_calls()
# expose the same data through the queryable EventLog wrapper
company = make_company()
manager = company.hire("Manager1", RoleRank.MANAGER)
manager.agent._send_request = FakeResponder([text_response("all done")])
company.run("a simple task", entry_point=manager)
kinds = [entry["kind"] for entry in company.activity_log]
assert kinds == ["hire", "task_start", "task_end"]
assert len(company.activity()) == 3
assert [e["kind"] for e in company.activity().by_employee("Manager1")] == ["hire", "task_start", "task_end"]
assert len(company.activity().by_kind("task_start")) == 1
assert len(company.tool_calls()) == 0, "no delegation happened in this run, so tool_call_log should be empty"
print("PASS: activity_log records the lifecycle, and activity()/tool_calls() query it as EventLogs")

print("\nAll checks passed.")
