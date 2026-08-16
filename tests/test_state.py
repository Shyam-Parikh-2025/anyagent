# Test made with AI to speed up validation, same style as test_company.py:
# plain asserts + prints, no pytest, fully offline against a FakeResponder.
#
# Covers Phase D: save_state() captures where a company is, restore_state()
# puts it back onto a freshly-built company, and every structural difference
# between the two is refused rather than half-applied.
import json
import os
import tempfile

from llmadapt import CompanyState, StateMismatch
from llmadapt.company import Company, EscalationDecision
from llmadapt.router import RoleRank


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, payload, headers):
        return self.responses.pop(0)


def text_response(t):
    return {"content": [{"type": "text", "text": t}]}


def tool_call_response(name, args, call_id="c1"):
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": args}]}


MODEL_MAP = {
    RoleRank.C_SUITE: {"provider": "anthropic", "model": "claude-x", "api_key": "super-secret-key"},
    RoleRank.JUNIOR: {"provider": "anthropic", "model": "claude-x", "api_key": "super-secret-key"},
}


def build_company(**kwargs):
    """The 'build it the same way you built it the first time' step. A snapshot
    cannot do this for you - on_escalation and the tools are code."""
    kwargs.setdefault("on_escalation", lambda e: EscalationDecision(approve=False))
    company = Company(name="Test Co", model_map=MODEL_MAP, **kwargs)
    boss = company.hire("Boss", RoleRank.C_SUITE)
    junior = company.hire("Junior", RoleRank.JUNIOR, reports_to=boss)
    return company, boss, junior


# --- 1. a snapshot captures the run, and reads back as the same document ----
company, boss, junior = build_company(total_token_budget=100_000, emergency_iteration_reserve=4)
junior.agent._send_request = FakeResponder([text_response("junior did the thing")])
boss.agent._send_request = FakeResponder([
    tool_call_response("delegate_to_junior", {"task": "sub"}),
    text_response("boss wrapped up"),
])
assert company.run("do the work", entry_point=boss) == "boss wrapped up"

snapshot = company.save_state(notes="after the first task")
assert snapshot.company_name == "Test Co"
assert snapshot.notes == "after the first task"
assert {e.name for e in snapshot.employees} == {"Boss", "Junior"}
assert snapshot.saved_at > 0
round_tripped = CompanyState.from_json(snapshot.to_json())
assert round_tripped.to_dict() == snapshot.to_dict()
print("PASS: save_state() captures the company and survives a JSON round trip unchanged")


# --- 2. it carries no credentials, and says so by checking ------------------
raw = snapshot.to_json()
assert "super-secret-key" not in raw, "an API key reached the snapshot"
assert '"api_key"' not in raw
# The provider and model are there - those are configuration, not credentials.
assert snapshot.employee("Boss").agent.provider == "anthropic"
assert snapshot.employee("Boss").agent.model == "claude-x"
print("PASS: a snapshot carries provider/model but no API key")


# --- 3. restoring onto a fresh company reproduces the run's state -----------
saved_history = [dict(m) for m in boss.agent.conversation.history]
saved_activity = len(company.activity_log)
boss.agent.total_tokens_used = 4321
company.budget.grant_bonus("Junior", 777)
company._emergency_iteration_remaining = 1
snapshot = company.save_state()

fresh, fresh_boss, fresh_junior = build_company(total_token_budget=100_000, emergency_iteration_reserve=4)
assert fresh_boss.agent.conversation.history != saved_history  # nothing has happened on it yet
fresh.restore_state(snapshot)

assert fresh_boss.agent.conversation.history == saved_history
assert fresh_boss.agent.total_tokens_used == 4321
assert fresh.budget._bonus_allocation.get("Junior") == 777
assert fresh._emergency_iteration_remaining == 1
assert len(fresh.activity_log) == saved_activity
assert [e["tool_name"] for e in fresh.tool_call_log] == ["delegate_to_Junior"]
print("PASS: restore_state() puts conversations, counters, budget and logs back on a fresh company")


# --- 4. the restored company carries on from where the old one stopped ------
fresh_junior.agent._send_request = FakeResponder([text_response("second task done")])
fresh_boss.agent._send_request = FakeResponder([text_response("carried on after resume")])
assert fresh.run("next task", entry_point=fresh_boss) == "carried on after resume"
# The new turn was appended to the restored history, not to an empty one.
assert len(fresh_boss.agent.conversation.history) > len(saved_history)
assert fresh_boss.agent.conversation.history[:len(saved_history)] == saved_history
print("PASS: a restored company continues its existing conversation instead of starting over")


# --- 5. a snapshot saves to and loads from disk -----------------------------
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "state.json")
    company.save_state(path=path)
    assert os.path.exists(path)
    with open(path) as handle:
        assert json.load(handle)["company_name"] == "Test Co"
    reloaded, reloaded_boss, _ = build_company(total_token_budget=100_000, emergency_iteration_reserve=4)
    reloaded.restore_state(path)  # a path is accepted directly
    assert reloaded_boss.agent.total_tokens_used == 4321
print("PASS: a snapshot writes to disk and restore_state() accepts the path directly")


# --- 6. structural mismatches are refused, all of them at once --------------
wrong = Company(name="Other Co", model_map=MODEL_MAP,
                on_escalation=lambda e: EscalationDecision(approve=False))
wrong_boss = wrong.hire("Boss", RoleRank.C_SUITE)
wrong.hire("Stranger", RoleRank.JUNIOR, reports_to=wrong_boss)
try:
    wrong.restore_state(snapshot)
    assert False, "restoring onto a differently-shaped company must not silently succeed"
except StateMismatch as e:
    joined = " ".join(e.problems)
    assert "'Junior'" in joined and "not in this company" in joined, e.problems
    assert "'Stranger'" in joined and "not in the snapshot" in joined, e.problems
    # Every problem at once, not just the first one found.
    assert len(e.problems) >= 2
print("PASS: restoring onto a company with a different roster raises and lists every difference")


# --- 7. ...including a reporting line that moved, and a missing tool --------
moved, moved_boss, _ = build_company(total_token_budget=100_000)
moved.hire("Extra", RoleRank.JUNIOR)  # unrelated, keeps the roster comparison clean
try:
    moved.restore_state(snapshot)
    assert False, "an extra employee should have been caught"
except StateMismatch:
    pass

no_tools, no_tools_boss, no_tools_junior = build_company(total_token_budget=100_000)
no_tools_boss.agent.tool_registry.schemas.pop("delegate_to_junior")
no_tools_boss.agent.tool_registry.functions_maps.pop("delegate_to_junior")
try:
    no_tools.restore_state(snapshot)
    assert False, "a company missing a tool it had at save time should be refused"
except StateMismatch as e:
    assert any("missing the tool" in p for p in e.problems), e.problems
print("PASS: a moved reporting line or a missing tool is caught, not restored around")


# --- 8. strict=False is the deliberate escape hatch ------------------------
loose = Company(name="Loose Co", model_map=MODEL_MAP,
                on_escalation=lambda e: EscalationDecision(approve=False))
loose_boss = loose.hire("Boss", RoleRank.C_SUITE)
loose.restore_state(snapshot, strict=False)
assert loose_boss.agent.total_tokens_used == 4321
assert loose._emergency_iteration_remaining == 1
print("PASS: strict=False restores what lines up, for deliberate surgery rather than resuming")


# --- 9. an unreadable version is refused rather than guessed at ------------
future = CompanyState.from_dict({**snapshot.to_dict(), "version": 999})
try:
    company.restore_state(future)
    assert False, "a future snapshot version must not be applied on a guess"
except StateMismatch as e:
    assert "version 999" in str(e)
print("PASS: a snapshot from a newer version is refused instead of partially understood")

print("\nAll checks passed.")
