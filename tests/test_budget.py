# Test made with AI to speed up validation, same style as the other test_*.py
# files: plain asserts + prints, no pytest. Covers three layers bottom-up:
# (1) core.py's real per-provider usage capture, (2) budget.py's BudgetLedger
# in isolation, (3) Company-level integration - the hard ceiling, rank-share
# gating, sibling/manager reallocation, and emergency_budget_tokens auto-grant,
# all exercised through a real Company/Employee/Agent stack against a
# FakeResponder standing in for the HTTP layer (same pattern test_company.py
# already uses).
import json
from types import SimpleNamespace

from llmadapt.core import Agent
from llmadapt.budget import DEFAULT_RANK_BUDGET_SHARES, normalize_shares, BudgetLedger
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


def usage_response(text, input_tokens, output_tokens):
    """An anthropic-shaped response carrying real usage numbers, the way the
    live API actually returns them."""
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def tool_call_response(name, args, call_id="c1"):
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": args}]}


MODEL_MAP = {
    RoleRank.C_SUITE: {"provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
    RoleRank.MANAGER: {"provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
    RoleRank.JUNIOR: {"provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
}


def make_company(on_escalation=None, **kwargs):
    if on_escalation is None:
        def on_escalation(event):
            return EscalationDecision(approve=False)
    return Company(name="Test Co", model_map=MODEL_MAP, on_escalation=on_escalation, **kwargs)


# ---------------------------------------------------------------------------
# Layer 1: core.py real usage capture
# ---------------------------------------------------------------------------

# Test 1: a single chat() turn records real usage from the provider response
agent = Agent(provider="anthropic", model="claude-x", api_key="test-key")
agent._send_request = FakeResponder([usage_response("hi there", 100, 50)])
result = agent.chat("hello")
assert result == "hi there"
assert agent.total_tokens_used == 150
assert agent.usage_log == [{"input": 100, "output": 50, "total": 150}]
print("PASS: a single chat() turn records real per-provider usage onto total_tokens_used/usage_log")

# Test 2: usage accumulates across multiple turns, including tool-call round-trips
agent = Agent(provider="anthropic", model="claude-x", api_key="test-key")
agent.add_tool(lambda: "42", schema={"name": "get_number", "description": "d",
                                      "parameters": {"type": "object", "properties": {}, "required": []}})
agent._send_request = FakeResponder([
    {**tool_call_response("get_number", {}), "usage": {"input_tokens": 30, "output_tokens": 5}},
    usage_response("the number is 42", 40, 10),
])
result = agent.chat("what's the number?")
assert result == "the number is 42"
assert agent.total_tokens_used == 30 + 5 + 40 + 10
assert len(agent.usage_log) == 2, "each round-trip (including the tool-call turn) should record its own usage entry"
print("PASS: usage accumulates across every turn of a tool-calling round-trip, not just the final one")

# Test 3: reset_usage() zeroes both counters
agent.reset_usage()
assert agent.total_tokens_used == 0
assert agent.usage_log == []
print("PASS: reset_usage() zeroes total_tokens_used and usage_log")

# Test 4: a response with no usage field defaults to 0 rather than raising
agent = Agent(provider="anthropic", model="claude-x", api_key="test-key")
agent._send_request = FakeResponder([text_response("no usage field here")])
agent.chat("hello")
assert agent.total_tokens_used == 0
print("PASS: a response missing usage data defaults to 0 instead of raising")


# ---------------------------------------------------------------------------
# Layer 2: budget.py's BudgetLedger in isolation
# ---------------------------------------------------------------------------

# Test 5: normalize_shares rescales arbitrary weights to sum to 1.0
raw = {"A": 2, "B": 6}
norm = normalize_shares(raw)
assert abs(sum(norm.values()) - 1.0) < 1e-9
assert abs(norm["A"] - 0.25) < 1e-9 and abs(norm["B"] - 0.75) < 1e-9
assert abs(sum(normalize_shares(DEFAULT_RANK_BUDGET_SHARES).values()) - 1.0) < 1e-9
print("PASS: normalize_shares() rescales arbitrary relative weights to sum to 1.0")

# Test 6: base_allocation is None (unlimited) with no total_token_budget, and
# importance scales the rank's share from 0.5x to 1.5x
ledger = BudgetLedger(total_token_budget=None)
assert ledger.base_allocation(RoleRank.MANAGER) is None
ledger = BudgetLedger(total_token_budget=1000)
low = ledger.base_allocation(RoleRank.MANAGER, importance=0.0)
mid = ledger.base_allocation(RoleRank.MANAGER, importance=0.5)
high = ledger.base_allocation(RoleRank.MANAGER, importance=1.0)
share = normalize_shares(DEFAULT_RANK_BUDGET_SHARES)[RoleRank.MANAGER]
assert low == int(1000 * share * 0.5)
assert mid == int(1000 * share * 1.0)
assert high == int(1000 * share * 1.5)
assert low < mid < high, "higher importance should scale allocation up, not down"
print("PASS: base_allocation() is unlimited (None) with no budget, and importance scales it 0.5x-1.5x")

# Test 7: allocated_for adds any granted bonus on top of the base share, and
# remaining() subtracts real spend
ledger = BudgetLedger(total_token_budget=1000)
base = ledger.base_allocation(RoleRank.JUNIOR, importance=0.5)
assert ledger.allocated_for(RoleRank.JUNIOR, "J1", importance=0.5) == base
ledger.grant_bonus("J1", 50)
assert ledger.allocated_for(RoleRank.JUNIOR, "J1", importance=0.5) == base + 50
ledger.grant_bonus("J1", 25)
assert ledger.allocated_for(RoleRank.JUNIOR, "J1", importance=0.5) == base + 75, "bonuses should accumulate"
assert ledger.remaining(RoleRank.JUNIOR, "J1", 0.5, spent=base + 75) == 0
assert ledger.remaining(RoleRank.JUNIOR, "J1", 0.5, spent=base) == 75
print("PASS: allocated_for() layers granted bonuses on top of the base share, remaining() nets out real spend")

# Test 8: report() builds a per-employee allocated/spent/remaining snapshot
fake_employees = {
    "J1": SimpleNamespace(name="J1", rank=RoleRank.JUNIOR, importance=0.5, agent=SimpleNamespace(total_tokens_used=10)),
    "M1": SimpleNamespace(name="M1", rank=RoleRank.MANAGER, importance=0.5, agent=SimpleNamespace(total_tokens_used=0)),
}
ledger = BudgetLedger(total_token_budget=1000)
rpt = ledger.report(fake_employees)
assert rpt["J1"]["spent"] == 10 and rpt["J1"]["allocated"] == ledger.base_allocation(RoleRank.JUNIOR, 0.5)
assert rpt["J1"]["remaining"] == rpt["J1"]["allocated"] - 10
assert rpt["M1"]["spent"] == 0
print("PASS: BudgetLedger.report() builds a correct allocated/spent/remaining snapshot per employee")


# ---------------------------------------------------------------------------
# Layer 3: Company-level integration
# ---------------------------------------------------------------------------

# Test 9: the company-wide hard ceiling is checked BEFORE a new task starts,
# not mid-generation - one task can push spend over the ceiling, but the
# *next* one is gated and (with on_escalation denying) raises
company = make_company(total_token_budget=100, emergency_budget_tokens=0)
manager = company.hire("Manager1", RoleRank.MANAGER)
manager.agent._send_request = FakeResponder([usage_response("first result", 60, 60)])
result = company.run("first task", entry_point=manager)
assert result == "first result"
assert manager.agent.total_tokens_used == 120
assert company.total_tokens_spent() == 120, "one in-flight task is allowed to cross the ceiling - not interruptible mid-response"

denied = []
company.on_escalation = lambda event: denied.append(event) or EscalationDecision(approve=False)
try:
    manager.run("second task", company=company)
    assert False, "should have raised - the hard ceiling was already crossed before this task started"
except EscalationUnresolved as e:
    assert e.event.kind == "budget_exhausted"
    assert e.event.detail == "hard_ceiling"
assert len(denied) == 1
print("PASS: the company-wide hard ceiling gates the NEXT task, not the one already in flight, and always asks a human")

# Test 10: a human approving a hard-ceiling escalation with extra_token_budget
# raises the ceiling itself and lets the employee retry
manager.agent._send_request = FakeResponder([usage_response("second result after approval", 10, 10)])
company.on_escalation = lambda event: EscalationDecision(approve=True, extra_token_budget=50)
result = manager.run("second task", company=company)
assert result == "second result after approval"
assert company.budget.total_token_budget == 150, "the ceiling itself should have been raised by the granted amount"
assert manager.agent.total_tokens_used == 140
print("PASS: approving a hard-ceiling escalation with extra_token_budget raises the ceiling and lets the retry through")

# Test 11: rank-allocation exhaustion triggers manager-slack reallocation
# BEFORE ever bothering the human
company = make_company(total_token_budget=1000, emergency_budget_tokens=0)
manager = company.hire("Manager1", RoleRank.MANAGER)
junior = company.hire("Junior1", RoleRank.JUNIOR, reports_to=manager)

junior.agent._send_request = FakeResponder([usage_response("first junior task", 200, 150)])
result = junior.run("task1", company=company)
assert result == "first junior task"
assert junior.agent.total_tokens_used == 350  # over its own ~300-token rank share, allowed for the same reason as test 9

escalation_calls = []
company.on_escalation = lambda event: escalation_calls.append(event) or EscalationDecision(approve=False)
junior.agent._send_request = FakeResponder([usage_response("second task via borrowed slack", 5, 5)])
result = junior.run("task2", company=company)
assert result == "second task via borrowed slack"
assert len(escalation_calls) == 0, "on_escalation should not be bothered while the manager still has slack to lend"
assert company.budget.report(company.employees)["Junior1"]["allocated"] > company.budget.base_allocation(RoleRank.JUNIOR, junior.importance)
print("PASS: an exhausted rank allocation is quietly covered by borrowing manager slack, before ever escalating")

# Test 12: rank-allocation exhaustion with NO slack anywhere up the chain
# (a top-of-chain employee has nowhere to borrow from) falls through to the
# emergency_budget_tokens reserve, still without bothering the human
company = make_company(total_token_budget=1000, emergency_budget_tokens=40)
csuite = company.hire("CEO", RoleRank.C_SUITE)  # no reports_to - nothing to borrow from
csuite.agent._send_request = FakeResponder([usage_response("first ceo task", 30, 30)])
result = csuite.run("task1", company=company)
assert csuite.agent.total_tokens_used == 60  # over its own 50-token rank share

escalation_calls = []
company.on_escalation = lambda event: escalation_calls.append(event) or EscalationDecision(approve=False)
csuite.agent._send_request = FakeResponder([usage_response("second ceo task via emergency reserve", 3, 2)])
result = csuite.run("task2", company=company)
assert result == "second ceo task via emergency reserve"
assert len(escalation_calls) == 0, "the emergency_budget_tokens reserve should cover this before asking a human"
assert company._emergency_budget_spent > 0
print("PASS: with no slack to borrow, an exhausted rank allocation auto-recovers from emergency_budget_tokens instead")

# Test 13: emergency_budget_tokens = 0 always asks the human immediately for
# a rank-allocation exhaustion, same cost-control default as the iteration reserve
company = make_company(total_token_budget=1000, emergency_budget_tokens=0)
csuite = company.hire("CEO", RoleRank.C_SUITE)
csuite.agent._send_request = FakeResponder([usage_response("first task", 60, 60)])
csuite.run("task1", company=company)

asked = []
company.on_escalation = lambda event: asked.append(event) or EscalationDecision(approve=False)
try:
    csuite.run("task2", company=company)
    assert False, "should have raised - no slack, no emergency reserve, human declined"
except EscalationUnresolved as e:
    assert e.event.kind == "budget_exhausted" and e.event.detail == "rank_allocation"
assert len(asked) == 1
print("PASS: emergency_budget_tokens=0 always asks the human immediately for a rank-allocation exhaustion too")

# Test 14: hire() accepts per-employee provider/model/api_key overrides that
# take precedence over the rank's model_map default, and importance is
# clamped into [0, 1] the same way Employee.__init__ already clamps it
company = make_company()
special = company.hire("SpecialManager", RoleRank.MANAGER, provider="openai", model="gpt-4o-mini",
                        api_key="special-key", importance=0.9)
assert special.agent.provider == "openai" and special.agent.model == "gpt-4o-mini" and special.agent.api_key == "special-key"
assert special.importance == 0.9
default_ranked = company.hire("PlainManager", RoleRank.MANAGER)
assert default_ranked.agent.provider == "anthropic" and default_ranked.agent.model == "claude-x", \
    "with no override, hire() should still fall back to the rank's model_map default"
clamped_low = company.hire("LowImportance", RoleRank.MANAGER, importance=-1.0)
clamped_high = company.hire("HighImportance", RoleRank.MANAGER, importance=5.0)
assert clamped_low.importance == 0.0 and clamped_high.importance == 1.0
print("PASS: hire() supports per-employee provider/model/api_key overrides and clamps importance into [0, 1]")

# Test 15: budget_report() reflects company-wide + per-employee state
company = make_company(total_token_budget=500, emergency_budget_tokens=20)
manager = company.hire("Manager1", RoleRank.MANAGER)
manager.agent._send_request = FakeResponder([usage_response("done", 20, 10)])
company.run("a task", entry_point=manager)
report = company.budget_report()
assert report["total_token_budget"] == 500
assert report["total_spent"] == 30
assert report["emergency_budget_tokens"] == 20
assert report["emergency_budget_spent"] == 0
assert report["employees"]["Manager1"]["spent"] == 30
print("PASS: budget_report() reflects company-wide and per-employee allocation/spend state")

print("\nAll checks passed.")
