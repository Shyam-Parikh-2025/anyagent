# Phase 7 tests - compaction.py (the "keep broad concepts, drop specifics" pass
# over activity_log/tool_call_log) and its Company wiring. Plain asserts +
# prints, no pytest, no network. Layers, bottom-up:
#   (1) activity-log compaction: thresholds, the untouched tail, the rollup
#   (2) the protected audit trail - what compaction is NOT allowed to lose
#   (3) tool-call compaction: grouping, errors kept, preview trimming
#   (4) agent mode and its fallback
#   (5) Company integration - automatic compaction at a task boundary
import json
import time

from llmadapt.compaction import ALWAYS_KEEP_KINDS, COMPACTED_KIND, LogCompactionPolicy
from llmadapt.company import Company, EscalationDecision
from llmadapt.router import RoleRank


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self._last = None

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        if self.responses:
            self._last = self.responses.pop(0)
        return self._last


def text_response(text):
    return {"content": [{"type": "text", "text": text}]}


MODEL_MAP = {rank: {"provider": "anthropic", "model": "claude-x", "api_key": "k"}
             for rank in RoleRank.ORDER}


def make_company(on_escalation=None, **kwargs):
    if on_escalation is None:
        def on_escalation(event):
            return EscalationDecision(approve=False)

    return Company(name="Compact Co", model_map=MODEL_MAP, on_escalation=on_escalation, **kwargs)


def activity_events(n, kind="task_start", employee="Worker"):
    now = time.time()
    return [{"time": now + i, "kind": kind, "employee": employee,
             "task": f"a fairly long task description number {i} " * 3} for i in range(n)]


def tool_entries(n, employee="Mgr", tool_name="delegate_to_worker", error=None):
    now = time.time()
    return [{"time": now + i, "employee": employee, "tool_name": tool_name,
             "args": {"task": f"t{i}"}, "result_preview": "x" * 200,
             "duration_s": 0.5, "error": error} for i in range(n)]


# ---------------------------------------------------------------------------
# Layer 1: activity-log compaction
# ---------------------------------------------------------------------------

off = LogCompactionPolicy()  # mode="off" is the default
events = activity_events(500)
assert off.compact_activity(events) == events
print("PASS 1: mode='off' is the default and returns the log untouched")

policy = LogCompactionPolicy(mode="algorithmic", keep_recent=10, trigger_events=50)
small = activity_events(20)
assert policy.compact_activity(small) == small
print("PASS 2: below trigger_events, compaction is a no-op")

big = activity_events(200)
result = policy.compact_activity(big)
assert len(result) < len(big)
assert result[0]["kind"] == COMPACTED_KIND
assert result[0]["covers"] == 190
assert result[0]["by_kind"] == {"task_start": 190}
assert result[0]["employees"] == ["Worker"]
assert result[-10:] == big[-10:], "the recent tail must survive verbatim"
print("PASS 3: old events collapse into one rollup carrying counts, recent tail kept verbatim")

# The input list must never be mutated - callers hold live EventLog wrappers.
original = activity_events(200)
snapshot = list(original)
policy.compact_activity(original)
assert original == snapshot
print("PASS 4: compact_activity never mutates the list it was given")

# The rollup keeps the *shape* and drops the *specifics* - that is the rule.
rolled = result[0]
assert "task" not in rolled, "individual task text is exactly what should be dropped"
assert rolled["from_time"] <= rolled["to_time"]
print("PASS 5: the rollup keeps counts and time span, and drops per-event text")

mixed = (activity_events(100, kind="hire", employee="A")
         + activity_events(100, kind="task_start", employee="B"))
rolled = policy.compact_activity(mixed)[0]
assert rolled["by_kind"]["hire"] == 100
assert set(rolled["employees"]) == {"A", "B"}
print("PASS 6: the rollup counts every kind and lists every employee involved")


# ---------------------------------------------------------------------------
# Layer 2: the protected audit trail
# ---------------------------------------------------------------------------

audit = [{"time": 1, "kind": "escalation", "employee": "A", "message": "ran out"},
         {"time": 2, "kind": "escalation_decision", "employee": "A", "approved": True},
         {"time": 3, "kind": "emergency_budget_used", "employee": "A", "tokens_granted": 500},
         {"time": 4, "kind": "budget_reallocated", "employee": "B", "amount": 10}]
log = activity_events(300) + audit + activity_events(5)
compacted = policy.compact_activity(log)
for protected in audit:
    assert protected in compacted, f"{protected['kind']} must survive compaction"
print("PASS 7: escalation and budget-authority events survive compaction verbatim")

assert set(ALWAYS_KEEP_KINDS) >= {"escalation", "escalation_decision",
                                  "emergency_budget_used", "budget_reallocated"}
print("PASS 8: ALWAYS_KEEP_KINDS covers the whole spend/authority audit trail")

# A log of nothing but protected events can't be compacted at all.
only_audit = audit * 100
assert policy.compact_activity(only_audit) == only_audit
print("PASS 9: a log made entirely of protected events is returned unchanged")

# protect_kinds is overridable, and passing your own replaces the default.
custom = LogCompactionPolicy(mode="algorithmic", keep_recent=5, trigger_events=10,
                             protect_kinds=("hire",))
mixed2 = activity_events(50, kind="hire") + activity_events(50, kind="task_start")
out = custom.compact_activity(mixed2)
assert sum(1 for e in out if e["kind"] == "hire") == 50
assert any(e["kind"] == COMPACTED_KIND for e in out)
print("PASS 10: protect_kinds is overridable and replaces (not extends) the default")


# ---------------------------------------------------------------------------
# Layer 3: tool-call compaction
# ---------------------------------------------------------------------------

calls = tool_entries(200)
out = policy.compact_tool_calls(calls)
rollups = [e for e in out if e.get("kind") == COMPACTED_KIND]
assert len(rollups) == 1, rollups
assert rollups[0]["calls"] == 190
assert rollups[0]["employee"] == "Mgr" and rollups[0]["tool_name"] == "delegate_to_worker"
assert rollups[0]["total_duration_s"] == 95.0
assert rollups[0]["mean_duration_s"] == 0.5
assert len(rollups[0]["example_result_preview"]) <= 200
print("PASS 11: tool calls group by (employee, tool) into one row with counts and timings")

two_tools = tool_entries(100, tool_name="delegate_to_a") + tool_entries(100, tool_name="delegate_to_b")
rollups = [e for e in policy.compact_tool_calls(two_tools) if e.get("kind") == COMPACTED_KIND]
assert len(rollups) == 2
assert {r["tool_name"] for r in rollups} == {"delegate_to_a", "delegate_to_b"}
print("PASS 12: different tools roll up separately rather than being merged")

# Errors are what you go back to a log to find - never collapsed.
with_errors = tool_entries(150) + tool_entries(3, error="boom") + tool_entries(60)
out = policy.compact_tool_calls(with_errors)
assert sum(1 for e in out if e.get("error") == "boom") == 3
print("PASS 13: entries that recorded an error survive compaction verbatim")

# Even entries too recent to collapse get their bulky previews trimmed.
tail_entry = [e for e in out if e.get("error") is None and e.get("kind") != COMPACTED_KIND][-1]
assert len(tail_entry["result_preview"]) < 200
print("PASS 14: kept entries still have their oversized result previews trimmed")

originals = tool_entries(200)
snapshot = json.dumps(originals)
policy.compact_tool_calls(originals)
assert json.dumps(originals) == snapshot
print("PASS 15: compact_tool_calls never mutates the list it was given")


# ---------------------------------------------------------------------------
# Layer 4: agent mode
# ---------------------------------------------------------------------------

seen = {}


def summarizer(text, budget):
    seen["text"] = text
    seen["budget"] = budget
    return "The team did a lot of routine work."


agent_policy = LogCompactionPolicy(mode="agent", keep_recent=10, trigger_events=50,
                                   summarizer=summarizer)
rolled = agent_policy.compact_activity(activity_events(200))[0]
assert rolled["summary"] == "The team did a lot of routine work."
assert rolled["by_kind"] == {"task_start": 190}, "agent mode keeps the structural counts too"
assert seen["budget"] == agent_policy.summary_budget_chars
print("PASS 16: agent mode adds a prose summary alongside the structural counts")


def exploding_summarizer(text, budget):
    raise RuntimeError("summarizer died")


falling_back = LogCompactionPolicy(mode="agent", keep_recent=10, trigger_events=50,
                                   summarizer=exploding_summarizer)
rolled = falling_back.compact_activity(activity_events(200))[0]
assert "summary" not in rolled
assert "summarizer died" in rolled["summary_error"]
assert rolled["covers"] == 190, "the structural rollup must still be complete"
print("PASS 17: a failing summarizer degrades to the structural rollup, never losing the logs")

no_summarizer = LogCompactionPolicy(mode="agent", keep_recent=10, trigger_events=50)
rolled = no_summarizer.compact_activity(activity_events(200))[0]
assert "summary" not in rolled and rolled["covers"] == 190
print("PASS 18: agent mode without a summarizer behaves as algorithmic")

typo = LogCompactionPolicy(mode="algorythmic", keep_recent=10, trigger_events=50)
assert typo.compact_activity(activity_events(200))[0]["kind"] == COMPACTED_KIND
print("PASS 19: a typo'd mode degrades to the free/safe path instead of no-op'ing or crashing")


# ---------------------------------------------------------------------------
# Layer 5: Company integration
# ---------------------------------------------------------------------------

co = make_company()
co.hire("Solo", RoleRank.C_SUITE)
assert co.compact_logs() is None
print("PASS 20: with no policy configured, compact_logs() is a no-op returning None")

co2 = make_company(log_compaction=LogCompactionPolicy(mode="algorithmic",
                                                     keep_recent=5, trigger_events=20))
worker = co2.hire("Solo", RoleRank.C_SUITE)
co2.activity_log.extend(activity_events(200))
co2.tool_call_log.extend(tool_entries(200))
before_events = len(co2.activity_log)
report = co2.compact_logs()
assert report["compacted"] and report["events_dropped"] > 0
assert len(co2.activity_log) < before_events
assert report["before"]["approx_tokens"] > report["after"]["approx_tokens"]
print("PASS 21: compact_logs() shrinks both logs and reports what it saved")

# EventLog wrappers taken BEFORE compaction must still see the compacted list -
# compaction assigns in place precisely so live wrappers stay valid.
log_view = co2.activity()
co2.activity_log.extend(activity_events(200))
co2.compact_logs()
assert len(log_view) == len(co2.activity_log)
assert log_view.by_kind(COMPACTED_KIND)
print("PASS 22: a live EventLog taken before compaction still tracks the compacted list")

# Compaction runs automatically at a task boundary.
co3 = make_company(log_compaction=LogCompactionPolicy(mode="algorithmic",
                                                     keep_recent=5, trigger_events=20))
lead = co3.hire("Lead", RoleRank.C_SUITE)
co3.activity_log.extend(activity_events(200))
lead.agent._send_request = FakeResponder([text_response("done")])
assert co3.run("a task") == "done"
assert any(e["kind"] == COMPACTED_KIND for e in co3.activity_log)
assert any(e["kind"] == "logs_compacted" for e in co3.activity_log)
print("PASS 23: a completed task triggers compaction automatically and logs that it happened")

# An escalation raised during a real run survives the automatic compaction.
approvals = {"n": 0}


def approve(event):
    approvals["n"] += 1
    return EscalationDecision(approve=False)


co4 = make_company(on_escalation=approve, total_token_budget=1,
                   log_compaction=LogCompactionPolicy(mode="algorithmic",
                                                      keep_recent=5, trigger_events=20))
boss = co4.hire("Boss", RoleRank.C_SUITE)
boss.agent.total_tokens_used = 100  # already over the hard ceiling
boss.agent._send_request = FakeResponder([text_response("never reached")])
co4.activity_log.extend(activity_events(300))
try:
    co4.run("a task")
except Exception:
    pass
co4.compact_logs()
assert any(e["kind"] == "escalation" for e in co4.activity_log), \
    "a real escalation must survive a real compaction pass"
print("PASS 24: an escalation raised during a live run survives automatic compaction")

print("\nAll Phase 7 log-compaction checks passed.")
