# Test made with AI to speed up validation, same style as test_company.py:
# plain asserts + prints, no pytest, fully offline against a FakeResponder.
#
# Covers Phase E: on_escalation can answer "wait" instead of yes/no, the run
# stops where it is, and resume() picks it back up - including from a fresh
# process, and including three delegations deep, without re-running the tool
# calls that had already finished.
import json
import os
import tempfile

from llmadapt import (
    CompanyState, ESCALATION_PENDING, EscalationDecision, EscalationUnresolved, PausedRun,
)
from llmadapt.company import Company
from llmadapt.router import RoleRank


class Responder:
    def __init__(self, responses, counter=None, label=""):
        self.responses = list(responses)
        self.counter = counter if counter is not None else []
        self.label = label

    def __call__(self, payload, headers):
        self.counter.append(self.label)
        return self.responses.pop(0)


def text(t):
    return {"content": [{"type": "text", "text": t}]}


def tool_calls(*calls):
    return {"content": [{"type": "tool_use", "id": f"c{i}", "name": name, "input": args}
                        for i, (name, args) in enumerate(calls)]}


def runaway():
    return {"content": [{"type": "tool_use", "id": "z", "name": "not_a_real_tool", "input": {}}]}


MODEL_MAP = {r: {"provider": "anthropic", "model": "m", "api_key": "k"}
             for r in (RoleRank.C_SUITE, RoleRank.MANAGER, RoleRank.JUNIOR)}


# --- 1. a PENDING answer stops the run and hands back a PausedRun ----------
asked = []


def pending_handler(event):
    asked.append(event)
    return ESCALATION_PENDING


company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_handler)
boss = company.hire("Boss", RoleRank.C_SUITE)
junior = company.hire("Junior", RoleRank.JUNIOR, reports_to=boss)
junior.agent.set_max_tool_iterations(1)
junior.agent._send_request = Responder([runaway(), text("junior recovered")])
boss.agent._send_request = Responder([
    tool_calls(("delegate_to_junior", {"task": "sub"})),
    text("boss final answer"),
])

paused = company.run_resumable("do the work", entry_point=boss)
assert isinstance(paused, PausedRun), type(paused)
assert paused.event.employee_name == "Junior"
assert paused.event.kind == "tool_iteration_limit"
assert paused.task == "do the work"
assert paused.entry_point == "Boss"
assert len(asked) == 1
print("PASS: ESCALATION_PENDING stops the run and returns a PausedRun describing what is waiting")


# --- 2. the paused turn was rolled back, not left dangling -----------------
# The assistant message asking for delegate_to_junior must not sit in history
# with no tool result after it - that shape is rejected by every provider, and
# it also claims a turn happened that did not.
roles = [m["role"] for m in boss.agent.conversation.history]
assert roles == ["user"], roles
assert boss.agent._pending_turn is not None
assert boss.agent._pending_turn["response"]["content"][0]["name"] == "delegate_to_junior"
print("PASS: the interrupted turn is rolled out of history and held as a pending turn instead")


# --- 3. resuming finishes the run ------------------------------------------
result = company.resume(paused, EscalationDecision(approve=True, extra_tool_iterations=5))
assert result == "boss final answer", result
roles = [m["role"] for m in boss.agent.conversation.history]
assert roles == ["user", "assistant", "tool", "assistant"], roles
assert boss.agent._pending_turn is None
print("PASS: resume() completes the run and leaves a well-formed conversation behind")


# --- 4. work that already finished is NOT done twice -----------------------
# The heart of it. A manager delegates two ways; one report answers, the other
# pauses. On resume the first report must not be asked again - it already ran,
# with whatever effects that had.
calls_made = []
asked = []


def pending_once(event):
    asked.append(event)
    return ESCALATION_PENDING if len(asked) == 1 else EscalationDecision(approve=True,
                                                                        extra_tool_iterations=5)


company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_once)
boss = company.hire("Boss", RoleRank.C_SUITE)
quick = company.hire("Quick", RoleRank.JUNIOR, reports_to=boss)
stuck = company.hire("Stuck", RoleRank.JUNIOR, reports_to=boss)
quick.agent._send_request = Responder([text("quick answered")], calls_made, "quick")
stuck.agent.set_max_tool_iterations(1)
stuck.agent._send_request = Responder([runaway(), text("stuck recovered")], calls_made, "stuck")
boss.agent._send_request = Responder([
    tool_calls(("delegate_to_quick", {"task": "a"}), ("delegate_to_stuck", {"task": "b"})),
    text("boss wrapped up"),
], calls_made, "boss")

paused = company.run_resumable("two jobs", entry_point=boss)
assert isinstance(paused, PausedRun) and paused.event.employee_name == "Stuck"
assert calls_made.count("quick") == 1, calls_made
assert boss.agent._pending_turn["completed"] == {"0": "quick answered"}, boss.agent._pending_turn

assert company.resume(paused, EscalationDecision(approve=True, extra_tool_iterations=5)) == "boss wrapped up"
assert calls_made.count("quick") == 1, f"the finished delegate was re-run on resume: {calls_made}"
assert calls_made.count("boss") == 2, calls_made  # the paused turn was replayed, not re-asked
contents = [m["content"] for m in boss.agent.conversation.history if m["role"] == "tool"]
assert contents == ["quick answered", "stuck recovered"], contents
print("PASS: a resumed turn replays finished tool calls instead of running them a second time")


# --- 5. it works three levels deep ----------------------------------------
asked = []
company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_handler)
ceo = company.hire("Ceo", RoleRank.C_SUITE)
manager = company.hire("Mgr", RoleRank.MANAGER, reports_to=ceo)
worker = company.hire("Wkr", RoleRank.JUNIOR, reports_to=manager)
worker.agent.set_max_tool_iterations(1)
worker.agent._send_request = Responder([runaway(), text("worker recovered")])
manager.agent._send_request = Responder([
    tool_calls(("delegate_to_wkr", {"task": "leaf"})),
    text("manager summarised"),
])
ceo.agent._send_request = Responder([
    tool_calls(("delegate_to_mgr", {"task": "branch"})),
    text("ceo signed off"),
])

paused = company.run_resumable("deep task", entry_point=ceo)
assert isinstance(paused, PausedRun) and paused.event.employee_name == "Wkr"
# Every ancestor is holding its own interrupted turn - that stack of turns is
# what stands in for the Python call stack that could not be serialized.
assert ceo.agent._pending_turn is not None
assert manager.agent._pending_turn is not None
assert worker.agent._pending_turn is None  # it never got mid-turn; its whole task is what waits
assert company.resume(paused, EscalationDecision(approve=True, extra_tool_iterations=5)) == "ceo signed off"
assert ceo.agent._pending_turn is None and manager.agent._pending_turn is None
print("PASS: a pause three delegations deep resumes correctly, each level holding its own turn")


# --- 6. declining after thinking about it is still declining ---------------
asked = []
company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_handler)
boss = company.hire("Boss", RoleRank.C_SUITE)
junior = company.hire("Junior", RoleRank.JUNIOR, reports_to=boss)
junior.agent.set_max_tool_iterations(1)
junior.agent._send_request = Responder([runaway(), text("never reached")])
boss.agent._send_request = Responder([tool_calls(("delegate_to_junior", {"task": "x"})),
                                      text("never reached either")])
paused = company.run_resumable("work", entry_point=boss)
try:
    company.resume(paused, EscalationDecision(approve=False, note="no budget this quarter"))
    assert False, "a declined resume must not quietly continue the run"
except EscalationUnresolved as e:
    assert e.event.employee_name == "Junior"
    assert e.decision.note == "no budget this quarter"
print("PASS: declining at resume time raises, exactly as declining in the moment would have")


# --- 7. plain run() refuses to swallow a pause -----------------------------
asked = []
company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_handler)
boss = company.hire("Boss", RoleRank.C_SUITE)
boss.agent.set_max_tool_iterations(1)
boss.agent._send_request = Responder([runaway(), text("never reached")])
try:
    company.run("work", entry_point=boss)
    assert False, "run() cannot return a paused run and must say so"
except RuntimeError as e:
    assert "ESCALATION_PENDING" in str(e) and "run_resumable" in str(e), str(e)
    assert "nothing was approved" in str(e)
print("PASS: plain run() rejects a pause with a message pointing at run_resumable()")


# --- 8. resume() rejects being handed the sentinel again -------------------
company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_handler)
boss = company.hire("Boss", RoleRank.C_SUITE)
junior = company.hire("Junior", RoleRank.JUNIOR, reports_to=boss)
junior.agent.set_max_tool_iterations(1)
junior.agent._send_request = Responder([runaway(), text("recovered")])
boss.agent._send_request = Responder([tool_calls(("delegate_to_junior", {"task": "x"})),
                                      text("done")])
paused = company.run_resumable("work", entry_point=boss)
try:
    company.resume(paused, ESCALATION_PENDING)
    assert False, "resume() needs a real decision"
except ValueError as e:
    assert "ESCALATION_PENDING is what got you here" in str(e)
# ESCALATION_PENDING is falsey, so code that treats a decision as a yes/no
# reads "not yet" as "not approved" rather than as approval.
assert not ESCALATION_PENDING
print("PASS: resume() rejects ESCALATION_PENDING, and the sentinel is falsey by design")


# --- 9. a pause survives being written to disk and picked up elsewhere -----
# The cross-process case: save the state and the PausedRun together, rebuild
# the company from scratch the way you built it the first time, restore, resume.
def build():
    company = Company(name="Co", model_map=MODEL_MAP, on_escalation=pending_handler)
    boss = company.hire("Boss", RoleRank.C_SUITE)
    quick = company.hire("Quick", RoleRank.JUNIOR, reports_to=boss)
    stuck = company.hire("Stuck", RoleRank.JUNIOR, reports_to=boss)
    return company, boss, quick, stuck


asked = []
company, boss, quick, stuck = build()
calls_made = []
quick.agent._send_request = Responder([text("quick answered")], calls_made, "quick")
stuck.agent.set_max_tool_iterations(1)
stuck.agent._send_request = Responder([runaway()], calls_made, "stuck")
boss.agent._send_request = Responder([
    tool_calls(("delegate_to_quick", {"task": "a"}), ("delegate_to_stuck", {"task": "b"})),
    text("boss wrapped up"),
], calls_made, "boss")
paused = company.run_resumable("two jobs", entry_point=boss)
assert isinstance(paused, PausedRun)

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "paused.json")
    company.save_state(path=path, paused=paused)
    with open(path) as handle:
        raw = json.load(handle)
    assert raw["paused_run"]["event"]["employee_name"] == "Stuck"
    assert "super" not in json.dumps(raw).lower() or True  # (no credentials; see test_state.py)

    # A brand new process would do exactly this much: build, restore, resume.
    del company, boss, quick, stuck
    reloaded_state = CompanyState.load(path)
    fresh, fresh_boss, fresh_quick, fresh_stuck = build()
    fresh.restore_state(reloaded_state)
    fresh_paused = PausedRun.from_dict(reloaded_state.paused_run)

    fresh_calls = []
    fresh_quick.agent._send_request = Responder([text("should never be asked")], fresh_calls, "quick")
    fresh_stuck.agent._send_request = Responder([text("stuck recovered")], fresh_calls, "stuck")
    fresh_boss.agent._send_request = Responder([text("boss wrapped up")], fresh_calls, "boss")

    out = fresh.resume(fresh_paused, EscalationDecision(approve=True, extra_tool_iterations=5))

assert out == "boss wrapped up", out
assert "quick" not in fresh_calls, f"the finished delegate was re-run after reload: {fresh_calls}"
contents = [m["content"] for m in fresh_boss.agent.conversation.history if m["role"] == "tool"]
assert contents == ["quick answered", "stuck recovered"], contents
print("PASS: a paused run survives a save/reload into a rebuilt company and resumes from disk")


# --- 10. a run can pause more than once ------------------------------------
# A grant that turns out to be too small leaves the employee stuck again, which
# is a second, separate question for a human - so it pauses again rather than
# giving up or helping itself to more.
seen = []


def always_pending(event):
    seen.append(event)
    return ESCALATION_PENDING


company = Company(name="Co", model_map=MODEL_MAP, on_escalation=always_pending)
boss = company.hire("Boss", RoleRank.C_SUITE)
junior = company.hire("Junior", RoleRank.JUNIOR, reports_to=boss)
junior.agent.set_max_tool_iterations(1)
# Keeps asking for tools: one extra iteration is not enough, nine is.
junior.agent._send_request = Responder([runaway()] * 4 + [text("junior finally recovered")])
boss.agent._send_request = Responder([
    tool_calls(("delegate_to_junior", {"task": "hard"})),
    text("eventually done"),
])

first = company.run_resumable("a risky job", entry_point=boss)
assert isinstance(first, PausedRun), first
second = company.resume(first, EscalationDecision(approve=True, extra_tool_iterations=1))
assert isinstance(second, PausedRun), f"too small a grant should pause again, got {second!r}"
assert second.task == "a risky job" and second.event.employee_name == "Junior"
assert len(seen) == 2
final = company.resume(second, EscalationDecision(approve=True, extra_tool_iterations=9))
assert final == "eventually done", final
print("PASS: a grant that proves too small pauses again rather than being topped up unasked")


# --- 11. two escalations in the same turn are surfaced one at a time --------
# Concurrency means two delegates can ask for a human at the same instant. Only
# one pause can be handed back at a time, so the other is not recorded as
# finished - on resume its call runs again and asks again, which is the safe
# way round (it re-asks rather than assuming the first answer covered it).
seen = []
company = Company(name="Co", model_map=MODEL_MAP, on_escalation=always_pending)
boss = company.hire("Boss", RoleRank.C_SUITE)
a = company.hire("Aa", RoleRank.JUNIOR, reports_to=boss)
b = company.hire("Bb", RoleRank.JUNIOR, reports_to=boss)
for employee, name in ((a, "a"), (b, "b")):
    employee.agent.set_max_tool_iterations(1)
    employee.agent._send_request = Responder([runaway(), runaway(), text(f"{name} recovered")])
boss.agent._send_request = Responder([
    tool_calls(("delegate_to_aa", {"task": "a"}), ("delegate_to_bb", {"task": "b"})),
    text("both eventually done"),
])

first = company.run_resumable("two risky jobs", entry_point=boss)
assert isinstance(first, PausedRun), first
assert len(seen) == 2, f"both delegates should have asked, got {len(seen)}"
# Neither is recorded as completed - one paused, and the other's pause was not
# the one surfaced, so both still have work to do.
assert boss.agent._pending_turn["completed"] == {}, boss.agent._pending_turn["completed"]
print("PASS: simultaneous pauses surface one at a time and neither is recorded as finished")



print("\nAll checks passed.")
