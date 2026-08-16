# Test made with AI to speed up validation, same style as test_company.py:
# plain asserts + prints, no pytest, fully offline against a FakeResponder.
#
# Covers Phase C: a manager's delegations actually run at the same time, one
# employee still only does one thing at a time, the budget gate does not
# deadlock or double-grant under concurrency, and on_escalation may be sync or
# async.
import asyncio
import json
import threading
import time

from llmadapt.company import Company, EscalationDecision, EscalationUnresolved
from llmadapt.router import RoleRank


def text_response(t):
    return {"content": [{"type": "text", "text": t}]}


def tool_calls_response(*calls):
    """One assistant turn asking for several tools at once."""
    return {"content": [{"type": "tool_use", "id": f"c{i}", "name": name, "input": args}
                        for i, (name, args) in enumerate(calls)]}


class SlowResponder:
    """Stands in for the HTTP layer, taking real wall-clock time and recording
    when it was inside a call so overlap can be measured rather than assumed."""

    def __init__(self, responses, delay=0.0, log=None, label=""):
        self.responses = list(responses)
        self.delay = delay
        self.log = log if log is not None else []
        self.label = label
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        self.log.append(("enter", self.label, time.monotonic()))
        if self.delay:
            time.sleep(self.delay)
        self.log.append(("exit", self.label, time.monotonic()))
        return self.responses.pop(0)


MODEL_MAP = {
    RoleRank.C_SUITE: {"provider": "anthropic", "model": "claude-x", "api_key": "k"},
    RoleRank.MANAGER: {"provider": "anthropic", "model": "claude-x", "api_key": "k"},
    RoleRank.JUNIOR: {"provider": "anthropic", "model": "claude-x", "api_key": "k"},
}


def make_company(**kwargs):
    kwargs.setdefault("on_escalation", lambda e: EscalationDecision(approve=False))
    return Company(name="Test Co", model_map=MODEL_MAP, **kwargs)


DELAY = 0.15

# --- 1. three delegations in one turn overlap in time ----------------------
company = make_company()
boss = company.hire("Boss", RoleRank.C_SUITE)
juniors = [company.hire(f"J{i}", RoleRank.JUNIOR, reports_to=boss) for i in range(3)]
shared_log = []
for i, j in enumerate(juniors):
    j.agent._send_request = SlowResponder([text_response(f"j{i} done")],
                                          delay=DELAY, log=shared_log, label=f"j{i}")
boss.agent._send_request = SlowResponder([
    tool_calls_response(("delegate_to_j0", {"task": "a"}),
                        ("delegate_to_j1", {"task": "b"}),
                        ("delegate_to_j2", {"task": "c"})),
    text_response("all three came back"),
])

started = time.monotonic()
assert company.run("split this three ways", entry_point=boss) == "all three came back"
elapsed = time.monotonic() - started

assert elapsed < DELAY * 3, f"delegations still look sequential: {elapsed:.3f}s for 3x{DELAY}s"
# Overlap directly, not just via the clock: at some moment more than one junior
# was inside its request.
depth = concurrent_peak = 0
for kind, _label, _t in sorted(shared_log, key=lambda e: e[2]):
    depth += 1 if kind == "enter" else -1
    concurrent_peak = max(concurrent_peak, depth)
assert concurrent_peak == 3, f"expected 3 juniors in flight at once, peaked at {concurrent_peak}"
print(f"PASS: three delegations in one turn ran concurrently ({elapsed:.2f}s, not {DELAY * 3:.2f}s)")

# ...and every result still landed against the right call, in order.
tool_msgs = [m for m in boss.agent.conversation.history if m.get("role") == "tool"]
assert [m["content"] for m in tool_msgs] == ["j0 done", "j1 done", "j2 done"], tool_msgs
print("PASS: concurrent tool results are written back in call order, not completion order")


# --- 2. one employee still only does one thing at a time -------------------
# A manager is free to name the same report twice in one turn. Their Agent is
# one Conversation, so those two tasks must not interleave into it.
company = make_company()
boss = company.hire("Boss", RoleRank.C_SUITE)
solo = company.hire("Solo", RoleRank.JUNIOR, reports_to=boss)
solo_log = []
solo.agent._send_request = SlowResponder([text_response("first"), text_response("second")],
                                         delay=DELAY, log=solo_log, label="solo")
boss.agent._send_request = SlowResponder([
    tool_calls_response(("delegate_to_solo", {"task": "one"}),
                        ("delegate_to_solo", {"task": "two"})),
    text_response("both done"),
])
assert company.run("give the same person two jobs", entry_point=boss) == "both done"

depth = peak = 0
for kind, _label, _t in sorted(solo_log, key=lambda e: e[2]):
    depth += 1 if kind == "enter" else -1
    peak = max(peak, depth)
assert peak == 1, f"one employee ran two tasks at once (peak {peak}) - their history would interleave"
roles = [m["role"] for m in solo.agent.conversation.history]
assert roles == ["user", "assistant", "user", "assistant"], roles
print("PASS: two tasks handed to the same employee serialize, keeping their conversation coherent")


# --- 3. a declined escalation in one of several parallel delegates stops it --
company = make_company(on_escalation=lambda e: EscalationDecision(approve=False))
boss = company.hire("Boss", RoleRank.C_SUITE)
good = company.hire("Good", RoleRank.JUNIOR, reports_to=boss)
bad = company.hire("Bad", RoleRank.JUNIOR, reports_to=boss)
good.agent._send_request = SlowResponder([text_response("fine")], delay=0.02)
bad.agent.set_max_tool_iterations(1)
bad.agent._send_request = SlowResponder([
    {"content": [{"type": "tool_use", "id": "x", "name": "nope", "input": {}}]},
    {"content": [{"type": "tool_use", "id": "x", "name": "nope", "input": {}}]},
])
boss.agent._send_request = SlowResponder([
    tool_calls_response(("delegate_to_good", {"task": "a"}), ("delegate_to_bad", {"task": "b"})),
    text_response("the manager should never answer"),
])
try:
    company.run("two jobs, one goes wrong", entry_point=boss)
    assert False, "a declined escalation must still stop the run when it happens in parallel"
except EscalationUnresolved as e:
    assert e.event.employee_name == "Bad"
print("PASS: a declined escalation among parallel delegations still stops the run")


# --- 4. an ordinary failure in one delegate does NOT stop its siblings ------
def explodes(x: str) -> str:
    """Always raises."""
    raise ValueError("nope")


company = make_company()
boss = company.hire("Boss", RoleRank.C_SUITE)
ok_junior = company.hire("Okj", RoleRank.JUNIOR, reports_to=boss)
ok_junior.agent._send_request = SlowResponder([text_response("still fine")], delay=0.02)
boss.agent.add_tool(explodes)
boss.agent._send_request = SlowResponder([
    tool_calls_response(("delegate_to_okj", {"task": "a"}), ("explodes", {"x": "1"})),
    text_response("carried on"),
])
assert company.run("one good, one broken", entry_point=boss) == "carried on"
contents = [m["content"] for m in boss.agent.conversation.history if m.get("role") == "tool"]
assert contents[0] == "still fine", contents
assert "Tool Execution Failure (ValueError)" in contents[1], contents
print("PASS: a crashing tool becomes a message for the model without cancelling its siblings")


# --- 5. the budget gate survives concurrency without deadlocking ------------
# This is the exact shape that would deadlock if the budget lock were held
# across an escalation: the entry point is gated, a human approves, the retry
# runs - and that retry delegates three ways, so three more gates open while
# the first one's call is still on the stack. A lock held across the escalation
# would have that task waiting on a lock it already owns, which hangs silently
# rather than raising, so this is bounded by a watchdog thread.
#
# Only the entry point escalates here: once it is granted headroom, the
# delegates borrow from it through _try_reallocate, which is the documented
# behaviour (that function only ever adds headroom and never debits the lender
# - the company-wide ceiling is the real backstop, not this bookkeeping).
approvals = []


def approve(event):
    approvals.append(event)
    return EscalationDecision(approve=True, extra_token_budget=10_000)


company = make_company(total_token_budget=1, on_escalation=approve)
boss = company.hire("Boss", RoleRank.C_SUITE)
kids = [company.hire(f"K{i}", RoleRank.JUNIOR, reports_to=boss) for i in range(3)]
for i, k in enumerate(kids):
    k.agent._send_request = SlowResponder([text_response(f"k{i} ok"), text_response(f"k{i} ok")],
                                          delay=0.02)
boss.agent._send_request = SlowResponder([
    tool_calls_response(("delegate_to_k0", {"task": "a"}),
                        ("delegate_to_k1", {"task": "b"}),
                        ("delegate_to_k2", {"task": "c"})),
    text_response("budget survived"),
])

done = []
runner = threading.Thread(target=lambda: done.append(company.run("go", entry_point=boss)), daemon=True)
runner.start()
runner.join(timeout=30)
assert not runner.is_alive(), "the budget gate deadlocked under concurrent delegation"
assert done == ["budget survived"], done
assert len(approvals) == 1 and approvals[0].employee_name == "Boss", [e.employee_name for e in approvals]
# All three delegates got through the gate and ran, off the headroom the
# approval gave their manager.
delegated = sorted(e["tool_name"] for e in company.tool_call_log)
assert delegated == ["delegate_to_K0", "delegate_to_K1", "delegate_to_K2"], delegated
assert all(e["error"] is None for e in company.tool_call_log)
print("PASS: an approved escalation whose retry delegates three ways does not deadlock the budget gate")


# --- 6. on_escalation may be an async function ------------------------------
seen = []


async def async_handler(event):
    await asyncio.sleep(0)
    seen.append(event)
    return EscalationDecision(approve=True, extra_tool_iterations=5)


company = make_company(on_escalation=async_handler)
boss = company.hire("Boss", RoleRank.C_SUITE)
junior = company.hire("Junior", RoleRank.JUNIOR, reports_to=boss)
junior.agent.set_max_tool_iterations(1)
junior.agent._send_request = SlowResponder([
    {"content": [{"type": "tool_use", "id": "x", "name": "nope", "input": {}}]},
    text_response("recovered after async approval"),
])
assert junior.run("task", company=company) == "recovered after async approval"
assert len(seen) == 1
print("PASS: an `async def` on_escalation handler is awaited and its decision honoured")


# --- 7. a blocking sync on_escalation does not freeze the other employees ---
# The realistic handler sits waiting on a human. It runs on a worker thread, so
# an unrelated employee keeps working while it waits.
progress = []


def blocking_handler(event):
    time.sleep(DELAY)
    progress.append("human answered")
    return EscalationDecision(approve=True, extra_tool_iterations=5)


company = make_company(on_escalation=blocking_handler)
boss = company.hire("Boss", RoleRank.C_SUITE)
stuck = company.hire("Stuck", RoleRank.JUNIOR, reports_to=boss)
busy = company.hire("Busy", RoleRank.JUNIOR, reports_to=boss)
stuck.agent.set_max_tool_iterations(1)
stuck.agent._send_request = SlowResponder([
    {"content": [{"type": "tool_use", "id": "x", "name": "nope", "input": {}}]},
    text_response("stuck recovered"),
])


def note(_payload, _headers):
    progress.append("busy worked")
    return text_response("busy done")


busy.agent._send_request = note
boss.agent._send_request = SlowResponder([
    tool_calls_response(("delegate_to_stuck", {"task": "a"}), ("delegate_to_busy", {"task": "b"})),
    text_response("both back"),
])
assert company.run("go", entry_point=boss) == "both back"
assert progress[0] == "busy worked", (
    f"the other employee should have finished while the human was deciding: {progress}")
print("PASS: a blocking sync on_escalation runs off the loop, so other employees keep working")

# --- 8. max_parallel_tools caps how many tool calls are in flight ----------
# A model may ask for twenty tools in one turn, and with delegation wired as an
# ordinary tool that is twenty employees starting at once. The cap turns that
# into a queue. Without one the limit is not "none" - it is asyncio's default
# thread pool size, which depends on the machine.
in_flight = []
peak = [0]


def tracked(x: str) -> str:
    """A slow tool that records how many copies of itself are running."""
    in_flight.append(x)
    peak[0] = max(peak[0], len(in_flight))
    time.sleep(0.05)
    in_flight.remove(x)
    return f"done:{x}"


company = make_company()
boss = company.hire("Boss", RoleRank.C_SUITE)
boss.agent.add_tool(tracked)
assert boss.agent.max_parallel_tools == 8, "the default cap should be a number, not None"
boss.agent.set_max_parallel_tools(2)
boss.agent._send_request = SlowResponder([
    tool_calls_response(*[("tracked", {"x": str(i)}) for i in range(6)]),
    text_response("all six ran"),
])
assert company.run("six tools at once", entry_point=boss) == "all six ran"
assert peak[0] <= 2, f"max_parallel_tools=2 was exceeded: {peak[0]} in flight"
assert peak[0] == 2, f"the cap should still allow 2 at once, saw {peak[0]}"
contents = [m["content"] for m in boss.agent.conversation.history if m.get("role") == "tool"]
assert contents == [f"done:{i}" for i in range(6)], contents
print("PASS: max_parallel_tools bounds how many of one turn's tool calls run at once")


# --- 9. a cap of 1 gives back strictly sequential behaviour ---------------
in_flight.clear()
peak[0] = 0
company = make_company()
boss = company.hire("Boss", RoleRank.C_SUITE)
boss.agent.add_tool(tracked)
boss.agent.set_max_parallel_tools(1)
boss.agent._send_request = SlowResponder([
    tool_calls_response(*[("tracked", {"x": str(i)}) for i in range(4)]),
    text_response("one at a time"),
])
assert company.run("four tools", entry_point=boss) == "one at a time"
assert peak[0] == 1, f"a cap of 1 should serialize, saw {peak[0]} in flight"
print("PASS: max_parallel_tools=1 restores strictly sequential tool calls, async machinery intact")

print("\nAll checks passed.")
