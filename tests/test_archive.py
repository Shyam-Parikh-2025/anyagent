"""test_archive.py - archive.RunArchive, and the two writers that feed it.

The archive exists because compaction is lossy in two places: Phase 7 collapses
the company's activity/tool-call logs, and HistoryCompactionPolicy rewrites the
conversation. So the assertions that matter here are not "does it write a file"
but "does the record survive the thing that destroys the in-memory copy", and
"does it stay out of the way when nobody asked for it".

Same style as the rest of the suite: plain asserts + prints, no pytest, no
network - a FakeResponder stands in for the HTTP layer.
"""

import json
import os
import shutil
import tempfile

from llmadapt import Agent, Company, EscalationDecision, HistoryCompactionPolicy, RunArchive
from llmadapt.compaction import LogCompactionPolicy
from llmadapt.router import RoleRank

checks = 0
TMP = tempfile.mkdtemp(prefix="llmadapt-archive-")


def check(label, condition):
    global checks
    assert condition, label
    checks += 1
    print(f"PASS: {label}")


def path_for(name):
    return os.path.join(TMP, name)


def text_response(text, tokens=5):
    return {"content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": tokens, "output_tokens": tokens}}


class FakeResponder:
    def __init__(self, response=None):
        self.response = response or text_response("ok")
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.response


MODEL_MAP = {
    RoleRank.MANAGER: {"provider": "anthropic", "model": "claude-x", "api_key": "k"},
    RoleRank.JUNIOR: {"provider": "anthropic", "model": "claude-x", "api_key": "k"},
}


def make_company(**kwargs):
    return Company(name="Archive Co", model_map=MODEL_MAP,
                   on_escalation=lambda e: EscalationDecision(approve=False), **kwargs)


# --- 1. the mechanism ------------------------------------------------------

archive = RunArchive(path_for("basic.jsonl"))
archive.append("activity", {"kind": "hire", "employee": "Ada"})
archive.append("tool_call", {"tool_name": "delegate_to_ada", "error": None})
check("records are readable back, filtered by stream",
      len(archive.read("activity")) == 1 and len(archive.read("tool_call")) == 1)

# An activity event's own "kind" field must survive. An earlier version used
# "kind" for the archive's own label too, which silently overwrote it and made
# every archived activity record unfilterable.
check("a payload's own 'kind' field is not clobbered by the archive's label",
      archive.read("activity")[0]["kind"] == "hire")

check("every record carries a timestamp and its stream",
      all("time" in r and "stream" in r for r in archive.read()))

# JSONL, not JSON: a truncated file must still be readable up to the last
# complete line, which is the whole reason for the format.
with open(archive.path, "a", encoding="utf-8") as handle:
    handle.write('{"stream": "activity", "kind": "trunc')  # a crash mid-write
check("a half-written final line is skipped, not fatal", len(archive.read()) == 2)

# Nothing may break a run - not even the archive's own failure.
broken = RunArchive(path_for("nested/dir/is/fine.jsonl"))
broken.append("activity", {"kind": "ok"})
check("a nested path is created rather than failing", broken.records_written == 1)

unwritable = RunArchive(os.path.join(TMP, "basic.jsonl", "not-a-dir.jsonl"))
unwritable.append("activity", {"kind": "doomed"})
check("an unwritable path disables the archive instead of raising",
      unwritable.disabled and unwritable.records_written == 0)

# Lazy: configured but never written to means no file on disk.
unused = RunArchive(path_for("never-used.jsonl"))
check("an archive that is never written to creates no file",
      not os.path.exists(unused.path))

# --- 2. deletion is opt-in and narrow -------------------------------------

keeper = RunArchive(path_for("keeper.jsonl"))          # no delete_on_clean_exit
keeper.append("activity", {"kind": "x"})
keeper.close(clean=True)
check("a clean exit does NOT delete an archive that wasn't asked to self-delete",
      os.path.exists(keeper.path))

deleter = RunArchive(path_for("deleter.jsonl"), delete_on_clean_exit=True)
deleter.append("activity", {"kind": "x"})
deleter.close(clean=False)
check("an unclean exit keeps the file - that is the run you most want it for",
      os.path.exists(deleter.path))

deleter2 = RunArchive(path_for("deleter2.jsonl"), delete_on_clean_exit=True)
deleter2.append("activity", {"kind": "x"})
deleter2.close(clean=True)
check("a clean exit deletes it when asked", not os.path.exists(deleter2.path))

with RunArchive(path_for("ctx.jsonl"), delete_on_clean_exit=True) as ctx:
    ctx.append("activity", {"kind": "x"})
check("the context manager treats a clean block as a clean exit",
      not os.path.exists(ctx.path))

raised = RunArchive(path_for("raised.jsonl"), delete_on_clean_exit=True)
try:
    with raised:
        raised.append("activity", {"kind": "x"})
        raise RuntimeError("boom")
except RuntimeError:
    pass
check("an exception out of the context manager is never a clean exit",
      os.path.exists(raised.path))

# --- 3. it survives what compaction destroys ------------------------------

live = RunArchive(path_for("live.jsonl"))
company = make_company(
    archive=live,
    log_compaction=LogCompactionPolicy(mode="algorithmic", keep_recent=1, trigger_events=1),
)
manager = company.hire("Manager", RoleRank.MANAGER)
manager.agent._send_request = FakeResponder()
for i in range(4):
    company.run(f"task {i}", entry_point=manager)

archived_activity = live.read("activity")
check("the archive keeps every activity event the in-memory log compacted away",
      len(archived_activity) > len(company.activity_log))
check("the conversation transcript is archived too, from the hired agent",
      len(live.read("message")) >= 4)
check("archived messages carry role and content, not a summary",
      any(m.get("role") == "user" and m.get("content", "").startswith("task")
          for m in live.read("message")))

# --- 4. off by default -----------------------------------------------------

quiet = make_company()
quiet_manager = quiet.hire("Quiet", RoleRank.MANAGER)
quiet_manager.agent._send_request = FakeResponder()
quiet.run("a task", entry_point=quiet_manager)
check("with no archive, Company and Agent write nothing anywhere",
      quiet.archive is None and quiet_manager.agent.archive is None)
quiet.finish()  # a no-op, and must not raise
check("finish() on a company with no archive is a harmless no-op", True)

# --- 5. Company.finish() decides what "clean" means ------------------------

finisher = RunArchive(path_for("finisher.jsonl"), delete_on_clean_exit=True)
finished_co = make_company(archive=finisher)
finished_co.hire("Someone", RoleRank.JUNIOR)
finished_co._log("escalation", employee="Someone", resolved=False)
finished_co.finish(clean=True)
check("finish() refuses to call a run clean when an escalation went unresolved",
      os.path.exists(finisher.path))

ok_archive = RunArchive(path_for("ok.jsonl"), delete_on_clean_exit=True)
ok_co = make_company(archive=ok_archive)
ok_co.hire("Someone", RoleRank.JUNIOR)
ok_co._log("escalation", employee="Someone", resolved=True)
ok_co.finish(clean=True)
check("a run whose escalations were all resolved does count as clean",
      not os.path.exists(ok_archive.path))

# --- 6. the history-compaction writer -------------------------------------

hist_archive = RunArchive(path_for("history.jsonl"))
agent = Agent(provider="anthropic", model="claude-x", api_key="k",
              system_instruction="You are a build assistant.",
              history_compaction=HistoryCompactionPolicy(
                  mode="algorithmic", keep_recent_rounds=2, trigger_tokens=50),
              archive=hist_archive)
agent._send_request = FakeResponder()

# A conversation with real tool rounds in it, which is the shape compaction is
# actually for - two short turns would not exceed the trigger, and a test that
# passed on those would not be testing compaction.
for i in range(6):
    agent.conversation.add_user_msg(f"Please check on task {i}.")
    agent.conversation.add_model_msg(
        tool_calls=[{"id": f"c{i}", "function": {"name": "run_tests",
                                                 "arguments": {"suite": f"suite_{i}"}}}])
    agent.conversation.add_tool_response(
        "run_tests", f"Running suite_{i}...\n" + "PASS test_case\n" * 30 + "2 failures.",
        tool_call_id=f"c{i}")

before = len(agent.conversation.history)
agent._maybe_compact_history()
after = len(agent.conversation.history)
check("history compaction actually runs once wired into Agent", after < before)
check("the lossy step is recorded in the archive",
      len(hist_archive.read("history_compacted")) == 1)

# The point of the archive: what compaction removed is still recoverable.
# Algorithmic compaction keeps the user's question and collapses the bulky tool
# output around it, so the tool result is the thing to check - asserting on the
# question would pass without compaction having removed anything at all.
live_contents = [str(m.get("content", "")) for m in agent.conversation.history]
check("the compacted history no longer holds the early tool output in full",
      not any("PASS test_case\nPASS test_case" in c for c in live_contents[:4]))
check("but the archive still holds it verbatim",
      any("PASS test_case" in str(m.get("content", "")) and "suite_0" in str(m.get("content", ""))
          for m in hist_archive.read("message")))
check("the system instruction is never compacted away",
      agent.conversation.history[0].get("role") == "system")

# The default has to stay exactly as it was: no policy, nothing touched.
plain_agent = Agent(provider="anthropic", model="claude-x", api_key="k")
plain_agent._send_request = FakeResponder()
for i in range(4):
    plain_agent.chat(f"question {i}")
check("an Agent with no history_compaction never compacts anything",
      plain_agent.history_compaction is None and len(plain_agent.conversation.history) == 8)

# A broken policy must not be a way to break a run.
class ExplodingPolicy:
    mode = "algorithmic"

    def compact(self, history):
        raise RuntimeError("policy is broken")


exploding_agent = Agent(provider="anthropic", model="claude-x", api_key="k",
                        history_compaction=ExplodingPolicy())
exploding_agent._send_request = FakeResponder()
check("a compaction policy that raises is logged and skipped, not fatal",
      exploding_agent.chat("still works") == "ok")

# --- 7. pinned context: what compaction may never touch --------------------
# Before pins, the ONLY protected thing was a leading system message - so an
# employee's identity survived and everything agreed since did not. These check
# the guarantee itself, and the ordering, which is the part that would go wrong
# quietly: a pinned turn from step 2 re-emitted after a digest covering steps
# 1-9 reads to the model as though it happened later.

pin_agent = Agent(provider="anthropic", model="claude-x", api_key="k",
                  system_instruction="You are Ada, a senior engineer.",
                  history_compaction=HistoryCompactionPolicy(
                      mode="algorithmic", keep_recent_rounds=2, trigger_tokens=50))
pin_agent._send_request = FakeResponder()
pin_agent.pin_context("HARD CONSTRAINT: the public API must stay compatible with v1.",
                      reason="agreed with the user at the start")

for i in range(6):
    pin_agent.conversation.add_user_msg(f"Please check on task {i}.")
    pin_agent.conversation.add_model_msg(
        tool_calls=[{"id": f"p{i}", "function": {"name": "run_tests",
                                                 "arguments": {"suite": f"suite_{i}"}}}])
    pin_agent.conversation.add_tool_response(
        "run_tests", "PASS test_case\n" * 40, tool_call_id=f"p{i}")

pin_before = len(pin_agent.conversation.history)
pin_agent._maybe_compact_history()
pin_history = pin_agent.conversation.history

check("compaction still shrinks a history containing a pin",
      len(pin_history) < pin_before)
check("the pinned message survives compaction verbatim",
      any("HARD CONSTRAINT" in str(m.get("content", "")) for m in pin_history))
check("the system instruction survives too", pin_history[0].get("role") == "system")

# Chronology: the pin was added before any of the tasks, so it must still come
# before them - not after the digest that covers them.
pin_index = next(i for i, m in enumerate(pin_history)
                 if "HARD CONSTRAINT" in str(m.get("content", "")))
first_task_index = next(i for i, m in enumerate(pin_history)
                        if "task" in str(m.get("content", "")).lower())
check("a pin keeps its place in the conversation rather than being re-appended",
      pin_index < first_task_index)

# The unpinned bulk around it still got collapsed - otherwise the pin would be
# "working" only because nothing was compacted at all.
check("the unpinned tool output around the pin was still compacted",
      not any("PASS test_case\nPASS test_case" in str(m.get("content", ""))
              for m in pin_history[:pin_index + 2]))

# A pin is bookkeeping, not payload - it must never reach a provider.
for provider in ("openai", "ollama", "anthropic", "gemini"):
    exported = pin_agent.conversation.export_for(provider)
    check(f"the pin marker never reaches the {provider} payload",
          "_pinned" not in json.dumps(exported, default=str))

check("pinned() reports the system instruction and the pins together",
      len(pin_agent.conversation.pinned()) == 2)

# pin_last, for a turn that turns out to matter after it has already arrived.
pin_agent.conversation.add_user_msg("Remember: ship on Friday.")
pin_agent.conversation.pin_last(reason="deadline")
check("pin_last protects a message that already arrived normally",
      pin_agent.conversation.history[-1].get("_pinned") is True)

# Company-wide, for a constraint the whole org has to respect.
pin_co = make_company()
lead = pin_co.hire("Lead", RoleRank.MANAGER)
worker = pin_co.hire("Worker", RoleRank.JUNIOR)
told = pin_co.pin_context("All output must be valid JSON.", reason="downstream parser")
check("Company.pin_context reaches every employee", sorted(told) == ["Lead", "Worker"])
check("...and each employee's own history carries it",
      all(any("valid JSON" in str(m.get("content", "")) for m in e.agent.conversation.history)
          for e in (lead, worker)))
check("pinning is recorded in the activity log, so provenance survives",
      any(e["kind"] == "pinned_context" for e in pin_co.activity()))

named = pin_co.pin_context("Only the lead needs this.", employees=["Lead"])
check("pinning can target named employees", named == ["Lead"])
check("an unknown employee name is skipped, not fatal",
      pin_co.pin_context("nobody", employees=["Ghost"]) == [])


shutil.rmtree(TMP, ignore_errors=True)
print(f"\nOK - {checks} checks")
