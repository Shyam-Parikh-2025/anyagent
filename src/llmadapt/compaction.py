"""compaction.py - Phase 7: the "keep broad concepts, drop specifics" cleanup
pass over a Company's activity_log and tool_call_log.

Phase 2 made those logs queryable; nothing pruned them. A long-running company
accumulates one activity entry per hire/task/escalation and one tool-call entry
per delegation, each carrying a 200-character result preview - which is fine
for an afternoon and a problem for a process that runs for days.

**This reuses compressor.py rather than growing a second compression system**,
which was the explicit instruction for this phase. Two kinds of reuse:

- *The code*: `ContextCompressor.compress_tool_output` shrinks the result
  previews, `_dedupe_repeated_lines` collapses repeated output, and
  `token_estimate` measures what compaction saved.
- *The pattern*: `LogCompactionPolicy` is deliberately shaped like
  `HistoryCompactionPolicy` - same three modes ("off"/"algorithmic"/"agent"),
  same `keep_recent` untouched tail, same trigger threshold, same
  summarizer-optional contract, same "a typo degrades to the free/safe path"
  rule, same "agent mode falls back to algorithmic if the summarizer fails, so
  choosing it can never break a run". Someone who has configured history
  compaction already knows how to configure this.

The one genuinely new decision here is what compaction is *not allowed* to
touch. Log compaction is lossy and irreversible, and these logs are the only
record of where the money went and who authorized crossing a budget ceiling.
`ALWAYS_KEEP_KINDS` protects that audit trail: escalations, escalation
decisions, emergency-reserve draws and budget reallocations are never folded
into a rollup, at any mode, regardless of age. Everything else is fair game.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .compressor import ContextCompressor

# Event kinds that are never compacted away, no matter how old.
#
# The rule for what belongs here: if losing the individual event would make it
# impossible to answer "who approved this spend?" or "what went wrong?" after
# the fact, it stays. Volume events (hire, task_start/end, plan_step,
# stub_filled) are exactly what compaction exists to collapse; authority and
# failure events are exactly what it must not.
ALWAYS_KEEP_KINDS: Tuple[str, ...] = (
    "escalation",
    "escalation_decision",
    "emergency_reserve_used",
    "emergency_budget_used",
    "budget_reallocated",
    "review_unresolved",
)

# The rollup event kind produced by compaction. Queryable like any other event
# via EventLog.by_kind("compacted").
COMPACTED_KIND = "compacted"


def _render_events(events: Sequence[Dict[str, Any]]) -> str:
    """Old events as plain text, for the agent-mode summarizer to read."""
    lines = []
    for event in events:
        kind = event.get("kind", "?")
        who = event.get("employee", "")
        detail = " ".join(
            f"{k}={v}" for k, v in event.items()
            if k not in ("time", "kind", "employee") and v not in (None, "", [], {})
        )
        lines.append(f"{kind} {who} {detail}".strip())
    return "\n".join(lines)


def _summarize_structurally(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The free, can't-fail summary: counts, not content.

    This is the "broad concepts, drop specifics" rule made literal - how many
    of each kind of thing happened, and who was involved, with every task
    string, message and result dropped. It cannot know that "the failure in
    event 12 caused the escalation in event 19" the way a real summary can,
    and it never mangles anything or costs a token.
    """
    by_kind: Dict[str, int] = {}
    employees: List[str] = []
    for event in events:
        kind = event.get("kind", "?")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        who = event.get("employee")
        if who and who not in employees:
            employees.append(who)
    return {"by_kind": by_kind, "employees": employees}


@dataclass
class LogCompactionPolicy:
    """Config + dispatch for compacting a Company's logs. Deliberately the same
    shape as compressor.HistoryCompactionPolicy - see the module docstring.

    mode="off" (the default): never touches the logs. Exactly the pre-Phase-7
        behavior if nothing sets this.
    mode="algorithmic": collapses old events into one rollup carrying counts
        per kind and the employees involved. No model call, can't fail, free.
    mode="agent": additionally asks a summarizer for a prose digest of what
        those old events were about, stored alongside the counts. Better (a
        model can tell what mattered), at the cost of one model call whenever
        compaction triggers. Falls back to algorithmic if the summarizer is
        missing or raises, so choosing "agent" is never a way to break a run.

    keep_recent: how many of the most recent events are left completely
        untouched. Recent events are the ones most likely to still be
        relevant to what is happening right now.
    trigger_events: compaction only runs once there are more than this many
        events - below it, compact() is a no-op, because paying anything to
        tidy a 30-entry log is not worth it.
    result_preview_chars: tool-call previews in the *kept* tail are trimmed to
        this via ContextCompressor.compress_tool_output. Previews are where
        tool_call_log's bulk actually lives.
    protect_kinds: event kinds never compacted. Defaults to ALWAYS_KEEP_KINDS
        (the spend/authority audit trail). Passing your own *replaces* the
        default - pass `ALWAYS_KEEP_KINDS + ("mine",)` to extend it.
    """

    mode: str = "off"
    keep_recent: int = 25
    trigger_events: int = 100
    result_preview_chars: int = 80
    summarizer: Optional[Callable[[str, int], str]] = None
    summary_budget_chars: int = 800
    protect_kinds: Sequence[str] = field(default_factory=lambda: ALWAYS_KEEP_KINDS)

    # -- activity log ------------------------------------------------------

    def compact_activity(self, events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns a compacted copy of an activity log. Never mutates the input.

        Split is: the last `keep_recent` events are kept verbatim, protected
        kinds are kept verbatim wherever they appear, and everything else older
        than the tail is folded into a single rollup event placed where those
        events were.
        """
        if self.mode == "off" or not events:
            return list(events)
        if len(events) <= max(self.trigger_events, self.keep_recent):
            return list(events)

        head = list(events[: len(events) - self.keep_recent])
        tail = list(events[len(events) - self.keep_recent:])

        protected = [e for e in head if e.get("kind") in self.protect_kinds]
        collapsible = [e for e in head if e.get("kind") not in self.protect_kinds]
        if not collapsible:
            return list(events)

        rollup = self._rollup(collapsible)
        # Protected events keep their original relative order, and the rollup
        # goes first so the log still reads chronologically enough to follow.
        return [rollup] + protected + tail

    def _rollup(self, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        structural = _summarize_structurally(events)
        rollup: Dict[str, Any] = {
            "time": events[0].get("time", 0),
            "kind": COMPACTED_KIND,
            "covers": len(events),
            "from_time": events[0].get("time", 0),
            "to_time": events[-1].get("time", 0),
            "by_kind": structural["by_kind"],
            "employees": structural["employees"],
        }
        if self.mode == "agent" and self.summarizer is not None:
            try:
                text = ContextCompressor._dedupe_repeated_lines(_render_events(events))
                summary = self.summarizer(text, self.summary_budget_chars)
                if summary and str(summary).strip():
                    rollup["summary"] = str(summary).strip()[: self.summary_budget_chars]
            except Exception as e:
                # Never let a summarizer failure cost the caller their logs -
                # the structural rollup is already complete and correct.
                rollup["summary_error"] = f"{e.__class__.__name__}: {e}"
        return rollup

    # -- tool-call log -----------------------------------------------------

    def compact_tool_calls(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns a compacted copy of a tool-call log. Never mutates the input.

        Older entries are grouped by (employee, tool_name) and collapsed into
        one row per pair carrying the call count, total and mean duration, the
        error count, and one representative (compressed) preview - which is
        the "broad concepts" a caller actually wants from a thousand identical
        delegate_to_worker calls. Any entry that recorded an error is kept
        verbatim regardless of age: an error is the specific thing you go back
        to the log to find.
        """
        if self.mode == "off" or not entries:
            return list(entries)
        if len(entries) <= max(self.trigger_events, self.keep_recent):
            return list(entries)

        head = list(entries[: len(entries) - self.keep_recent])
        tail = [self._trim_entry(e) for e in entries[len(entries) - self.keep_recent:]]

        errored = [e for e in head if e.get("error")]
        groupable = [e for e in head if not e.get("error")]
        if not groupable:
            return list(entries)

        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for entry in groupable:
            key = (entry.get("employee", "?"), entry.get("tool_name", "?"))
            groups.setdefault(key, []).append(entry)

        rolled: List[Dict[str, Any]] = []
        for (employee, tool_name), group in groups.items():
            durations = [float(e.get("duration_s") or 0.0) for e in group]
            preview = ContextCompressor.compress_tool_output(
                str(group[0].get("result_preview", "")), max_chars=self.result_preview_chars
            )
            rolled.append({
                "time": group[0].get("time", 0),
                "kind": COMPACTED_KIND,
                "employee": employee,
                "tool_name": tool_name,
                "calls": len(group),
                "total_duration_s": round(sum(durations), 4),
                "mean_duration_s": round(sum(durations) / len(durations), 4) if durations else 0.0,
                "errors": 0,
                "example_result_preview": preview,
            })
        rolled.sort(key=lambda r: r["time"])
        return rolled + errored + tail

    def _trim_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Shrink a kept tool-call entry's result preview without dropping the
        entry. Previews are where this log's bulk lives, so trimming them is
        most of the win even for entries too recent to collapse."""
        preview = entry.get("result_preview")
        if not isinstance(preview, str) or len(preview) <= self.result_preview_chars:
            return dict(entry)
        trimmed = dict(entry)
        trimmed["result_preview"] = ContextCompressor.compress_tool_output(
            preview, max_chars=self.result_preview_chars
        )
        return trimmed

    # -- the one call a Company makes --------------------------------------

    def compact_company(self, company: Any) -> Dict[str, Any]:
        """Compact both of a company's logs in place, returning a report.

        In place because `EventLog` wraps the company's list live (Phase 2's
        deliberate choice), so rebinding `company.activity_log` to a new list
        would leave any EventLog a caller is holding pointed at the old one.
        Slice-assignment keeps every existing reference valid.
        """
        before = {
            "activity_events": len(company.activity_log),
            "tool_calls": len(company.tool_call_log),
            "approx_tokens": self.estimate_tokens(company),
        }
        if self.mode == "off":
            return {"mode": "off", "compacted": False, "before": before, "after": before}

        company.activity_log[:] = self.compact_activity(company.activity_log)
        company.tool_call_log[:] = self.compact_tool_calls(company.tool_call_log)

        after = {
            "activity_events": len(company.activity_log),
            "tool_calls": len(company.tool_call_log),
            "approx_tokens": self.estimate_tokens(company),
        }
        return {
            "mode": self.mode,
            "compacted": after != before,
            "before": before,
            "after": after,
            "events_dropped": before["activity_events"] - after["activity_events"],
            "tool_calls_dropped": before["tool_calls"] - after["tool_calls"],
        }

    @staticmethod
    def estimate_tokens(company: Any) -> int:
        """Rough token size of both logs, via ContextCompressor's own estimator
        - the same one CompressionPolicy budgets against, so the numbers are
        comparable across the library."""
        text = "\n".join(
            str(e) for e in list(company.activity_log) + list(company.tool_call_log)
        )
        return ContextCompressor.token_estimate(text)


__all__ = ["LogCompactionPolicy", "ALWAYS_KEEP_KINDS", "COMPACTED_KIND"]
