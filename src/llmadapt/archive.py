"""archive.py - one append-only JSONL record of everything a run did, written
through *before* anything lossy touches it.

Two things in this library are deliberately lossy and irreversible:

- `compaction.LogCompactionPolicy` (Phase 7) collapses old activity events and
  tool calls into rollups at the end of every completed task.
- `compressor.HistoryCompactionPolicy` replaces old conversation rounds with a
  digest - and in `mode="agent"`, with prose a summarizer wrote over them.

Both are the right default. Both mean that "what exactly did this company do
for six hours?" and "what did that model actually say at turn 12?" stop being
answerable once a run gets long. Compaction already refuses to touch the spend
and authority trail (`ALWAYS_KEEP_KINDS`, and any tool call that errored), so
"who approved this?" and "what broke?" survive - what does not survive is the
ordinary texture, which is exactly what you want back when reconstructing a run
after the fact.

**One archive, two writers.** The company log sink and the conversation
transcript could each have grown their own file handling, their own disk-error
path and their own flag. They share this instead - the same "one mechanism,
instantiated N times" move Phase 5 made with `PresetRegistry`, for the same
reason: two half-identical implementations of the same thing is how they drift.

Design rules, each of which is load-bearing:

- **Off by default.** `archive_path=None` and nothing is created. This is the
  first thing in llmadapt that writes to a user's disk in normal operation
  (`benchmark.py` writes a *cache*, which is different in kind), so it happens
  only when asked.
- **Write-through, at log time.** Records are appended when the event happens,
  not when compaction runs. Archiving at compaction time would archive a view
  that is already partly collapsed, and would lose everything if the process
  died before the first compaction - which is precisely the long unattended run
  this is for.
- **It can never break a run.** Any I/O failure disables the archive with a
  warning and the run continues. A run must not die because its own audit log
  could not be written.
- **Deleting the archive is opt-in and narrow.** `delete_on_clean_exit` maps to
  "if everything went fine, clean up after yourself", and "fine" is defined
  strictly: an explicit `close(clean=True)` from a caller that got through its
  work without unresolved escalations. Deliberately not an `atexit` hook -
  `atexit` also fires on the paths where things went wrong, and silently
  deleting the record of a bad run is the exact opposite of the point.
- **JSON Lines, not JSON.** One self-contained object per line means a file
  truncated by a crash is still readable up to the last complete line, and it
  can be `grep`ped and streamed without loading it all. A single top-level JSON
  array would be unparseable the moment a run didn't finish.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["RunArchive"]


class RunArchive:
    """An append-only JSONL file for one run.

    Typical use is to hand the same instance to both writers, so one file holds
    the whole picture in timestamp order:

        archive = RunArchive("run.jsonl", delete_on_clean_exit=True)
        company = Company(..., archive=archive,
                          log_compaction=LogCompactionPolicy(mode="algorithmic"))
        agent = Agent(..., archive=archive,
                      history_compaction=HistoryCompactionPolicy(mode="agent"))
        ...
        archive.close(clean=True)   # deletes the file, having got through cleanly

    Every record carries `stream`, `time` and whatever fields the writer
    passed. `stream` is the writer's label - "activity", "tool_call",
    "message" - so one file can be filtered back apart by whoever reads it.

    It is called `stream` and not `kind` on purpose: activity events already
    carry their own `kind` field ("hire", "escalation", "model_policy"), and
    an archive field by the same name silently overwrote it, turning every
    archived activity record into an unfilterable one. The archive's own
    label has to live in a namespace the payloads do not use.
    """

    def __init__(self, path: str, delete_on_clean_exit: bool = False):
        self.path = str(path)
        self.delete_on_clean_exit = delete_on_clean_exit
        self.closed = False
        self.disabled = False
        self.records_written = 0
        # Both writers can be driven from different threads (gui.py serves
        # requests on its own thread; an async agent interleaves), and a torn
        # line would make the file unparseable at exactly the point of
        # interest. One lock around the append is enough - this is not a hot
        # path next to an HTTP round-trip to a model.
        self._lock = threading.Lock()
        self._handle = None

    # -- writing -----------------------------------------------------------

    def append(self, stream: str, record: Optional[Dict[str, Any]] = None, **fields: Any) -> None:
        """Append one record to `stream`. Never raises."""
        if self.closed or self.disabled:
            return
        # Payload first, then the archive's own fields, so a record carrying
        # its own "kind" (every activity event does) keeps it while `stream`
        # and `time` are guaranteed to be the archive's.
        entry: Dict[str, Any] = {}
        entry.update(record or {})
        entry.update(fields)
        entry["time"] = entry.get("time", time.time())
        entry["stream"] = stream
        line = self._encode(entry)
        with self._lock:
            handle = self._open()
            if handle is None:
                return
            try:
                handle.write(line + "\n")
                handle.flush()
                self.records_written += 1
            except OSError as e:
                self._disable(f"could not write to {self.path!r}: {e}")

    def append_many(self, stream: str, records: Sequence[Dict[str, Any]], **fields: Any) -> None:
        for record in records or ():
            self.append(stream, record, **fields)

    @staticmethod
    def _encode(entry: Dict[str, Any]) -> str:
        """One line of JSON, with anything unserializable stringified.

        `default=str` rather than dropping the field: a record whose payload is
        a live object should still say *something* about it. An archive that
        silently omits the interesting half of an event is worse than one whose
        entries are occasionally reprs.
        """
        try:
            return json.dumps(entry, default=str)
        except (TypeError, ValueError):  # pragma: no cover - default=str covers ~everything
            return json.dumps({"time": entry.get("time"), "stream": entry.get("stream"),
                               "error": "record could not be serialized"})

    def _open(self):
        if self._handle is not None:
            return self._handle
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Line-buffered append. Opened lazily so a configured-but-unused
            # archive never creates a file.
            self._handle = open(self.path, "a", encoding="utf-8")
        except OSError as e:
            self._disable(f"could not open {self.path!r}: {e}")
            return None
        return self._handle

    def _disable(self, message: str) -> None:
        """Give up on archiving, loudly, without touching the run."""
        self.disabled = True
        self._handle = None
        logger.warning(
            "RunArchive disabled - %s. The run continues; nothing further will be archived.",
            message,
        )

    # -- reading back ------------------------------------------------------

    def read(self, stream: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every archived record, oldest first, optionally filtered by stream.

        For tests and for a caller reconstructing a finished run. A line that
        doesn't parse (a crash mid-write) is skipped rather than raising, since
        the whole point of JSONL is that a truncated file is still useful.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return []
        out: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if stream is None or entry.get("stream") == stream:
                out.append(entry)
        return out

    # -- finishing ---------------------------------------------------------

    def close(self, clean: bool = False) -> None:
        """Close the file, and delete it if this was a clean finish.

        `clean=True` is the caller stating that the run got through its work -
        it is not inferred here, and it is not inferred from the absence of an
        exception either. `Company.finish()` decides it by checking for
        unresolved escalations, which is a question this class has no business
        answering.
        """
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:  # pragma: no cover - closing a broken handle
                    pass
                self._handle = None
            self.closed = True
            if clean and self.delete_on_clean_exit:
                try:
                    os.remove(self.path)
                    logger.debug("RunArchive: clean exit, removed %s", self.path)
                except OSError as e:  # pragma: no cover
                    logger.warning("RunArchive: could not remove %s: %s", self.path, e)

    def __enter__(self) -> "RunArchive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # An exception on the way out is never a clean exit - that is the run
        # you most want the file for.
        self.close(clean=exc_type is None)

    def __repr__(self) -> str:
        state = "disabled" if self.disabled else ("closed" if self.closed else "open")
        return f"RunArchive(path={self.path!r}, {state}, {self.records_written} records)"
