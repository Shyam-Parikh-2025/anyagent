"""state.py - a serializable snapshot of where a company currently *is*,
as opposed to what it did.

`archive.RunArchive` answers "what happened": an append-only JSONL log written
through as events occur, so a run leaves a record even if it dies. This answers
a different question - "where were we?" - and the two are deliberately not the
same file and not derived from each other:

- The archive is **off by default**. A state document that only worked for runs
  that had opted into archiving would be a resume story with a footnote.
- The archive is a *report* log, not an event-sourcing log. It records that a
  grant happened (`emergency_budget_used`), not the invariants that grant left
  behind (`_bonus_allocation`, `_emergency_budget_spent`). Replaying it would
  mean reconstructing state from prose about state.
- JSONL truncated by a crash is still readable, which is exactly right for an
  audit trail and exactly wrong for a snapshot: half a snapshot is not a
  smaller snapshot, it is a corrupt one.

**What a snapshot cannot contain, and what follows from that.** A Company holds
things that are not data: the `on_escalation` callable, every user-registered
tool in `ToolRegistry.functions_maps` (arbitrary Python functions), the model
policy, the preset registries (which support runtime `.register()`), the
compaction policies, the archive's open file handle. None of those can be
written to JSON and read back, so `CompanyState` does not pretend to be a
`Company` - it cannot be loaded on its own into a working company.

Resuming is therefore *rehydration into a live company*: build the company the
same way you built it the first time (in code, or from a `CompanySpec`), then
apply the snapshot to it. Everything that can be checked, is - the roster, each
employee's rank and reporting line, and the set of registered tool names - and a
mismatch raises rather than restoring what happens to line up. A company that is
missing a tool it had before, or has an employee reporting somewhere new, is not
the company that was saved, and silently continuing on it would produce a run
that looks resumed and is not.

**No credentials, ever.** A snapshot carries provider and model names but never
an API key, and `Company.restore_state` never sets one - keys are re-resolved
from the environment by `env.resolve_api_key` exactly as they were at hire time.
That is what makes a state file safe to commit, attach to a bug report, or move
between machines.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "STATE_VERSION",
    "CompanyState",
    "StateMismatch",
    "capture_state",
    "apply_state",
]

# Bumped whenever the document's shape changes in a way an older reader would
# get wrong. apply_state() refuses a version it does not know rather than
# guessing at fields that may have moved.
STATE_VERSION = 1

# Keys that must never appear in a snapshot, checked on the way out rather than
# trusted. Cheap insurance against a future field quietly carrying a credential
# into a file people are told is safe to share.
_FORBIDDEN_KEYS = ("api_key", "apikey", "authorization", "secret", "token_value")


class StateMismatch(Exception):
    """Raised when a snapshot does not describe the company it is being
    applied to. Carries every difference found, not just the first, because
    the useful question when this fires is "how far apart are these?" and one
    line of answer makes that take several rounds."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        detail = "\n  - ".join(self.problems)
        super().__init__(
            f"this snapshot does not match the company it was applied to:\n  - {detail}"
        )


@dataclass
class AgentState:
    """One Agent's resumable state. Everything here is either plain data or a
    counter - the tools and policies the Agent also holds are code, and are
    expected to be rebuilt by whoever rebuilds the company."""

    system_instruction: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens_used: int = 0
    usage_log: List[Dict[str, Any]] = field(default_factory=list)
    max_tool_iterations: int = 6
    max_context_tokens: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # The names only - enough to verify the same tools are present on resume,
    # and useless for smuggling anything executable into a data file.
    tool_names: List[str] = field(default_factory=list)
    # The turn this agent was in the middle of when a run paused, if any:
    # the provider response that asked for tools, plus the outputs of the tool
    # calls that had already finished. This is the piece that makes a pause
    # survive the process it happened in - without it, a resumed run would
    # re-ask the provider and re-run tools whose effects already happened.
    pending_turn: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_instruction": self.system_instruction,
            "history": self.history,
            "total_tokens_used": self.total_tokens_used,
            "usage_log": self.usage_log,
            "max_tool_iterations": self.max_tool_iterations,
            "max_context_tokens": self.max_context_tokens,
            "provider": self.provider,
            "model": self.model,
            "tool_names": sorted(self.tool_names),
            "pending_turn": self.pending_turn,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class EmployeeState:
    """One Employee's resumable state, plus the structural facts that are
    checked rather than restored (rank, reporting line) - those describe the
    company you must have rebuilt, not something a snapshot may change."""

    name: str
    rank: str = ""
    reports_to: Optional[str] = None
    importance: float = 0.5
    effort: Optional[str] = None
    specialty: Optional[str] = None
    cost_weight: Optional[float] = None
    skills: List[str] = field(default_factory=list)
    personality: Optional[str] = None
    agent: AgentState = field(default_factory=AgentState)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "rank": self.rank, "reports_to": self.reports_to,
            "importance": self.importance, "effort": self.effort,
            "specialty": self.specialty, "cost_weight": self.cost_weight,
            "skills": list(self.skills), "personality": self.personality,
            "agent": self.agent.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmployeeState":
        data = dict(data or {})
        agent = AgentState.from_dict(data.pop("agent", {}) or {})
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(agent=agent, **known)


@dataclass
class CompanyState:
    """Where a company currently is, as plain JSON-able data.

    Mirrors the `CompanySpec.to_dict()/from_dict()` shape in builder.py on
    purpose: a design file and a state file are read by the same kinds of tool
    and by the same people, and there is no reason for them to disagree about
    what a document looks like.
    """

    version: int = STATE_VERSION
    company_name: str = ""
    saved_at: float = 0.0
    # Set when the snapshot was taken while a run was paused waiting on a
    # human - `company.PausedRun.to_dict()`. Carried here so that "where are
    # we?" and "what were we waiting for?" travel as one document rather than
    # as two files somebody has to remember to keep together.
    paused_run: Optional[Dict[str, Any]] = None
    total_token_budget: Optional[int] = None
    bonus_allocation: Dict[str, int] = field(default_factory=dict)
    emergency_iteration_remaining: int = 0
    emergency_budget_spent: int = 0
    employees: List[EmployeeState] = field(default_factory=list)
    activity_log: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_log: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "company_name": self.company_name,
            "saved_at": self.saved_at,
            "paused_run": self.paused_run,
            "total_token_budget": self.total_token_budget,
            "bonus_allocation": dict(self.bonus_allocation),
            "emergency_iteration_remaining": self.emergency_iteration_remaining,
            "emergency_budget_spent": self.emergency_budget_spent,
            "employees": [e.to_dict() for e in self.employees],
            "activity_log": self.activity_log,
            "tool_call_log": self.tool_call_log,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompanyState":
        data = dict(data or {})
        employees = [EmployeeState.from_dict(e) for e in data.pop("employees", []) or []]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(employees=employees, **known)

    def to_json(self, indent: int = 2) -> str:
        """`default=str` for the same reason builder.py uses it: a snapshot must
        not fail to save because one log entry picked up an exotic value. A
        stringified oddity in a log line is recoverable; losing the whole
        snapshot at the moment you needed it is not."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, text: str) -> "CompanyState":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        return path

    @classmethod
    def load(cls, path: str) -> "CompanyState":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_json(handle.read())

    def employee(self, name: str) -> Optional[EmployeeState]:
        return next((e for e in self.employees if e.name == name), None)


def _agent_state_of(agent: Any) -> AgentState:
    return AgentState(
        system_instruction=agent.conversation.system_instruction,
        # Copied, not referenced: a snapshot taken mid-run must not keep
        # changing underneath whoever is holding it.
        history=[dict(m) for m in agent.conversation.history],
        total_tokens_used=getattr(agent, "total_tokens_used", 0),
        usage_log=[dict(u) for u in getattr(agent, "usage_log", [])],
        max_tool_iterations=getattr(agent, "max_tool_iterations", 6),
        max_context_tokens=getattr(agent, "max_context_tokens", None),
        provider=getattr(agent, "provider", None),
        model=getattr(agent, "model", None),
        tool_names=sorted(agent.tool_registry.schemas),
        pending_turn=getattr(agent, "_pending_turn", None),
    )


def capture_state(company: Any, notes: str = "") -> CompanyState:
    """Everything about `company` that can be written down. See the module
    docstring for what deliberately cannot be."""
    employees = []
    for employee in company.employees.values():
        employees.append(EmployeeState(
            name=employee.name,
            rank=employee.rank,
            reports_to=(employee.reports_to.name if employee.reports_to else None),
            importance=employee.importance,
            effort=employee.effort,
            specialty=employee.specialty,
            cost_weight=employee.cost_weight,
            skills=[getattr(s, "name", str(s)) for s in (employee.skills or [])],
            personality=(getattr(employee.personality, "name", None)
                         if employee.personality is not None else None),
            agent=_agent_state_of(employee.agent),
        ))

    state = CompanyState(
        version=STATE_VERSION,
        company_name=company.name,
        saved_at=time.time(),
        total_token_budget=company.budget.total_token_budget,
        bonus_allocation=dict(company.budget._bonus_allocation),
        emergency_iteration_remaining=company._emergency_iteration_remaining,
        emergency_budget_spent=company._emergency_budget_spent,
        employees=employees,
        activity_log=[dict(e) for e in company.activity_log],
        tool_call_log=[dict(e) for e in company.tool_call_log],
        notes=notes,
    )
    _assert_no_credentials(state)
    return state


def _assert_no_credentials(state: CompanyState) -> None:
    """A snapshot is documented as safe to share. Check that rather than
    assert it: this walks the finished document looking for anything that
    looks like a credential, so a field added later cannot quietly make the
    documentation false."""
    found: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and key.lower() in _FORBIDDEN_KEYS and value:
                    found.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(state.to_dict(), "state")
    if found:
        raise ValueError(
            "refusing to produce a company snapshot containing credential-shaped fields: "
            + ", ".join(found)
        )


def _structural_problems(company: Any, state: CompanyState) -> List[str]:
    """Every way `state` fails to describe `company`, in one pass."""
    problems: List[str] = []

    live = set(company.employees)
    saved = {e.name for e in state.employees}
    for missing in sorted(saved - live):
        problems.append(f"employee {missing!r} is in the snapshot but not in this company")
    for extra in sorted(live - saved):
        problems.append(f"employee {extra!r} is in this company but not in the snapshot")

    for saved_employee in state.employees:
        employee = company.employees.get(saved_employee.name)
        if employee is None:
            continue
        if employee.rank != saved_employee.rank:
            problems.append(
                f"{employee.name} is a {employee.rank} here but was a {saved_employee.rank} "
                f"when the snapshot was taken")
        live_manager = employee.reports_to.name if employee.reports_to else None
        if live_manager != saved_employee.reports_to:
            problems.append(
                f"{employee.name} reports to {live_manager!r} here but reported to "
                f"{saved_employee.reports_to!r} in the snapshot")
        live_tools = set(employee.agent.tool_registry.schemas)
        saved_tools = set(saved_employee.agent.tool_names)
        for missing in sorted(saved_tools - live_tools):
            problems.append(f"{employee.name} is missing the tool {missing!r} it had in the snapshot")
        for extra in sorted(live_tools - saved_tools):
            problems.append(f"{employee.name} has a tool {extra!r} that the snapshot does not know about")

    return problems


def apply_state(company: Any, state: CompanyState, strict: bool = True) -> CompanyState:
    """Restores `state` onto an already-built `company`, in place.

    `strict=True` (the default) refuses to restore anything unless the company
    matches the snapshot structurally - same roster, same ranks, same reporting
    lines, same registered tool names. Passing `strict=False` downgrades those
    to a warning-free best effort and restores whatever lines up; it exists for
    deliberate surgery (moving one employee's history into a differently-shaped
    company) and is the wrong default for resuming a run, because a partial
    restore produces a company that looks resumed and is not.
    """
    if state.version != STATE_VERSION:
        raise StateMismatch([
            f"snapshot is version {state.version}, this llmadapt reads version {STATE_VERSION}"
        ])

    if strict:
        problems = _structural_problems(company, state)
        if problems:
            raise StateMismatch(problems)

    company.budget.total_token_budget = state.total_token_budget
    company.budget._bonus_allocation = dict(state.bonus_allocation)
    company._emergency_iteration_remaining = state.emergency_iteration_remaining
    company._emergency_budget_spent = state.emergency_budget_spent

    # Assigned in place, the same way compaction.compact_company() does it and
    # for the same reason: anything already holding a reference to these lists
    # (an EventLog view, a caller mid-inspection) must not be left pointing at
    # the pre-restore copy.
    company.activity_log[:] = [dict(e) for e in state.activity_log]
    company.tool_call_log[:] = [dict(e) for e in state.tool_call_log]

    for saved_employee in state.employees:
        employee = company.employees.get(saved_employee.name)
        if employee is None:
            continue  # only reachable with strict=False
        agent = employee.agent
        agent.conversation.system_instruction = saved_employee.agent.system_instruction
        agent.conversation.history[:] = [dict(m) for m in saved_employee.agent.history]
        agent.total_tokens_used = saved_employee.agent.total_tokens_used
        agent.usage_log = [dict(u) for u in saved_employee.agent.usage_log]
        agent.set_max_tool_iterations(saved_employee.agent.max_tool_iterations)
        agent.set_max_context_tokens(saved_employee.agent.max_context_tokens)
        # Restored last, and only ever as data: an agent holding a pending turn
        # will continue it instead of asking the provider again the next time
        # anyone calls it.
        agent._pending_turn = saved_employee.agent.pending_turn
        employee.importance = saved_employee.importance
        employee.effort = saved_employee.effort
        employee.specialty = saved_employee.specialty
        employee.cost_weight = saved_employee.cost_weight

    return state
