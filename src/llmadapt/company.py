"""company.py - Phase 0-3 of the multi-agent hierarchy: Employee/Team/Company
skeleton classes, config-driven Agent creation per rank (with per-employee
overrides), delegation wired as a normal tool call, tool-iteration AND
token-budget failures both escalating up the reporting chain instead of
raising straight to the caller, (Phase 2) queryable activity/tool-call logs
plus org-chart rendering (ascii/mermaid/svg) - see observability.py - and
(Phase 3) real percentage-based token budget governance - see budget.py.

Deliberately NOT in this file yet (later phases, see the architecture doc /
company-roadmap.md):
  - the auto mode local-vs-API model policy layer (Phase 4) - for now every
    rank's provider/model is explicit in model_map (or a per-employee
    override at hire() time), "mode" is accepted but unused
  - skills/personality presets, default org templates, stub-and-fill delegation,
    and swappable color palettes for render_org_chart (Phase 5 - all four are
    meant to share one named-preset registry pattern, not be built separately)
  - the compressor.py-based log cleanup/compaction pass (still Phase 7) - logs
    are recorded and queryable now, but nothing prunes or summarizes them yet
  - set_company_up() text/GUI company builder (Phase 8)

This stays a single file for now (observability.py and budget.py split out
already, same reasoning as hardware.py/router.py/compressor.py being separate
concerns). Once Phase 4+ lands (model_policy.py, skills/, templates.py) this
should become a company/ package - flagged here so that split isn't a
surprise later.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .budget import BudgetLedger
from .core import Agent
from .hardware import ResourceQuota
from .router import RoleRank

# Default number of extra tool iterations granted per automatic emergency-reserve
# draw for a tool_iteration_limit escalation. Kept small and fixed - unrelated to
# emergency_budget_tokens below, which is the equivalent reserve for
# budget_exhausted escalations, denominated in real tokens instead of iterations.
_EMERGENCY_GRANT_ITERATIONS = 3

# What fraction of the *remaining* emergency_budget_tokens pool a single
# budget_exhausted escalation can draw in one grant. A fraction rather than a
# fixed constant (unlike _EMERGENCY_GRANT_ITERATIONS above) because token
# reserves span orders of magnitude across different companies' configs, while
# tool-iteration reserves don't - this keeps one runaway employee from
# draining the whole reserve in a single automatic grant.
_EMERGENCY_BUDGET_DRAW_FRACTION = 0.25


@dataclass
class EscalationEvent:
    """Something an Employee couldn't resolve on its own and is passing up
    the chain. kind is "tool_iteration_limit" (a runaway-loop safety cap) or
    "budget_exhausted" (a resource/cost cap) - genuinely different failure
    modes, handled by separate reserves and separate recovery logic in
    Company._handle_top_level_escalation. More kinds (unhandled tool errors,
    explicit "I don't know") are future work."""

    kind: str
    employee_name: str
    rank: str
    message: str
    detail: Optional[str] = None


@dataclass
class EscalationDecision:
    """What the human (via Company.on_escalation) decided to do about an
    EscalationEvent that made it all the way to the top with no automatic
    resolution left. extra_tool_iterations answers a tool_iteration_limit
    event; extra_token_budget answers a budget_exhausted event - set
    whichever matches event.kind."""

    approve: bool
    note: Optional[str] = None
    extra_tool_iterations: Optional[int] = None
    extra_token_budget: Optional[int] = None


class EscalationUnresolved(Exception):
    """Raised when an EscalationEvent reaches the top of the company and
    on_escalation declines to approve a way forward."""

    def __init__(self, event: EscalationEvent, decision: Optional[EscalationDecision] = None):
        self.event = event
        self.decision = decision
        super().__init__(f"Unresolved escalation from {event.employee_name} ({event.rank}): {event.message}")


class Employee:
    """One Agent bound to a rank, with a reporting line for escalation and
    delegation wired through it."""

    def __init__(
        self,
        name: str,
        rank: str,
        agent: Agent,
        reports_to: Optional["Employee"] = None,
        importance: float = 0.5,
    ):
        self.name = name
        self.rank = rank
        self.agent = agent
        self.reports_to = reports_to
        self.importance = max(0.0, min(1.0, importance))
        self._subordinates: List["Employee"] = []
        if reports_to is not None:
            reports_to._subordinates.append(self)

    @property
    def subordinates(self) -> List["Employee"]:
        return list(self._subordinates)

    def delegate_tool(self, company: Optional["Company"] = None) -> Callable[[str], str]:
        """A function shaped for Agent.add_tool - gives whoever holds it a way
        to hand this employee a task directly through the normal tool-calling
        loop, instead of any new delegation-specific machinery. If a company
        is given, every call is recorded to its tool_call_log (Phase 2)."""

        def _delegate(task: str) -> str:
            """Delegate a task to {name} ({rank})."""
            start = time.time()
            error = None
            result = ""
            try:
                result = self.run(task, company=company)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                if company is not None:
                    from .observability import record_tool_call

                    record_tool_call(
                        company.tool_call_log,
                        employee=self.reports_to.name if self.reports_to else "?",
                        tool_name=f"delegate_to_{self.name}",
                        args={"task": task, "delegated_to": self.name},
                        result=result,
                        duration_s=time.time() - start,
                        error=error,
                    )

        safe_name = "".join(c if c.isalnum() else "_" for c in self.name.lower())
        _delegate.__name__ = f"delegate_to_{safe_name}"
        _delegate.__doc__ = f"Delegate a task to {self.name} ({self.rank})."
        return _delegate

    def run(self, task: str, company: Optional["Company"] = None) -> str:
        """Runs task through this employee's Agent. Checked against the
        company's token budget BEFORE starting (see Company._budget_gate) -
        if that gate already had to escalate and resolve things, its result
        is returned directly rather than running the task twice. Otherwise, a
        tool-iteration overflow is caught here and routed up the reporting
        chain instead of raising straight out to whoever called run() (a
        manager delegating, or Company.run() itself)."""
        if company is not None:
            gated_result = company._budget_gate(self, task)
            if gated_result is not None:
                return gated_result
        try:
            return self.agent.chat(task)
        except RuntimeError as e:
            if "max_tool_iterations" not in str(e):
                raise  # not an iteration-limit failure - not ours to handle here
            event = EscalationEvent(
                kind="tool_iteration_limit",
                employee_name=self.name,
                rank=self.rank,
                message=f"{self.name} ({self.rank}) hit its tool-iteration limit working on: {task[:120]!r}",
                detail=str(e),
            )
            return self._escalate(event, task, company)

    def _escalate(self, event: EscalationEvent, task: str, company: Optional["Company"]) -> str:
        if self.reports_to is not None:
            return self.reports_to.handle_escalation(event, task, company)
        if company is not None:
            return company._handle_top_level_escalation(event, task)
        raise EscalationUnresolved(event)

    def handle_escalation(self, event: EscalationEvent, task: str, company: Optional["Company"]) -> str:
        """Default behavior is to keep passing the event up the chain. A
        subclass (or a future Company-level policy hook) can override this to
        actually try something at this rank before escalating further -
        deliberately left as the simplest possible thing for Phase 0-1."""
        return self._escalate(event, task, company)


class Team:
    """A company inside a company: a bounded group of Employees with a lead,
    and - per the plan - at least one reviewer by default."""

    def __init__(self, name: str, lead: Employee, reviewer: Optional[Employee] = None):
        self.name = name
        self.lead = lead
        self.reviewer = reviewer
        self.members: List[Employee] = [lead] + ([reviewer] if reviewer else [])

    def add_member(self, employee: Employee) -> None:
        self.members.append(employee)

    def run(self, task: str, company: Optional["Company"] = None) -> str:
        return self.lead.run(task, company=company)


class Company:
    """Owns the org chart, the emergency escalation reserve, and the single
    callback the whole hierarchy eventually reports to when nothing
    automatic is left to try."""

    def __init__(
        self,
        name: str,
        model_map: Dict[str, Dict[str, Any]],
        on_escalation: Callable[[EscalationEvent], EscalationDecision],
        emergency_iteration_reserve: int = 0,
        total_token_budget: Optional[int] = None,
        emergency_budget_tokens: int = 0,
        rank_budget_shares: Optional[Dict[str, float]] = None,
        quota: Optional[ResourceQuota] = None,
    ):
        """
        model_map: rank -> {"provider": ..., "model": ..., "api_key": ..., "mode": "local"|"api"|"auto"}.
            "mode" is accepted but not acted on yet - Phase 4 wires it to
            ModelRouter.allocate_local_auto()/benchmark.py for the local-vs-API
            decision. Per-rank defaults here; hire() also takes a per-employee
            provider/model/api_key override for when the ranking-style default
            isn't what a specific employee should use.
        on_escalation: called once nothing automatic is left for EITHER
            escalation kind - the relevant reserve is empty, or a retry from
            it still failed. Must return an EscalationDecision.
        emergency_iteration_reserve: an integer reserve of *tool iterations*
            (a runaway-loop safety valve, unrelated to cost) drawn down
            automatically for tool_iteration_limit escalations before
            on_escalation is ever invoked. 0 means always ask a human first.
        total_token_budget: the company-wide hard ceiling on total real tokens
            spent (summed across every employee's Agent.total_tokens_used).
            None means unlimited - no budget gate runs at all. Once the
            company's cumulative spend reaches this ceiling, EVERY further
            request for more - internal reallocation included - must go
            through on_escalation; nothing silently crosses it. This is the
            "user stays at the top" cost-control default.
        emergency_budget_tokens: a real-token reserve drawn down automatically
            for budget_exhausted escalations, but ONLY while total spend is
            still below total_token_budget (see above) - once the hard
            ceiling itself is reached, this reserve is skipped too and
            on_escalation is always called. 0 means always ask a human first.
        rank_budget_shares: rank -> relative share of total_token_budget (see
            budget.DEFAULT_RANK_BUDGET_SHARES for the default table and
            budget.BudgetLedger for how a per-employee importance weight
            scales it). Only matters when total_token_budget is set.
        """
        self.name = name
        self.model_map = model_map
        self.on_escalation = on_escalation
        self.emergency_iteration_reserve = emergency_iteration_reserve
        self._emergency_iteration_remaining = emergency_iteration_reserve
        self.emergency_budget_tokens = emergency_budget_tokens
        self._emergency_budget_spent = 0
        self.budget = BudgetLedger(total_token_budget, rank_budget_shares)
        self.quota = quota
        self.teams: List[Team] = []
        self.employees: Dict[str, Employee] = {}
        self.activity_log: List[Dict[str, Any]] = []
        self.tool_call_log: List[Dict[str, Any]] = []

    def hire(
        self,
        name: str,
        rank: str,
        reports_to: Optional[Employee] = None,
        system_instruction: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        importance: float = 0.5,
        **agent_kwargs: Any,
    ) -> Employee:
        """Builds an Agent for `rank` from model_map, wraps it as an Employee,
        registers it with the company, and - if reports_to is given - adds a
        delegate_to_<name> tool onto the manager's Agent so it can hand this
        employee tasks through the ordinary tool-calling loop.

        provider/model/api_key: per-employee overrides. The ranking-style
        model_map[rank] default is used unless one of these is given, in
        which case it just changes it for this one employee - nothing else
        about hiring changes.
        importance: this employee's weight in BudgetLedger's percentage-based
        allocation (0.0-1.0, see budget.py) - defaults to a neutral 0.5.
        """
        if name in self.employees:
            raise ValueError(f"'{name}' is already hired - names must be unique within a Company.")
        rank_config = self.model_map.get(rank, {})
        if not rank_config:
            raise ValueError(f"No model_map entry for rank '{rank}' - add one before hiring for it.")

        agent = Agent(
            provider=provider or rank_config.get("provider", "ollama"),
            model=model or rank_config.get("model"),
            api_key=api_key or rank_config.get("api_key"),
            system_instruction=system_instruction,
            **agent_kwargs,
        )
        employee = Employee(name=name, rank=rank, agent=agent, reports_to=reports_to, importance=importance)
        if reports_to is not None:
            reports_to.agent.add_tool(employee.delegate_tool(company=self))

        self.employees[name] = employee
        self._log("hire", employee=name, rank=rank, reports_to=(reports_to.name if reports_to else None))
        return employee

    def add_team(self, team: Team) -> None:
        self.teams.append(team)
        for member in team.members:
            self.employees.setdefault(member.name, member)

    def run(self, task: str, entry_point: Optional[Employee] = None) -> str:
        """Runs a task starting from entry_point, defaulting to the
        highest-ranked employee registered (typically your C-suite)."""
        start = entry_point or self._default_entry_point()
        if start is None:
            raise ValueError("Company has no employees - call hire() before run().")
        self._log("task_start", employee=start.name, task=task[:200])
        result = start.run(task, company=self)
        self._log("task_end", employee=start.name)
        return result

    def _default_entry_point(self) -> Optional[Employee]:
        for rank in RoleRank.ORDER:
            for employee in self.employees.values():
                if employee.rank == rank:
                    return employee
        return next(iter(self.employees.values()), None)

    def _handle_top_level_escalation(self, event: EscalationEvent, task: str) -> str:
        """Dispatches to the kind-specific handler. Both kinds share the same
        overall shape (try an automatic reserve first, only then call
        on_escalation) but draw from genuinely different reserves and mean
        different things when approved, so they're separate methods rather
        than one branchy one."""
        self._log("escalation", employee=event.employee_name, event_kind=event.kind, message=event.message)
        if event.kind == "budget_exhausted":
            return self._handle_budget_escalation(event, task, hard_ceiling=(event.detail == "hard_ceiling"))
        return self._handle_iteration_escalation(event, task)

    def _handle_iteration_escalation(self, event: EscalationEvent, task: str) -> str:
        if self._emergency_iteration_remaining > 0:
            grant = min(self._emergency_iteration_remaining, _EMERGENCY_GRANT_ITERATIONS)
            self._emergency_iteration_remaining -= grant
            self._log("emergency_reserve_used", employee=event.employee_name, iterations_granted=grant,
                       remaining=self._emergency_iteration_remaining)
            employee = self.employees.get(event.employee_name)
            if employee is not None:
                employee.agent.set_max_tool_iterations(employee.agent.max_tool_iterations + grant)
                try:
                    return employee.agent.chat(task)
                except RuntimeError:
                    pass  # still failing even with the extra room - fall through to the human

        decision = self.on_escalation(event)
        self._log("escalation_decision", employee=event.employee_name, approved=decision.approve, note=decision.note)
        if not decision.approve:
            raise EscalationUnresolved(event, decision)

        employee = self.employees.get(event.employee_name)
        if employee is not None and decision.extra_tool_iterations:
            employee.agent.set_max_tool_iterations(employee.agent.max_tool_iterations + decision.extra_tool_iterations)
            return employee.agent.chat(task)

        return f"[Escalation for {event.employee_name} was approved, but no concrete recovery action was given.]"

    def _handle_budget_escalation(self, event: EscalationEvent, task: str, hard_ceiling: bool = False) -> str:
        """hard_ceiling=True means total_token_budget itself has been
        reached (see _budget_gate) - emergency_budget_tokens is skipped
        entirely in that case (it exists for an individual employee/rank
        running dry while the company as a whole still has room, not for
        crossing the company-wide ceiling), and on_escalation is always
        called."""
        if not hard_ceiling:
            reserve_left = self.emergency_budget_tokens - self._emergency_budget_spent
            if reserve_left > 0:
                grant = min(reserve_left, max(1, int(reserve_left * _EMERGENCY_BUDGET_DRAW_FRACTION)))
                self._emergency_budget_spent += grant
                employee = self.employees.get(event.employee_name)
                if employee is not None:
                    self.budget.grant_bonus(employee.name, grant)
                    self._log("emergency_budget_used", employee=event.employee_name, tokens_granted=grant,
                               remaining=self.emergency_budget_tokens - self._emergency_budget_spent)
                    try:
                        return employee.agent.chat(task)
                    except RuntimeError:
                        pass  # still failing (e.g. hit its iteration cap too) - fall through to the human

        decision = self.on_escalation(event)
        self._log("escalation_decision", employee=event.employee_name, approved=decision.approve, note=decision.note)
        if not decision.approve:
            raise EscalationUnresolved(event, decision)

        employee = self.employees.get(event.employee_name)
        if employee is None:
            return f"[Escalation for {event.employee_name} was approved, but no concrete recovery action was given.]"

        if decision.extra_token_budget:
            self.budget.grant_bonus(employee.name, decision.extra_token_budget)
        if hard_ceiling:
            # Human approved crossing the company-wide ceiling itself, not just
            # this employee's rank share - raise the ceiling by the granted
            # amount (or a nominal 1 token if none given, so the retry isn't
            # immediately re-gated at the exact same ceiling).
            self.budget.total_token_budget = (self.budget.total_token_budget or 0) + (decision.extra_token_budget or 1)
        return employee.agent.chat(task)

    def _budget_gate(self, employee: Employee, task: str) -> Optional[str]:
        """Pre-flight check run by Employee.run() before it calls
        agent.chat(). Returns None to let run() proceed normally, or a
        finished result string if a budget escalation had to happen (and got
        resolved) instead.

        Two distinct checks, in order:
        1. The company-wide hard ceiling (total_token_budget) - once total
           spend reaches it, nothing proceeds without on_escalation; internal
           reallocation can never cross this line, by design.
        2. This employee's own rank-based allocation - if it's used up,
           try borrowing slack from the reports_to chain (_try_reallocate)
           before escalating.
        """
        if self.budget.total_token_budget is None:
            return None  # unlimited - no budget configured, gate is a no-op

        total_spent = self.total_tokens_spent()
        if total_spent >= self.budget.total_token_budget:
            event = EscalationEvent(
                kind="budget_exhausted",
                employee_name=employee.name,
                rank=employee.rank,
                message=(f"{employee.name} ({employee.rank}) can't start a new task - the company-wide token "
                         f"budget ({self.budget.total_token_budget}) has already been reached ({total_spent} spent)."),
                detail="hard_ceiling",
            )
            return employee._escalate(event, task, self)

        remaining = self.budget.remaining(employee.rank, employee.name, employee.importance, employee.agent.total_tokens_used)
        if remaining is not None and remaining <= 0:
            if self._try_reallocate(employee):
                return None  # borrowed some room - let run() proceed with a normal chat() call
            event = EscalationEvent(
                kind="budget_exhausted",
                employee_name=employee.name,
                rank=employee.rank,
                message=(f"{employee.name} ({employee.rank}) has used its full token allocation "
                         f"({employee.agent.total_tokens_used} tokens) and no manager up the chain had slack to lend."),
                detail="rank_allocation",
            )
            return employee._escalate(event, task, self)

        return None

    def _try_reallocate(self, employee: Employee) -> bool:
        """Simplified (non-water-filling) reallocation for v1: walk the
        employee's reports_to chain looking for anyone with unspent
        allocation to spare, and if found, grant the employee a bonus
        conceptually drawn from that slack. This does NOT literally debit the
        lender's own ledger entry - see BudgetLedger's docstring in budget.py
        for why that's a safe simplification here (it only ever adds
        headroom, so the company-wide hard ceiling in _budget_gate is the
        actual backstop, not this bookkeeping). Returns True if a bonus was
        granted (caller should retry), False if nobody up the chain had room."""
        manager = employee.reports_to
        while manager is not None:
            mgr_remaining = self.budget.remaining(manager.rank, manager.name, manager.importance, manager.agent.total_tokens_used)
            if mgr_remaining is not None and mgr_remaining > 0:
                # Lend at most half the manager's own remaining slack, so one
                # starved report can't fully consume a manager's future headroom.
                grant = max(1, mgr_remaining // 2)
                self.budget.grant_bonus(employee.name, grant)
                self._log("budget_reallocated", employee=employee.name, from_employee=manager.name, amount=grant)
                return True
            manager = manager.reports_to
        return False

    def total_tokens_spent(self) -> int:
        """Real tokens spent so far, summed across every hired employee's
        Agent.total_tokens_used (core.py) - the number _budget_gate checks
        against total_token_budget."""
        return sum(e.agent.total_tokens_used for e in self.employees.values())

    def budget_report(self) -> Dict[str, Any]:
        """A snapshot of company-wide + per-employee budget state - allocated
        (None means unlimited), spent, and remaining for each employee, plus
        the emergency reserve's own state."""
        return {
            "total_token_budget": self.budget.total_token_budget,
            "total_spent": self.total_tokens_spent(),
            "emergency_budget_tokens": self.emergency_budget_tokens,
            "emergency_budget_spent": self._emergency_budget_spent,
            "employees": self.budget.report(self.employees),
        }

    def _log(self, kind: str, **fields: Any) -> None:
        self.activity_log.append({"time": time.time(), "kind": kind, **fields})

    def org_chart(self) -> Dict[str, Any]:
        """A nested view of who reports to whom - the data render_org_chart()
        and the standalone observability.render_org_chart_* functions draw
        from."""

        def node(employee: Employee) -> Dict[str, Any]:
            return {
                "name": employee.name,
                "rank": employee.rank,
                "reports": [node(sub) for sub in employee.subordinates],
            }

        roots = [e for e in self.employees.values() if e.reports_to is None]
        return {"company": self.name, "org": [node(r) for r in roots]}

    def render_org_chart(self, fmt: str = "ascii", theme: str = "light") -> str:
        """Renders the current org chart as text or a graphic.

        fmt="ascii"   - a tree-command-style text rendering, zero deps, prints
                         straight to a terminal or log (the default).
        fmt="mermaid" - Mermaid `graph TD` text, pastes directly into a
                         GitHub/GitLab markdown file or any Mermaid renderer.
        fmt="svg"     - a self-contained SVG tree diagram (no JS, no external
                         renderer) - write the returned string to a .svg file
                         and open it in any browser. theme="dark" switches to
                         the dark-surface palette.
        """
        from . import observability

        chart = self.org_chart()
        if fmt == "ascii":
            return observability.render_org_chart_ascii(chart)
        if fmt == "mermaid":
            return observability.render_org_chart_mermaid(chart, theme=theme)
        if fmt == "svg":
            return observability.render_org_chart_svg(chart, theme=theme)
        raise ValueError(f"Unknown render_org_chart fmt '{fmt}' - use 'ascii', 'mermaid', or 'svg'.")

    def activity(self) -> "EventLog":
        """A queryable view over activity_log (hire/task/escalation lifecycle
        events) - see observability.EventLog for by_employee()/by_kind()/since()."""
        from .observability import EventLog

        return EventLog(self.activity_log)

    def tool_calls(self) -> "EventLog":
        """A queryable view over tool_call_log (every delegation call so far -
        the args, a result preview, duration, and any error). Stored for
        later reference, not consulted automatically by anything yet."""
        from .observability import EventLog

        return EventLog(self.tool_call_log)
