"""company/employee.py - one Agent bound to a rank, with a reporting
line for escalation and delegation wired through it.

Split out of the old single-file company.py. Employee knows about escalation
(it raises events up its reports_to chain) and about the Company only through
the optional `company` argument threaded into run() - it never imports Company,
which is what keeps the dependency arrow pointing one way."""

import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ..core import Agent, loop_local_lock, run_coroutine_blocking
from .escalation import EscalationEvent, EscalationUnresolved

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from .company import Company


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
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        model_decision: Optional[Any] = None,
        skills: Optional[List[Any]] = None,
        personality: Optional[Any] = None,
        cost_weight: Optional[float] = None,
    ):
        self.name = name
        self.rank = rank
        self.agent = agent
        self.reports_to = reports_to
        self.importance = max(0.0, min(1.0, importance))
        # An explicit multiplier on this employee's token spend, overriding
        # everything budget.CostModel would otherwise work out. None means
        # "derive it", which is the normal case; this exists for the employee
        # whose real cost the library has no way to know - a fine-tuned model,
        # a reseller endpoint, a provider not in any catalog. Deliberately
        # separate from `importance`: importance is how large a slice this
        # employee gets, cost_weight is how fast they consume it, and the same
        # reasoning that kept effort out of importance in Phase 4 keeps these
        # two apart. Not clamped to [0, 1] - a model 12x the baseline is a
        # real thing to express.
        self.cost_weight = None if cost_weight is None else max(0.0, float(cost_weight))
        # Phase 4: the standing effort/priority hint for this employee, and the
        # PolicyDecision (policy.py) that produced their current model - kept on
        # the Employee so `why is Bob on gpt-4o-mini?` is answerable months later
        # without re-deriving it. Both are None when no ModelPolicy is in play.
        self.effort = effort
        self.specialty = specialty
        self.model_decision = model_decision
        # Phase 5: the presets this employee was built from, kept for the
        # record (and for Phase 8's "what got built" return value). Changing
        # them here does NOT re-template the system instruction - that is
        # composed once at hire() time.
        self.skills: List[Any] = list(skills or [])
        self.personality = personality
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
        is given, every call is recorded to its tool_call_log (Phase 2).

        The returned function is `async def`, so ToolRegistry.execute_async
        awaits it directly instead of pushing it onto a worker thread. That is
        what makes a manager's three delegations cost one wait instead of
        three: they are coroutines on one event loop, not three OS threads, so
        everything they touch on the way - the company's budget gate, the
        activity and tool-call logs, an employee's own conversation - is
        guarded by asyncio primitives that actually apply to them.
        """

        async def _delegate(task: str) -> str:
            """Delegate a task to {name} ({rank})."""
            start = time.time()
            error = None
            result = ""
            try:
                result = await self.run_async(task, company=company)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                if company is not None:
                    from ..observability import record_tool_call

                    record_tool_call(
                        company.tool_call_log,
                        archive=getattr(company, "archive", None),
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

    def pin_context(self, text: str, reason: str = "") -> dict:
        """Give this employee a fact history compaction may never drop.

        Their skills and personality are already safe - those live in the
        system instruction, which compaction leaves alone. This is for what
        arrives *after* hiring: the spec they were handed, a constraint the
        manager set mid-task, a correction they got wrong once. See
        Conversation.pin for what does and doesn't belong here.
        """
        return self.agent.pin_context(text, reason=reason)

    def run(self, task: str, company: Optional["Company"] = None) -> str:
        """Runs task through this employee's Agent, synchronously.

        A wrapper over run_async(), which is the implementation - see
        Agent.chat() for why the library keeps one of each rather than two."""
        return run_coroutine_blocking(
            lambda: self.run_async(task, company=company),
            what=f"Employee.run() for {self.name}",
        )

    async def run_async(self, task: str, company: Optional["Company"] = None) -> str:
        """Runs task through this employee's Agent. Checked against the
        company's token budget BEFORE starting (see Company._budget_gate) -
        if that gate already had to escalate and resolve things, its result
        is returned directly rather than running the task twice. Otherwise, a
        tool-iteration overflow is caught here and routed up the reporting
        chain instead of raising straight out to whoever called run() (a
        manager delegating, or Company.run() itself).

        One task at a time, per employee. Now that a manager's delegations run
        concurrently, the same employee can be handed two tasks at once - a
        manager splitting work three ways can perfectly well name the same
        report twice. An Employee is one Agent, and one Agent is one
        Conversation; two tasks interleaving turns into it would produce a
        history where neither task's messages follow each other, which is both
        wrong and very hard to read afterwards. The lock makes a second task
        wait for the first, which is also what the org-chart metaphor says
        should happen: a person finishes one thing before starting the next.
        """
        async with loop_local_lock(self, "_run_lock"):
            if company is not None:
                gated_result = await company._budget_gate_async(self, task)
                if gated_result is not None:
                    return gated_result
            try:
                return await self.agent.chat_async(task)
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
                return await self._escalate_async(event, task, company)

    def _escalate(self, event: EscalationEvent, task: str, company: Optional["Company"]) -> str:
        return run_coroutine_blocking(
            lambda: self._escalate_async(event, task, company),
            what=f"Employee._escalate() for {self.name}",
        )

    async def _escalate_async(self, event: EscalationEvent, task: str,
                              company: Optional["Company"]) -> str:
        if self.reports_to is not None:
            return await self.reports_to.handle_escalation_async(event, task, company)
        if company is not None:
            return await company._handle_top_level_escalation_async(event, task)
        raise EscalationUnresolved(event)

    def handle_escalation(self, event: EscalationEvent, task: str, company: Optional["Company"]) -> str:
        return run_coroutine_blocking(
            lambda: self.handle_escalation_async(event, task, company),
            what=f"Employee.handle_escalation() for {self.name}",
        )

    async def handle_escalation_async(self, event: EscalationEvent, task: str,
                                      company: Optional["Company"]) -> str:
        """Default behavior is to keep passing the event up the chain. A
        subclass (or a future Company-level policy hook) can override this to
        actually try something at this rank before escalating further -
        deliberately left as the simplest possible thing for Phase 0-1.

        Override this one, not the synchronous `handle_escalation` - that is
        only a wrapper, and the chain walks through the async form."""
        return await self._escalate_async(event, task, company)


__all__ = ["Employee"]
