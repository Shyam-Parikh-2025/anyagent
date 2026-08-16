"""company/escalation.py - the escalation vocabulary shared by every
layer of the hierarchy.

Split out of the old single-file company.py so that Employee, Team and Company
can each import the event/decision types without importing each other. These
three types are the whole contract between "something went wrong at some rank"
and "a human (or a callback standing in for one) decided what to do about it" -
they deliberately carry no behaviour beyond that."""

from dataclasses import dataclass
from typing import Callable, Optional

from ..core import RunPaused, ToolControlFlow

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


class _EscalationPending:
    """The type of ESCALATION_PENDING. A distinct object rather than a field on
    EscalationDecision, because `approve` already means something here and
    anything built on top of it would be read wrong exactly once, in the worst
    place: `EscalationDecision(approve=False, pending=True)` looks like a
    decline to every existing `if not decision.approve` check in this file, and
    a decline raises. "Wait" and "no" must not be one keystroke apart."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ESCALATION_PENDING"

    def __bool__(self) -> bool:
        # Deliberately falsey: any code that treats a returned decision as a
        # yes/no will read "not yet" as "not approved", which is the safe way
        # round for something this library would rather stop than assume.
        return False


# Return this from on_escalation instead of an EscalationDecision to say "hold
# on, a human is looking at it" - the run stops where it is, and everything
# needed to pick it up again is handed back by Company.run_resumable().
#
# It is the third answer, alongside approve and decline, and it is the one the
# other two could not express: a decline is final, an approval is immediate, and
# neither one covers a person who will get back to you in an hour from a
# different process. Nothing here weakens the default - a company with no
# handler still declines (see default_on_escalation), and a handler that returns
# this without anyone ever calling resume() leaves the run stopped, which is the
# conservative outcome rather than a quiet approval.
ESCALATION_PENDING = _EscalationPending()


class EscalationUnresolved(ToolControlFlow):
    """Raised when an EscalationEvent reaches the top of the company and
    on_escalation declines to approve a way forward.

    Subclasses core.ToolControlFlow rather than Exception so that a decline
    raised inside a *delegated* employee still reaches the caller. Because
    `delegate_to_<name>` is an ordinary tool, this used to be caught by
    ToolRegistry.execute()'s catch-all and handed to the delegating manager's
    model as a "Tool Execution Failure" string - the manager then routed
    around it and the run returned a normal answer, so a human declining an
    escalation stopped nothing unless it happened at the entry point itself.
    See ToolControlFlow in core.py for the full reasoning.
    """

    def __init__(self, event: EscalationEvent, decision: Optional[EscalationDecision] = None):
        self.event = event
        self.decision = decision
        super().__init__(f"Unresolved escalation from {event.employee_name} ({event.rank}): {event.message}")


# ---------------------------------------------------------------------------
# Ready-made handlers
# ---------------------------------------------------------------------------
#
# Lives here (not builder.py, where it used to be) so Company itself can fall
# back to it when a caller constructs one with no on_escalation at all. That
# used to be a hard requirement of Company.__init__ - every hand-written
# Company(...) call had to spell out a callback, even a throwaway one, which
# is exactly the kind of boilerplate a caller shouldn't have to write for the
# safe behavior. Moving the default here (rather than duplicating it in
# builder.py) means Company() and build_company()/set_company_up() share the
# same fallback, instead of two definitions that could quietly drift apart.


def default_on_escalation(event: EscalationEvent) -> EscalationDecision:
    """The escalation callback used when a caller supplies none - for both a
    bare `Company(...)` and the builder.py entry points.

    Declines. That is the deliberate choice, matching the 0-means-always-ask
    defaults the budget/emergency-reserve system uses everywhere else: no
    on_escalation handler means no human (or a callback standing in for one)
    is actually watching, and "nobody's watching" must mean "stop", never
    "approve yourself".
    """
    return EscalationDecision(
        approve=False,
        note="No on_escalation handler was configured for this company, so the request was declined.",
    )


# Same behavior as default_on_escalation, under a name you can hand to
# Company(on_escalation=...) when you want "never approve" to read as a
# decision you made rather than an omission you forgot to fill in.
always_decline = default_on_escalation


def always_approve(
    extra_token_budget: Optional[int] = None,
    extra_tool_iterations: Optional[int] = None,
    note: str = "auto-approved by always_approve() - no human actually reviewed this.",
) -> Callable[[EscalationEvent], EscalationDecision]:
    """Factory for an on_escalation handler that approves every escalation it
    is handed, granting the same fixed extra_token_budget /
    extra_tool_iterations regardless of which kind of event it is.

    This exists to cut prototyping boilerplate - a company on local Ollama
    models with no API key has nothing to lose, and writing a real handler
    just to get past a demo is friction with no corresponding safety benefit.
    It is named for exactly what it does: approving every escalation with no
    human in the loop is the specific failure mode this library's
    0-means-always-ask defaults exist to prevent everywhere else, so reach
    for this only when that risk is genuinely zero. The moment real spend or
    a real budget is on the line, write your own on_escalation instead.
    """

    def _handler(event: EscalationEvent) -> EscalationDecision:
        return EscalationDecision(
            approve=True,
            note=note,
            extra_token_budget=extra_token_budget,
            extra_tool_iterations=extra_tool_iterations,
        )

    return _handler


__all__ = [
    "EscalationEvent",
    "EscalationDecision",
    "EscalationUnresolved",
    "ESCALATION_PENDING",
    "RunPaused",
    "default_on_escalation",
    "always_decline",
    "always_approve",
]
