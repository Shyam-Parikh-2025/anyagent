"""company/team.py - a lead, an optional reviewer, and the review
loop between them.

Split out of the old single-file company.py. The review prompts stay
module-level constants (rather than inline f-strings) so a caller can read -
and if they must, monkeypatch - the exact wording driving the loop."""

from typing import TYPE_CHECKING, Any, List, Optional

from ..core import run_coroutine_blocking
from .employee import Employee

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from .company import Company


class ReviewMode:
    """The three modes `Team.review_mode` accepts, as named constants - via
    the `review` alias below (`review.CRITIQUE`), the same pattern
    `role`/`mode` use for RoleRank/PolicyMode (see router.py's `role` for the
    full reasoning on why this is a plain class of strings, not a real Enum).

    This used to be three separate hardcoded ("critique", "append", "off")
    tuples/lists - one here, one in builder.py's CompanySpec.validate(), one
    in gui.py's options payload - so adding a fourth mode would have meant
    remembering to update all three in lockstep, with nothing that would
    catch a missed one. REVIEW_MODES below is now the one place that exists;
    builder.py and gui.py import it rather than retyping the literal.
    """

    CRITIQUE = "critique"
    APPEND = "append"
    OFF = "off"


REVIEW_MODES = (ReviewMode.CRITIQUE, ReviewMode.APPEND, ReviewMode.OFF)

# Not re-exported from the top-level `llmadapt` package, for the same reason
# `mode` isn't - "review" is common enough to want scoped rather than bare.
# `from llmadapt.company import review` is the explicit way to get it;
# `ReviewMode.CRITIQUE` (exported at the top level same as RoleRank/PolicyMode)
# needs no second import at all.
review = ReviewMode


# Wording of the review request handed to a Team's reviewer, and of the
# revision request handed back to the lead. Module-level constants rather than
# inline f-strings so a caller can see (and, if they must, monkeypatch) the
# exact prompts driving the review loop.
_REVIEW_PROMPT = (
    "You are reviewing a colleague's work before it is delivered.\n\n"
    "ORIGINAL TASK:\n{task}\n\n"
    "THEIR WORK:\n{draft}\n\n"
    "If the work adequately completes the task, reply with exactly: APPROVED\n"
    "Otherwise, reply with a short, specific list of what must change. Do not "
    "rewrite the work yourself, and do not raise problems that are not there."
)

_REVISION_PROMPT = (
    "A reviewer asked for changes to your work.\n\n"
    "ORIGINAL TASK:\n{task}\n\n"
    "YOUR PREVIOUS ANSWER:\n{draft}\n\n"
    "REVIEWER'S NOTES:\n{critique}\n\n"
    "Produce the corrected version. Output only the corrected work."
)


def _looks_approved(text: str) -> bool:
    """Whether a reviewer's reply counts as sign-off.

    A convention, not a protocol: the reviewer is *asked* to reply exactly
    "APPROVED", and this checks the opening of the reply for that word while
    rejecting the obvious negations ("not approved", "cannot be approved").
    A structured tool call would be stricter, but it would force every
    reviewer onto a tool-calling-capable model, which the cheap local tiers
    this library targets often are not. Flagged as a v1 simplification: a
    reviewer that buries "APPROVED" in the middle of a critique will be read
    as approving.
    """
    head = (text or "").strip()[:200].lower()
    if not head:
        return False
    for negation in ("not approved", "cannot approve", "can't approve", "do not approve", "isn't approved"):
        if negation in head:
            return False
    return head.startswith("approved") or head.startswith("**approved") or head.startswith("# approved")


class Team:
    """A company inside a company: a bounded group of Employees with a lead,
    and - per the plan - at least one reviewer by default.

    (Phase 5) The reviewer is finally *used*. Until now `reviewer` was a
    structural field that `run()` ignored, which the handoff notes flagged as
    needing a decision rather than just an implementation. The decision:

      review_mode="critique" (default) - the lead produces a draft, the
        reviewer is asked to either sign off or list what must change, and a
        requested change goes back to the lead for a bounded number of
        revision rounds. Chosen over the alternatives because it is the only
        one where the reviewer's opinion can actually change the delivered
        output, which is what "reviewer" means outside this codebase.
      review_mode="append" - the reviewer independently answers the same task
        and both answers are returned, labelled. Cheaper to reason about and
        occasionally what you want (two opinions, human picks), but it is a
        second opinion, not a review.
      review_mode="off" - the pre-Phase-5 behavior: lead only.

    `max_review_rounds` defaults to **1**, not unlimited. An unbounded
    critique loop is a budget hazard of exactly the kind Phase 3 exists to
    prevent - two agents can disagree forever - and the token cost of a
    review round is real. If the reviewer still objects after the last round,
    the lead's final answer is returned *with the outstanding critique
    appended*, rather than silently dropping the objection or raising: the
    caller gets the work plus the unresolved concern, and can decide.
    """

    def __init__(
        self,
        name: str,
        lead: Employee,
        reviewer: Optional[Employee] = None,
        review_mode: str = "critique",
        max_review_rounds: int = 1,
    ):
        if review_mode not in REVIEW_MODES:
            raise ValueError(f"unknown review_mode {review_mode!r} - use one of {', '.join(REVIEW_MODES)}")
        self.name = name
        self.lead = lead
        self.reviewer = reviewer
        self.review_mode = review_mode
        self.max_review_rounds = max(0, int(max_review_rounds))
        self.members: List[Employee] = [lead] + ([reviewer] if reviewer and reviewer is not lead else [])

    def add_member(self, employee: Employee) -> None:
        if employee not in self.members:
            self.members.append(employee)

    def run(self, task: str, company: Optional["Company"] = None) -> str:
        """Synchronous wrapper over run_async(), which is the implementation."""
        return run_coroutine_blocking(
            lambda: self.run_async(task, company=company),
            what=f"Team.run() for {self.name}",
        )

    async def run_async(self, task: str, company: Optional["Company"] = None) -> str:
        draft = await self.lead.run_async(task, company=company)
        if self.reviewer is None or self.reviewer is self.lead or self.review_mode == "off":
            return draft
        if self.review_mode == "append":
            # The one genuinely parallel branch here: in "append" mode the
            # reviewer answers the same task independently, so it does not have
            # to wait for the lead's draft at all. "critique" mode below cannot
            # be parallelized - a review of a draft needs the draft.
            second = await self.reviewer.run_async(task, company=company)
            self._log(company, "review_append", rounds=0)
            return f"## {self.lead.name}\n{draft}\n\n## {self.reviewer.name} (reviewer)\n{second}"

        # max_review_rounds counts how many times work can be sent *back*, so
        # there is always one more review than revision: the final revision
        # gets looked at before it ships. Without that last pass, a revised
        # draft would go out carrying a critique that was written about the
        # version before it - misleading in exactly the place honesty matters.
        critique = ""
        for round_index in range(self.max_review_rounds + 1):
            critique = await self.reviewer.run_async(
                _REVIEW_PROMPT.format(task=task, draft=draft), company=company
            )
            if _looks_approved(critique):
                self._log(company, "review_approved", rounds=round_index)
                return draft
            if round_index == self.max_review_rounds:
                break  # no revisions left
            draft = await self.lead.run_async(
                _REVISION_PROMPT.format(task=task, draft=draft, critique=critique), company=company
            )
            self._log(company, "review_revised", rounds=round_index + 1)

        # Out of rounds with an objection still standing: deliver the work and
        # the objection together rather than hiding either one.
        self._log(company, "review_unresolved", rounds=self.max_review_rounds)
        return (
            f"{draft}\n\n---\n[Unresolved reviewer note from {self.reviewer.name} after "
            f"{self.max_review_rounds} revision round(s):]\n{critique}"
        )

    def _log(self, company: Optional["Company"], kind: str, **fields: Any) -> None:
        if company is not None:
            company._log(kind, employee=self.lead.name, team=self.name,
                         reviewer=(self.reviewer.name if self.reviewer else None), **fields)


__all__ = [
    "Team", "ReviewMode", "REVIEW_MODES", "review",
    "_looks_approved", "_REVIEW_PROMPT", "_REVISION_PROMPT",
]
