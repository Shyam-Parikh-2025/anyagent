"""company.py - Phase 0-3 of the multi-agent hierarchy: Employee/Team/Company
skeleton classes, config-driven Agent creation per rank (with per-employee
overrides), delegation wired as a normal tool call, tool-iteration AND
token-budget failures both escalating up the reporting chain instead of
raising straight to the caller, (Phase 2) queryable activity/tool-call logs
plus org-chart rendering (ascii/mermaid/svg) - see observability.py - and
(Phase 3) real percentage-based token budget governance - see budget.py.

(Phase 4) "mode" in model_map / hire() is now real: with a ModelPolicy
attached (see policy.py), a rank or an individual employee can be routed
automatically between local and API models, steered by an effort/priority
hint that is accepted at hire time and per task.

(Phase 5) Skills, personalities, org templates and org-chart palettes all
come from ONE named-preset registry (presets.py) - hire(skills=, personality=),
build_from_template(), render_org_chart(palette=). Team's long-dormant
`reviewer` field now actually reviews; see Team's docstring for the decision.

(Phase 6) run_structured(task, strategy="direct"|"plan"|"stub_fill") routes a
task through delegation.py's decomposition strategies.

Deliberately NOT in this file yet (later phases, see the architecture doc /
company-roadmap.md):
  - the compressor.py-based log cleanup/compaction pass (Phase 7) - logs
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        model_decision: Optional[Any] = None,
        skills: Optional[List[Any]] = None,
        personality: Optional[Any] = None,
    ):
        self.name = name
        self.rank = rank
        self.agent = agent
        self.reports_to = reports_to
        self.importance = max(0.0, min(1.0, importance))
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
        if review_mode not in ("critique", "append", "off"):
            raise ValueError(f"unknown review_mode {review_mode!r} - use 'critique', 'append', or 'off'")
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
        draft = self.lead.run(task, company=company)
        if self.reviewer is None or self.reviewer is self.lead or self.review_mode == "off":
            return draft
        if self.review_mode == "append":
            second = self.reviewer.run(task, company=company)
            self._log(company, "review_append", rounds=0)
            return f"## {self.lead.name}\n{draft}\n\n## {self.reviewer.name} (reviewer)\n{second}"

        # max_review_rounds counts how many times work can be sent *back*, so
        # there is always one more review than revision: the final revision
        # gets looked at before it ships. Without that last pass, a revised
        # draft would go out carrying a critique that was written about the
        # version before it - misleading in exactly the place honesty matters.
        critique = ""
        for round_index in range(self.max_review_rounds + 1):
            critique = self.reviewer.run(
                _REVIEW_PROMPT.format(task=task, draft=draft), company=company
            )
            if _looks_approved(critique):
                self._log(company, "review_approved", rounds=round_index)
                return draft
            if round_index == self.max_review_rounds:
                break  # no revisions left
            draft = self.lead.run(
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
        model_policy: Optional[Any] = None,
        presets: Optional[Any] = None,
    ):
        """
        model_map: rank -> {"provider": ..., "model": ..., "api_key": ...,
            "mode": "local"|"api"|"auto", "effort": ..., "specialty": ...}.
            Per-rank defaults; hire() also takes a per-employee
            provider/model/api_key override for when the ranking-style default
            isn't what a specific employee should use.
            (Phase 4) "mode" is now acted on when a model_policy is attached:
            "local"/"api"/"auto" are handed to ModelPolicy.decide(), which
            consults benchmark.py/selector.py for the local side and its own
            cost/specialty catalog for the API side. Without a model_policy,
            "mode" is still inert and provider/model are taken literally, so
            existing configs behave exactly as before.
        presets: an optional presets.PresetBundle - the four named-preset
            registries (skills, personalities, palettes, org templates) this
            company resolves names against. Defaults to the shared built-in
            registries; pass `default_bundle().fork()` for a private copy that
            custom registrations won't leak out of.
        model_policy: an optional policy.ModelPolicy. When present, any rank
            or employee whose resolved mode is "local"/"api"/"auto" gets its
            provider/model chosen by the policy instead of read from
            model_map. An explicit provider+model (in model_map or at hire())
            always wins over the policy - the policy fills gaps, it doesn't
            override what you asked for.
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
        self.model_policy = model_policy
        if presets is None:
            from .presets import default_bundle

            presets = default_bundle()
        self.presets = presets
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
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        mode: Optional[str] = None,
        skills: Sequence[Any] = (),
        personality: Optional[Any] = None,
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

        (Phase 4) effort/specialty/mode steer the ModelPolicy, if this Company
        has one:
          effort:    "cheap" | "balanced" | "effort" (aliases like
                     "needs effort" / "keep it cheap" are accepted). Falls
                     back to model_map[rank]["effort"], then to the rank's
                     default in policy.DEFAULT_RANK_EFFORT.
          specialty: a tag matched against ApiModelSpec.specialties, e.g.
                     "code" / "reasoning" / "vision".
          mode:      "auto" | "local" | "api"; falls back to
                     model_map[rank]["mode"]. Anything else (or no policy
                     attached) means "take provider/model literally", which
                     is the pre-Phase-4 behavior.

        Note that effort deliberately does NOT change `importance` - see
        policy.suggested_importance() for the explicit opt-in bridge and
        policy.py's module docstring for why they stay separate knobs.

        (Phase 5) skills/personality are names from this company's
        PresetBundle (or Skill/Personality objects directly, for one-offs that
        need no registration). They are templated into the system instruction
        by presets.compose_system_instruction(), appended after whatever
        `system_instruction` you passed - your text stays first and is never
        rewritten. A skill may also *suggest* an effort/specialty for the
        Phase 4 policy, used only when you did not pass your own.
        """
        if name in self.employees:
            raise ValueError(f"'{name}' is already hired - names must be unique within a Company.")
        rank_config = self.model_map.get(rank, {})
        if not rank_config:
            raise ValueError(f"No model_map entry for rank '{rank}' - add one before hiring for it.")

        effort = effort if effort is not None else rank_config.get("effort")
        specialty = specialty if specialty is not None else rank_config.get("specialty")

        skills = list(skills or rank_config.get("skills", ()))
        personality = personality if personality is not None else rank_config.get("personality")
        if skills or personality:
            from .presets import compose_system_instruction, skill_hints

            system_instruction = compose_system_instruction(
                base=system_instruction, personality=personality, skills=skills,
                bundle=self.presets, role_line=f"You are {name}, a {rank} at {self.name}.",
            )
            hints = skill_hints(skills, bundle=self.presets)
            # A skill's suggestion is a fallback only - an explicit argument,
            # a rank_config entry, or a rank default all outrank it.
            effort = effort if effort is not None else hints["effort"]
            specialty = specialty if specialty is not None else hints["specialty"]

        resolved_provider = provider or rank_config.get("provider", "ollama")
        resolved_model = model or rank_config.get("model")
        resolved_api_key = api_key or rank_config.get("api_key")
        resolved_base_url = rank_config.get("base_url")
        decision = self._decide_model(
            rank=rank, effort=effort, specialty=specialty, mode=mode,
            explicit_provider=provider or rank_config.get("provider"),
            explicit_model=resolved_model,
            employee_name=name,
        )
        if decision is not None and decision.ok:
            resolved_provider = decision.provider or resolved_provider
            resolved_model = decision.model or resolved_model
            resolved_base_url = decision.base_url or resolved_base_url
            resolved_api_key = resolved_api_key or decision.api_key

        agent = Agent(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            system_instruction=system_instruction,
            **agent_kwargs,
        )
        employee = Employee(
            name=name, rank=rank, agent=agent, reports_to=reports_to, importance=importance,
            effort=effort, specialty=specialty, model_decision=decision,
        )
        employee.skills = list(skills)
        employee.personality = personality
        if reports_to is not None:
            reports_to.agent.add_tool(employee.delegate_tool(company=self))

        self.employees[name] = employee
        self._log("hire", employee=name, rank=rank, reports_to=(reports_to.name if reports_to else None))
        return employee

    def _decide_model(
        self,
        rank: str,
        effort: Optional[str],
        specialty: Optional[str],
        mode: Optional[str],
        explicit_provider: Optional[str],
        explicit_model: Optional[str],
        employee_name: str,
    ) -> Optional[Any]:
        """Runs the ModelPolicy for one hire, if a policy is attached and the
        resolved mode asks for one. Returns the PolicyDecision (which the
        caller applies) or None to mean "no policy involvement, use the
        literal model_map values".

        An explicit provider AND model together are treated as "the user
        already decided" and skip the policy entirely - the policy fills gaps
        rather than overruling. A mode of "local"/"api"/"auto" with only one
        of the two given still goes through the policy, since that config is
        asking to be routed.

        A policy that returns kind="unavailable" is logged and then ignored
        (the literal model_map values are used) rather than raising: a routing
        miss at hire time should be visible in the activity log, not fatal
        halfway through building an org chart. It will surface again as a real
        error on the first request, with the provider's own message.
        """
        if self.model_policy is None:
            return None
        rank_config = self.model_map.get(rank, {})
        resolved_mode = (mode or rank_config.get("mode") or "").strip().lower()
        if resolved_mode not in ("auto", "local", "api"):
            return None
        if explicit_provider and explicit_model:
            return None

        decision = self.model_policy.decide(
            rank=rank, effort=effort, specialty=specialty, mode=resolved_mode,
            requested_provider=explicit_provider, requested_model=explicit_model,
        )
        self._log(
            "model_policy", employee=employee_name, rank=rank, mode=resolved_mode,
            decision=decision.kind, effort=decision.effort,
            provider=decision.provider, model=decision.model, reason=decision.reason,
        )
        return decision

    def reassign_model(
        self,
        employee: Employee,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        mode: str = "auto",
    ) -> Optional[Any]:
        """Re-run the ModelPolicy for an already-hired employee and swap their
        Agent onto the newly chosen model, in place.

        This is what makes the effort hint *per-task* and not merely
        per-employee: `Company.run(task, effort="needs effort")` calls this on
        the entry point before starting. It uses `Agent.change_api()`, so the
        employee keeps their conversation history, their registered tools
        (including every delegate_to_* tool), and their usage counters - only
        the endpoint changes. Returns the PolicyDecision, or None if this
        company has no ModelPolicy.

        Caveat worth stating plainly: switching provider mid-conversation
        re-exports the existing history into the new provider's format
        (Conversation.export_for handles that), but provider-native blocks
        recorded from the old provider are replayed through the generic path.
        For a fresh task on a fresh employee - the intended use - this is
        clean; mid-conversation swaps on a long tool-heavy history are not
        something this has been hardened for.
        """
        if self.model_policy is None:
            return None
        decision = self.model_policy.decide(
            rank=employee.rank,
            effort=effort if effort is not None else employee.effort,
            specialty=specialty if specialty is not None else employee.specialty,
            mode=mode,
        )
        self._log(
            "model_policy", employee=employee.name, rank=employee.rank, mode=mode,
            decision=decision.kind, effort=decision.effort,
            provider=decision.provider, model=decision.model, reason=decision.reason,
        )
        if not decision.ok:
            return decision
        employee.model_decision = decision
        if effort is not None:
            employee.effort = effort
        if specialty is not None:
            employee.specialty = specialty
        employee.agent.change_api(
            provider=decision.provider or employee.agent.provider,
            model=decision.model,
            base_url=decision.base_url,
            api_key=decision.api_key or employee.agent.api_key,
        )
        return decision

    def add_team(self, team: Team) -> None:
        self.teams.append(team)
        for member in team.members:
            self.employees.setdefault(member.name, member)

    def build_from_template(
        self,
        template: Any,
        size: str = "small",
        name_prefix: str = "",
        review_mode: str = "critique",
        max_review_rounds: int = 1,
        **hire_overrides: Any,
    ) -> Team:
        """Instantiate a whole org shape from a named org template (Phase 5).

        template: a name in this company's org-template registry, or an
            OrgTemplate object.
        size: "small" | "medium" | "large". **Opt-in scaling** - the size is
            an argument, never inferred from a task string. A template's roles
            declare how many copies of themselves exist at each size, so
            "large" typically multiplies the worker tiers while leaving the
            oversight roles alone. Inferring this from the task would mean
            guessing at spend on the user's behalf, which is the thing Phase
            3's budget governance exists to prevent.
        name_prefix: prepended to every employee name, so the same template
            can be instantiated twice in one Company without name collisions
            (names must be unique - hire() enforces that).
        review_mode / max_review_rounds: passed to the Team - see Team's
            docstring for what "review" means and why the round count is
            bounded.
        **hire_overrides: forwarded to every hire() call (e.g. mode="auto" to
            put the whole template under the Phase 4 model policy).

        Returns the Team, already registered with this Company. Its lead is
        the role marked lead=True (falling back to the highest-ranked role),
        and its reviewer is the first role marked reviewer=True. Employees are
        created parents-first so `reports_to` is always already hired.
        """
        template = self.presets.org_templates.resolve(template)
        template.validate()
        roles = template.roles_for(size)
        role_keys = {r.key for r in roles}

        # Parents before children, so reports_to always resolves. A role whose
        # manager doesn't exist at this size reports to that manager's own
        # manager instead (walking up until something exists, or None) - that
        # keeps a template usable at "small" without needing a separate
        # small-only copy of it with rewritten reporting lines.
        def effective_manager(role: Any) -> Optional[str]:
            cursor = role.reports_to
            while cursor is not None and cursor not in role_keys:
                cursor = template.role(cursor).reports_to
            return cursor

        ordered: List[Any] = []
        placed: set = set()
        remaining = list(roles)
        while remaining:
            progressed = False
            for role in list(remaining):
                manager = effective_manager(role)
                if manager is None or manager in placed:
                    ordered.append(role)
                    placed.add(role.key)
                    remaining.remove(role)
                    progressed = True
            if not progressed:  # validate() rules this out, but never loop forever
                raise ValueError(f"template {template.name!r} could not be ordered - check reports_to")

        by_key: Dict[str, Employee] = {}
        for role in ordered:
            manager_key = effective_manager(role)
            manager = by_key.get(manager_key) if manager_key else None
            count = role.count_for(size)
            for index in range(count):
                title = role.display_title()
                suffix = f" {index + 1}" if count > 1 else ""
                employee = self.hire(
                    name=f"{name_prefix}{title}{suffix}",
                    rank=role.rank,
                    reports_to=manager,
                    skills=role.skills,
                    personality=role.personality,
                    effort=role.effort,
                    specialty=role.specialty,
                    importance=role.importance,
                    **hire_overrides,
                )
                if index == 0:
                    by_key[role.key] = employee

        lead_role = next((r for r in ordered if r.lead), None)
        if lead_role is None:
            lead_role = min(ordered, key=lambda r: RoleRank.ORDER.index(r.rank)
                            if r.rank in RoleRank.ORDER else len(RoleRank.ORDER))
        reviewer_role = next((r for r in ordered if r.reviewer), None)

        team = Team(
            name=f"{name_prefix}{template.team_name or template.name}",
            lead=by_key[lead_role.key],
            reviewer=by_key[reviewer_role.key] if reviewer_role else None,
            review_mode=review_mode,
            max_review_rounds=max_review_rounds,
        )
        for role in ordered:
            for employee in self.employees.values():
                if employee not in team.members and employee.name.startswith(f"{name_prefix}{role.display_title()}"):
                    team.add_member(employee)
        self.add_team(team)
        self._log("team_built", employee=team.lead.name, team=team.name,
                  template=template.name, size=size, headcount=len(team.members))
        return team

    def run(
        self,
        task: str,
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> str:
        """Runs a task starting from entry_point, defaulting to the
        highest-ranked employee registered (typically your C-suite).

        (Phase 4) effort/specialty are the *per-task* form of the model
        policy hint. When either is given and this Company has a ModelPolicy,
        the entry point is re-routed for this task via reassign_model() before
        it starts - so "this one is worth spending on" is a per-call argument,
        not something baked in at hire time. The hint applies to the entry
        point only; employees it delegates to keep their own standing hints,
        since a manager deciding a task is hard doesn't mean every intern
        touching it needs a frontier model.
        """
        start = entry_point or self._default_entry_point()
        if start is None:
            raise ValueError("Company has no employees - call hire() before run().")
        if (effort is not None or specialty is not None) and self.model_policy is not None:
            self.reassign_model(start, effort=effort, specialty=specialty)
        self._log("task_start", employee=start.name, task=task[:200])
        result = start.run(task, company=self)
        self._log("task_end", employee=start.name)
        return result

    def run_structured(
        self,
        task: str,
        strategy: str = "direct",
        entry_point: Optional[Employee] = None,
        effort: Optional[str] = None,
        specialty: Optional[str] = None,
        **strategy_kwargs: Any,
    ) -> Any:
        """(Phase 6) Run a task through a decomposition strategy.

        strategy="direct"    - exactly `run()`: the entry point handles it,
                               delegating through its normal tools. Returns a
                               string.
        strategy="plan"      - plan-then-execute (delegation.plan_then_execute).
                               Returns a PlanResult.
        strategy="stub_fill" - stub-and-fill (delegation.stub_and_fill). The
                               senior tier writes signatures + docstrings, the
                               cheap tiers write bodies. Returns a
                               StubAndFillResult.

        The two structured strategies return result *objects*, not strings,
        because "did every step succeed?" and "which stubs came back unfilled?"
        are the questions a caller actually has, and a string answer can't
        carry them. `.answer` / `.code` hold the deliverable.

        effort/specialty behave as in run(): they re-route the entry point
        (here, the architect/planner) through the Phase 4 policy for this task.
        """
        start = entry_point or self._default_entry_point()
        if start is None:
            raise ValueError("Company has no employees - call hire() before run_structured().")
        if (effort is not None or specialty is not None) and self.model_policy is not None:
            self.reassign_model(start, effort=effort, specialty=specialty)

        if strategy == "direct":
            return self.run(task, entry_point=start)

        from . import delegation

        self._log("strategy_start", employee=start.name, strategy=strategy, task=task[:200])
        if strategy == "plan":
            result = delegation.plan_then_execute(self, task, planner=start, **strategy_kwargs)
        elif strategy == "stub_fill":
            result = delegation.stub_and_fill(self, task, architect=start, **strategy_kwargs)
        else:
            raise ValueError(
                f"unknown strategy {strategy!r} - use 'direct', 'plan', or 'stub_fill'"
            )
        self._log("strategy_done", employee=start.name, strategy=strategy, ok=result.ok)
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

    def render_org_chart(self, fmt: str = "ascii", theme: str = "light", palette: Optional[Any] = None) -> str:
        """Renders the current org chart as text or a graphic.

        fmt="ascii"   - a tree-command-style text rendering, zero deps, prints
                         straight to a terminal or log (the default).
        fmt="mermaid" - Mermaid `graph TD` text, pastes directly into a
                         GitHub/GitLab markdown file or any Mermaid renderer.
        fmt="svg"     - a self-contained SVG tree diagram (no JS, no external
                         renderer) - write the returned string to a .svg file
                         and open it in any browser. theme="dark" switches to
                         the dark-surface palette.
        palette       - (Phase 5) a name from this company's palette registry
                         ("dataviz" default, plus "grayscale"/"ocean"/"ember"),
                         or a Palette object. Resolved through the same
                         PresetRegistry mechanism as skills and templates -
                         palettes are not a separate system. Ignored by
                         fmt="ascii", which has no colors.
        """
        from . import observability

        chart = self.org_chart()
        if palette is not None:
            palette = self.presets.palettes.resolve(palette)
        if fmt == "ascii":
            return observability.render_org_chart_ascii(chart)
        if fmt == "mermaid":
            return observability.render_org_chart_mermaid(chart, theme=theme, palette=palette)
        if fmt == "svg":
            return observability.render_org_chart_svg(chart, theme=theme, palette=palette)
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
