"""presets/org_templates.py - ready-made org shapes.

Templates are declarative: roles refer to each other by `key`, and `validate()`
catches dangling reporting lines, cycles and duplicate leads before anything is
half-built.

**Scaling is opt-in.** `build_from_template(name, size=...)` takes the size as
an argument and never infers it from the task text, because inferring it means
guessing at spend on the user's behalf - the thing Phase 3 exists to prevent.
Roles declare `count_by_size`, so one template works at every size: worker
tiers multiply while oversight roles do not, and a role whose manager does not
exist at a smaller size reports to that manager's manager instead.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..router import RoleRank
from .registry import Preset, PresetRegistry

# Ordered smallest-first. A template's roles declare how many copies of
# themselves exist at each size; a size not listed for a role means zero, i.e.
# that role doesn't exist at that size.
TASK_SIZES: Tuple[str, ...] = ("small", "medium", "large")


@dataclass
class RoleSpec:
    """One position in an org template.

    `key` identifies the role within the template (and is what `reports_to`
    points at - templates are declarative, so roles refer to each other by key
    rather than by object). `title` is the employee name; when a role has
    more than one copy at a given size, names are suffixed " 2", " 3", ...
    """

    key: str
    rank: str
    title: Optional[str] = None
    reports_to: Optional[str] = None
    skills: Sequence[str] = ()
    personality: Optional[str] = None
    effort: Optional[str] = None
    specialty: Optional[str] = None
    importance: float = 0.5
    reviewer: bool = False
    lead: bool = False
    count_by_size: Dict[str, int] = field(default_factory=lambda: {"small": 1, "medium": 1, "large": 1})

    def count_for(self, size: str) -> int:
        return max(0, int(self.count_by_size.get(size, 0)))

    def display_title(self) -> str:
        return self.title or self.key.replace("_", " ").title()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "rank": self.rank, "title": self.display_title(),
            "reports_to": self.reports_to, "skills": list(self.skills),
            "personality": self.personality, "effort": self.effort,
            "specialty": self.specialty, "importance": self.importance,
            "reviewer": self.reviewer, "lead": self.lead,
            "count_by_size": dict(self.count_by_size),
        }


@dataclass
class OrgTemplate(Preset):
    """A ready-made org shape. `Company.build_from_template()` instantiates it.

    Scaling is opt-in and explicit: `size` is an argument the caller passes,
    never inferred from the task text. Inferring it would mean guessing at
    spend on the user's behalf, which is exactly the thing Phase 3's budget
    work exists to prevent.
    """

    roles: Sequence[RoleSpec] = ()
    team_name: Optional[str] = None
    notes: str = ""

    def roles_for(self, size: str = "small") -> List[RoleSpec]:
        if size not in TASK_SIZES:
            raise ValueError(f"unknown task size {size!r} - use one of {', '.join(TASK_SIZES)}")
        return [r for r in self.roles if r.count_for(size) > 0]

    def role(self, key: str) -> RoleSpec:
        for r in self.roles:
            if r.key == key:
                return r
        raise KeyError(f"template {self.name!r} has no role {key!r}")

    def validate(self) -> None:
        """Checks the template is internally consistent - every reports_to
        points at a real key, no cycles, at most one lead. Called by
        build_from_template() so a malformed template fails at build time with
        a clear message instead of half-constructing an org."""
        keys = {r.key for r in self.roles}
        if len(keys) != len(self.roles):
            raise ValueError(f"template {self.name!r} has duplicate role keys")
        leads = [r for r in self.roles if r.lead]
        if len(leads) > 1:
            raise ValueError(f"template {self.name!r} marks more than one role as lead")
        for role in self.roles:
            if role.reports_to is not None and role.reports_to not in keys:
                raise ValueError(
                    f"template {self.name!r}: role {role.key!r} reports_to {role.reports_to!r}, "
                    f"which is not a role in this template"
                )
        for role in self.roles:  # cycle check
            seen = {role.key}
            cursor = role.reports_to
            while cursor is not None:
                if cursor in seen:
                    raise ValueError(f"template {self.name!r} has a reporting cycle through {cursor!r}")
                seen.add(cursor)
                cursor = self.role(cursor).reports_to

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "team_name": self.team_name,
            "notes": self.notes,
            "roles": [r.to_dict() for r in self.roles],
            "sizes": list(TASK_SIZES),
        })
        return out


BUILTIN_ORG_TEMPLATES: Tuple[OrgTemplate, ...] = (
    OrgTemplate(
        name="solo",
        description="One senior generalist. The cheapest thing that can still do work.",
        team_name="Solo",
        roles=(
            RoleSpec(key="worker", rank=RoleRank.SENIOR, title="Worker", lead=True,
                     skills=("python",), personality="concise", importance=0.6),
        ),
    ),
    OrgTemplate(
        name="small-coding-team",
        description="1 C-suite + 1 manager + a reviewer and workers. The roadmap's reference template.",
        team_name="Coding Team",
        notes="Worker count is what scales with size; the oversight roles do not.",
        roles=(
            RoleSpec(key="exec", rank=RoleRank.C_SUITE, title="Chief", personality="concise",
                     effort="needs effort", importance=0.4),
            RoleSpec(key="manager", rank=RoleRank.MANAGER, title="Manager", reports_to="exec",
                     lead=True, personality="concise", importance=0.5),
            RoleSpec(key="reviewer", rank=RoleRank.SENIOR, title="Reviewer", reports_to="manager",
                     reviewer=True, skills=("code-review",), personality="skeptical", importance=0.6),
            RoleSpec(key="dev", rank=RoleRank.JUNIOR, title="Developer", reports_to="manager",
                     skills=("python",), personality="concise", importance=0.5,
                     count_by_size={"small": 1, "medium": 2, "large": 4}),
            RoleSpec(key="intern", rank=RoleRank.INTERN, title="Intern", reports_to="manager",
                     skills=("python",), personality="concise", importance=0.3,
                     count_by_size={"medium": 1, "large": 2}),
        ),
    ),
    OrgTemplate(
        name="research-pod",
        description="A researcher-heavy shape: a lead analyst, researchers, and a skeptical reviewer.",
        team_name="Research Pod",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Lead Analyst", lead=True,
                     skills=("research", "writing"), personality="thorough",
                     effort="needs effort", importance=0.6),
            RoleSpec(key="reviewer", rank=RoleRank.SENIOR, title="Reviewer", reports_to="lead",
                     reviewer=True, skills=("research",), personality="skeptical", importance=0.6),
            RoleSpec(key="researcher", rank=RoleRank.JUNIOR, title="Researcher", reports_to="lead",
                     skills=("research",), personality="thorough", importance=0.5,
                     count_by_size={"small": 1, "medium": 3, "large": 5}),
        ),
    ),
    OrgTemplate(
        name="writing-desk",
        description="A writer plus an editor, for prose deliverables rather than code.",
        team_name="Writing Desk",
        roles=(
            RoleSpec(key="editor", rank=RoleRank.MANAGER, title="Editor", lead=True,
                     skills=("writing",), personality="skeptical", reviewer=True, importance=0.6),
            RoleSpec(key="writer", rank=RoleRank.SENIOR, title="Writer", reports_to="editor",
                     skills=("writing",), personality="thorough", importance=0.6,
                     count_by_size={"small": 1, "medium": 2, "large": 3}),
        ),
    ),
    OrgTemplate(
        name="support-desk",
        description="Front-line responders with an escalation tier above them.",
        team_name="Support Desk",
        notes="The responder tier scales; the escalation engineer does not - that is the point of it.",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Support Lead", lead=True,
                     personality="executive", importance=0.5),
            RoleSpec(key="escalation", rank=RoleRank.SENIOR, title="Escalation Engineer",
                     reports_to="lead", skills=("debugging",), personality="thorough",
                     effort="needs effort", importance=0.7),
            RoleSpec(key="responder", rank=RoleRank.JUNIOR, title="Responder", reports_to="lead",
                     skills=("summarization",), personality="diplomatic", importance=0.4,
                     count_by_size={"small": 2, "medium": 4, "large": 8}),
        ),
    ),
    OrgTemplate(
        name="data-pod",
        description="Analysts working over data, with a reviewer who checks the claims.",
        team_name="Data Pod",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Analytics Lead", lead=True,
                     skills=("data-analysis",), personality="thorough",
                     effort="needs effort", importance=0.6),
            RoleSpec(key="reviewer", rank=RoleRank.SENIOR, title="Reviewer", reports_to="lead",
                     reviewer=True, skills=("data-analysis",), personality="skeptical",
                     importance=0.6),
            RoleSpec(key="analyst", rank=RoleRank.JUNIOR, title="Analyst", reports_to="lead",
                     skills=("sql", "data-analysis"), personality="thorough", importance=0.5,
                     count_by_size={"small": 1, "medium": 3, "large": 5}),
        ),
    ),
    OrgTemplate(
        name="incident-response",
        description="An incident commander, an investigator and responders. Built for speed, not thoroughness.",
        team_name="Incident Response",
        notes=(
            "Personalities are deliberately terse here - during an incident, a long answer is a "
            "worse answer. The commander is high-effort because triage decisions are the "
            "expensive ones to get wrong."
        ),
        roles=(
            RoleSpec(key="commander", rank=RoleRank.MANAGER, title="Incident Commander", lead=True,
                     personality="executive", effort="needs effort", importance=0.7),
            RoleSpec(key="investigator", rank=RoleRank.SENIOR, title="Investigator",
                     reports_to="commander", skills=("debugging",), personality="blunt",
                     effort="needs effort", importance=0.7),
            RoleSpec(key="responder", rank=RoleRank.JUNIOR, title="Responder",
                     reports_to="commander", skills=("devops",), personality="concise",
                     importance=0.5, count_by_size={"small": 1, "medium": 2, "large": 4}),
            RoleSpec(key="scribe", rank=RoleRank.INTERN, title="Scribe", reports_to="commander",
                     skills=("summarization",), personality="concise", importance=0.3,
                     count_by_size={"medium": 1, "large": 1}),
        ),
    ),
    OrgTemplate(
        name="qa-team",
        description="Testers under a QA lead, with an adversarial reviewer trying to break the work.",
        team_name="QA Team",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="QA Lead", lead=True,
                     skills=("testing",), personality="skeptical", importance=0.6),
            RoleSpec(key="breaker", rank=RoleRank.SENIOR, title="Adversarial Reviewer",
                     reports_to="lead", reviewer=True, skills=("testing", "security-review"),
                     personality="adversarial", effort="needs effort", importance=0.7),
            RoleSpec(key="tester", rank=RoleRank.JUNIOR, title="Tester", reports_to="lead",
                     skills=("testing",), personality="thorough", importance=0.5,
                     count_by_size={"small": 1, "medium": 3, "large": 5}),
        ),
    ),
    OrgTemplate(
        name="product-trio",
        description="A product lead, an engineer and a designer-writer. Three perspectives, no hierarchy below the lead.",
        team_name="Product Trio",
        notes="Stays three at every size - the shape is the point, so nothing here scales.",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Product Lead", lead=True,
                     personality="socratic", effort="needs effort", importance=0.6),
            RoleSpec(key="engineer", rank=RoleRank.SENIOR, title="Engineer", reports_to="lead",
                     skills=("python", "api-design"), personality="pragmatic", importance=0.6),
            RoleSpec(key="writer", rank=RoleRank.SENIOR, title="Design Writer", reports_to="lead",
                     skills=("writing", "technical-docs"), personality="thorough", importance=0.5),
        ),
    ),
    OrgTemplate(
        name="security-review-board",
        description="A security lead with reviewers, weighted toward senior judgment rather than volume.",
        team_name="Security Review Board",
        notes=(
            "The only template whose worker tier is SENIOR rather than JUNIOR. Security findings "
            "are expensive when wrong in either direction - a missed hole, or a false alarm that "
            "burns a team's trust - so this one deliberately does not economize on the tier."
        ),
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Security Lead", lead=True,
                     skills=("security-review",), personality="skeptical",
                     effort="needs effort", importance=0.7),
            RoleSpec(key="reviewer", rank=RoleRank.SENIOR, title="Security Reviewer",
                     reports_to="lead", reviewer=True, skills=("security-review", "code-review"),
                     personality="adversarial", effort="needs effort", importance=0.7,
                     count_by_size={"small": 1, "medium": 2, "large": 3}),
            RoleSpec(key="scribe", rank=RoleRank.JUNIOR, title="Findings Writer",
                     reports_to="lead", skills=("technical-docs",), personality="concise",
                     importance=0.4, count_by_size={"medium": 1, "large": 1}),
        ),
    ),
    OrgTemplate(
        name="editorial-desk",
        description="A researcher-writer-editor chain for pieces that need their facts checked.",
        team_name="Editorial Desk",
        roles=(
            RoleSpec(key="editor", rank=RoleRank.MANAGER, title="Editor", lead=True,
                     skills=("writing",), personality="skeptical", reviewer=True, importance=0.6),
            RoleSpec(key="fact_checker", rank=RoleRank.SENIOR, title="Fact Checker",
                     reports_to="editor", skills=("research",), personality="skeptical",
                     effort="needs effort", importance=0.6),
            RoleSpec(key="writer", rank=RoleRank.JUNIOR, title="Writer", reports_to="editor",
                     skills=("writing", "research"), personality="thorough", importance=0.5,
                     count_by_size={"small": 1, "medium": 2, "large": 4}),
        ),
    ),
    # --- Additional built-ins ----------------------------------------------
    # Every template here follows the same rules the ones above do: exactly one
    # lead at every size, oversight roles that do not multiply, and only the
    # doing roles scaling with `size`. The skills and personalities named are
    # all real entries in the other two registries - test_presets.py checks
    # that, because a template naming a preset that no longer exists fails at
    # build time with a KeyError from a registry, which is a confusing place to
    # find out about it.
    OrgTemplate(
        name="pair",
        description="Two seniors, one writing and one reviewing. The smallest thing with a second opinion.",
        team_name="Pair",
        notes="No manager on purpose - at this size the review loop is the whole process.",
        roles=(
            RoleSpec(key="driver", rank=RoleRank.SENIOR, title="Driver", lead=True,
                     skills=("python",), personality="pragmatic", importance=0.6),
            RoleSpec(key="navigator", rank=RoleRank.SENIOR, title="Navigator", reports_to="driver",
                     reviewer=True, skills=("code-review",), personality="skeptical", importance=0.6),
        ),
    ),
    OrgTemplate(
        name="platform-team",
        description="Infrastructure and tooling: a lead, an SRE-minded reviewer, and platform engineers.",
        team_name="Platform Team",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Platform Lead", lead=True,
                     skills=("devops",), personality="pragmatic", importance=0.6),
            RoleSpec(key="reliability", rank=RoleRank.SENIOR, title="Reliability Engineer",
                     reports_to="lead", reviewer=True, skills=("devops", "incident-postmortem"),
                     personality="cautious", importance=0.6),
            RoleSpec(key="engineer", rank=RoleRank.SENIOR, title="Platform Engineer", reports_to="lead",
                     skills=("devops", "python"), personality="methodical", importance=0.5,
                     count_by_size={"small": 1, "medium": 2, "large": 3}),
            RoleSpec(key="oncall", rank=RoleRank.JUNIOR, title="On-call Engineer", reports_to="lead",
                     skills=("debugging",), personality="concise", importance=0.4,
                     count_by_size={"medium": 1, "large": 2}),
        ),
    ),
    OrgTemplate(
        name="migration-crew",
        description="Moving a system from one shape to another without losing anything on the way.",
        team_name="Migration Crew",
        notes="The verifier is the point - a migration nobody checked is a migration you find out about later.",
        roles=(
            RoleSpec(key="architect", rank=RoleRank.SENIOR, title="Migration Architect", lead=True,
                     skills=("api-design", "database-design"), personality="thorough",
                     effort="needs effort", importance=0.7),
            RoleSpec(key="verifier", rank=RoleRank.SENIOR, title="Verifier", reports_to="architect",
                     reviewer=True, skills=("testing",), personality="skeptical", importance=0.6),
            RoleSpec(key="porter", rank=RoleRank.JUNIOR, title="Porter", reports_to="architect",
                     skills=("refactoring", "python"), personality="methodical", importance=0.5,
                     count_by_size={"small": 1, "medium": 3, "large": 5}),
        ),
    ),
    OrgTemplate(
        name="docs-team",
        description="Technical documentation with an editor and a reader-advocate.",
        team_name="Docs Team",
        roles=(
            RoleSpec(key="editor", rank=RoleRank.MANAGER, title="Docs Editor", lead=True,
                     skills=("technical-docs",), personality="detail-oriented", importance=0.6),
            RoleSpec(key="reader", rank=RoleRank.SENIOR, title="Reader Advocate", reports_to="editor",
                     reviewer=True, skills=("technical-docs",), personality="plain-language",
                     importance=0.5),
            RoleSpec(key="writer", rank=RoleRank.JUNIOR, title="Technical Writer", reports_to="editor",
                     skills=("technical-docs", "writing"), personality="plain-language", importance=0.5,
                     count_by_size={"small": 1, "medium": 2, "large": 4}),
        ),
    ),
    OrgTemplate(
        name="design-studio",
        description="Interface work: a lead, an accessibility reviewer, and designers.",
        team_name="Design Studio",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Design Lead", lead=True,
                     skills=("frontend-ui",), personality="big-picture", importance=0.6),
            RoleSpec(key="a11y", rank=RoleRank.SENIOR, title="Accessibility Reviewer",
                     reports_to="lead", reviewer=True, skills=("accessibility",),
                     personality="detail-oriented", importance=0.6),
            RoleSpec(key="designer", rank=RoleRank.JUNIOR, title="Designer", reports_to="lead",
                     skills=("frontend-ui",), personality="collaborative", importance=0.5,
                     count_by_size={"small": 1, "medium": 2, "large": 3}),
        ),
    ),
    OrgTemplate(
        name="analytics-desk",
        description="Questions in, defensible numbers out - with someone checking the method.",
        team_name="Analytics Desk",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Analytics Lead", lead=True,
                     skills=("data-analysis",), personality="analytical", importance=0.6),
            RoleSpec(key="reviewer", rank=RoleRank.SENIOR, title="Method Reviewer", reports_to="lead",
                     reviewer=True, skills=("data-analysis", "fact-checking"),
                     personality="skeptical", importance=0.6),
            RoleSpec(key="analyst", rank=RoleRank.JUNIOR, title="Analyst", reports_to="lead",
                     skills=("sql", "data-analysis"), personality="analytical", importance=0.5,
                     count_by_size={"small": 1, "medium": 2, "large": 3}),
            RoleSpec(key="viz", rank=RoleRank.JUNIOR, title="Visualisation Analyst", reports_to="lead",
                     skills=("data-visualization",), personality="plain-language", importance=0.4,
                     count_by_size={"medium": 1, "large": 2}),
        ),
    ),
    OrgTemplate(
        name="localization-desk",
        description="Taking finished material into other languages without losing the register.",
        team_name="Localization Desk",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Localization Lead", lead=True,
                     skills=("translation",), personality="detail-oriented", importance=0.6),
            RoleSpec(key="reviewer", rank=RoleRank.SENIOR, title="Native Reviewer", reports_to="lead",
                     reviewer=True, skills=("translation",), personality="skeptical", importance=0.6),
            RoleSpec(key="translator", rank=RoleRank.JUNIOR, title="Translator", reports_to="lead",
                     skills=("translation",), personality="formal", importance=0.5,
                     count_by_size={"small": 1, "medium": 3, "large": 5}),
        ),
    ),
    OrgTemplate(
        name="architecture-council",
        description="A small group of seniors deciding one hard design question, adversarially.",
        team_name="Architecture Council",
        notes=(
            "Deliberately expensive and deliberately small. This is for the decision you only "
            "get to make once, not for ordinary work."
        ),
        roles=(
            RoleSpec(key="chair", rank=RoleRank.C_SUITE, title="Chair", lead=True,
                     skills=("api-design",), personality="decisive", effort="needs effort",
                     importance=0.8),
            RoleSpec(key="challenger", rank=RoleRank.SENIOR, title="Challenger", reports_to="chair",
                     reviewer=True, skills=("code-review",), personality="adversarial",
                     effort="needs effort", importance=0.7),
            RoleSpec(key="member", rank=RoleRank.SENIOR, title="Council Member", reports_to="chair",
                     skills=("api-design", "planning"), personality="thorough", importance=0.6,
                     count_by_size={"small": 1, "medium": 2, "large": 3}),
        ),
    ),
    OrgTemplate(
        name="discovery-pod",
        description="Works out what is actually being asked for, before anyone builds it.",
        team_name="Discovery Pod",
        roles=(
            RoleSpec(key="lead", rank=RoleRank.MANAGER, title="Discovery Lead", lead=True,
                     skills=("requirements-gathering",), personality="socratic", importance=0.6),
            RoleSpec(key="challenger", rank=RoleRank.SENIOR, title="Scope Challenger",
                     reports_to="lead", reviewer=True, skills=("requirements-gathering",),
                     personality="skeptical", importance=0.6),
            RoleSpec(key="planner", rank=RoleRank.SENIOR, title="Planner", reports_to="lead",
                     skills=("planning",), personality="methodical", importance=0.5,
                     count_by_size={"small": 1, "medium": 1, "large": 2}),
            RoleSpec(key="researcher", rank=RoleRank.JUNIOR, title="Researcher", reports_to="lead",
                     skills=("research", "fact-checking"), personality="curious", importance=0.4,
                     count_by_size={"medium": 1, "large": 2}),
        ),
    ),
)

ORG_TEMPLATES = PresetRegistry("org template", OrgTemplate, BUILTIN_ORG_TEMPLATES)


__all__ = ["TASK_SIZES", "RoleSpec", "OrgTemplate", "BUILTIN_ORG_TEMPLATES", "ORG_TEMPLATES"]
