"""presets.py - Phase 5: ONE named-preset registry pattern, reused identically
for skills, personality presets, org templates, and color palettes.

This is the whole point of the phase, and it was an explicit correction from
the project owner: colour palettes are not a bespoke system bolted onto
`observability.py`, and org templates are not a bespoke system bolted onto
`company.py`. All four are *named things a user picks by name, ships defaults
for, and can extend with their own* - which is one mechanism, `PresetRegistry`,
instantiated four times.

Consequences of committing to that:

- Every registry behaves the same way: `.get("name")`, `.register(obj)`,
  `.names()`, `.resolve(name_or_object)`, `.copy()`. Learn it once.
- Anything that takes a preset takes *either* a name or the object, via
  `resolve()`. `hire(skills=["python"])` and `hire(skills=[my_skill])` both
  work, and the object form needs no registration at all - useful for a
  one-off skill you don't want to name globally.
- `PresetRegistry.names()` gives Phase 8's `set_company_up` schema its enum
  values for free, for all four kinds, from one method.
- A user who wants to change the built-ins forks them with `.copy()` and hands
  the fork to their `Company` in a `PresetBundle`, instead of mutating module
  globals that another Company in the same process is also using.

Two deliberate limits, stated rather than hidden:

- Skills are *prompt-level* things: instructions plus constraints that get
  templated into the system instruction. A skill cannot register tools or
  change the tool-calling loop. "Don't use library X" is a sentence in a
  prompt, not an enforced sandbox - a model can ignore it, and nothing here
  verifies compliance.
- Org templates instantiate a *shape* (who exists, who reports to whom, what
  each is good at). They do not size a task; `size=` is an argument the caller
  opts into, not something inferred from the task text.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .router import RoleRank

# ---------------------------------------------------------------------------
# The one mechanism
# ---------------------------------------------------------------------------


@dataclass
class Preset:
    """Base for everything in a registry: something with a name you can look
    up, and a description a UI (or an LLM reading a Phase 8 schema) can show
    to explain what picking it would do."""

    name: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


class PresetRegistry:
    """A named collection of one kind of preset.

    Deliberately not a dict subclass: the useful surface here is narrow
    (register/get/resolve/names) and the error messages matter more than
    dict's full API. A missing name is the single most likely user mistake in
    this whole subsystem - a typo'd skill name would otherwise silently
    produce an employee with no skills - so `get()` raises with the available
    names listed, and never returns None.
    """

    def __init__(self, kind: str, item_type: type = Preset, presets: Iterable[Preset] = ()):
        self.kind = kind
        self.item_type = item_type
        self._items: Dict[str, Preset] = {}
        for preset in presets:
            self.register(preset)

    def register(self, preset: Preset, overwrite: bool = False) -> Preset:
        """Adds a preset. Refuses to silently replace an existing name unless
        overwrite=True - shadowing a built-in by accident (two skills both
        called "python") is a bug worth surfacing, while shadowing one on
        purpose is a normal thing to want."""
        if not isinstance(preset, self.item_type):
            raise TypeError(
                f"{self.kind} registry takes {self.item_type.__name__} objects, got {type(preset).__name__}"
            )
        if preset.name in self._items and not overwrite:
            raise ValueError(
                f"a {self.kind} named {preset.name!r} is already registered - "
                f"pass overwrite=True to replace it deliberately"
            )
        self._items[preset.name] = preset
        return preset

    def remove(self, name: str) -> None:
        self._items.pop(name, None)

    def get(self, name: str) -> Preset:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "(none registered)"
            raise KeyError(f"unknown {self.kind} {name!r}. Available: {available}") from None

    def resolve(self, value: Union[str, Preset]) -> Preset:
        """Accepts a registered name OR an already-built preset object.

        The object path is what makes one-off presets ergonomic: a caller can
        pass `Skill(name="just-this-once", ...)` straight into `hire()`
        without polluting a global registry. It is not registered as a side
        effect - passing an object means "use this", not "and remember it".
        """
        if isinstance(value, self.item_type):
            return value
        if isinstance(value, Preset):
            raise TypeError(f"{self.kind} registry cannot resolve a {type(value).__name__}")
        return self.get(str(value))

    def resolve_all(self, values: Iterable[Union[str, Preset]]) -> List[Preset]:
        return [self.resolve(v) for v in (values or ())]

    def names(self) -> List[str]:
        """Sorted names - also the enum values Phase 8's generated schema
        offers an LLM caller for this kind."""
        return sorted(self._items)

    def all(self) -> List[Preset]:
        return [self._items[n] for n in self.names()]

    def copy(self) -> "PresetRegistry":
        """A fork holding the same preset objects. Presets are treated as
        immutable value objects (build a new one with dataclasses.replace
        rather than mutating), so sharing the objects is safe and the fork
        only isolates *membership*."""
        clone = PresetRegistry(self.kind, self.item_type)
        clone._items = dict(self._items)
        return clone

    def describe(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.all()]

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Preset]:
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"PresetRegistry(kind={self.kind!r}, {len(self)} registered)"


# ---------------------------------------------------------------------------
# Kind 1: skills
# ---------------------------------------------------------------------------


@dataclass
class Skill(Preset):
    """A capability an employee is told they have.

    `instructions` is prose spliced into the system instruction.
    `constraints` are hard "don't" rules, collected across every skill an
    employee holds and rendered as one deduplicated list - so two skills both
    saying "no third-party dependencies" produce one line, not two.

    `specialty` and `effort` are the Phase 4 hook: a skill can declare that
    work of this kind wants a model tagged e.g. "code", or that it is
    inherently high-effort. `Company.hire()` uses them only as a *fallback*
    when the caller didn't pass their own - an explicit argument always beats
    a skill's suggestion.
    """

    instructions: str = ""
    constraints: Sequence[str] = ()
    specialty: Optional[str] = None
    effort: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "instructions": self.instructions,
            "constraints": list(self.constraints),
            "specialty": self.specialty,
            "effort": self.effort,
        })
        return out


BUILTIN_SKILLS: Tuple[Skill, ...] = (
    Skill(
        name="python",
        description="Writes idiomatic, well-typed Python.",
        instructions=(
            "Write idiomatic Python 3. Use type hints on public functions, prefer the standard "
            "library over new dependencies, and keep functions small enough to test."
        ),
        constraints=("Do not add a third-party dependency without saying why it is unavoidable.",),
        specialty="code",
    ),
    Skill(
        name="code-review",
        description="Reviews code for correctness, clarity, and risk.",
        instructions=(
            "Review code for correctness first, then clarity, then style. Point at specific lines. "
            "Say plainly when something is fine - do not invent problems to seem thorough."
        ),
        constraints=("Do not rewrite the whole file when a targeted fix would do.",),
        specialty="code",
        effort="balanced",
    ),
    Skill(
        name="research",
        description="Gathers and cites evidence before concluding.",
        instructions=(
            "Gather evidence before concluding. Distinguish what a source states from what you "
            "infer, and say when the evidence is thin rather than filling the gap with confidence."
        ),
        constraints=("Do not present an inference as a cited fact.",),
        specialty="reasoning",
        effort="effort",
    ),
    Skill(
        name="writing",
        description="Produces clear prose for a stated audience.",
        instructions=(
            "Write for the stated audience. Lead with the conclusion, keep sentences short, and "
            "cut any sentence that does not carry information."
        ),
        constraints=("Do not pad with filler introductions or restate the prompt back.",),
        specialty="writing",
    ),
    Skill(
        name="data-analysis",
        description="Analyzes tabular/numeric data and states its limits.",
        instructions=(
            "State the question, the method, and the limitations of the data before the answer. "
            "Show the numbers you relied on."
        ),
        constraints=("Do not report a correlation as a cause.",),
        specialty="reasoning",
        effort="balanced",
    ),
)

SKILLS = PresetRegistry("skill", Skill, BUILTIN_SKILLS)


# ---------------------------------------------------------------------------
# Kind 2: personalities
# ---------------------------------------------------------------------------


@dataclass
class Personality(Preset):
    """How an employee talks, as opposed to what it can do. Separate from
    Skill because the two compose independently - a terse researcher and a
    verbose researcher are both coherent, and forcing them into one preset
    would multiply the built-in list instead of adding to it."""

    instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out["instructions"] = self.instructions
        return out


BUILTIN_PERSONALITIES: Tuple[Personality, ...] = (
    Personality(
        name="concise",
        description="Short, direct answers with no preamble.",
        instructions="Answer directly and briefly. No preamble, no summary of what you are about to do.",
    ),
    Personality(
        name="thorough",
        description="Explores edge cases and states assumptions.",
        instructions=(
            "Work through the problem carefully. Name your assumptions and the edge cases you "
            "considered, but keep the final answer at the top."
        ),
    ),
    Personality(
        name="mentor",
        description="Explains the reasoning so the reader learns from it.",
        instructions=(
            "Explain your reasoning as you go, so someone reading can follow why - not just what. "
            "Prefer teaching the rule over handing over the answer."
        ),
    ),
    Personality(
        name="skeptical",
        description="Challenges premises before accepting them.",
        instructions=(
            "Challenge the premise before acting on it. If the task as stated would not achieve "
            "what the asker seems to want, say so first."
        ),
    ),
)

PERSONALITIES = PresetRegistry("personality", Personality, BUILTIN_PERSONALITIES)


# ---------------------------------------------------------------------------
# Kind 3: color palettes (same mechanism - this was the explicit correction)
# ---------------------------------------------------------------------------


@dataclass
class Palette(Preset):
    """Colors for `render_org_chart`. One hue per rank tier, indexed by
    position in `RoleRank.ORDER`, plus the surface/ink/connector tokens each
    theme needs.

    Only the "dataviz" palette carries the dataviz skill's colorblind-safe
    adjacent-contrast validation. The others are stylistic alternatives and
    are NOT validated for that - said plainly here rather than implied by
    their presence in the same list.
    """

    ranks_light: Sequence[str] = ()
    ranks_dark: Sequence[str] = ()
    surface: Dict[str, str] = field(default_factory=lambda: {"light": "#fcfcfb", "dark": "#1a1a19"})
    ink_primary: Dict[str, str] = field(default_factory=lambda: {"light": "#0b0b0b", "dark": "#ffffff"})
    ink_secondary: Dict[str, str] = field(default_factory=lambda: {"light": "#52514e", "dark": "#c3c2b7"})
    connector: Dict[str, str] = field(default_factory=lambda: {"light": "#c3c2b7", "dark": "#383835"})
    colorblind_validated: bool = False

    def ranks_for(self, theme: str) -> Sequence[str]:
        return self.ranks_dark if theme == "dark" else self.ranks_light

    def color_for_rank(self, rank: str, theme: str) -> str:
        colors = self.ranks_for(theme)
        if not colors:
            return self.ink_primary.get(theme, "#000000")
        try:
            idx = RoleRank.ORDER.index(rank)
        except ValueError:
            idx = len(colors) - 1  # unrecognized rank - last slot rather than crash
        return colors[idx % len(colors)]

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "ranks_light": list(self.ranks_light),
            "ranks_dark": list(self.ranks_dark),
            "colorblind_validated": self.colorblind_validated,
        })
        return out


BUILTIN_PALETTES: Tuple[Palette, ...] = (
    Palette(
        name="dataviz",
        description="Default. Fixed-order categorical hues, validated for colorblind-safe adjacent contrast.",
        ranks_light=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
        ranks_dark=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
        colorblind_validated=True,
    ),
    Palette(
        name="grayscale",
        description="Print-safe monotone ramp, light to dark by seniority. Not hue-coded.",
        ranks_light=("#0b0b0b", "#2f2f2e", "#4a4947", "#666461", "#83817c", "#a09d97", "#bdbab3"),
        ranks_dark=("#ffffff", "#e3e2dd", "#c8c6c0", "#adaba4", "#928f88", "#77746d", "#5c5952"),
    ),
    Palette(
        name="ocean",
        description="Cool blue-green ramp. Stylistic - not contrast-validated.",
        ranks_light=("#0b3d63", "#11577f", "#17729a", "#1e8fa8", "#28a89f", "#4dbd91", "#84cf8e"),
        ranks_dark=("#7fc4ea", "#5db0dd", "#3d9cce", "#2b8ab8", "#2ba396", "#4dbd91", "#96d99c"),
    ),
    Palette(
        name="ember",
        description="Warm red-amber ramp. Stylistic - not contrast-validated.",
        ranks_light=("#6b1616", "#8c2410", "#ad3a0d", "#c75512", "#dd7519", "#eb9a28", "#f2bd4d"),
        ranks_dark=("#f6b48a", "#ef9463", "#e37845", "#d65f30", "#c9822a", "#e0a63a", "#f0c766"),
    ),
)

PALETTES = PresetRegistry("palette", Palette, BUILTIN_PALETTES)


# ---------------------------------------------------------------------------
# Kind 4: org templates
# ---------------------------------------------------------------------------

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
)

ORG_TEMPLATES = PresetRegistry("org template", OrgTemplate, BUILTIN_ORG_TEMPLATES)


# ---------------------------------------------------------------------------
# Bundling the four registries
# ---------------------------------------------------------------------------


@dataclass
class PresetBundle:
    """The four registries a Company consults, in one object.

    Defaults to the module-level built-in registries, which means every
    Company in a process shares them - fine, since presets are read-only value
    objects in normal use. `fork()` gives a Company its own membership so
    registering a custom skill doesn't leak into other Companies (or other
    tests).
    """

    skills: PresetRegistry = field(default_factory=lambda: SKILLS)
    personalities: PresetRegistry = field(default_factory=lambda: PERSONALITIES)
    palettes: PresetRegistry = field(default_factory=lambda: PALETTES)
    org_templates: PresetRegistry = field(default_factory=lambda: ORG_TEMPLATES)

    def fork(self) -> "PresetBundle":
        return PresetBundle(
            skills=self.skills.copy(),
            personalities=self.personalities.copy(),
            palettes=self.palettes.copy(),
            org_templates=self.org_templates.copy(),
        )

    def describe(self) -> Dict[str, List[Dict[str, Any]]]:
        """Everything available, by kind - the payload Phase 8's text mode
        hands an LLM caller so it knows what names it may reference."""
        return {
            "skills": self.skills.describe(),
            "personalities": self.personalities.describe(),
            "palettes": self.palettes.describe(),
            "org_templates": self.org_templates.describe(),
        }

    def names(self) -> Dict[str, List[str]]:
        return {
            "skills": self.skills.names(),
            "personalities": self.personalities.names(),
            "palettes": self.palettes.names(),
            "org_templates": self.org_templates.names(),
        }


def default_bundle() -> PresetBundle:
    return PresetBundle()


# ---------------------------------------------------------------------------
# System-instruction templating
# ---------------------------------------------------------------------------

_SKILL_HEADER = "## Your skills"
_CONSTRAINT_HEADER = "## Hard constraints"
_PERSONALITY_HEADER = "## How you work"


def compose_system_instruction(
    base: str = "",
    personality: Optional[Union[str, Personality]] = None,
    skills: Sequence[Union[str, Skill]] = (),
    bundle: Optional[PresetBundle] = None,
    role_line: str = "",
) -> str:
    """Builds one system instruction from a base string plus presets.

    Deterministic section order - base, role line, personality, skills,
    constraints - because a system instruction that reshuffles between runs
    makes prompt-caching useless and behavior changes impossible to attribute.
    Constraints from every skill are pooled and deduplicated (first occurrence
    wins) so overlapping skills don't repeat the same rule.

    Returns "" for no inputs rather than a header-only skeleton, so an
    employee with no presets gets exactly the empty system instruction it
    would have had before Phase 5.
    """
    bundle = bundle or default_bundle()
    parts: List[str] = []
    if base:
        parts.append(base.strip())
    if role_line:
        parts.append(role_line.strip())

    if personality is not None:
        resolved = bundle.personalities.resolve(personality)
        if resolved.instructions:
            parts.append(f"{_PERSONALITY_HEADER}\n{resolved.instructions.strip()}")

    resolved_skills = bundle.skills.resolve_all(skills)
    skill_blocks = [f"- **{s.name}**: {s.instructions.strip()}" for s in resolved_skills if s.instructions]
    if skill_blocks:
        parts.append(f"{_SKILL_HEADER}\n" + "\n".join(skill_blocks))

    constraints: List[str] = []
    for skill in resolved_skills:
        for constraint in skill.constraints:
            text = constraint.strip()
            if text and text not in constraints:
                constraints.append(text)
    if constraints:
        parts.append(f"{_CONSTRAINT_HEADER}\n" + "\n".join(f"- {c}" for c in constraints))

    return "\n\n".join(parts)


def skill_hints(
    skills: Sequence[Union[str, Skill]], bundle: Optional[PresetBundle] = None
) -> Dict[str, Optional[str]]:
    """The Phase 4 hints a set of skills implies: {"specialty":..., "effort":...}.

    First skill that declares each wins - skills are an ordered list and the
    first one is the primary reason the employee exists. Used by
    `Company.hire()` only when the caller passed no explicit effort/specialty.
    """
    bundle = bundle or default_bundle()
    specialty: Optional[str] = None
    effort: Optional[str] = None
    for skill in bundle.skills.resolve_all(skills):
        if specialty is None and skill.specialty:
            specialty = skill.specialty
        if effort is None and skill.effort:
            effort = skill.effort
    return {"specialty": specialty, "effort": effort}


__all__ = [
    "Preset", "PresetRegistry", "PresetBundle", "default_bundle",
    "Skill", "SKILLS", "BUILTIN_SKILLS",
    "Personality", "PERSONALITIES", "BUILTIN_PERSONALITIES",
    "Palette", "PALETTES", "BUILTIN_PALETTES",
    "RoleSpec", "OrgTemplate", "ORG_TEMPLATES", "BUILTIN_ORG_TEMPLATES", "TASK_SIZES",
    "compose_system_instruction", "skill_hints",
]
