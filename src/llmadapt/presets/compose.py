"""presets/compose.py - turning presets into one system instruction.

Section order is fixed - base, role line, personality, skills, constraints -
because an instruction that reshuffles between runs makes prompt caching
useless and behaviour changes impossible to attribute.
"""

from typing import Dict, List, Optional, Sequence, Union

from .bundle import PresetBundle, default_bundle
from .personalities import Personality
from .skills import Skill

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

    **Why this stays in the system instruction rather than being said once in
    the first message.** It reads like the cheaper option - say it once, let
    the compressor hold on to it - but it is the more expensive one on every
    axis that matters here:

    - This function already runs exactly once, at `hire()`. The composed string
      is handed to `Agent(system_instruction=...)` and stored; nothing
      recomposes it per turn. There is no repeated work to remove.
    - A system message is what the providers actually cache. Anthropic, OpenAI
      and Gemini all cache on a stable prefix, so an unchanging system block is
      close to free after the first call - while an instruction buried in the
      first *user* message sits inside the part of the conversation that
      changes, and so is re-billed at full rate every turn.
    - It would put the employee's identity inside the compactable region.
      `HistoryCompactionPolicy` deliberately never touches `history[0]` when it
      is a system message; move these rules into a user turn and a long
      conversation will eventually summarize away the personality and
      constraints the employee was hired with. Silently becoming a different
      employee at turn 40 is a far worse failure than a few repeated tokens.

    So: composed once, lives in the system instruction, never compacted.
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


# Only what this module actually defines. The package-wide list that used to
# live at the bottom of presets.py now lives in presets/__init__.py, which is
# the only place that can honestly claim all of those names.
__all__ = ["compose_system_instruction", "skill_hints"]
