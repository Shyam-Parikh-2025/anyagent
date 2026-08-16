"""llmadapt.presets - ONE named-preset registry pattern, reused identically for
skills, personality presets, org templates and colour palettes.

That reuse is the whole point of Phase 5, and it was an explicit correction
from the project owner: palettes are not a bespoke system bolted onto
`observability.py`, and org templates are not a bespoke system bolted onto
`company.py`. All four are *named things a user picks by name, ships defaults
for, and can extend with their own* - which is one mechanism, `PresetRegistry`,
instantiated four times.

Consequences of committing to that:

- Every registry behaves the same way: `.get("name")`, `.register(obj)`,
  `.names()`, `.resolve(name_or_object)`, `.copy()`. Learn it once.
- Anything taking a preset takes *either* a name or the object, via
  `resolve()`. `hire(skills=["python"])` and `hire(skills=[my_skill])` both
  work, and the object form needs no registration at all - useful for a
  one-off skill you don't want to name globally.
- `PresetRegistry.names()` gives Phase 8's `set_company_up` schema its enum
  values for free, for all four kinds, from one method.
- A user who wants to change the built-ins forks them with `.copy()` and hands
  the fork to their `Company` in a `PresetBundle`, instead of mutating module
  globals another Company in the same process is also using.

**This used to be a single 748-line module and is now a package**, because
roughly half of it was *content* rather than mechanism - and content is what
grows. The split is along the seam that was already there:

    registry.py        Preset + PresetRegistry - the mechanism, alone
    skills.py          the Skill type and the built-in skill catalog
    personalities.py   the Personality type and its catalog
    palettes.py        the Palette type and its catalog
    org_templates.py   RoleSpec / OrgTemplate and the built-in shapes
    bundle.py          PresetBundle - the four registries held together
    compose.py         compose_system_instruction / skill_hints

Adding a built-in is now an edit to one small catalog file rather than a diff
in the middle of everything, and the mechanism cannot accidentally grow a
per-kind special case, because the mechanism no longer sits next to any one
kind's data.

Every name that was importable from `llmadapt.presets` before the split still
is - `tests/test_import_paths.py` asserts it - so no caller changed.

Two deliberate limits, stated rather than hidden:

- Skills are *prompt-level* things: instructions plus constraints templated
  into the system instruction. A skill cannot register tools or change the
  tool-calling loop. "Don't use library X" is a sentence in a prompt, not an
  enforced sandbox - a model can ignore it, and nothing here verifies
  compliance.
- Org templates instantiate a *shape* (who exists, who reports to whom, what
  each is good at). They do not size a task; `size=` is an argument the caller
  opts into, never inferred from the task text.
"""

from .bundle import PresetBundle, default_bundle
from .compose import compose_system_instruction, skill_hints
from .org_templates import (
    BUILTIN_ORG_TEMPLATES,
    ORG_TEMPLATES,
    TASK_SIZES,
    OrgTemplate,
    RoleSpec,
)
from .palettes import BUILTIN_PALETTES, PALETTES, Palette
from .personalities import BUILTIN_PERSONALITIES, PERSONALITIES, Personality
from .registry import Preset, PresetRegistry
from .skills import BUILTIN_SKILLS, SKILLS, Skill

__all__ = [
    "Preset", "PresetRegistry", "PresetBundle", "default_bundle",
    "Skill", "SKILLS", "BUILTIN_SKILLS",
    "Personality", "PERSONALITIES", "BUILTIN_PERSONALITIES",
    "Palette", "PALETTES", "BUILTIN_PALETTES",
    "RoleSpec", "OrgTemplate", "ORG_TEMPLATES", "BUILTIN_ORG_TEMPLATES", "TASK_SIZES",
    "compose_system_instruction", "skill_hints",
]
