"""presets/bundle.py - the four registries, held together.

A `PresetBundle` is what a Company resolves names against. `fork()` gives one
Company private membership, so registering a custom skill cannot leak into
another Company in the same process - or into another test.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .org_templates import ORG_TEMPLATES
from .palettes import PALETTES
from .personalities import PERSONALITIES
from .registry import PresetRegistry
from .skills import SKILLS

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


__all__ = ["PresetBundle", "default_bundle"]
