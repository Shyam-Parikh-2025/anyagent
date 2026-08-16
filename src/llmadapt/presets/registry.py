"""presets/registry.py - the one mechanism, on its own.

`PresetRegistry` is instantiated four times - skills, personalities, palettes,
org templates - and Phase 5's whole point was that it is the *same* class each
time rather than four bespoke systems. Keeping the mechanism in its own module
makes that literal: the four catalogs are data files that import from here, and
none of them can quietly grow behaviour the others don't have.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Union

from ..suggest import did_you_mean

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
            # Listing everything is what makes a typo recoverable; naming the
            # nearest match is what makes it recoverable at a glance, once the
            # list is longer than a line. Same helper policy.py uses for a
            # misspelled effort hint, so both halves of the library answer a
            # typo the same way.
            suggestion = did_you_mean(str(name), self._items)
            raise KeyError(
                f"unknown {self.kind} {name!r}{suggestion}. Available: {available}"
            ) from None

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


__all__ = ["Preset", "PresetRegistry"]
