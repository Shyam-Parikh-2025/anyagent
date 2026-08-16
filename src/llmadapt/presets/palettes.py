"""presets/palettes.py - org-chart colour schemes, picked by name.

Palettes go through the same registry as skills and templates. That was an
explicit correction during Phase 5: colour is not a bespoke system living
inside `observability.py`, it is another named thing a user picks, ships
defaults for, and can extend.

**Only `dataviz` carries the colourblind-safe adjacent-contrast validation.**
Every other palette here is stylistic and says so in its own description,
rather than implying validation by sitting in the same list.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

from ..router import RoleRank
from .registry import Preset, PresetRegistry

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
    Palette(
        name="forest",
        description="Green-through-moss ramp. Stylistic - not contrast-validated.",
        ranks_light=("#14401f", "#1c5a2c", "#24743a", "#3a8c4b", "#5aa361", "#7fb97c", "#a6cf9c"),
        ranks_dark=("#a6cf9c", "#8bc189", "#6fb374", "#57a05f", "#48874f", "#3a6e40", "#2c5531"),
    ),
    Palette(
        name="violet",
        description="Indigo-through-orchid ramp. Stylistic - not contrast-validated.",
        ranks_light=("#2e1065", "#43168f", "#5b21b6", "#7c3aed", "#9b64f0", "#b98df4", "#d4b5f8"),
        ranks_dark=("#d4b5f8", "#c19bf6", "#ab7cf2", "#9560ee", "#7f4ad8", "#6838b4", "#512a90"),
    ),
    Palette(
        name="slate",
        description="Cool blue-grey ramp - quieter than grayscale but still low-chroma. Stylistic.",
        ranks_light=("#101828", "#1d2939", "#344054", "#475467", "#667085", "#98a2b3", "#c4cbd6"),
        ranks_dark=("#e6eaf0", "#cdd4e0", "#b0bacb", "#94a0b5", "#78859c", "#5d6a80", "#455065"),
    ),
    Palette(
        name="high-contrast",
        description=(
            "Maximum luminance separation against the background, for projection or low-quality "
            "screens. Note this is a *luminance* choice, not the colourblind validation - only "
            "the dataviz palette carries that."
        ),
        ranks_light=("#000000", "#8f0000", "#00458f", "#005c2e", "#6b3f00", "#5c0080", "#333333"),
        ranks_dark=("#ffffff", "#ff8080", "#8fc4ff", "#79e0a5", "#ffcc66", "#e2a3ff", "#cccccc"),
    ),

    # --- Additional stylistic ramps -------------------------------------
    # Same contract as the ones above: seven steps, light and dark, ordered
    # by seniority. None of these claim the colourblind validation - that is
    # dataviz's alone, and it is a claim you earn by checking, not by taste.
    Palette(
        name="sunset",
        description="Magenta-through-gold ramp. Stylistic - not contrast-validated.",
        ranks_light=("#6f2054", "#872450", "#a02843", "#ba2b2b", "#d3542f", "#db8744", "#e2b55a"),
        ranks_dark=("#e9a5d2", "#e592b7", "#e37f95", "#e16b6b", "#df7556", "#de8742", "#dda22c"),
    ),
    Palette(
        name="teal",
        description="Deep teal to pale aqua. Stylistic - not contrast-validated.",
        ranks_light=("#11525f", "#1a727c", "#249497", "#30b1ab", "#41c5b6", "#62c9b7", "#81cfbc"),
        ranks_dark=("#b0dfe8", "#9cdae1", "#88d7d9", "#75d1cd", "#63c9bd", "#50c0ab", "#43b196"),
    ),
    Palette(
        name="rose",
        description="Wine-through-blush ramp. Stylistic - not contrast-validated.",
        ranks_light=("#602030", "#7e2838", "#9c303c", "#bc373c", "#cd4f4b", "#d77268", "#e09485"),
        ranks_dark=("#eab8c4", "#e5a4af", "#e08f98", "#db7a7e", "#d76966", "#d25d50", "#ce543b"),
    ),
    Palette(
        name="sand",
        description="Warm neutral ramp, low chroma. Stylistic - not contrast-validated.",
        ranks_light=("#534128", "#6b5635", "#826b43", "#988152", "#aa9566", "#b5a780", "#c2b899"),
        ranks_dark=("#e4d9c8", "#daccb5", "#d0c0a3", "#c5b490", "#baa87e", "#af9d6d", "#a3915c"),
    ),
    Palette(
        name="midnight",
        description="Near-black through indigo. Stylistic - not contrast-validated.",
        ranks_light=("#12163f", "#1c1e5b", "#282677", "#393091", "#4b3cab", "#644ebf", "#826bc7"),
        ranks_dark=("#babee8", "#a3a5e0", "#8f8dd8", "#7e76d0", "#6e60c8", "#6049c0", "#593dae"),
    ),
    Palette(
        name="mint",
        description="Cool green ramp, lighter than forest. Stylistic - not contrast-validated.",
        ranks_light=("#24604c", "#2e7c5e", "#37986d", "#40b57a", "#56c487", "#71cf95", "#8cd9a6"),
        ranks_dark=("#bce6d8", "#a8dfca", "#94d8ba", "#7fd1a8", "#6acb95", "#55c581", "#40bf6a"),
    ),
    Palette(
        name="copper",
        description="Brown-through-apricot ramp. Stylistic - not contrast-validated.",
        ranks_light=("#5f301c", "#7d4123", "#9b552a", "#ba6a31", "#cf8241", "#d89a5e", "#e0b17b"),
        ranks_dark=("#e6c2b3", "#e0b59e", "#dba98a", "#d79e75", "#d3945f", "#cf8c49", "#cc8533"),
    ),
    Palette(
        name="plum",
        description="Aubergine through lilac. Stylistic - not contrast-validated.",
        ranks_light=("#491f51", "#682a71", "#893591", "#ac3fb1", "#c456c4", "#d075cc", "#db94d5"),
        ranks_dark=("#e2c4e9", "#daade1", "#d396d9", "#ce7fd1", "#ca68ca", "#c350bf", "#b83dad"),
    ),
    Palette(
        name="steel",
        description="Blue-grey ramp, cooler and flatter than slate. Stylistic.",
        ranks_light=("#243342", "#33475a", "#435b72", "#537088", "#66849d", "#8098aa", "#99abb8"),
        ranks_dark=("#cbd6e2", "#b7c6d5", "#a3b7c8", "#90a8bb", "#7e98ad", "#6c899f", "#5e798d"),
    ),
    Palette(
        name="autumn",
        description="Rust-through-ochre ramp. Stylistic - not contrast-validated.",
        ranks_light=("#6a281b", "#823b21", "#995128", "#b16b30", "#c88837", "#cda250", "#d3b969"),
        ranks_dark=("#e6b2a8", "#e1a995", "#dca382", "#d79f6f", "#d29e5c", "#cea049", "#c9a436"),
    ),
    Palette(
        name="arctic",
        description="Ice blue through near-white. Stylistic - not contrast-validated.",
        ranks_light=("#20546f", "#2e6f8c", "#3d8aa7", "#54a1ba", "#77b1c2", "#97c2cc", "#b6d2d8"),
        ranks_dark=("#d1e5f0", "#bbdae7", "#a6cedd", "#92c3d3", "#7eb8c8", "#6cacbc", "#5aa1af"),
    ),
    Palette(
        name="sepia",
        description="Photographic warm monotone. Print-friendly, hue-flat.",
        ranks_light=("#3c2e20", "#574431", "#725a43", "#8b7055", "#a1866c", "#b09c89", "#c0b2a5"),
        ranks_dark=("#e5dbd1", "#d8c9bb", "#cab8a6", "#bba690", "#ac947c", "#9d8267", "#87705a"),
    ),

)

PALETTES = PresetRegistry("palette", Palette, BUILTIN_PALETTES)


__all__ = ["Palette", "BUILTIN_PALETTES", "PALETTES"]
