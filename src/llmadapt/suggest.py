"""suggest.py - "did you mean ...?" for the library's named things.

One tiny function, in its own module for one reason: both `presets.py` (skill,
personality, palette and template names) and `policy.py` (effort hints) need
it, and neither imports the other. Putting it in either would create a
dependency between two modules that are deliberately independent - presets
knows nothing about routing, policy knows nothing about presets - so it lives
where both can reach it and nothing is coupled.

Why it exists at all: llmadapt takes plain strings for all of these, and that
is on purpose, because the strings have to survive a JSON spec file and an LLM
filling in a tool schema, which an enum cannot. The cost of that choice is
typos, and the two halves of the library had drifted into handling them
differently - a bad skill name raised a KeyError listing every valid name,
while a bad effort hint silently fell back to the lowest tier. Neither
behaviour was wrong on its own; the asymmetry was. This is the shared piece
that lets both say the same helpful thing.

Uses `difflib` from the standard library - no dependency, and the same
matching the interpreter itself uses for its own "did you mean" hints.
"""

import difflib
from typing import Iterable, Optional

__all__ = ["did_you_mean", "closest_name"]


def closest_name(value: str, options: Iterable[str], cutoff: float = 0.6) -> Optional[str]:
    """The single closest option to `value`, or None if nothing is close.

    `cutoff` is difflib's similarity ratio. 0.6 is difflib's own default and
    sits about right for names of this length: it catches "balnced" ->
    "balanced" and "pyhton" -> "python" while refusing to suggest "python" for
    an unrelated word, which is worse than saying nothing.
    """
    if not value:
        return None
    candidates = [option for option in options if option]
    matches = difflib.get_close_matches(value, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def did_you_mean(value: str, options: Iterable[str], cutoff: float = 0.6) -> str:
    """A ready-to-append suggestion clause, or "" when there is nothing to say.

    Returns the leading space and punctuation too - `f"unknown thing{...}"` -
    so callers can concatenate unconditionally instead of writing the same
    "if there is a suggestion" branch at every call site.
    """
    match = closest_name(value, options, cutoff=cutoff)
    return f" (did you mean {match!r}?)" if match else ""
