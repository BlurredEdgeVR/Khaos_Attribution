"""What a prompt may ask for: the sound, never another artist. A request that names a real artist or singer to imitate
("in the style of X", "sounds like X", "the voice of X", "featuring X") is refused in both rooms; the model's own
artist may be named. Imitating an identifiable person's voice or performance is not the artist's to give.
"""

from __future__ import annotations

import re

IMITATION_REFUSED = ("Describe the sound you want, not another artist: Khaos makes music only in the voice of the artist "
                     "whose model this is. Refused: {hits}.")

# Each pattern ends in the named object; a lowercase object is a description ("sounds like rain"), never a name.
_NAME = r"(?:the\s+)?(?P<name>[A-Z][\w'’.&-]*(?:\s+[A-Z][\w'’.&-]*){0,3})"
_PATTERNS = [
    re.compile(r"\b(?i:in\s+the\s+(?:style|vein|manner|spirit|voice)\s+of)\s+" + _NAME),
    re.compile(r"\b(?:sounds?|sounding|sing(?:s|ing)?|sung|play(?:s|ing|ed)?)\s+(?:a\s+(?:lot|bit)\s+)?like\s+" + _NAME),
    re.compile(r"\b(?:the\s+)?(?:voice|vocals?|singing|style)\s+of\s+" + _NAME),
    re.compile(r"\breminiscent\s+of\s+" + _NAME),
    re.compile(r"\b(?:similar|comparable|akin)\s+to\s+" + _NAME),
    re.compile(r"\b(?:à|a)\s+la\s+" + _NAME),
    re.compile(r"\binspired\s+by\s+" + _NAME),
    re.compile(r"\b(?i:featuring|feat\.|ft\.)\s+" + _NAME),
    re.compile(r"\b(?:cover|remake|version|remix)\s+of\s+" + _NAME),
    re.compile(r"\b(?i:imitat(?:e|es|ing|ion\s+of)|impersonat(?:e|es|ing|ion\s+of)|mimic(?:s|king)?)\s+" + _NAME),
    re.compile(r"\b(?:by|from|like|of)\s+(?:the\s+)?(?:band|group|artist|composer|producer|singer|rapper|dj)\s+"
               r"(?P<name>[\w'’.&-]+(?:\s+[A-Z][\w'’.&-]*){0,3})", re.IGNORECASE),
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def imitation_hits(prompt: str, allowed: tuple[str, ...] | list[str] = ()) -> list[str]:
    """The spans of a prompt that ask for another artist's voice or style; empty when it asks only for a sound.
    `allowed` are the model's own artist's names, which a prompt may use."""
    own = {_norm(a) for a in allowed if a and _norm(a)}
    hits = []
    for rx in _PATTERNS:
        for m in rx.finditer(prompt or ""):
            name, span = _norm(m.group("name")), f" {_norm(m.group(0))} "
            if not name or any(name == o or name.startswith(o + " ") or f" {o} " in span for o in own):
                continue
            hits.append(m.group(0).strip())
    return hits


def refusal(prompt: str, allowed: tuple[str, ...] | list[str] = ()) -> str | None:
    """The sentence a refused prompt is answered with, or None when it may be rendered."""
    hits = imitation_hits(prompt, allowed)
    return IMITATION_REFUSED.format(hits="; ".join(f"“{h}”" for h in hits)) if hits else None
