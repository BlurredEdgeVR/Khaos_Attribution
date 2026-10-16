"""What a prompt may ask for: the sound, never another artist. A request that names a real artist or singer to imitate
("in the style of X", "sounds like X", "X's voice", "X type beat") or a work to reproduce ("a cover of X") is refused in
both rooms; the model's own artist may be named. A capitalised run counts as a name unless every word of it is a genre,
a place, an era, an instrument or an occasion, so a person named in lowercase can slip through: a guard, not a filter.
"""

from __future__ import annotations

import re

IMITATION_REFUSED = ("Describe the sound you want, not another artist: Khaos makes music only in the voice of the artist "
                     "whose model this is. Refused: {hits}.")

# Words that are never someone to imitate: genres, places and peoples, eras, instruments and gear, occasions, roles.
NON_NAMES = frozenset("""
afrobeat afrobeats afro ambient americana amapiano arabic baroque bebop bhangra big bluegrass blues bollywood bossa
breakbeat britpop broadway calypso celtic chamber chillwave chiptune christmas classical country cumbia dancehall
darkwave disco dixieland doo drill drum dub dubstep easter edm electro electronic emo eurobeat eurodance flamenco folk
funk gamelan garage gospel gothic gregorian grime grunge hardcore highlife hip hop house idm indie industrial italo
j-pop j-rock jazz jungle k-pop krautrock latin lo-fi lofi lounge lullaby madchester mambo mariachi merengue metal
minimal motown neo new noise northern nu orchestral psychedelic punk qawwali r&b ragtime rap reggae reggaeton
renaissance rnb rock rockabilly romantic salsa samba schlager shoegaze ska soul southern surf swing synthpop
synthwave tango techno trance trap tropical uk us vaporwave western world yacht zouk
african american arab argentinian asian atlantic australian balkan berlin brazilian british caribbean chicago chinese
cuban detroit dutch east eastern english european french german greek hawaiian indian irish italian jamaican japanese
korean london los memphis mexican miami middle nashville nordic orleans paris persian portuguese russian scandinavian
scottish spanish swedish texas tokyo turkish vienna welsh
50s 60s 70s 80s 90s 00s 2000s 1950s 1960s 1970s 1980s 1990s
808 909 303 tr-808 tr-909 tb-303 rhodes wurlitzer hammond moog juno prophet mellotron fender gibson les stratocaster
telecaster rickenbacker marshall vox steinway yamaha roland korg arp oberheim minimoog linn mpc sp-1200 theremin
strings
god christmas easter halloween valentine diwali ramadan advent hanukkah
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september
october november december spring summer autumn fall winter morning evening midnight
dj band group artist composer producer singer rapper choir orchestra trio quartet quintet ensemble drummer guitarist
pianist bassist vocalist crooner
""".split())
_NAME = r"(?P<name>[A-Z0-9][\w'’.&-]*(?:\s+[A-Z0-9][\w'’.&-]*){0,3})"
_ARTICLE = r"(?:(?i:a|an|the)\s+)?"
_TRIGGERS = (r"in\s+the\s+(?:style|vein|manner|spirit|voice)\s+of|(?:the\s+)?(?:voice|vocals?|singing|style|sound)\s+of"
             r"|sounds?\s+(?:a\s+(?:lot|bit)\s+)?like|sounding\s+like|sing(?:s|ing)?\s+like|sung\s+(?:like|by)"
             r"|as\s+(?:if\s+)?sung\s+by|vocals?\s+like|play(?:s|ing|ed)?\s+like|reminiscent\s+of"
             r"|(?:similar|comparable|akin)\s+to|(?:à|a)\s+la|inspired\s+by|influenced\s+by|styled\s+after"
             r"|featuring|feat\.|ft\.|(?:cover|remake|version|remix|copy)\s+of"
             r"|imitat(?:e|es|ing|ion\s+of)|impersonat(?:e|es|ing|ion\s+of)|mimic(?:s|king)?|emulat(?:e|es|ing)"
             r"|channel(?:s|ing|ling)?|(?:by|from|like)\s+(?:the\s+)?(?:band|group|artist|composer|producer|singer|rapper|dj)"
             r"|like")
_PATTERNS = [
    re.compile(r"\b(?i:" + _TRIGGERS + r")\s+" + _ARTICLE + _NAME),
    re.compile(_NAME + r"(?i:'s|’s)\s+(?i:voice|vocals?|style|sound|singing|flow)\b"),
    re.compile(_NAME + r"(?i:-style|\s+style|\s+type\s+beat|-like|-esque|\s+vibes?|\s+energy)\b"),
    re.compile(_NAME + r"\s+(?i:would|might|could)\s+(?i:play|sing|write|make|do|produce)\b"),
    re.compile(r"\b(?i:with)\s+" + _NAME + r"\s+(?i:vocals?|on\s+vocals)\b"),
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9&-]+", " ", (text or "").lower()).strip()


# Small words inside a capitalised run mean title case ("A Ballad In The"), not a name.
_FUNCTION = frozenset("a an the in of on to for with and at by from into".split())


def _is_description(name: str) -> bool:
    """A capitalised object that is a genre, a place, an era, an instrument or an occasion, or a run of title case,
    not a person or a band."""
    words = _norm(name).split()
    while words and words[0] in ("a", "an", "the"):
        words = words[1:]
    # A run of description words is a description ("New Order" is a name; "New Orleans Jazz" is not).
    return not words or all(w in NON_NAMES for w in words) or any(w in _FUNCTION for w in words)


def imitation_hits(prompt: str, allowed: tuple[str, ...] | list[str] = ()) -> list[str]:
    """The spans of a prompt that ask for another artist's voice or style; empty when it asks only for a sound.
    `allowed` are the model's own artist's names, which a prompt may use."""
    own = {_norm(a) for a in allowed if a and _norm(a)}
    found = []
    for rx in _PATTERNS:
        for m in rx.finditer(prompt or ""):
            name = _norm(m.group("name"))
            if _is_description(m.group("name")):
                continue
            # The own name exempts only itself; a one-word own name never exempts a longer name that starts with it.
            if any(name == o or (len(o.split()) > 1 and name.startswith(o + " ")) for o in own):
                continue
            found.append((m.start(), m.end(), m.group(0).strip()))
    hits = []
    for start, end, text in sorted(found, key=lambda h: (h[0], -(h[1] - h[0]))):
        if hits and start < hits[-1][1]:
            continue
        hits.append((start, end, text))
    return [t for _, _, t in hits]


def refusal(prompt: str, allowed: tuple[str, ...] | list[str] = ()) -> str | None:
    """The sentence a refused prompt is answered with, or None when it may be rendered."""
    hits = imitation_hits(prompt, allowed)
    return IMITATION_REFUSED.format(hits="; ".join(f"“{h}”" for h in hits)) if hits else None
