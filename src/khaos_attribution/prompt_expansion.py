"""Prompt expansion — one idea in, the adapter's caption dialect out.

Shared by the Workshop (Inference tab) and the Listening Space (Simple
mode): both rewrite a listener's short request into the prose dialect an
adapter was trained in, by few-shotting a small LM on the adapter's own
training captions. Everything the two apps must agree on lives here —
the exemplar file shape, which exemplars to show the LM, the query
template, the tidy-up of what comes back, and the trigger rule — so a
tuning lands in both apps at once instead of through "keep in sync"
comments.

The exemplar file (``prompt_exemplars.json``, written by the Workshop at
training completion, read-only everywhere else)::

    {"exemplars_version": "0.2.0", "artist_id", "trigger_phrase",
     "created_at_utc",
     "embedding": {"model": "Qwen3-Embedding-0.6B", "dim": 1024} | null,
     "exemplars": [{"text", "section", "track", "vector": [...] | null}, ...]}

Version 0.1.0 files carry plain strings in ``exemplars`` and no vectors;
``normalise_exemplars`` reads both.

Relevance: the LM copies the *shape* of the examples it sees, so the
examples should be the captions closest to the request — not the first
three in the file (those were three intros). Ranking is by cosine over
``Qwen3-Embedding-0.6B`` vectors when the file carries them and the caller
can embed the request (``embed_texts``; the same encoder ACE-Step already
holds as its text encoder), else by lexical overlap. "Try another" rotates
the window over the ranked list (``attempt``).

Heavy imports stay inside functions: importing this module costs nothing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

EXEMPLARS_VERSION = "0.2.0"
EXEMPLARS_FILENAME = "prompt_exemplars.json"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"

# Tuned live against the 1.7B: six exemplars at temperature 0.85 made the
# LM copy the examples' musical content wholesale; three at 0.5 keep the
# dialect while the request stays in charge.
FEW_SHOT_COUNT = 3
TEMPERATURE = 0.5

# Training captions sit in a tight band (47–66 words across every adapter
# exported so far); the LM is told the band and the result is clipped to
# the exporter's own limit so nothing longer than a training caption ever
# reaches the engine.
CAPTION_WORDS = (45, 70)
MIN_CAPTION_CHARS = 80
MAX_CAPTION_CHARS = 420

_STOPWORDS = frozenset("""
a an and the of for to in on at with by from into over under as is are be
it its this that these those some something very really bit kind sort more
most like music track piece song sound sounds feel feels feeling mood
""".split())
_WORD_RE = re.compile(r"[a-z']+")


# ── exemplars ────────────────────────────────────────────────────────────

def normalise_exemplars(payload: dict | None) -> list[dict]:
    """The file's exemplars as dicts ``{text, section, track, vector}``
    whatever version wrote them. Empty texts are dropped."""
    out: list[dict] = []
    for item in (payload or {}).get("exemplars") or []:
        if isinstance(item, str):
            entry = {"text": item, "section": None, "track": None, "vector": None}
        elif isinstance(item, dict):
            entry = {"text": str(item.get("text") or ""),
                     "section": item.get("section"),
                     "track": item.get("track"),
                     "vector": item.get("vector")}
        else:
            continue
        entry["text"] = " ".join(entry["text"].split())
        if entry["text"]:
            out.append(entry)
    return out


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for w in _WORD_RE.findall(text.lower()):
        w = w.strip("'")
        if len(w) < 3 or w in _STOPWORDS:
            continue
        # Crude stem so "driving" meets "drive" and "drums" meets "drum".
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def lexical_score(request: str, text: str) -> float:
    """How much of the request's vocabulary the caption covers, with a small
    credit for the caption being about that and little else."""
    q, t = _tokens(request), _tokens(text)
    if not q or not t:
        return 0.0
    shared = len(q & t)
    return shared / len(q) + 0.25 * shared / len(t)


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_exemplars(request: str, exemplars: list[dict],
                   request_vector: Iterable[float] | None = None) -> tuple[list[int], str]:
    """Exemplar indices, most relevant first, and the method used:
    ``"embedding"`` when every exemplar carries a vector and the request's
    vector is given, else ``"lexical"``. Stable: ties keep file order."""
    vec = list(request_vector) if request_vector is not None else None
    if vec and exemplars and all(e.get("vector") for e in exemplars):
        scores = [_cosine(vec, e["vector"]) for e in exemplars]
        method = "embedding"
    else:
        scores = [lexical_score(request, e["text"]) for e in exemplars]
        method = "lexical"
    order = sorted(range(len(exemplars)), key=lambda i: -scores[i])
    return order, method


def select_exemplars(request: str, exemplars: list[dict], attempt: int = 0,
                     count: int = FEW_SHOT_COUNT,
                     request_vector: Iterable[float] | None = None) -> tuple[list[int], str]:
    """The ``count`` exemplars to show the LM for this ``attempt``: the top
    of the ranking first, then the next window for each retry, wrapping —
    every attempt sees a different set while the catalogue allows it."""
    order, method = rank_exemplars(request, exemplars, request_vector)
    if not order:
        return [], method
    count = max(1, min(count, len(order)))
    start = (max(0, attempt) * count) % len(order)
    chosen = [order[(start + i) % len(order)] for i in range(count)]
    return chosen, method


# ── the query ────────────────────────────────────────────────────────────

def build_query(request: str, shots: list[str], instrumental: bool = True) -> str:
    """The text handed to the 5Hz LM's Simple-mode entry. Style and content
    are separated on purpose: an early draft let the LM copy the
    exemplars' musical content along with their voice ("dark hypnotic
    groove" came back as bright indie-pop)."""
    lo, hi = CAPTION_WORDS
    examples = "\n".join(f"- {s}" for s in shots)
    vocals = ("The music is instrumental — no vocals, no singing."
              if instrumental else "The music has sung vocals.")
    return (
        "Here are example descriptions of this artist's music. Copy their "
        "writing STYLE — vocabulary, sentence rhythm, level of detail — but "
        "NOT their musical content:\n"
        f"{examples}\n\n"
        f"Request: {request.strip()}\n\n"
        "Write one flowing prose music caption describing music that matches "
        "the request's mood and character, in exactly the writing style of the "
        f"examples: a single paragraph of roughly {lo}–{hi} words. {vocals} "
        "Every musical detail must serve the request. Do not mention BPM, key, "
        "time signature, duration, timestamps or clock times, or any artist, "
        "band or song names."
    )


# ── tidying what comes back ──────────────────────────────────────────────

# The 1.7B leaks clock-time anchors ("At 1:26, the song shifts…") despite
# the instruction, and training captions never contain them (the
# Workshop's cleanup strips all time references). Longer alternatives
# first, or "At" would match inside "At around" and leave the stamp behind.
_TIMESTAMP = re.compile(
    r"(?:,\s*)?\b(?:[Aa]t\s+around|[Aa]round|[Aa]t|[Aa]fter|[Bb]y|[Nn]ear)\s+"
    r"(?:the\s+)?\d{1,2}:\d{2}(?:\s+mark)?,?\s*"
    r"|\(\s*\d{1,2}:\d{2}\s*\)"
)
_TIDY_SPACES = re.compile(r"\s{2,}")
_TIDY_PUNCT = re.compile(r"\s+([.,;!?])")


def strip_timestamps(text: str) -> str:
    text = _TIMESTAMP.sub("", text)
    text = _TIDY_SPACES.sub(" ", text)
    return _TIDY_PUNCT.sub(r"\1", text).strip(" ,;")


def clip_caption(text: str, limit: int = MAX_CAPTION_CHARS) -> str:
    """Clip to the last full sentence inside ``limit`` (the whole caption
    when it fits) — the exporter's rule, applied to the LM's output too."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot > MIN_CAPTION_CHARS else cut.rstrip() + "…"


def tidy_caption(text: str) -> str:
    return clip_caption(strip_timestamps(text or ""))


def with_trigger(caption: str, trigger: str | None) -> str:
    """``trigger`` in front of ``caption`` unless it is already there —
    a rework's base caption may carry the trigger (even a longer, per-track
    one) from the generation it derives from."""
    caption = (caption or "").strip()
    trigger = (trigger or "").strip()
    if not trigger:
        return caption
    artist = trigger.split()[0]
    if caption == artist or caption.startswith(artist + " "):
        return caption
    return f"{trigger} {caption}".strip()


# ── harvesting the LM's whole answer ─────────────────────────────────────

_FLAT_TO_SHARP = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#",
                  "CB": "B", "FB": "E"}
_KEYSCALE = re.compile(r"^\s*([A-Ga-g])\s*([#♯b♭]?)\s*[- ]?\s*(major|minor|maj|min|m)?\s*$", re.IGNORECASE)


def normalise_keyscale(value) -> str | None:
    """``"A minor"`` form — the form the apps' key pickers list (sharps
    only) — from the LM's spellings (``a min``, ``Bb Major``, ``F#m``…)."""
    if not isinstance(value, str):
        return None
    m = _KEYSCALE.match(value)
    if not m:
        return None
    note = m.group(1).upper()
    accidental = m.group(2)
    if accidental in ("#", "♯"):
        note += "#"
    elif accidental in ("b", "♭"):
        note = _FLAT_TO_SHARP.get(note + "B", note + "b")
    mode = (m.group(3) or "major").lower()
    mode = "minor" if mode in ("minor", "min", "m") else "major"
    return f"{note} {mode}"


def normalise_timesignature(value) -> str | None:
    """Beats per bar as the engines take it (``"4"``), from ``"4/4"``,
    ``"6/8"``, ``4``…"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        beats = int(value)
    elif isinstance(value, str):
        m = re.match(r"^\s*(\d{1,2})(?:\s*/\s*\d{1,2})?\s*$", value)
        if not m:
            return None
        beats = int(m.group(1))
    else:
        return None
    return str(beats) if 2 <= beats <= 12 else None


def _int_in(value, lo: int, hi: int) -> int | None:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def harvest(metadata: dict | None, instrumental: bool = True) -> dict:
    """Everything usable from the LM's answer, normalised: the caption
    tidied; tempo, key and time signature in the apps' own forms (``None``
    when absent or implausible); a lyrics draft only for a sung request
    (the LM's ``[Instrumental]`` placeholder is not a draft)."""
    md = metadata or {}
    lyrics = md.get("lyrics")
    if instrumental or not isinstance(lyrics, str) or not lyrics.strip() \
            or lyrics.strip().lower() in ("[instrumental]", "instrumental"):
        lyrics = None
    language = md.get("language")
    return {
        "caption": tidy_caption(str(md.get("caption") or "")),
        "bpm": _int_in(md.get("bpm"), 30, 300),
        "keyscale": normalise_keyscale(md.get("keyscale")),
        "timesignature": normalise_timesignature(md.get("timesignature")),
        "duration": _int_in(md.get("duration"), 10, 600),
        "lyrics": lyrics.strip() if lyrics else None,
        "language": language if isinstance(language, str) and language.strip()
                    and language.strip().lower() != "unknown" else None,
    }


# ── embeddings (optional; torch + transformers) ──────────────────────────

def embed_texts(model, tokenizer, texts: list[str], max_length: int = 256):
    """L2-normalised mean-pooled embeddings from a loaded Qwen3-Embedding
    encoder (``transformers`` AutoModel + AutoTokenizer — the pair ACE-Step
    holds as its text encoder). Returns a ``(len(texts), dim)`` numpy
    array. Caller supplies the model; this function loads nothing."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for text in texts:
            enc = tokenizer(text, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask.to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            rows.append(torch.nn.functional.normalize(pooled, dim=-1)[0].cpu().numpy())
    return np.stack(rows).astype(np.float32)


class TextEmbedder:
    """Lazy owner of a Qwen3-Embedding encoder for processes that do not
    already hold one (the Workshop's exporter). ``available()`` says
    whether the weights and libraries are present without loading them."""

    def __init__(self, model_dir, device: str | None = None):
        from pathlib import Path  # noqa: PLC0415
        self.model_dir = Path(model_dir)
        self.device = device
        self._pair = None

    def available(self) -> bool:
        if not (self.model_dir / "config.json").exists():
            return False
        try:
            import torch  # noqa: F401, PLC0415
            import transformers  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def _load(self):
        if self._pair is None:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
            device = self.device
            if device is None:
                device = ("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            model = AutoModel.from_pretrained(str(self.model_dir)).to(device).eval()
            self._pair = (model, tokenizer)
        return self._pair

    def embed(self, texts: list[str]):
        model, tokenizer = self._load()
        return embed_texts(model, tokenizer, texts)
