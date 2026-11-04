"""Arabic text normalisation for matching (never for display).

STT engines spell the same Gulf word several ways (ابي / أبي / ابى, بكرة / بكره,
الساعة / الساعه). Matching is done on a normalised form; the original text is
kept for anything shown to people or written into a booking.
"""

from __future__ import annotations

import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TASHKEEL = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = "\u0640"
_LETTERS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي", "ئ": "ي", "ؤ": "و",
    "ک": "ك", "ی": "ي", "گ": "ك", "پ": "ب", "چ": "ج", "ڤ": "ف",
    "،": ",", "؛": ";", "؟": "?",
})
_PUNCT = re.compile(r"[^\w\s:/+\-]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_digits(text: str) -> str:
    """Arabic-Indic and Persian digits -> ASCII digits."""
    return text.translate(_DIGITS)


def normalize(text: str) -> str:
    """Lower-cased, diacritic-free, letter-folded form used for matching."""
    t = normalize_digits(text or "")
    t = _TASHKEEL.sub("", t).replace(_TATWEEL, "")
    t = t.translate(_LETTERS).lower()
    t = _PUNCT.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def strip_prefixes(word: str) -> str:
    """Drop clitics that glue onto Gulf words: و (and), ب (at/with), ل (for), ال (the)."""
    w = word
    for p in ("وال", "بال", "لل", "ال"):
        if w.startswith(p) and len(w) > len(p) + 1:
            return w[len(p):]
    if len(w) > 3 and w[0] in "وبل":
        return w[1:]
    return w


ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str, default: str = "ar") -> str:
    """Script-based EN/AR detection; good enough for short STT transcripts."""
    if not text or not text.strip():
        return default
    a = len(ARABIC_SCRIPT.findall(text))
    lat = len(LATIN.findall(text))
    if a == 0 and lat == 0:
        return default
    return "en" if lat > a else "ar"
