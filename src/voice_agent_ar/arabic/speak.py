"""Make Arabic text safe to send to a TTS engine.

LLMs write "حجزك الساعة 20:30 لـ 4 أشخاص، الكشف 150 درهم، رقمك 0501234567". Arabic
TTS voices read that unpredictably (English digits, "عشرون نقطتان ثلاثون", or a
phone number as "خمسمئة مليون..."). ``to_speech`` rewrites it before synthesis:

    الساعة ثمان ونص بالليل لأربعة أشخاص، الكشف مية وخمسين درهم،
    رقمك صفر خمسة صفر، واحد اثنين ثلاثة، أربعة خمسة ستة سبعة

The transcript and call log keep the original text; only the audio uses this form.
"""

from __future__ import annotations

import re
from datetime import time

from .normalize import normalize_digits
from .numbers import number_to_words, people, price, time_to_words
from .phone import parse_uae_phone, speak_phone

_PHONE = re.compile(r"(?<![\d])(\+?\d[\d\s\-]{7,16}\d)(?![\d])")
_TIME = re.compile(r"(?:(?:ال)?ساعة\s*)?\b([01]?\d|2[0-3])[:٫.]([0-5]\d)\b(\s*(?:صباحاً|صباحا|مساءً|مساء|am|pm|AM|PM|ص|م)(?![\wً-ْ]))?")
_PRICE = re.compile(r"\b(\d+(?:\.\d{1,2})?)\s*(درهم|دراهم|درهماً|AED|aed|د\.إ)")
_PRICE_PRE = re.compile(r"(AED|Dhs?\.?)\s*(\d+(?:\.\d{1,2})?)\b")
_PEOPLE = re.compile(r"(لـ\s*|ل\s*)?\b(\d{1,3})\s*(أشخاص|اشخاص|شخص|نفر|ضيوف)")
_NUMBER = re.compile(r"\b\d+\b")


def _time_sub(style: str):
    def f(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        suffix = (m.group(3) or "").strip().lower()
        if suffix in ("مساءً", "مساء", "م", "pm") and h < 12:
            h += 12
        elif suffix in ("صباحاً", "صباحا", "ص", "am") and h == 12:
            h = 0
        return time_to_words(time(h, mi), style)
    return f


def to_speech(text: str, style: str = "gulf") -> str:
    """Rewrite digits, times, prices, party sizes and phone numbers as spoken Arabic."""
    if not text:
        return text
    t = normalize_digits(text).replace("ـ", "")

    def phone(m: re.Match) -> str:
        p = parse_uae_phone(m.group(1))
        return speak_phone(p.e164, style) if p else m.group(0)
    t = _PHONE.sub(phone, t)
    t = _TIME.sub(_time_sub(style), t)
    # "الساعة الساعة" can appear when the LLM wrote "الساعة 8:30"
    t = re.sub(r"(الساعة\s+)+الساعة", "الساعة", t)
    t = _PRICE.sub(lambda m: price(float(m.group(1)), "AED", style), t)
    t = _PRICE_PRE.sub(lambda m: price(float(m.group(2)), "AED", style), t)

    def ppl(m: re.Match) -> str:
        n = int(m.group(2))
        if m.group(1):
            # after the preposition the dual is genitive in MSA too: "لشخصين"
            return "ل" + ("شخصين" if n == 2 else people(n, style))
        return people(n, style)
    t = _PEOPLE.sub(ppl, t)
    t = _NUMBER.sub(lambda m: number_to_words(int(m.group(0)), style) if len(m.group(0)) <= 6 else
                    " ".join(number_to_words(int(c), style) for c in m.group(0)), t)
    return re.sub(r"\s{2,}", " ", t).strip()
