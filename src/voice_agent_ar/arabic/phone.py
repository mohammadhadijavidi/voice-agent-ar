"""UAE phone numbers: parse what callers say or type, store E.164, read back in Arabic.

Covers the formats people actually use in the UAE:

* mobile  ``050 123 4567``, ``0501234567``, ``+971 50 123 4567``, ``00971501234567``,
  ``971501234567`` and ``50 123 4567`` (prefixes 050 052 054 055 056 058)
* landline ``04 123 4567`` (Dubai), 02 Abu Dhabi, 03 Al Ain, 06 Sharjah / Ajman / UAQ,
  07 Ras Al Khaimah, 09 Fujairah
* digits in Arabic-Indic (٠٥٥...) or spoken as words ("صفر خمسة خمسة ...", "خمسين" for 50)

WhatsApp and SMS need E.164 (``+971501234567``), so that is the canonical form.
Numbers outside the UAE are accepted if they are already in international format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import normalize, normalize_digits
from .numbers import digits_to_words, word_value

MOBILE_PREFIXES = {"50": "Etisalat (e&)", "52": "du", "54": "Etisalat (e&)", "55": "du", "56": "Etisalat (e&)",
                   "58": "du"}
LANDLINE_AREAS = {"2": "Abu Dhabi", "3": "Al Ain", "4": "Dubai", "6": "Sharjah / Ajman / UAQ",
                  "7": "Ras Al Khaimah", "9": "Fujairah"}


@dataclass
class UAEPhone:
    e164: str            # +971501234567
    kind: str            # "mobile" | "landline" | "international"
    national: str        # 050 123 4567
    area: str = ""

    @property
    def whatsapp_capable(self) -> bool:
        """Only mobiles can receive WhatsApp / SMS confirmations."""
        return self.kind in ("mobile", "international")


def _spoken_digits(text: str) -> str:
    """Turn "صفر خمسة خمسة واحد ..." into "0551..."; keeps real digits as they are."""
    out = []
    for tok in normalize(text).split():
        if tok.isdigit():
            out.append(tok)
            continue
        if tok.startswith("+") and tok[1:].isdigit():
            out.append(tok)
            continue
        v = word_value(tok)
        if v is not None and v < 100:
            out.append(str(v))
    return "".join(out)


def parse_uae_phone(text: str) -> UAEPhone | None:
    """Best-effort parse. Returns None when the input is not a valid number."""
    if not text:
        return None
    raw = normalize_digits(text)
    digits = re.sub(r"[^\d+]", "", raw)
    if len(re.sub(r"\D", "", digits)) < 7:
        digits = _spoken_digits(text)
    plus = digits.startswith("+")
    d = re.sub(r"\D", "", digits)
    if d.startswith("00"):
        d, plus = d[2:], True
    if d.startswith("971"):
        local = d[3:]
    elif plus:
        if 8 <= len(d) <= 15:
            return UAEPhone(e164="+" + d, kind="international", national="+" + d)
        return None
    elif d.startswith("0"):
        local = d[1:]
    elif len(d) == 9 and d[:2] in MOBILE_PREFIXES:
        local = d
    else:
        return None
    local = local.lstrip("0")
    if len(local) == 9 and local[:2] in MOBILE_PREFIXES:
        return UAEPhone(e164="+971" + local, kind="mobile",
                        national=f"0{local[:2]} {local[2:5]} {local[5:]}", area=MOBILE_PREFIXES[local[:2]])
    if len(local) == 8 and local[0] in LANDLINE_AREAS:
        return UAEPhone(e164="+971" + local, kind="landline",
                        national=f"0{local[0]} {local[1:4]} {local[4:]}", area=LANDLINE_AREAS[local[0]])
    return None


def format_uae_phone(e164: str) -> str:
    p = parse_uae_phone(e164)
    return p.national if p else e164


def speak_phone(e164: str, style: str = "gulf") -> str:
    """Read a number back the way it is written locally, in groups: "صفر خمسة خمسة، واحد اثنين ثلاثة، ..."."""
    p = parse_uae_phone(e164)
    groups = p.national.split() if p and p.kind != "international" else [e164.lstrip("+")]
    return "، ".join(digits_to_words(g, style) for g in groups)


def last_digits(e164: str, n: int = 4) -> str:
    return re.sub(r"\D", "", e164)[-n:]
