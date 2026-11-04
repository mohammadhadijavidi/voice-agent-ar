"""Gulf (Emirati / Khaleeji) Arabic understanding for booking calls.

Deterministic extractors for the slots a restaurant or clinic booking needs:
day, time, party size, name, yes/no, seating, and caller intents. They run on STT
transcripts, which arrive in many spellings (باجر / بكرة / بكره, ابي / أبغى / ابغي,
ثمان ونص / 8 ونص / ٨:٣٠), so everything is matched on ``normalize()``-d text.

These helpers are used by the mock LLM (tests, offline demo), by the tools (to
accept whatever the LLM passes, e.g. "باجر" or "8 ونص بالليل"), and are handy for
your own rules (e.g. detecting an emergency before the LLM answers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .normalize import normalize, strip_prefixes
from .numbers import MONTHS_AR, WEEKDAYS_AR, parse_number, word_value


def _n(words) -> list[str]:
    return [normalize(w) for w in words]


def _has(text_n: str, phrases: list[str]) -> bool:
    padded = f" {text_n} "
    return any(f" {p} " in padded for p in phrases)


# ---------------------------------------------------------------------------
# intents
# ---------------------------------------------------------------------------

YES = _n(["ايه", "اي", "إي", "ايوه", "ايوا", "أيوه", "هيه", "هي نعم", "نعم", "اكيد", "أكيد", "تمام", "زين",
          "اوكي", "اوك", "ok", "okay", "yes", "yeah", "يب", "صح", "صحيح", "مضبوط", "ماشي", "طيب", "اكيد اكيد",
          "ان شاء الله", "إن شاء الله", "يصير", "عادي", "بالضبط", "تمام التمام", "حلو"])
NO = _n(["لا", "لاء", "لأ", "مب", "مو", "no", "nope", "ما ابي", "ما ابا", "ما بغيت", "مب لازم", "لا لا", "غلط"])
GOODBYE = _n(["مع السلامه", "مع السلامة", "فمان الله", "في امان الله", "الله يحفظك", "الله يعطيك العافيه",
              "يعطيك العافيه", "مشكور", "مشكوره", "مشكورة", "شكرا", "خلاص شكرا", "باي", "bye", "goodbye",
              "thank you bye", "سلام", "تسلم", "تسلمين", "ما قصرت", "ما قصرتي"])
HUMAN = _n(["موظف", "موظفه", "حد يرد", "ابي اكلم حد", "اكلم حد", "انسان", "شخص حقيقي", "المدير", "المسؤول",
            "الريسبشن", "الاستقبال", "خدمه العملاء", "خدمة العملاء", "operator", "human", "real person",
            "manager", "staff"])
BOOK = _n(["احجز", "حجز", "موعد", "طاوله", "طاولة", "ميز", "table", "book", "booking", "reservation",
           "reserve", "appointment", "ابي اجي", "بنجي", "بنيي", "نبي نجي", "كشف", "مراجعه", "مراجعة"])
HOURS_Q = _n(["متى تفتحون", "متى تسكرون", "متى تبطلون", "الدوام", "اوقات", "أوقات", "ساعات العمل",
              "فاتحين", "مسكرين", "opening hours", "what time do you open", "مواعيد العمل"])
LOCATION_Q = _n(["وين مكانكم", "وين المكان", "وين موقعكم", "وينكم", "العنوان", "اللوكيشن", "لوكيشن",
                 "where are you", "address", "location", "باركنج", "مواقف"])
EMERGENCY = _n(["طوارئ", "طوارىء", "اسعاف", "إسعاف", "نزيف", "ما اقدر اتنفس", "ما يقدر يتنفس", "الم شديد",
                "ألم شديد", "حادث", "اغماء", "إغماء", "طاح", "emergency", "ambulance", "can't breathe"])
WHATSAPP = _n(["واتساب", "واتس", "الواتس", "الواتساب", "whatsapp", "وتساب"])
SMS = _n(["رساله نصيه", "رسالة نصية", "sms", "مسج", "مسج عادي", "رساله عاديه", "رسالة عادية", "اس ام اس"])
NO_MESSAGE = _n(["لا ترسل", "ما ابي رساله", "ما ابي رسالة", "لا ترسلون", "بدون رساله", "بدون رسالة",
                 "مب لازم رساله"])
CANCEL = _n(["الغي", "ألغي", "الغاء", "إلغاء", "كنسل", "cancel"])


def is_yes(text: str) -> bool:
    t = normalize(text)
    if is_no(text):
        return False
    return _has(t, YES)


def is_no(text: str) -> bool:
    t = normalize(text)
    first = t.split()[0] if t.split() else ""
    return first in ("لا", "لاء", "لا،", "مب", "مو", "no", "nope") or _has(t, NO[5:])


def is_goodbye(text: str) -> bool:
    return _has(normalize(text), GOODBYE)


def wants_human(text: str) -> bool:
    return _has(normalize(text), HUMAN)


def wants_booking(text: str) -> bool:
    t = normalize(text)
    return _has(t, BOOK) or any(strip_prefixes(w) in ("حجز", "موعد", "طاوله") for w in t.split())


def is_emergency(text: str) -> bool:
    return _has(normalize(text), EMERGENCY)


def asks_hours(text: str) -> bool:
    return _has(normalize(text), HOURS_Q)


def asks_location(text: str) -> bool:
    return _has(normalize(text), LOCATION_Q)


def channel_preference(text: str) -> str | None:
    """'whatsapp' | 'sms' | 'none' if the caller states how they want the confirmation."""
    t = normalize(text)
    if _has(t, NO_MESSAGE):
        return "none"
    if _has(t, SMS):
        return "sms"
    if _has(t, WHATSAPP):
        return "whatsapp"
    return None


# ---------------------------------------------------------------------------
# day
# ---------------------------------------------------------------------------

_WEEKDAY_FORMS = {i: _n([w, w.removeprefix("ال")]) for i, w in enumerate(WEEKDAYS_AR)}
_WEEKDAY_FORMS[0] += _n(["الاثنين", "الإثنين", "اثنين", "monday"])
_WEEKDAY_FORMS[1] += _n(["الثلاثاء", "ثلاثا", "الثلاثا", "tuesday"])
_WEEKDAY_FORMS[2] += _n(["الاربعاء", "اربعا", "الاربعا", "الربوع", "wednesday"])
_WEEKDAY_FORMS[3] += _n(["الخميس", "thursday"])
_WEEKDAY_FORMS[4] += _n(["الجمعه", "الجمعة", "جمعه", "friday"])
_WEEKDAY_FORMS[5] += _n(["السبت", "saturday"])
_WEEKDAY_FORMS[6] += _n(["الاحد", "الأحد", "احد", "sunday"])
_MONTHS = {normalize(m): i + 1 for i, m in enumerate(MONTHS_AR)}
_MONTHS.update({normalize(m): i + 1 for i, m in enumerate(
    ["كانون الثاني", "شباط", "اذار", "نيسان", "ايار", "حزيران", "تموز", "اب", "ايلول", "تشرين الاول",
     "تشرين الثاني", "كانون الاول"])})
_MONTHS.update({"ابريل": 4, "إبريل": 4, "اغسطس": 8, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
                "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12})

TODAY = _n(["اليوم", "الحين", "الليله", "الليلة", "هالليله", "today", "tonight", "العصر اليوم"])
TOMORROW = _n(["باجر", "بكره", "بكرة", "بكرا", "غدا", "غداً", "بجر", "tomorrow"])
AFTER_TOMORROW = _n(["عقب باجر", "بعد باجر", "بعد بكره", "بعد بكرة", "بعد بكرا", "بعد غد", "عقب بكره",
                     "day after tomorrow"])
NEXT_WEEK = _n(["الاسبوع الياي", "الأسبوع الياي", "الاسبوع الجاي", "الاسبوع القادم", "الأسبوع القادم",
                "next week"])


def parse_day(text: str, today: date) -> date | None:
    """Day the caller means, or None. Weekdays resolve to the next occurrence after today."""
    t = normalize(text)
    if _has(t, AFTER_TOMORROW):
        return today + timedelta(days=2)
    if _has(t, TOMORROW):
        return today + timedelta(days=1)
    # explicit dates: 2026-11-05, 5/11, 5 نوفمبر, خمسه نوفمبر
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        y = y + 2000 if y < 100 else y
        try:
            res = date(y, mo, d)
            if res < today and not m.group(3):
                res = date(y + 1, mo, d)
            return res
        except ValueError:
            pass
    toks = t.split()
    for i, tok in enumerate(toks):
        if tok in _MONTHS and i > 0:
            d = word_value(toks[i - 1])
            if d and 1 <= d <= 31:
                try:
                    res = date(today.year, _MONTHS[tok], d)
                    return res if res >= today else date(today.year + 1, _MONTHS[tok], d)
                except ValueError:
                    return None
    next_week = _has(t, NEXT_WEEK)
    for tok in toks:
        base = tok[1:] if tok.startswith(("ي", "ب")) and normalize("ال") in tok[:3] else tok
        for wd, forms in _WEEKDAY_FORMS.items():
            if tok in forms or base in forms or strip_prefixes(tok) in forms:
                ahead = (wd - today.weekday()) % 7 or 7
                res = today + timedelta(days=ahead)
                if next_week and ahead < 7 and res.isocalendar()[1] == today.isocalendar()[1]:
                    res += timedelta(days=7)
                return res
    if _has(t, TODAY):
        return today
    return None


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

@dataclass
class TimeExpr:
    hour: int                 # 0-23 if part is known, else 1-12 as said
    minute: int
    part: str | None          # "am" | "pm" | "noon" | None (ambiguous)

    def candidates(self) -> list[tuple[int, int]]:
        """24h interpretations, most likely first."""
        h = self.hour
        if self.part == "am":
            return [(0 if h == 12 else h, self.minute)]
        if self.part == "pm":
            return [(h + 12 if h < 12 else h, self.minute)]
        if self.part == "noon":
            return [(12 if h == 12 else (h + 12 if h < 6 else h), self.minute)]
        if h >= 13:
            return [(h, self.minute)]
        # "الساعة ثمان" with no day part: evening first for 1-11 (bookings skew late in the Gulf)
        pm = (h + 12 if h < 12 else 12, self.minute)
        am = (0 if h == 12 else h, self.minute)
        return [pm, am] if h != 12 else [(12, self.minute), (0, self.minute)]

    def resolve(self, is_open=None) -> str:
        """Pick the first interpretation the business is open for (``is_open(h, m) -> bool``)."""
        cands = self.candidates()
        if is_open:
            for h, m in cands:
                if is_open(h, m):
                    return f"{h:02d}:{m:02d}"
        h, m = cands[0]
        return f"{h:02d}:{m:02d}"


_PART_PM = _n(["بالليل", "الليل", "ليل", "مساء", "مساءً", "مسا", "المسا", "المساء", "العشا", "العشاء",
               "المغرب", "مغرب", "العصر", "عصر", "عقب الظهر", "بعد الظهر", "pm", "p.m", "evening", "night",
               "بليل"])
_PART_AM = _n(["الصبح", "صبح", "الصباح", "صباحا", "صباحاً", "الفجر", "الضحى", "am", "a.m", "morning",
               "الصباحيه"])
_PART_NOON = _n(["الظهر", "ظهر", "الظهريه", "ظهرا", "ظهراً", "noon"])
_FRACTIONS = {normalize(k): v for k, v in {
    "ونص": 30, "والنص": 30, "والنصف": 30, "ونصف": 30, "وربع": 15, "والربع": 15, "وثلث": 20, "والثلث": 20,
    "الا ربع": -15, "إلا ربع": -15, "الا ثلث": -20, "إلا ثلث": -20, "الا ربعا": -15, "إلا ربعاً": -15,
    "الا خمس": -5, "الا عشر": -10, "وخمس": 5, "وعشر": 10, "وخمسه": 5, "وعشره": 10,
}.items()}
_HOUR_WORDS = {normalize(k): v for k, v in {
    "وحده": 1, "وحدة": 1, "الوحده": 1, "واحد": 1, "الواحده": 1, "ثنتين": 2, "اثنين": 2, "الثانيه": 2,
    "ثلاث": 3, "الثالثه": 3, "اربع": 4, "الرابعه": 4, "خمس": 5, "الخامسه": 5, "ست": 6, "السادسه": 6,
    "سبع": 7, "السابعه": 7, "ثمان": 8, "ثمانيه": 8, "الثامنه": 8, "تسع": 9, "التاسعه": 9, "عشر": 10,
    "العاشره": 10, "احدعش": 11, "حدعش": 11, "الحاديه عشر": 11, "اثنعش": 12, "اطنعش": 12, "ثنعش": 12,
    "الثانيه عشر": 12,
}.items()}


_ANCHORS = set(_n(["الساعه", "ساعه", "على", "عالساعه", "بالساعه", "at", "around", "حوالي", "تقريبا", "الحين"]))


def part_of_day(text: str) -> str | None:
    t = normalize(text)
    if _has(t, _PART_NOON) and not _has(t, _n(["عقب الظهر", "بعد الظهر"])):
        return "noon"
    if _has(t, _PART_PM):
        return "pm"
    if _has(t, _PART_AM):
        return "am"
    return None


def parse_time(text: str) -> TimeExpr | None:
    """"الساعه ثمان ونص بالليل" -> TimeExpr(8, 30, 'pm'); "20:30" -> TimeExpr(20, 30, None)."""
    t = normalize(text)
    part = part_of_day(t)
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if h < 24 and mi < 60:
            return TimeExpr(h, mi, part if h <= 12 else None)
    toks = t.split()
    hour = None
    idx = -1
    anchored = False
    for i, tok in enumerate(toks):
        prev = toks[i - 1] if i else ""
        after_anchor = prev in _ANCHORS
        val = None
        if tok.isdigit() and 1 <= int(tok) <= 24:
            val = int(tok)
        elif tok in _HOUR_WORDS:
            val = _HOUR_WORDS[tok]
        elif tok.startswith("ع") and tok[1:] in _HOUR_WORDS:      # "عثمان" never; "عالثمان" rare
            val = None
        if val is None:
            continue
        nxt = " ".join(toks[i + 1:i + 3])
        has_frac = any(nxt.startswith(f) for f in _FRACTIONS)
        # A bare number is only a time when anchored ("الساعه 8", "على 8"), fractional ("8 ونص"),
        # or followed by a day part ("8 بالليل"); otherwise "احنا 8" is a party size.
        follows_part = bool(part) and any(toks[j] in _PART_PM + _PART_AM + _PART_NOON
                                          for j in range(i + 1, min(i + 4, len(toks))))
        if after_anchor or has_frac or follows_part:
            hour, idx, anchored = val, i, True
            break
    if hour is None:
        return None
    minute = 0
    rest = " ".join(toks[idx + 1:idx + 3])
    for f, v in sorted(_FRACTIONS.items(), key=lambda kv: -len(kv[0])):
        if rest.startswith(f):
            minute = v
            break
    else:
        # "ثمان و عشرين دقيقه" / "8 و 10"
        if idx + 1 < len(toks) and toks[idx + 1].startswith("و"):
            v = word_value(toks[idx + 1]) if toks[idx + 1] != "و" else (
                word_value(toks[idx + 2]) if idx + 2 < len(toks) else None)
            if v is not None and 0 < v < 60:
                minute = v
    if minute < 0:
        hour = hour - 1 if hour > 1 else 12
        minute = 60 + minute
    if part and hour > 12:
        part = None
    return TimeExpr(hour, minute, part) if anchored else None


# ---------------------------------------------------------------------------
# party size, seating, name
# ---------------------------------------------------------------------------

_PARTY_HINTS = _n(["شخص", "اشخاص", "أشخاص", "نفر", "انفار", "نفرين", "شخصين", "people", "persons", "pax",
                   "guests", "for", "of us", "عددنا", "احنا", "إحنا", "بنكون", "بنصير", "نكون", "طاوله ل", "لعدد", "حق"])
_SEATING = {
    "outdoor": _n(["برا", "بره", "برع", "خارج", "خارجي", "الخارجيه", "الجلسات الخارجيه", "تراس", "الترس",
                   "التراس", "outdoor", "outside", "terrace", "على البحر", "تيراس"]),
    "indoor": _n(["داخل", "جوا", "جوه", "داخلي", "الداخل", "الداخليه", "indoor", "inside", "صاله", "الصاله"]),
    "family": _n(["عائلي", "عايلي", "العوايل", "قسم العائلات", "قسم العوايل", "عائلات", "عوايل", "family",
                  "خاص", "بارتشن", "partition"]),
}


def parse_party_size(text: str, asked: bool = False) -> int | None:
    """"احنا خمسة" -> 5, "طاولة لشخصين" -> 2, "انا وزوجتي" -> 2. Bare numbers only when ``asked``."""
    t = normalize(text)
    toks = t.split()
    if _has(t, _n(["شخصين", "نفرين", "لشخصين", "لنفرين", "couple", "two of us"])):
        return 2
    if _has(t, _n(["انا وزوجتي", "انا وزوجي", "انا والمدام", "انا وياه", "انا وياها", "انا وخويي"])):
        return 2
    if _has(t, _n(["بروحي", "لحالي", "شخص واحد", "لشخص واحد", "just me"])):
        return 1
    for i, tok in enumerate(toks):
        v = word_value(tok)
        if v is None or not (1 <= v <= 60):
            continue
        window = " ".join(toks[max(0, i - 2):i + 3])
        prefixed = tok[0] in "ل" and not tok.isdigit()   # "لخمسه" = for five
        if prefixed or _has(window, _PARTY_HINTS) or any(h in window for h in ("شخص", "نفر", "اشخاص")):
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if nxt.startswith("و") and word_value(nxt):
                v += word_value(nxt)
            return v
    if asked and toks and len(toks) <= 4 and parse_time(text) is None:
        v = parse_number(text)
        if v and 1 <= v <= 60:
            return v
    return None


def parse_seating(text: str) -> str | None:
    t = normalize(text)
    for area, forms in _SEATING.items():
        if _has(t, forms) or any(strip_prefixes(w) in forms for w in t.split()):
            return area
    return None


_NAME_LEADS = [normalize(x) for x in ("على اسم", "باسم", "الاسم", "اسمي", "اسمه", "اسمها", "معك", "معاك",
                                      "my name is", "name is", "under", "this is", "انا اسمي", "المريض",
                                      "اسم المريض")]
_NAME_STOP = set(_n(["و", "من", "بس", "لو", "رقم", "رقمي", "الرقم", "باجر", "اليوم", "الساعه", "ابي", "ابا",
                     "عندي", "لو سمحت", "سمحت", "please", "and", "tomorrow", "شكرا", "مشكور", "يعطيك",
                     "والرقم", "ورقمي", "اتصل", "متصل"]))
_NAME_FILLER = set(_n(["ايه", "اي", "نعم", "تمام", "طيب", "اوكي", "اسمي", "اسم", "هو", "هي", "الحجز", "على",
                       "حطه", "حطها", "سجله", "سجلها", "خلاص", "اكيد", "يب", "the", "it's", "its", "is",
                       "my", "name", "باسم", "الاسم", "انا", "هيه"]))


def parse_name(text: str, asked: bool = False) -> str | None:
    """Name from "على اسم محمد الكعبي" / "اسمي فاطمة"; bare "فاطمة علي" only when ``asked``.

    Returns the caller's original spelling (not the normalised form).
    """
    orig = re.sub(r"[.,!?؟،]", " ", text).split()
    norm = [normalize(w) for w in orig]
    joined = " ".join(norm)
    for lead in sorted(_NAME_LEADS, key=len, reverse=True):
        pos = f" {joined} ".find(f" {lead} ")
        if pos < 0:
            continue
        start = len(f" {joined} "[:pos + len(lead) + 1].split())
        words = []
        for w_o, w_n in zip(orig[start:], norm[start:]):
            if w_n in _NAME_STOP or word_value(w_n) is not None or w_n.isdigit():
                break
            words.append(w_o)
            if len(words) == 3:
                break
        if words:
            return " ".join(words)
    if asked:
        words = [o for o, n in zip(orig, norm) if n not in _NAME_FILLER]
        cut = []
        for w in words:
            if normalize(w) in _NAME_STOP or word_value(w) is not None:
                break
            cut.append(w)
        phrase = " ".join(cut)
        if 1 <= len(cut) <= 3 and not (is_yes(phrase) or is_no(phrase) or is_goodbye(phrase)):
            return " ".join(cut)
    return None
