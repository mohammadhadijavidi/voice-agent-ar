"""Arabic numbers, times, dates and prices, written the way they are SAID.

Two styles:

* ``gulf`` (default): how a receptionist in Dubai talks on the phone.
  "الساعة ثمان ونص بالليل", "مية وخمسين درهم", "لخمسة أشخاص".
* ``msa``: Modern Standard Arabic for businesses that prefer a formal voice.
  "الساعة الثامنة والنصف مساءً", "مئة وخمسون درهماً".

Text-to-speech engines read "20:30" or "150 AED" unpredictably in Arabic (often as
English, or as "عشرون نقطتان ثلاثون"). Everything the agent says goes through this
module first, so the caller hears natural Arabic.

The reverse direction (``parse_number``) reads numbers that STT returns as words,
including Gulf forms such as "ثنتين", "ثمان", "اطنعش" and "عشرين".
"""

from __future__ import annotations

from datetime import date, time

from .normalize import normalize, normalize_digits

# ---------------------------------------------------------------------------
# integer -> words
# ---------------------------------------------------------------------------

_ONES = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_ONES_MSA = ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_TEENS_GULF = {11: "إحدعش", 12: "اثنعش", 13: "ثلطعش", 14: "أربعطعش", 15: "خمسطعش", 16: "سطعش",
               17: "سبعطعش", 18: "ثمنطعش", 19: "تسعطعش"}
_TEENS_MSA = {11: "أحد عشر", 12: "اثنا عشر", 13: "ثلاثة عشر", 14: "أربعة عشر", 15: "خمسة عشر",
              16: "ستة عشر", 17: "سبعة عشر", 18: "ثمانية عشر", 19: "تسعة عشر"}
_TENS = {20: "عشرين", 30: "ثلاثين", 40: "أربعين", 50: "خمسين", 60: "ستين", 70: "سبعين", 80: "ثمانين",
         90: "تسعين"}
_TENS_MSA = {20: "عشرون", 30: "ثلاثون", 40: "أربعون", 50: "خمسون", 60: "ستون", 70: "سبعون",
             80: "ثمانون", 90: "تسعون"}
_HUNDREDS = {100: "مية", 200: "ميتين", 300: "ثلاثمية", 400: "أربعمية", 500: "خمسمية", 600: "ستمية",
             700: "سبعمية", 800: "ثمانمية", 900: "تسعمية"}
_HUNDREDS_MSA = {100: "مئة", 200: "مئتان", 300: "ثلاثمئة", 400: "أربعمئة", 500: "خمسمئة", 600: "ستمئة",
                 700: "سبعمئة", 800: "ثمانمئة", 900: "تسعمئة"}


def number_to_words(n: int, style: str = "gulf") -> str:
    """0 <= n < 1,000,000 -> Arabic words. Larger numbers are returned as digits."""
    if n < 0:
        return "سالب " + number_to_words(-n, style)
    if n >= 1_000_000:
        return str(n)
    msa = style == "msa"
    ones = _ONES_MSA if msa else _ONES
    if n <= 10:
        return ones[n]
    if n < 20:
        return (_TEENS_MSA if msa else _TEENS_GULF)[n]
    if n < 100:
        t, o = divmod(n, 10)
        tens = (_TENS_MSA if msa else _TENS)[t * 10]
        return tens if o == 0 else f"{ones[o]} و{tens}"
    if n < 1000:
        h, rest = divmod(n, 100)
        hw = (_HUNDREDS_MSA if msa else _HUNDREDS)[h * 100]
        return hw if rest == 0 else f"{hw} و{number_to_words(rest, style)}"
    th, rest = divmod(n, 1000)
    if th == 1:
        tw = "ألف"
    elif th == 2:
        tw = "ألفان" if msa else "ألفين"
    elif 3 <= th <= 10:
        tw = f"{ones[th]} آلاف"
    else:
        tw = f"{number_to_words(th, style)} ألف"
    return tw if rest == 0 else f"{tw} و{number_to_words(rest, style)}"


def counted(n: int, singular: str, dual: str, plural: str, style: str = "gulf",
            acc: str | None = None) -> str:
    """Arabic counted noun: 1 -> "شخص واحد", 2 -> "شخصين", 3-10 -> "خمسة أشخاص", 11+ -> "عشرين شخص".

    ``acc`` is the MSA accusative singular used for 11-99 ("درهماً").
    """
    if n == 1:
        return f"{singular} واحد"
    if n == 2:
        return dual
    if 3 <= n <= 10:
        return f"{number_to_words(n, style)} {plural}"
    last2 = n % 100
    noun = acc if (style == "msa" and acc and 11 <= last2 <= 99) else singular
    return f"{number_to_words(n, style)} {noun}"


def people(n: int, style: str = "gulf") -> str:
    return counted(n, "شخص", "شخصين" if style == "gulf" else "شخصان", "أشخاص", style, acc="شخصاً")


def price(amount: float, currency: str = "AED", style: str = "gulf") -> str:
    """150 -> "مية وخمسين درهم"; 99.5 -> "تسعة وتسعين درهم ونص"."""
    whole = int(amount)
    frac = round(amount - whole, 2)
    if currency.upper() in ("AED", "درهم"):
        words = counted(whole, "درهم", "درهمين" if style == "gulf" else "درهمان", "دراهم", style, acc="درهماً")
    else:
        words = f"{number_to_words(whole, style)} {currency}"
    if frac == 0.5:
        words += " ونص" if style == "gulf" else " ونصف"
    elif frac:
        fils = int(round(frac * 100))
        words += f" و{number_to_words(fils, style)} فلس"
    return words


# ---------------------------------------------------------------------------
# time / date -> words
# ---------------------------------------------------------------------------

_HOURS_GULF = ["اثنعش", "وحدة", "ثنتين", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثمان", "تسع", "عشر",
               "إحدعش", "اثنعش"]
_HOURS_MSA = ["الثانية عشرة", "الواحدة", "الثانية", "الثالثة", "الرابعة", "الخامسة", "السادسة",
              "السابعة", "الثامنة", "التاسعة", "العاشرة", "الحادية عشرة", "الثانية عشرة"]


def day_part(h: int, style: str = "gulf") -> str:
    if style == "msa":
        return "صباحاً" if h < 12 else ("ظهراً" if h < 13 else "مساءً")
    if h < 5:
        return "بالليل"
    if h < 12:
        return "الصبح"
    if h < 14:
        return "الظهر"
    if h < 18:
        return "العصر"
    if h < 19:
        return "المغرب"
    return "بالليل"


def time_to_words(t: time | str, style: str = "gulf") -> str:
    """time(20, 30) -> "الساعة ثمان ونص بالليل" (gulf) / "الساعة الثامنة والنصف مساءً" (msa)."""
    if isinstance(t, str):
        t = time.fromisoformat(t)
    h, m = t.hour, t.minute
    if style == "msa":
        hour_words = _HOURS_MSA
        quarter, half, third = "والربع", "والنصف", "والثلث"
        less_q, less_t = "إلا ربعاً", "إلا ثلثاً"
    else:
        hour_words = _HOURS_GULF
        quarter, half, third = "وربع", "ونص", "وثلث"
        less_q, less_t = "إلا ربع", "إلا ثلث"
    part_h = h
    if m in (40, 45):
        # "nine less a quarter" names the NEXT hour
        part_h = (h + 1) % 24
    hw = hour_words[part_h % 12 or 12] if part_h % 12 else hour_words[12]
    if m == 0:
        core = hw
    elif m == 15:
        core = f"{hw} {quarter}"
    elif m == 30:
        core = f"{hw} {half}"
    elif m == 20:
        core = f"{hw} {third}"
    elif m == 45:
        core = f"{hw} {less_q}"
    elif m == 40:
        core = f"{hw} {less_t}"
    elif style == "gulf" and m in (5, 10):
        core = f"{hw} و{'خمس' if m == 5 else 'عشر'}"
    else:
        mins = counted(m, "دقيقة", "دقيقتين" if style == "gulf" else "دقيقتان", "دقايق" if style == "gulf"
                       else "دقائق", style, acc="دقيقة")
        if m in (1, 2):
            mins = {1: "دقيقة", 2: "دقيقتين" if style == "gulf" else "دقيقتان"}[m]
        core = f"{hw} و{mins}"
    return f"الساعة {core} {day_part(part_h if m in (40, 45) else h, style)}"


WEEKDAYS_AR = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر",
             "نوفمبر", "ديسمبر"]
_ORD_DAY_MSA = {1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع", 5: "الخامس", 6: "السادس", 7: "السابع",
                8: "الثامن", 9: "التاسع", 10: "العاشر"}


def date_to_words(d: date, style: str = "gulf", today: date | None = None) -> str:
    """date(2026, 11, 5) -> "الخميس خمسة نوفمبر"; with ``today`` adds "باجر"/"اليوم" where it applies."""
    wd = WEEKDAYS_AR[d.weekday()]
    if style == "msa":
        day = _ORD_DAY_MSA.get(d.day) or number_to_words(d.day, style)
        core = f"يوم {wd} {day} من {MONTHS_AR[d.month - 1]}"
    else:
        core = f"{wd} {number_to_words(d.day, style)} {MONTHS_AR[d.month - 1]}"
    if today is not None:
        delta = (d - today).days
        rel = {0: ("اليوم", "اليوم"), 1: ("باجر", "غداً"), 2: ("عقب باجر", "بعد غد")}.get(delta)
        if rel:
            core = f"{rel[1] if style == 'msa' else rel[0]}، {core}"
    return core


def digits_to_words(digits: str, style: str = "gulf") -> str:
    """"0551234567" -> "صفر خمسة خمسة ..." one digit at a time (phone numbers, booking codes)."""
    ones = _ONES_MSA if style == "msa" else _ONES
    return " ".join(ones[int(c)] for c in normalize_digits(digits) if c.isdigit())


# ---------------------------------------------------------------------------
# words -> integer (for STT output)
# ---------------------------------------------------------------------------

# normalised spellings (see normalize.normalize: ة->ه, أ->ا, ى->ي)
WORD_VALUES: dict[str, int] = {}
for _v, _forms in {
    0: ["صفر", "زيرو"],
    1: ["واحد", "وحده", "وحدة", "احد", "واحده"],
    2: ["اثنين", "اثنان", "ثنين", "ثنتين", "اثنتين", "اتنين", "ثنتينه"],
    3: ["ثلاث", "ثلاثه", "تلاته", "ثلاثة", "ثلث"],
    4: ["اربع", "اربعه"],
    5: ["خمس", "خمسه"],
    6: ["ست", "سته", "سِت"],
    7: ["سبع", "سبعه"],
    8: ["ثمان", "ثمانيه", "ثماني", "ثمن", "تمانيه", "ثمانية"],
    9: ["تسع", "تسعه"],
    10: ["عشر", "عشره"],
    11: ["احدعش", "احدعشر", "حدعش", "احداعش"],
    12: ["اثنعش", "اطنعش", "اثناعشر", "اثنعشر", "ثنعش", "طنعش"],
    13: ["ثلطعش", "ثلثطعش", "تلتاشر"], 14: ["اربعطعش"], 15: ["خمسطعش", "خمستاشر"], 16: ["سطعش"],
    17: ["سبعطعش"], 18: ["ثمنطعش"], 19: ["تسعطعش"],
    20: ["عشرين", "عشرون"], 30: ["ثلاثين", "ثلاثون"], 40: ["اربعين", "اربعون"], 50: ["خمسين", "خمسون"],
    60: ["ستين", "ستون"], 70: ["سبعين", "سبعون"], 80: ["ثمانين", "ثمانون"], 90: ["تسعين", "تسعون"],
    100: ["ميه", "مية", "مائه", "مئه", "ميت"], 200: ["ميتين", "مئتين", "مئتان"],
}.items():
    for _f in _forms:
        WORD_VALUES[normalize(_f)] = _v
for _v, _w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                         "eleven", "twelve"]):
    WORD_VALUES[_w] = _v
for _v, _forms in {}.items():
    for _f in _forms:
        WORD_VALUES[normalize(_f)] = _v


def word_value(word: str) -> int | None:
    """Value of one token ("خمسه", "٥", "وعشرين" -> 20), or None."""
    w = normalize(word)
    if w.isdigit():
        return int(w)
    if w in WORD_VALUES:
        return WORD_VALUES[w]
    if w.startswith("و") and w[1:] in WORD_VALUES:
        return WORD_VALUES[w[1:]]
    for p in ("ل", "ب"):
        if w.startswith(p) and w[1:] in WORD_VALUES:
            return WORD_VALUES[w[1:]]
    return None


def parse_number(text: str) -> int | None:
    """First number in ``text``, digits or words: "خمسة وعشرين" -> 25, "احنا 6" -> 6."""
    toks = normalize(text).split()
    for i, tok in enumerate(toks):
        v = word_value(tok)
        if v is None:
            continue
        total = v
        # "خمسه وعشرين" / "ميه وخمسين"
        j = i + 1
        while j < len(toks) and toks[j].startswith("و"):
            nxt = word_value(toks[j])
            if nxt is None:
                break
            total += nxt
            j += 1
        return total
    return None
