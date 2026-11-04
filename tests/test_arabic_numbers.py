"""Spoken Arabic: numbers, prices, times, dates (Gulf and MSA) and the TTS text normaliser."""

from datetime import date, time

import pytest

from voice_agent_ar.arabic.numbers import (
    date_to_words,
    number_to_words,
    parse_number,
    people,
    price,
    time_to_words,
)
from voice_agent_ar.arabic.speak import to_speech


@pytest.mark.parametrize("n,gulf,msa", [
    (0, "صفر", "صفر"), (2, "اثنين", "اثنان"), (11, "إحدعش", "أحد عشر"), (12, "اثنعش", "اثنا عشر"),
    (25, "خمسة وعشرين", "خمسة وعشرون"), (150, "مية وخمسين", "مئة وخمسون"), (200, "ميتين", "مئتان"),
    (1250, "ألف وميتين وخمسين", "ألف ومئتان وخمسون"), (3000, "ثلاثة آلاف", "ثلاثة آلاف"),
])
def test_number_to_words(n, gulf, msa):
    assert number_to_words(n) == gulf
    assert number_to_words(n, "msa") == msa


@pytest.mark.parametrize("t,gulf,msa", [
    ("20:30", "الساعة ثمان ونص بالليل", "الساعة الثامنة والنصف مساءً"),
    ("09:15", "الساعة تسع وربع الصبح", "الساعة التاسعة والربع صباحاً"),
    ("18:45", "الساعة سبع إلا ربع بالليل", "الساعة السابعة إلا ربعاً مساءً"),
    ("19:40", "الساعة ثمان إلا ثلث بالليل", "الساعة الثامنة إلا ثلثاً مساءً"),
    ("12:00", "الساعة اثنعش الظهر", "الساعة الثانية عشرة ظهراً"),
    ("13:00", "الساعة وحدة الظهر", "الساعة الواحدة مساءً"),
    ("16:00", "الساعة أربع العصر", "الساعة الرابعة مساءً"),
    ("18:00", "الساعة ست المغرب", "الساعة السادسة مساءً"),
    ("10:05", "الساعة عشر وخمس الصبح", "الساعة العاشرة وخمسة دقائق صباحاً"),
])
def test_time_to_words(t, gulf, msa):
    assert time_to_words(time.fromisoformat(t)) == gulf
    assert time_to_words(t, "msa") == msa


def test_counted_nouns_follow_arabic_rules():
    assert [people(n) for n in (1, 2, 5, 11, 20)] == ["شخص واحد", "شخصين", "خمسة أشخاص", "إحدعش شخص",
                                                       "عشرين شخص"]
    assert people(12, "msa") == "اثنا عشر شخصاً"
    assert price(150) == "مية وخمسين درهم"
    assert price(150, style="msa") == "مئة وخمسون درهماً"
    assert price(3) == "ثلاثة دراهم" and price(2) == "درهمين" and price(99.5) == "تسعة وتسعين درهم ونص"


def test_dates_with_relative_words():
    today = date(2026, 11, 4)
    assert date_to_words(date(2026, 11, 5), today=today) == "باجر، الخميس خمسة نوفمبر"
    assert date_to_words(date(2026, 11, 6), today=today) == "عقب باجر، الجمعة ستة نوفمبر"
    assert date_to_words(date(2026, 11, 5), "msa", today) == "غداً، يوم الخميس الخامس من نوفمبر"
    assert date_to_words(date(2026, 11, 12), today=today) == "الخميس اثنعش نوفمبر"


def test_parse_number_words_and_digits():
    assert parse_number("مية وخمسين") == 150
    assert parse_number("خمسه وعشرين") == 25
    assert parse_number("احنا ٦") == 6
    assert parse_number("ثنتين") == 2
    assert parse_number("ما في رقم") is None


def test_to_speech_rewrites_llm_text_for_tts():
    s = to_speech("حجزك الساعة 20:30 لـ 4 أشخاص، الكشف 150 درهم، رقمك 0501234567")
    assert s == ("حجزك الساعة ثمان ونص بالليل لأربعة أشخاص، الكشف مية وخمسين درهم، "
                 "رقمك صفر خمسة صفر، واحد اثنين ثلاثة، أربعة خمسة ستة سبعة")
    assert to_speech("موعدك 9:15 صباحاً") == "موعدك الساعة تسع وربع الصبح"
    assert to_speech("موعدك 7:00 م.") == "موعدك الساعة سبع بالليل."
    assert to_speech("طاولة ل 2 أشخاص", "msa") == "طاولة لشخصين"
    assert to_speech("AED 300", "msa") == "ثلاثمئة درهم"
    assert to_speech("٣ طاولات") == "ثلاثة طاولات"
    assert not any(c.isdigit() for c in to_speech("الكود 48213 والساعة ١٩:٣٠ والسعر 1200 AED"))
