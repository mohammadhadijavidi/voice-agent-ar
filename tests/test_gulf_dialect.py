"""Gulf / Emirati Arabic understanding: days, times, party size, names, yes/no, intents."""

from datetime import date

import pytest

from voice_agent_ar.arabic import gulf
from voice_agent_ar.arabic.normalize import normalize

TODAY = date(2026, 11, 4)  # Wednesday


def test_normalize_folds_stt_spelling_variants():
    assert normalize("أبي أحجز بُكرة السّاعة ٨") == normalize("ابي احجز بكره الساعه 8") == "ابي احجز بكره الساعه 8"
    assert normalize("طاولـــة") == "طاوله"


@pytest.mark.parametrize("text,expected", [
    ("باجر", date(2026, 11, 5)), ("بكرة", date(2026, 11, 5)), ("عقب باجر", date(2026, 11, 6)),
    ("بعد بكره", date(2026, 11, 6)), ("الليلة", TODAY), ("اليوم", TODAY),
    ("يوم الخميس", date(2026, 11, 5)), ("الأربعاء", date(2026, 11, 11)), ("السبت", date(2026, 11, 7)),
    ("5/11", date(2026, 11, 5)), ("خمسة نوفمبر", date(2026, 11, 5)), ("2026-12-01", date(2026, 12, 1)),
    ("ابي اجي", None),
])
def test_parse_day(text, expected):
    assert gulf.parse_day(text, TODAY) == expected


@pytest.mark.parametrize("text,expected", [
    ("الساعة ثمان ونص بالليل", "20:30"), ("الساعه سبع الا ربع", "18:45"), ("على 9", "21:00"),
    ("الساعة ٤ العصر", "16:00"), ("الساعة عشر الصبح", "10:00"), ("8:30", "20:30"), ("20:30", "20:30"),
    ("الساعة وحدة الظهر", "13:00"), ("الساعة اثنعش الظهر", "12:00"), ("تسع وربع بالليل", "21:15"),
    ("at 7:30 pm", "19:30"), ("الساعة ثمان وثلث", "20:20"),
])
def test_parse_time(text, expected):
    te = gulf.parse_time(text)
    assert te is not None
    assert te.resolve() == expected


def test_bare_number_is_not_a_time():
    assert gulf.parse_time("احنا 8 اشخاص") is None
    assert gulf.parse_party_size("احنا 8 اشخاص") == 8


def test_ambiguous_hour_uses_opening_hours():
    te = gulf.parse_time("الساعة عشر")
    # a clinic open 09:00-13:00 / 16:00-21:00 means 10 AM; a restaurant open until midnight means 10 PM
    assert te.resolve(lambda h, m: 9 * 60 <= h * 60 + m < 13 * 60 or 16 * 60 <= h * 60 + m < 21 * 60) == "10:00"
    assert te.resolve(lambda h, m: 12 * 60 <= h * 60 + m < 24 * 60) == "22:00"


@pytest.mark.parametrize("text,asked,expected", [
    ("طاولة لخمسة أشخاص", False, 5), ("بنكون ستة", False, 6), ("لشخصين", False, 2), ("انا وزوجتي", False, 2),
    ("احنا ٤ اشخاص", False, 4), ("عشرة", True, 10), ("عشرة", False, None), ("بروحي", False, 1),
    ("a table for two", False, 2),
])
def test_party_size(text, asked, expected):
    assert gulf.parse_party_size(text, asked=asked) == expected


def test_seating():
    assert gulf.parse_seating("برا في التراس") == "outdoor"
    assert gulf.parse_seating("قسم العوايل لو سمحت") == "family"
    assert gulf.parse_seating("داخل") == "indoor"
    assert gulf.parse_seating("عادي") is None


@pytest.mark.parametrize("text,asked,expected", [
    ("على اسم محمد الكعبي", False, "محمد الكعبي"), ("اسمي فاطمة", False, "فاطمة"),
    ("الاسم خالد ورقمي ٠٥٠", False, "خالد"), ("سارة احمد", True, "سارة احمد"), ("ايه", True, None),
    ("My name is Sarah", False, "Sarah"), ("ابي احجز", False, None),
])
def test_name(text, asked, expected):
    assert gulf.parse_name(text, asked=asked) == expected


def test_yes_no_goodbye_human_emergency():
    assert all(gulf.is_yes(x) for x in ("ايه", "إي صح", "تمام", "اكيد", "يب", "اوكي"))
    assert not gulf.is_yes("لا مب صح")
    assert all(gulf.is_no(x) for x in ("لا", "مب صح", "لا خلها الساعة تسع"))
    assert all(gulf.is_goodbye(x) for x in ("مع السلامة", "مشكور", "يعطيك العافية", "فمان الله"))
    assert gulf.wants_human("ابي اكلم موظف")
    assert gulf.is_emergency("عنده الم شديد في صدره")
    assert gulf.channel_preference("ارسلها رسالة نصية") == "sms"
    assert gulf.channel_preference("على الواتس") == "whatsapp"
    assert gulf.channel_preference("لا ترسل شي") == "none"
