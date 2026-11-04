import pytest

from voice_agent_ar.arabic.phone import format_uae_phone, parse_uae_phone, speak_phone


@pytest.mark.parametrize("raw", ["050 123 4567", "0501234567", "+971 50 123 4567", "00971501234567",
                                 "971501234567", "501234567", "٠٥٠١٢٣٤٥٦٧", "+971-50-123-4567",
                                 "صفر خمسة صفر واحد اثنين ثلاثة اربعة خمسة ستة سبعة"])
def test_mobile_formats(raw):
    p = parse_uae_phone(raw)
    assert p and p.e164 == "+971501234567" and p.kind == "mobile" and p.national == "050 123 4567"
    assert p.whatsapp_capable


def test_landline_and_operators():
    p = parse_uae_phone("04 123 4567")
    assert (p.e164, p.kind, p.area, p.whatsapp_capable) == ("+97141234567", "landline", "Dubai", False)
    assert parse_uae_phone("+97121234567").area == "Abu Dhabi"
    assert parse_uae_phone("0551234567").area == "du"


@pytest.mark.parametrize("raw", ["12345", "0571234567", "05012345", "", "hello"])
def test_invalid(raw):
    assert parse_uae_phone(raw) is None


def test_international_passthrough_and_reading():
    assert parse_uae_phone("+44 20 7946 0958").kind == "international"
    assert format_uae_phone("+971552223344") == "055 222 3344"
    assert speak_phone("+971552223344") == "صفر خمسة خمسة، اثنين اثنين اثنين، ثلاثة ثلاثة أربعة أربعة"
