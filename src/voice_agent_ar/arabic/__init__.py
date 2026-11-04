"""Arabic (Gulf-first) language layer: normalisation, spoken numbers/times, UAE phones, slot extraction."""

from .gulf import (
    TimeExpr,
    channel_preference,
    is_emergency,
    is_goodbye,
    is_no,
    is_yes,
    parse_day,
    parse_name,
    parse_party_size,
    parse_seating,
    parse_time,
    wants_booking,
    wants_human,
)
from .normalize import detect_language, normalize, normalize_digits
from .numbers import date_to_words, digits_to_words, number_to_words, parse_number, people, price, time_to_words
from .phone import UAEPhone, format_uae_phone, parse_uae_phone, speak_phone

__all__ = [
    "TimeExpr", "UAEPhone", "channel_preference", "date_to_words", "detect_language", "digits_to_words",
    "format_uae_phone", "is_emergency", "is_goodbye", "is_no", "is_yes", "normalize", "normalize_digits",
    "number_to_words", "parse_day", "parse_name", "parse_number", "parse_party_size", "parse_seating",
    "parse_time", "parse_uae_phone", "people", "price", "speak_phone", "time_to_words", "wants_booking",
    "wants_human",
]
