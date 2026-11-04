"""Booking tools against the sample restaurant and clinic (fixed clock: Wed 4 Nov 2026 09:00 Dubai)."""

import pytest

from voice_agent_ar.notify import Notifier, NotifyConfig
from voice_agent_ar.tools import ToolContext, default_tools, parse_time_arg
from voice_agent_ar.webhooks import WebhookDispatcher


def ctx(profile, store, caller="+971501234567", lang="ar"):
    cid = store.create_call(from_number=caller)
    return ToolContext(call_id=cid, caller_number=caller, language=lang, profile=profile, store=store,
                       webhooks=WebhookDispatcher(store), notifier=Notifier(NotifyConfig(), store, profile))


async def call(profile, c, name, **args):
    return await default_tools(profile).call(name, args, c)


async def test_restaurant_availability_is_spoken_in_gulf_arabic(restaurant, store):
    c = ctx(restaurant, store)
    r = await call(restaurant, c, "check_availability", date="2026-11-05", time="20:30", party_size=5)
    assert r["available"] is True and r["area"] == "indoor"
    assert r["spoken_date"] == "باجر، الخميس خمسة نوفمبر"
    assert r["spoken_requested_time"] == "الساعة ثمان ونص بالليل"
    assert r["spoken_party"] == "خمسة أشخاص"


async def test_tool_args_accept_gulf_words(restaurant, store):
    c = ctx(restaurant, store)
    r = await call(restaurant, c, "check_availability", date="باجر", time="ثمان ونص بالليل", party_size="خمسة")
    assert (r["date"], r["requested_time"], r["party_size"]) == ("2026-11-05", "20:30", 5)
    assert parse_time_arg("8:30 pm", restaurant).isoformat() == "20:30:00"
    assert parse_time_arg("الساعة عشر", restaurant).isoformat() == "22:00:00"


async def test_split_shift_gap_is_outside_hours(restaurant, store):
    c = ctx(restaurant, store)
    r = await call(restaurant, c, "check_availability", date="2026-11-05", time="16:30", party_size=2)
    assert r["available"] is False and r["reason"] == "outside_opening_hours"
    # lunch seating starts until 15:00 (closes 15:30), dinner from 18:00
    assert [a["time"] for a in r["alternatives"]] == ["14:30", "15:00", "18:00"]
    assert r["alternatives"][2]["spoken"] == "الساعة ست المغرب"


async def test_capacity_per_area_and_confirmation_guard(restaurant, store):
    c = ctx(restaurant, store)
    no = await call(restaurant, c, "create_booking", date="2026-11-05", time="20:00", party_size=12,
                    area="outdoor", customer_name="حمد")
    assert no == {"booked": False, "reason": "not_confirmed", "note": no["note"]}
    ok = await call(restaurant, c, "create_booking", date="2026-11-05", time="20:00", party_size=12,
                    area="outdoor", customer_name="حمد", confirmed_by_caller=True)
    assert ok["booked"] and len(ok["code"]) == 5 and ok["can_receive_whatsapp"]
    assert ok["phone"] == "+971501234567" and ok["spoken_last4"] == "أربعة خمسة ستة سبعة"
    ok2 = await call(restaurant, c, "create_booking", date="2026-11-05", time="20:30", party_size=8,
                     area="outdoor", customer_name="سعيد", confirmed_by_caller=True)
    assert ok2["booked"] and ok2["code"] != ok["code"]
    # 24 outdoor seats: 20 taken around 21:00, 6 more do not fit for an overlapping sitting (90 min)
    r = await call(restaurant, c, "check_availability", date="2026-11-05", time="21:00", party_size=6, area="برا")
    assert r["available"] is False and r["reason"] == "taken"
    assert "21:30" in [a["time"] for a in r["alternatives"]]   # the 20:00 table has left by then
    # without an area preference the indoor room still has space
    r = await call(restaurant, c, "check_availability", date="2026-11-05", time="21:00", party_size=6)
    assert r["available"] and r["area"] == "indoor"


async def test_party_too_large_and_closed_and_past(restaurant, clinic, store):
    c = ctx(restaurant, store)
    r = await call(restaurant, c, "check_availability", date="2026-11-05", time="20:00", party_size=20)
    assert r["reason"] == "party_too_large" and r["max_party_size"] == 12
    r = await call(restaurant, c, "check_availability", date="2026-11-03", time="20:00", party_size=2)
    assert r["reason"] == "date_in_past"
    c2 = ctx(clinic, store)
    r = await call(clinic, c2, "check_availability", date="2026-11-08", time="10:00", service="كشف")
    assert r["reason"] == "closed_that_day" and "الأحد مسكرين" in r["spoken_hours"]


async def test_min_notice_today(restaurant, clinic, store):
    c = ctx(clinic, store)
    r = await call(clinic, c, "check_availability", date="2026-11-04", time="09:30", service="consultation")
    assert r["available"] is False  # 60 min notice at 09:00
    r = await call(clinic, c, "check_availability", date="2026-11-04", time="10:00", service="consultation")
    assert r["available"] is True


async def test_clinic_doctor_assignment_and_price(clinic, store):
    c = ctx(clinic, store)
    r = await call(clinic, c, "check_availability", date="2026-11-07", time="10:00", service="تنظيف")
    assert r["available"] and r["doctor"] == "dr_salem" and r["spoken_price"] == "ثلاثمية درهم"
    # Dr. Salem does not work Fridays and is the only one doing cleanings
    r = await call(clinic, c, "check_availability", date="2026-11-06", time="17:00", service="cleaning")
    assert r["reason"] == "no_doctor_that_day"
    # two consultations at the same time go to two different doctors, the third is refused
    for name in ("أ", "ب"):
        b = await call(clinic, c, "create_booking", date="2026-11-05", time="10:00", service="consultation",
                       customer_name=name, confirmed_by_caller=True)
        assert b["booked"]
    docs = {x["doctor_id"] for x in store.list_bookings(c.call_id)}
    assert docs == {"dr_salem", "dr_huda"}
    r = await call(clinic, c, "check_availability", date="2026-11-05", time="10:00", service="consultation")
    assert r["available"] is False
    # a 45-minute cleaning must end before the 13:00 break
    r = await call(clinic, c, "check_availability", date="2026-11-05", time="12:30", service="cleaning")
    assert r["available"] is False


async def test_landline_caller_booking_flags_no_whatsapp(restaurant, store):
    c = ctx(restaurant, store, caller="+97143334444")
    b = await call(restaurant, c, "create_booking", date="2026-11-05", time="20:00", party_size=2,
                   customer_name="خالد", confirmed_by_caller=True)
    assert b["booked"] and b["can_receive_whatsapp"] is False
    r = await call(restaurant, c, "send_booking_summary", booking_code=b["code"])
    assert r["reason"] == "landline"
    r = await call(restaurant, c, "send_booking_summary", booking_code=b["code"], to_number="٠٥٢ ١١١ ٢٢٣٣")
    assert r["dry_run"] and r["to"] == "+971521112233" and r["channel"] == "whatsapp"


async def test_cancel_requires_same_caller(restaurant, store):
    c = ctx(restaurant, store)
    b = await call(restaurant, c, "create_booking", date="2026-11-05", time="20:00", party_size=2,
                   customer_name="مريم", confirmed_by_caller=True)
    other = ctx(restaurant, store, caller="+971559999999")
    assert (await call(restaurant, other, "cancel_booking", booking_code=b["code"],
                       confirmed_by_caller=True))["reason"] == "caller_mismatch"
    spoken_code = " ".join(["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية",
                            "تسعة"][int(d)] for d in b["code"])
    r = await call(restaurant, c, "cancel_booking", booking_code=spoken_code, confirmed_by_caller=True)
    assert r["cancelled"] is True
    assert store.get_booking(b["code"])["status"] == "cancelled"


async def test_transfer_and_message(restaurant, store):
    c = ctx(restaurant, store)
    r = await call(restaurant, c, "transfer_to_human", reason="x")
    assert r["transferred"] is False and r["reason"] == "no_staff_line"
    restaurant.transfer_number = "+97140000000"
    r = await call(restaurant, c, "transfer_to_human", reason="x")
    assert r["transferred"] and c.control["action"] == "transfer"


async def test_bad_arguments_are_reported_not_raised(restaurant, store):
    c = ctx(restaurant, store)
    r = await call(restaurant, c, "check_availability", date="someday", party_size=2)
    assert "error" in r
    r = await default_tools(restaurant).call("nope", {}, c)
    assert r["error"].startswith("unknown tool")
    with pytest.raises(AssertionError):
        assert "error" not in await call(restaurant, c, "check_availability", date="2026-11-05")
