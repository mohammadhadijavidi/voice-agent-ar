"""WhatsApp template / SMS confirmation: payloads, dry-run, fallback, segment maths."""

import json

import httpx

from voice_agent_ar.notify import Notifier, NotifyConfig, sms_body, sms_segments, template_params


def booking(store, **kw):
    b = dict(call_id="c1", code="48213", kind="restaurant", service_id="table", service_name="table",
             party_size=5, area="indoor", date="2026-11-05", time="20:30", duration_min=90,
             customer_name="محمد الكعبي", phone="+971501234567", notes="", language="ar")
    b.update(kw)
    return store.create_booking(**b)


def test_template_params_match_bundled_template(restaurant, store):
    b = booking(store)
    params = template_params(restaurant, b)
    assert params == ["محمد الكعبي", "مطعم الفنار", "طاولة لـ 5 أشخاص", "الخميس 5 نوفمبر 8:30 مساءً", "48213",
                      "040000000"]
    tpl = json.load(open("examples/whatsapp_template.json", encoding="utf-8"))
    body = next(c for c in tpl["components"] if c["type"] == "BODY")
    assert body["text"].count("{{") == len(params) == len(body["example"]["body_text"][0])
    assert tpl["language"] == "ar" and tpl["category"] == "UTILITY"


def test_sms_body_is_short_ucs2(restaurant, clinic, store):
    body = sms_body(restaurant, booking(store))
    enc, seg = sms_segments(body)
    assert enc == "UCS-2" and seg <= 2
    assert sms_segments("Hello")[0] == "GSM-7"
    assert sms_segments("ا" * 70) == ("UCS-2", 1) and sms_segments("ا" * 71) == ("UCS-2", 2)
    b = booking(store, code="11111", kind="clinic", service_id="cleaning", service_name="تنظيف أسنان",
                doctor_id="dr_salem", party_size=1, area=None)
    assert "تنظيف أسنان - الدكتور سالم" in sms_body(clinic, b)


async def test_dry_run_logs_exact_request(restaurant, store):
    n = Notifier(NotifyConfig(dry_run=True), store, restaurant)
    b = booking(store)
    r = await n.send_booking_summary(b, call_id="c1")
    assert r["dry_run"] and not r["sent"] and r["channel"] == "whatsapp"
    (row,) = store.notifications(b["id"])
    assert row["status"] == "dry_run"
    req = row["request"]
    assert req["url"].endswith("/<PHONE_NUMBER_ID>/messages")
    assert req["json"]["to"] == "971501234567" and req["json"]["type"] == "template"
    assert req["json"]["template"]["language"]["code"] == "ar"


async def test_whatsapp_sent_then_sms_fallback(restaurant, store):
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        if "graph.facebook.com" in str(req.url):
            return httpx.Response(400, json={"error": {"code": 131026, "message": "Message undeliverable"}})
        return httpx.Response(201, json={"sid": "SM123"})
    cfg = NotifyConfig(dry_run=False, wa_token="t", wa_phone_number_id="123", sms_from="ALFANAR",
                       twilio_account_sid="AC1", twilio_auth_token="tok")
    n = Notifier(cfg, store, restaurant, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    b = booking(store)
    r = await n.send_booking_summary(b)
    assert r["sent"] and r["channel"] == "sms" and r["provider_id"] == "SM123"
    assert [a["status"] for a in r["attempts"]] == ["failed", "sent"]
    assert seen[0].headers["authorization"] == "Bearer t"
    assert json.loads(seen[0].content)["template"]["name"] == "booking_confirmation_ar"
    assert b"To=%2B971501234567" in seen[1].content and b"From=ALFANAR" in seen[1].content
    assert [x["status"] for x in store.notifications(b["id"])] == ["failed", "sent"]


async def test_sms_only_and_invalid_numbers(restaurant, store):
    def handler(req):
        return httpx.Response(201, json={"sid": "SM9"})
    cfg = NotifyConfig(dry_run=False, sms_from="+15005550006", twilio_account_sid="AC1", twilio_auth_token="tok")
    n = Notifier(cfg, store, restaurant, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    b = booking(store)
    assert (await n.send_booking_summary(b, channel="sms"))["channel"] == "sms"
    assert (await n.send_booking_summary(b, to="123"))["reason"] == "invalid_number"
    assert (await n.send_booking_summary(b, to="04 123 4567"))["reason"] == "landline"
    # WhatsApp chosen but not configured -> logged as dry_run with the reason, never claimed as sent
    r = await n.send_booking_summary(b, channel="whatsapp")
    assert r["dry_run"] and store.notifications(b["id"])[-1]["error"] == "whatsapp not configured"
