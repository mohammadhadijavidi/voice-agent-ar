"""End-to-end: a real uvicorn server, a real WebSocket, and a simulated Twilio Media Streams caller.

Caller audio is FSK-encoded text (decoded by the mock STT); agent audio comes from the
mock TTS and is "played" by the simulator on a virtual clock, with marks echoed and
``clear`` honoured exactly like Twilio. The LLM is the deterministic mock receptionist.
"""

import asyncio
import json

import httpx
import pytest

from voice_agent_ar.cli import bundled_scenarios, running_server
from voice_agent_ar.providers.llm import LLMProvider, TextDelta, ToolCall, ToolCalls
from voice_agent_ar.providers.mock import MockTTS
from voice_agent_ar.server import create_app
from voice_agent_ar.simulator import Scenario, Step, TwilioSimulator
from voice_agent_ar.twilio import TwilioREST


async def run_call(app, scenario, **sim_kw):
    scenario.seed(app.state.store, app.state.profile)
    async with running_server(app) as base:
        result = await TwilioSimulator(base, speed=sim_kw.pop("speed", 40), **sim_kw).run(scenario)
        for _ in range(60):  # let finalize() finish
            calls = app.state.store.list_calls(1)
            if calls and calls[0]["status"] == "completed":
                break
            await asyncio.sleep(0.05)
    return result


def scenario(name):
    return Scenario.load(bundled_scenarios()[name])


async def test_restaurant_booking_whatsapp(settings, restaurant, store):
    tts = MockTTS()
    app = create_app(settings, profile=restaurant, store=store, tts=tts)
    result = await run_call(app, scenario("restaurant_booking"))
    said = result.agent_said()
    assert said[0] == "مساء الخير، مطعم الفنار، معك مريم. شلون أقدر أساعدك؟"
    assert said[1] == "حياك الله. كم شخص بتكونون؟"
    assert any("طاولة لخمسة أشخاص باجر، الخميس خمسة نوفمبر الساعة ثمان ونص بالليل" in s and "صح كذا؟" in s
               for s in said)
    assert any(s.startswith("تم الحجز يا محمد الكعبي، رقم الحجز") for s in said)
    assert result.closed_by_agent
    call = store.list_calls(1)[0]
    assert (call["outcome"], call["language"], call["simulated"]) == ("booked", "ar", 1)
    (b,) = store.list_bookings(call["id"])
    assert (b["date"], b["time"], b["party_size"], b["area"], b["customer_name"], b["phone"]) == \
        ("2026-11-05", "20:30", 5, "indoor", "محمد الكعبي", "+971501234567")
    # booking created only after the read-back yes: check, check, ..., create, send
    tools = [e["data"]["name"] for e in store.events(call["id"]) if e["kind"] == "tool_call"]
    assert tools.index("create_booking") > tools.index("check_availability")
    assert tools[-2:] == ["send_booking_summary", "end_call"]
    (n,) = store.notifications(call_id=call["id"])
    assert (n["channel"], n["to_number"], n["status"]) == ("whatsapp", "+971501234567", "dry_run")
    assert n["request"]["json"]["template"]["components"][0]["parameters"][4]["text"] == b["code"]
    # the TTS never received a digit: codes, times and phone digits were spoken as words
    assert tts.spoken and not any(ch.isdigit() for t in tts.spoken for ch in t)
    assert "[ملخص تجريبي من mock]" in call["summary"] and b["code"] in call["summary"]
    events = [o["event"] for o in store.outbox_all()]
    assert events == ["booking.created", "call.completed"]
    done = store.outbox_all()[-1]["payload"]["data"]
    assert done["notifications"] == [{"channel": "whatsapp", "to_number": "+971501234567", "status": "dry_run"}]


async def test_full_slot_barge_in_terrace_and_sms_to_other_number(settings, restaurant, store):
    app = create_app(settings, profile=restaurant, store=store)
    result = await run_call(app, scenario("restaurant_taken_bargein"))
    assert result.clears == 1
    assert [t.get("interrupted", False) for t in result.transcript if t["role"] == "agent"][1] is True
    call = store.list_calls(1)[0]
    assert call["barge_ins"] == 1 and call["outcome"] == "booked"
    (b,) = store.list_bookings(call["id"])
    assert (b["time"], b["area"], b["party_size"], b["customer_name"]) == ("21:30", "outdoor", 4, "فاطمة")
    (n,) = store.notifications(call_id=call["id"])
    assert (n["channel"], n["to_number"]) == ("sms", "+971567778899")
    assert "طاولة لـ 4 أشخاص" in n["body"] and b["code"] in n["body"]
    cut = [t for t in store.turns(call["id"]) if t["interrupted"]]
    assert len(cut) == 1 and cut[0]["text"].endswith("—")


async def test_clinic_booking_with_correction(settings, clinic, store):
    app = create_app(settings, profile=clinic, store=store)
    result = await run_call(app, scenario("clinic_booking"))
    said = result.agent_said()
    assert said[0].startswith("هلا والله، عيادة الواحة")
    assert any("الساعة عشر الصبح" in s and "صح كذا؟" in s and "ثلاثمية درهم" in s for s in said)
    assert any("الساعة إحدعش الصبح" in s and "صح كذا؟" in s for s in said)   # corrected read-back
    (b,) = store.list_bookings()
    assert (b["service_id"], b["doctor_id"], b["date"], b["time"], b["customer_name"]) == \
        ("cleaning", "dr_salem", "2026-11-07", "11:00", "سارة احمد")
    assert store.notifications()[0]["channel"] == "whatsapp"


async def test_clinic_emergency_gives_998_first(settings, clinic, store):
    app = create_app(settings, profile=clinic, store=store)
    result = await run_call(app, scenario("clinic_emergency"))
    said = result.agent_said()
    assert "الإسعاف تسعة تسعة ثمانية" in said[1]
    assert store.list_bookings() == []
    kinds = [e["kind"] for e in store.events(store.list_calls(1)[0]["id"])]
    assert "message" in kinds  # no staff line in the sample -> message taken


async def test_landline_caller_gets_asked_for_mobile(settings, restaurant, store):
    app = create_app(settings, profile=restaurant, store=store)
    result = await run_call(app, scenario("landline_caller"))
    said = result.agent_said()
    assert any("تبي أرسل لك التأكيد على رقم موبايل" in s for s in said)
    (n,) = store.notifications()
    assert n["to_number"] == "+971521112233" and n["channel"] == "whatsapp"
    assert store.list_bookings()[0]["phone"] == "+97143334444"


async def test_english_caller_switches_language(settings, restaurant, store):
    app = create_app(settings, profile=restaurant, store=store)
    result = await run_call(app, scenario("english_caller"))
    said = result.agent_said()
    assert any("a table for 2 is available on Saturday, November 7 at 7:30 PM" in s for s in said)
    assert any(s.startswith("You're booked, Sarah.") for s in said)
    call = store.list_calls(1)[0]
    assert call["language"] == "en"
    assert "language_switch" in [e["kind"] for e in store.events(call["id"])]
    assert store.notifications() == []   # caller said "No thanks"


async def test_double_booking_between_two_calls(settings, restaurant, store):
    restaurant.areas = {"indoor": 4}
    app = create_app(settings, profile=restaurant, store=store)
    steps = [Step("ابي طاولة لاربعة اشخاص باجر الساعة تسع بالليل"), Step("على اسم علي"), Step("ايه"),
             Step("لا"), Step("مع السلامة")]
    await run_call(app, Scenario(name="a", steps=steps))
    result = await run_call(app, Scenario(name="b", caller="+971509990000", steps=[steps[0], Step("مع السلامة")]))
    assert any("للأسف الساعة تسع بالليل محجوز" in s and "الساعة عشر ونص بالليل" in s for s in result.agent_said())
    assert len(store.list_bookings()) == 1


async def test_transfer_uses_twilio_rest_with_signed_webhooks(settings, restaurant, store):
    sent = []

    def handler(req):
        sent.append(req.content.decode())
        return httpx.Response(200, json={})
    rest = TwilioREST("AC1", "tok", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    restaurant.transfer_number = "+97140001111"
    settings.twilio_auth_token = "tok"   # signature validation ON: the simulator must sign like Twilio
    app = create_app(settings, profile=restaurant, store=store, twilio_rest=rest)
    result = await run_call(app, Scenario(name="t", steps=[Step("ابي اكلم موظف لو سمحت")]), auth_token="tok")
    assert result.closed_by_agent
    assert any("لحظة أحولك على الزملاء" in s for s in result.agent_said())
    assert len(sent) == 1 and "Dial" in sent[0] and "%2B97140001111" in sent[0]
    assert store.list_calls(1)[0]["outcome"] == "transferred"


async def test_bad_signature_blocks_call(settings, restaurant, store):
    settings.twilio_auth_token = "tok"
    app = create_app(settings, profile=restaurant, store=store)
    with pytest.raises(httpx.HTTPStatusError):
        await run_call(app, Scenario(name="x", steps=[Step("مرحبا")]), auth_token="wrong")
    assert store.list_calls() == []


class ScriptedLLM(LLMProvider):
    """Writes digits like a real model would; checks the TTS normaliser and barge-in truncation."""
    name = "scripted"

    def __init__(self):
        self.histories = []

    async def stream(self, messages, tools):
        self.histories.append(json.loads(json.dumps(messages)))
        if messages[0]["content"].startswith("SUMMARIZE"):
            yield TextDelta("ملخص")
            return
        last = messages[-1]
        text = last.get("content", "") if last["role"] == "user" else ""
        if "قائمة" in text:
            for s in ["عندنا مشويات مشكلة.", " وعندنا سمك هامور.", " وعندنا مجبوس دجاج.", " وعندنا لقيمات.",
                      " وعندنا شاي كرك."]:
                yield TextDelta(s)
                await asyncio.sleep(0)
            return
        if "خلل" in text:
            raise RuntimeError("provider down")
        if "مع السلامة" in text:
            yield TextDelta("مع السلامة.")
            yield ToolCalls([ToolCall("c1", "end_call", {})])
            return
        yield TextDelta("موعدك الساعة 20:30 والسعر 150 درهم.")


async def test_llm_digits_are_spoken_as_arabic_words(settings, restaurant, store):
    tts = MockTTS()
    app = create_app(settings, profile=restaurant, store=store, llm=ScriptedLLM(), tts=tts)
    result = await run_call(app, Scenario(name="d", steps=[Step("متى موعدي؟"), Step("مع السلامة")]))
    assert "موعدك الساعة ثمان ونص بالليل والسعر مية وخمسين درهم." in tts.spoken
    # the transcript / call log keep what the model wrote
    turns = store.turns(store.list_calls(1)[0]["id"])
    assert any(t["text"] == "موعدك الساعة 20:30 والسعر 150 درهم." for t in turns)
    assert result.closed_by_agent


async def test_msa_dialect_override(settings, restaurant, store):
    settings.dialect = "msa"
    tts = MockTTS()
    app = create_app(settings, profile=restaurant, store=store, llm=ScriptedLLM(), tts=tts)
    await run_call(app, Scenario(name="m", steps=[Step("متى موعدي؟"), Step("مع السلامة")]))
    assert "موعدك الساعة الثامنة والنصف مساءً والسعر مئة وخمسون درهماً." in tts.spoken
    assert "Modern Standard Arabic" in app.state.profile.system_prompt("ar")


async def test_barge_in_history_contains_only_heard_sentences(settings, restaurant, store):
    llm = ScriptedLLM()
    app = create_app(settings, profile=restaurant, store=store, llm=llm)
    sc = Scenario(name="menu", steps=[
        Step("شو عندكم في القائمة؟"),
        Step("بس بس، عندي سؤال", barge_in_after_ms=4500),
        Step("مع السلامة"),
    ])
    result = await run_call(app, sc)
    assert result.clears == 1
    hist = next(h for h in llm.histories if h[-1]["role"] == "user" and h[-1]["content"].startswith("بس بس"))
    cut = hist[-2]
    assert cut["role"] == "assistant" and cut["content"].endswith("—")
    heard = cut["content"].removesuffix("—").strip()
    assert heard.startswith("عندنا مشويات مشكلة.") and "شاي كرك" not in heard
    n_heard = heard.count("عندنا")
    played_full = [s for s in result.agent_segments if not s.interrupted and s.text and "عندنا" in s.text]
    assert 1 <= n_heard <= 4 and len(played_full) == n_heard


async def test_llm_failure_apologises_in_arabic_and_call_continues(settings, restaurant, store):
    app = create_app(settings, profile=restaurant, store=store, llm=ScriptedLLM())
    result = await run_call(app, Scenario(name="err", steps=[Step("صار خلل"), Step("مع السلامة")]))
    said = result.agent_said()
    assert any("صار عندي خلل بسيط" in s for s in said)
    assert said[-1] == "مع السلامة."
    errs = [e for e in store.events(store.list_calls(1)[0]["id"]) if e["kind"] == "error"]
    assert errs and "provider down" in errs[0]["data"]["error"]


async def test_dashboard_api_after_call(settings, restaurant, store):
    app = create_app(settings, profile=restaurant, store=store)
    await run_call(app, scenario("restaurant_booking"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        calls = (await c.get("/api/calls")).json()
        detail = (await c.get(f"/api/calls/{calls[0]['id']}")).json()
        stats = (await c.get("/api/stats")).json()
        info = (await c.get("/api/info")).json()
        page = (await c.get("/")).text
    assert calls[0]["bookings"] == 1
    assert detail["bookings"][0]["customer_name"] == "محمد الكعبي"
    assert detail["notifications"][0]["status"] == "dry_run"
    assert stats["calls"] == 1 and stats["bookings"] == 1 and stats["notifications_sent"] == 1
    assert info["business"] == "مطعم الفنار" and info["kind"] == "restaurant" and info["mock"] is True
    assert 'dir="rtl"' in page and 'lang="ar"' in page
