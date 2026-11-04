"""Deterministic mock providers for tests, CI and the offline demo.

* ``MockSTT`` / ``MockTTS`` use the FSK test modem (``fsk.py``): the simulator encodes
  the caller's words as tones and the mock STT decodes them. This is NOT speech
  recognition; it lets the whole audio path run without API keys.
* ``MockLLM`` is a rule-based Gulf-Arabic receptionist. It sees only what a real
  model sees (system prompt FACTS, chat history, tool results) and answers through
  the same streaming + tool-calling protocol, so the agent loop, tools, read-back
  confirmation and WhatsApp/SMS summary are exercised for real. Its sentences are
  hand-written templates, not model output; it is labelled ``mock`` everywhere.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import date

from .. import fsk
from ..arabic import gulf
from ..arabic.normalize import detect_language, normalize
from ..arabic.phone import parse_uae_phone
from ..notify import written_party
from .llm import LLMProvider, TextDelta, ToolCall, ToolCalls
from .stt import STTProvider, Transcript
from .tts import TTSProvider


class MockSTT(STTProvider):
    name = "mock"
    is_mock = True

    async def transcribe(self, ulaw, language=None, hints=None):
        text = fsk.decode(ulaw)
        if text is None:
            return Transcript(text="", language=None, confidence=0.0)
        return Transcript(text=text, language=detect_language(text), confidence=1.0)


class MockTTS(TTSProvider):
    name = "mock"
    is_mock = True
    chunk_bytes = 800  # 100 ms

    def __init__(self):
        self.spoken: list[str] = []      # exactly what was synthesised (after Arabic speech normalisation)

    async def synthesize(self, text, language="ar"):
        self.spoken.append(text)
        audio = fsk.encode(text)
        for i in range(0, len(audio), self.chunk_bytes):
            yield audio[i:i + self.chunk_bytes]
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Mock LLM: templates
# ---------------------------------------------------------------------------

T = {
    "ask_party": {"ar": "حياك الله. كم شخص بتكونون؟", "en": "Sure. How many people will you be?"},
    "ask_day": {"ar": "أبشر. أي يوم تبون؟", "en": "Great. Which day would you like?"},
    "ask_time": {"ar": "زين، {day}. أي ساعة تناسبكم؟", "en": "Great, {day}. What time suits you?"},
    "ask_service": {"ar": "حياك الله. شو الخدمة اللي تبيها: {services}؟",
                    "en": "Sure. Which service would you like: {services}?"},
    "ask_name": {"ar": "أبشر، عندي {what} {day} {time}. على أي اسم أسجل الحجز؟",
                 "en": "Good news, {what} is available on {day} at {time}. What name should I put the booking under?"},
    "readback": {"ar": "تمام، {what} {day} {time}{extra}، على اسم {name}. صح كذا؟",
                 "en": "So that's {what} on {day} at {time}{extra}, under {name}. Is that right?"},
    "what_change": {"ar": "ولا يهمك، شو تحب نغير؟", "en": "No problem, what would you like to change?"},
    "booked": {"ar": "تم الحجز يا {name}، رقم الحجز {code}. أرسل لك التأكيد على الواتساب على الرقم اللي "
                     "آخره {last4}؟",
               "en": "You're booked, {name}. Your booking number is {code}. Shall I send the confirmation "
                     "on WhatsApp to the number ending {last4}?"},
    "booked_landline": {"ar": "تم الحجز يا {name}، رقم الحجز {code}. تبي أرسل لك التأكيد على رقم موبايل؟",
                        "en": "You're booked, {name}. Your booking number is {code}. Would you like the "
                              "confirmation on a mobile number?"},
    "sent_whatsapp": {"ar": "أبشر، بيوصلك ملخص الحجز على الواتساب على الرقم اللي آخره {last4}. تبي شي ثاني؟",
                      "en": "Done, the booking summary is on its way to WhatsApp, number ending {last4}. "
                            "Anything else?"},
    "sent_sms": {"ar": "أبشر، بيوصلك ملخص الحجز برسالة نصية على الرقم اللي آخره {last4}. تبي شي ثاني؟",
                 "en": "Done, the booking summary is on its way by SMS to the number ending {last4}. "
                       "Anything else?"},
    "send_failed": {"ar": "سامحني، ما قدرت أرسل الرسالة الحين، بس حجزك مسجل. تبي شي ثاني؟",
                    "en": "Sorry, I couldn't send the message right now, but your booking is saved. "
                          "Anything else?"},
    "ask_mobile": {"ar": "هذا رقم أرضي وما يستقبل واتساب. عطني رقم موبايل لو سمحت.",
                   "en": "That's a landline, it can't receive WhatsApp. Could you give me a mobile number?"},
    "bad_number": {"ar": "سامحني، الرقم ما وصلني صح. ممكن تعيده؟ مثلاً صفر خمسة خمسة وبعدين سبعة أرقام.",
                   "en": "Sorry, I didn't get the number right. Could you repeat it?"},
    "no_summary": {"ar": "ولا يهمك. تبي شي ثاني؟", "en": "No problem. Anything else?"},
    "taken": {"ar": "للأسف {time} محجوز {day}. عندي {alts}. أيها يناسبك؟",
              "en": "Sorry, {time} on {day} is taken. I have {alts}. Which suits you?"},
    "outside": {"ar": "سامحني، {time} احنا مسكرين {day}. عندي {alts}. أيها يناسبك؟",
                "en": "Sorry, we're closed at {time} on {day}. I have {alts}. Which suits you?"},
    "full": {"ar": "للأسف {day} محجوز بالكامل. يناسبك يوم ثاني؟", "en": "Sorry, {day} is fully booked. "
                                                                        "Would another day work?"},
    "closed": {"ar": "سامحني، احنا مسكرين {day}. يناسبك يوم ثاني؟", "en": "Sorry, we're closed on {day}. "
                                                                         "Would another day work?"},
    "no_doctor": {"ar": "سامحني، {service} ما في طبيب له {day}. يناسبك يوم ثاني؟",
                  "en": "Sorry, no doctor offers {service} on {day}. Would another day work?"},
    "past": {"ar": "هذا اليوم راح. أي يوم ثاني يناسبك؟", "en": "That day has passed. Which other day works?"},
    "slots": {"ar": "{day} عندي {alts}. أيها يناسبك؟", "en": "On {day} I have {alts}. Which suits you?"},
    "too_big": {"ar": "للمجموعات أكثر من {n} شخص نرتب الحجز مع فريق الحجوزات. أحولك عليهم؟",
                "en": "For groups larger than {n} our reservations team handles the booking. "
                      "Shall I transfer you?"},
    "hours": {"ar": "أوقاتنا: {hours}. تبي تحجز؟", "en": "Our hours are: {hours}. Would you like to book?"},
    "location": {"ar": "مكاننا {address}. تبي تحجز؟", "en": "We're at {address}. Would you like to book?"},
    "transfer": {"ar": "أبشر، لحظة أحولك على الزملاء.", "en": "Of course, let me connect you to a colleague."},
    "no_transfer": {"ar": "سامحني، ما في أحد متوفر الحين. سجلت رسالتك وبيتصلون فيك الفريق.",
                    "en": "Sorry, no one is available right now. I've taken a message and the team will "
                          "call you back."},
    "emergency": {"ar": "سلامتك. إذا الحالة طارئة اتصل على الإسعاف تسعة تسعة ثمانية أو روح أقرب طوارئ الحين. "
                        "أحولك على الطاقم الطبي؟",
                  "en": "If this is an emergency, please call 998 or go to the nearest emergency department "
                        "now. Shall I transfer you to our medical staff?"},
    "goodbye": {"ar": "حياك الله، نتريّاكم. مع السلامة.", "en": "Thank you for calling. Goodbye!"},
    "fallback": {"ar": "أقدر أحجز لك، أو أعطيك أوقات الدوام والعنوان، أو أحولك على الزملاء. شو تحب؟",
                 "en": "I can make a booking, tell you our hours and address, or connect you to the team. "
                       "What would you like?"},
    "repeat": {"ar": "سامحني، ما سمعتك زين. ممكن تعيد؟", "en": "Sorry, I didn't catch that. Could you say it again?"},
    "anything_else": {"ar": "تبي شي ثاني؟", "en": "Anything else?"},
}
_OR = {"ar": " أو ", "en": " or "}
_COMMA = {"ar": "، ", "en": ", "}
# phrases the mock uses to recognise what it asked last
ASKED = {
    "party": ("كم شخص", "How many people"),
    "name": ("على أي اسم", "What name"),
    "readback": ("صح كذا", "Is that right"),
    "whatsapp": ("أرسل لك التأكيد على الواتساب", "Shall I send the confirmation", "على رقم موبايل",
                 "mobile number", "عطني رقم موبايل", "الرقم ما وصلني"),
    "transfer_offer": ("أحولك على الطاقم", "أحولك عليهم", "Shall I transfer"),
}


def _join(items: list[str], lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return _COMMA[lang].join(items[:-1]) + _OR[lang] + items[-1]


def _asked(prev: str, key: str) -> bool:
    return any(k in prev for k in ASKED[key])


def parse_facts(system: str) -> dict:
    f: dict = {"services": [], "doctors": [], "areas": []}
    m = re.search(r"Current local date and time: \w+ (\d{4}-\d{2}-\d{2})", system)
    f["today"] = date.fromisoformat(m.group(1)) if m else date.today()
    m = re.search(r"Caller speaks: (\w+)\. Dialect: (\w+)\. Business kind: (\w+)\.", system)
    f["lang"], f["dialect"], f["kind"] = (m.groups() if m else ("ar", "gulf", "restaurant"))
    m = re.search(r"Opening hours \(24h\): (.+)\.$", system, re.M)
    hours = {}
    for part in (m.group(1).split("; ") if m else []):
        day, _, spans = part.partition(" ")
        hours[day] = [] if spans == "closed" else [tuple(s.split("-")) for s in spans.split(",")]
    f["hours"] = hours
    m = re.search(r"Opening hours \(spoken\): (.+)\.$", system, re.M)
    f["hours_spoken"] = m.group(1) if m else ""
    m = re.search(r"Seating areas: (.+?)\. Max party size by phone: (\d+)\.", system)
    if m:
        f["areas"], f["max_party"] = m.group(1).split(", "), int(m.group(2))
    for sm in re.finditer(r"^- service (\w+): (.+?) / (.+?) \(", system, re.M):
        f["services"].append({"id": sm.group(1), "ar": sm.group(2), "en": sm.group(3)})
    for dm in re.finditer(r"^- doctor (\w+): (.+?) / (.+?) \(", system, re.M):
        f["doctors"].append({"id": dm.group(1), "ar": dm.group(2), "en": dm.group(3)})
    m = re.search(r"^Address: (.+)$", system, re.M)
    f["address"] = m.group(1) if m else ""
    m = re.search(r"^Caller ID: (.+)\.$", system, re.M)
    f["caller"] = m.group(1) if m else ""
    return f


def _opens(f: dict):
    def ok(h: int, mi: int) -> bool:
        mm = h * 60 + mi
        for spans in f["hours"].values():
            for a, b in spans:
                ah, am = map(int, a.split(":"))
                bh, bm = map(int, b.split(":"))
                if ah * 60 + am <= mm < bh * 60 + bm:
                    return True
        return False
    return ok


def _find_service(text: str, services: list[dict]) -> dict | None:
    t = f" {normalize(text)} "
    best = None
    aliases = {"consultation": ["كشف", "استشاره", "فحص", "checkup"], "cleaning": ["تنظيف", "cleaning"],
               "pediatrics": ["اطفال", "طفل", "ولدي", "بنتي", "الياهل", "kids"]}
    for s in services:
        keys = [s["ar"], s["en"], s["id"]] + aliases.get(s["id"], [])
        for k in keys:
            k = normalize(k)
            if k and (f" {k} " in t or f" ال{k} " in t or f" ل{k} " in t) and (best is None or len(k) > best[0]):
                best = (len(k), s)
    return best[1] if best else None


def _find_doctor(text: str, doctors: list[dict]) -> dict | None:
    t = normalize(text)
    for d in doctors:
        first = normalize(d["ar"]).split()[-1]
        if first in t.split() or normalize(d["en"]).split()[-1] in t:
            return d
    return None


class MockLLM(LLMProvider):
    name = "mock"
    is_mock = True

    def __init__(self, chunk_words: int = 3):
        self.chunk_words = chunk_words

    async def _emit(self, text: str):
        words = text.split(" ")
        for i in range(0, len(words), self.chunk_words):
            piece = " ".join(words[i:i + self.chunk_words])
            yield TextDelta(piece + (" " if i + self.chunk_words < len(words) else ""))
            await asyncio.sleep(0)

    async def stream(self, messages, tools):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        if system.startswith("SUMMARIZE"):
            async for ev in self._emit(self._summary(messages)):
                yield ev
            return
        text, calls = self.decide(messages)
        if text:
            async for ev in self._emit(text):
                yield ev
        if calls:
            yield ToolCalls(calls)

    # ------------------------------------------------------------------ logic
    def decide(self, messages: list[dict]) -> tuple[str, list[ToolCall]]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        f = parse_facts(system)
        lang = f["lang"] if f["lang"] in ("ar", "en") else "ar"
        clinic = f["kind"] == "clinic"

        def say(key, **kw):
            return T[key][lang].format(**kw)

        def call(_tool, **args):
            return ToolCall(id="call_" + uuid.uuid4().hex[:8], name=_tool, arguments=args)

        # --- reconstruct the conversation state, like a model would -------------
        st = {"party": None, "day": None, "time": None, "area": None, "name": None, "service": None,
              "doctor": None}
        tool_names: dict[str, tuple[str, dict]] = {}
        last_check = last_book = last_summary = None
        prev_assistant = ""
        for m in messages:
            if m["role"] == "user":
                text = m["content"]
                asked_party = _asked(prev_assistant, "party")
                asked_name = _asked(prev_assistant, "name")
                if f["areas"]:
                    p = gulf.parse_party_size(text, asked=asked_party)
                    st["party"] = p or st["party"]
                    a = gulf.parse_seating(text)
                    if a in f["areas"]:
                        st["area"] = a
                if clinic:
                    s = _find_service(text, f["services"])
                    st["service"] = s or st["service"]
                    d = _find_doctor(text, f["doctors"])
                    st["doctor"] = d or st["doctor"]
                d = gulf.parse_day(text, f["today"])
                st["day"] = d or st["day"]
                te = gulf.parse_time(text)
                if te:
                    st["time"] = te.resolve(_opens(f))
                if not _asked(prev_assistant, "whatsapp"):
                    n = gulf.parse_name(text, asked=asked_name)
                    st["name"] = n or st["name"]
            elif m["role"] == "assistant":
                prev_assistant = m.get("content") or prev_assistant
                for c in m.get("tool_calls") or []:
                    tool_names[c["id"]] = (c["name"], c["arguments"])
            elif m["role"] == "tool":
                res = json.loads(m["content"])
                kind = tool_names.get(m["tool_call_id"], ("?", {}))[0]
                if kind == "check_availability":
                    last_check = res
                elif kind == "create_booking":
                    last_book = res
                elif kind == "send_booking_summary":
                    last_summary = res

        last = messages[-1]
        if last["role"] == "tool":
            return self._after_tool(last, tool_names, st, f, lang, say, call, last_check)

        text = last.get("content") or ""
        if not text.strip():
            return say("repeat"), []
        if clinic and gulf.is_emergency(text):
            return say("emergency"), []
        if _asked(prev_assistant, "transfer_offer") and gulf.is_yes(text):
            return say("transfer"), [call("transfer_to_human", reason="caller accepted transfer")]
        if gulf.wants_human(text):
            return say("transfer"), [call("transfer_to_human", reason="caller asked for a person")]

        booked = bool(last_book and last_book.get("booked"))
        # --- confirmation summary offer --------------------------------------
        done_summary = bool(last_summary and (last_summary.get("sent") or last_summary.get("dry_run")))
        if booked and _asked(prev_assistant, "whatsapp") and not done_summary:
            pref = gulf.channel_preference(text)
            num = parse_uae_phone(text)
            numeric = sum(1 for w in normalize(text).split() if w.isdigit() or gulf.word_value(w) is not None)
            if pref == "none" or (gulf.is_no(text) and not num and pref is None):
                return say("no_summary"), []
            args = {"booking_code": last_book["code"]}
            if num:
                args["to_number"] = num.e164
            elif numeric >= 3:
                return say("bad_number"), []
            if pref in ("sms", "whatsapp"):
                args["channel"] = pref
            if num or pref or gulf.is_yes(text):
                return "", [call("send_booking_summary", **args)]
        if gulf.is_goodbye(text) or (gulf.is_no(text) and booked and len(normalize(text).split()) <= 3):
            return say("goodbye"), [call("end_call", reason="caller said goodbye")]
        wants = gulf.wants_booking(text) or any(st[k] for k in ("party", "day", "time", "service"))
        if gulf.asks_hours(text) and not gulf.wants_booking(text):
            return "", [call("get_business_info", topic="hours")]
        if gulf.asks_location(text) and not gulf.wants_booking(text):
            return say("location", address=f["address"]), []
        if booked and not gulf.wants_booking(text):
            return say("anything_else"), []
        if not wants:
            return say("fallback"), []

        # --- slot filling --------------------------------------------------------
        if clinic and not st["service"]:
            return say("ask_service", services=_join([s[lang] for s in f["services"]], lang)), []
        if not clinic and not st["party"]:
            return say("ask_party"), []
        if not st["day"]:
            return say("ask_day"), []
        if not st["time"]:
            return say("ask_time", day=_day_words(st["day"], f, lang)), []
        args = {"date": st["day"].isoformat(), "time": st["time"]}
        if clinic:
            args["service"] = st["service"]["id"]
            if st["doctor"]:
                args["doctor"] = st["doctor"]["id"]
        else:
            args["party_size"] = st["party"]
            if st["area"]:
                args["area"] = st["area"]
        if not _matches(last_check, args) or not last_check.get("available"):
            if booked and _matches(last_book, args):
                return say("anything_else"), []
            return "", [call("check_availability", **args)]
        # availability confirmed for exactly these details
        if not st["name"]:
            return self._ask_name(last_check, lang, say), []
        if _asked(prev_assistant, "readback") and gulf.is_yes(text):
            return "", [call("create_booking", customer_name=st["name"], confirmed_by_caller=True, **args)]
        if _asked(prev_assistant, "readback") and gulf.is_no(text) and not _changed_in(text, f):
            return say("what_change"), []
        return self._readback(last_check, st["name"], lang, say), []

    # ------------------------------------------------------------- tool replies
    def _after_tool(self, last, tool_names, st, f, lang, say, call, last_check):
        kind, args = tool_names.get(last["tool_call_id"], ("?", {}))
        res = json.loads(last["content"])
        if kind in ("end_call", "take_message"):
            return "", []
        if kind == "transfer_to_human":
            if res.get("transferred"):
                return "", []
            return say("no_transfer"), [call("take_message", name=st["name"] or "",
                                             message="caller asked for a person; no staff line available")]
        if kind == "get_business_info":
            return say("hours", hours=res.get("spoken_hours") or f["hours_spoken"]), []
        if "error" in res:
            return say("repeat"), []
        if kind == "check_availability":
            if res.get("available"):
                if st["name"]:
                    return self._readback(res, st["name"], lang, say), []
                return self._ask_name(res, lang, say), []
            return self._unavailable(res, lang, say), []
        if kind == "create_booking":
            if not res.get("booked"):
                return self._unavailable(res, lang, say), []
            key = "booked" if res.get("can_receive_whatsapp") else "booked_landline"
            return say(key, name=res["customer_name"], code=res["spoken_code"], last4=res.get("spoken_last4", "")), []
        if kind == "send_booking_summary":
            if res.get("sent") or res.get("dry_run"):
                key = "sent_sms" if res.get("channel") == "sms" else "sent_whatsapp"
                return say(key, last4=res.get("spoken_last4", "")), []
            if res.get("reason") == "landline":
                return say("ask_mobile"), []
            if res.get("reason") == "invalid_number":
                return say("bad_number"), []
            return say("send_failed"), []
        return say("fallback"), []

    @staticmethod
    def _what(res: dict, lang: str) -> str:
        if res.get("spoken_party"):
            if lang == "en":
                return f"a table for {res.get('party_size')}"
            sp = res["spoken_party"]
            return "طاولة ل" + sp
        what = res.get("spoken_service", "")
        if lang == "ar":
            what = "موعد " + what
        if res.get("spoken_doctor"):
            what += (" مع " if lang == "ar" else " with ") + res["spoken_doctor"]
        return what

    def _ask_name(self, res, lang, say):
        what = self._what(res, lang)
        return say("ask_name", what=what, day=res["spoken_date"], time=res["spoken_requested_time"])

    def _readback(self, res, name, lang, say):
        extra = ""
        if res.get("spoken_area") and lang == "ar":
            extra = f" في {res['spoken_area']}"
        elif res.get("area") and lang == "en":
            extra = f" ({res['area']})"
        if res.get("spoken_price") and lang == "ar":
            extra += f"، والسعر {res['spoken_price']}"
        return say("readback", what=self._what(res, lang), day=res["spoken_date"],
                   time=res["spoken_requested_time"], extra=extra, name=name)

    def _unavailable(self, res, lang, say):
        reason = res.get("reason")
        day = res.get("spoken_date", "")
        alts = [a["spoken"] for a in res.get("alternatives") or []]
        if reason == "party_too_large":
            return say("too_big", n=res["max_party_size"])
        if reason == "no_doctor_that_day":
            return say("no_doctor", service=res.get("spoken_service", ""), day=day)
        if reason == "closed_that_day":
            return say("closed", day=day)
        if reason == "date_in_past":
            return say("past")
        if reason in ("fully_booked", "too_far_ahead") or not alts and not res.get("slots"):
            return say("full", day=day)
        if res.get("slots") and "requested_time" not in res:
            return say("slots", day=day, alts=_join([s["spoken"] for s in res["slots"][:3]], lang))
        key = "outside" if reason == "outside_opening_hours" else "taken"
        return say(key, time=res.get("spoken_requested_time", ""), day=day, alts=_join(alts, lang))

    # --------------------------------------------------------------- summary
    def _summary(self, messages):
        text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        facts = []
        for line in text.splitlines():
            if line.startswith("TOOL create_booking") and '"booked": true' in line:
                d = json.loads(line.split("-> ", 1)[1])
                what = written_party(int(d["party_size"])) if d.get("party_size") else \
                    "موعد " + str(d.get("spoken_service", ""))
                facts.append(f"حجز {what} بتاريخ {d['date']} الساعة {d['time']} باسم {d['customer_name']} "
                             f"(رقم الحجز {d['code']})")
            elif line.startswith("TOOL send_booking_summary"):
                d = json.loads(line.split("-> ", 1)[1])
                ch = "واتساب" if d.get("channel") == "whatsapp" else "رسالة نصية"
                if d.get("sent"):
                    facts.append(f"أُرسل التأكيد عبر {ch}")
                elif d.get("dry_run"):
                    facts.append(f"جُهّز تأكيد {ch} في وضع التجربة (لم يُرسل)")
                else:
                    facts.append("تعذر إرسال التأكيد")
            elif line.startswith("TOOL transfer_to_human"):
                facts.append("طلب التحدث مع موظف" + (" وتم التحويل" if '"transferred": true' in line
                                                    else " (لا يوجد خط متاح، أُخذت رسالة)"))
        barge = text.count("[interrupted]")
        if barge:
            facts.append(f"قاطع المساعد {barge} مرة")
        if not facts:
            facts.append("لم يتم أي حجز")
        return "[ملخص تجريبي من mock] المتصل: " + "؛ ".join(facts) + "."


def _matches(res: dict | None, args: dict) -> bool:
    if not res:
        return False
    if res.get("date") != args["date"]:
        return False
    if res.get("requested_time", res.get("time")) != args["time"]:
        return False
    if "party_size" in args and res.get("party_size") != args["party_size"]:
        return False
    if args.get("area") and res.get("area") != args["area"]:
        return False
    if "service" in args and res.get("service") != args["service"]:
        return False
    if args.get("doctor") and res.get("doctor") != args["doctor"]:
        return False
    return True


def _changed_in(text: str, f: dict) -> bool:
    return bool(gulf.parse_time(text) or gulf.parse_day(text, f["today"]) or gulf.parse_party_size(text))


def _day_words(d: date, f: dict, lang: str) -> str:
    from ..arabic.numbers import date_to_words
    if lang == "en":
        return d.strftime("%A, %B ") + str(d.day)
    return date_to_words(d, "msa" if f["dialect"] == "msa" else "gulf", today=f["today"])
