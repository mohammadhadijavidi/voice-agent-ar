"""Tools the agent calls during a call: availability, booking, WhatsApp/SMS summary, transfer.

Every result that the agent will say out loud carries ``spoken_*`` fields already
written as Gulf (or MSA) Arabic words, so the model never has to read "20:30" or
"150 AED" itself. Inputs are tolerant: dates may be "2026-11-05", "باجر" or
"الخميس"; times may be "20:30", "8:30 pm" or "ثمان ونص بالليل".

Capacity model
* restaurant: each seating area has a number of seats (covers). A booking holds
  ``party_size`` seats in one area for ``dining_minutes``.
* clinic: each doctor sees one patient at a time; a service lasts ``duration_min``.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Awaitable, Callable

from .arabic import gulf
from .arabic.normalize import normalize, normalize_digits
from .arabic.numbers import digits_to_words, people
from .arabic.phone import last_digits, parse_uae_phone
from .profile import SEATING_AR, Profile
from .storage import Store
from .webhooks import WebhookDispatcher


@dataclass
class ToolContext:
    call_id: str
    caller_number: str
    language: str
    profile: Profile
    store: Store
    webhooks: WebhookDispatcher
    notifier: Any = None
    control: dict = field(default_factory=dict)   # {"action": "hangup"|"transfer", ...}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[dict] | dict]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register(self, name: str, description: str, parameters: dict):
        def deco(fn):
            self.add(Tool(name, description, parameters, fn))
            return fn
        return deco

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    async def call(self, name: str, args: dict, ctx: ToolContext) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        if "_raw" in args:
            return {"error": "arguments were not valid JSON", "raw": args["_raw"]}
        try:
            res = tool.handler(ctx, **args)
            if inspect.isawaitable(res):
                res = await res
            return res
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except ToolError as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# tolerant parsing of tool arguments
# ---------------------------------------------------------------------------

def parse_date_arg(value: str, profile: Profile) -> date:
    v = normalize_digits(str(value or "")).strip()
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        pass
    d = gulf.parse_day(v, profile.today())
    if d is None:
        raise ToolError(f"could not understand date {value!r}; pass YYYY-MM-DD")
    return d


def parse_time_arg(value: str, profile: Profile) -> time:
    v = normalize_digits(str(value or "")).strip().lower()
    if not v:
        raise ToolError("time is required (HH:MM, 24h)")
    raw = v.replace(" ", "")
    pm, am = raw.endswith(("pm", "p.m.")), raw.endswith(("am", "a.m."))
    core = raw.rstrip("apm.").strip()
    if core.replace(":", "").replace(".", "").isdigit():
        h, _, m = core.replace(".", ":").partition(":")
        h, m = int(h), int(m or 0)
        if pm and h < 12:
            h += 12
        if am and h == 12:
            h = 0
        if 0 <= h < 24 and 0 <= m < 60:
            return time(h, m)
        raise ToolError(f"invalid time {value!r}")
    te = gulf.parse_time(v if any(w in v for w in ("ساع", "على")) else "الساعة " + v)
    if te is None:
        raise ToolError(f"could not understand time {value!r}; pass HH:MM (24h)")
    return time.fromisoformat(te.resolve(lambda h, m: profile.opens_hour(h, m)))


def _mins(t: time) -> int:
    return t.hour * 60 + t.minute


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def candidate_starts(profile: Profile, day: date, hold_minutes: int, *, fit_whole: bool) -> list[int]:
    """Start minutes on ``day`` respecting shifts, slot grid and minimum notice."""
    out = []
    earliest = profile.now() + timedelta(minutes=profile.min_notice_minutes)
    for a, b in profile.shifts_on(day):
        start = _mins(a)
        end = 24 * 60 if (b.hour, b.minute, b.second) == (23, 59, 59) else _mins(b)
        last = end - (hold_minutes if fit_whole else profile.slot_minutes)
        t = start
        while t <= last:
            dt = datetime.combine(day, time(t // 60, t % 60), tzinfo=profile.tz)
            if dt >= earliest:
                out.append(t)
            t += profile.slot_minutes
    return out


# ---------------------------------------------------------------------------
# restaurant capacity
# ---------------------------------------------------------------------------

def _area_key(profile: Profile, area: str | None) -> str | None:
    if not area:
        return None
    a = normalize(area)
    if a in profile.areas:
        return a
    g = gulf.parse_seating(area)
    if g in profile.areas:
        return g
    for k in profile.areas:
        if normalize(SEATING_AR.get(k, k)) in a or a in k:
            return k
    raise ToolError(f"unknown seating area {area!r}; areas: {', '.join(profile.areas)}")


def table_free_area(profile: Profile, store: Store, day: date, start: int, party: int,
                    area: str | None) -> str | None:
    areas = [area] if area else list(profile.areas)
    for a in areas:
        used = sum(b["party_size"] or 0 for b in
                   store.overlapping(day.isoformat(), start, start + profile.dining_minutes, area=a))
        if used + party <= profile.areas[a]:
            return a
    return None


# ---------------------------------------------------------------------------
# clinic capacity
# ---------------------------------------------------------------------------

def clinic_free_doctor(profile: Profile, store: Store, day: date, start: int, service, doctor=None):
    docs = [doctor] if doctor else profile.doctors_for(service.id, day)
    if not profile.doctors:   # clinic without a doctor list: one chair per service
        busy = store.overlapping(day.isoformat(), start, start + service.duration_min, service_id=service.id)
        return "any" if not busy else None
    for d in docs:
        if d is None or service.id not in d.services:
            continue
        from .profile import WEEKDAYS
        if WEEKDAYS[day.weekday()] not in d.days:
            continue
        if not store.overlapping(day.isoformat(), start, start + service.duration_min, doctor_id=d.id):
            return d.id
    return None


# ---------------------------------------------------------------------------
# helpers for spoken results
# ---------------------------------------------------------------------------

def spoken_digits(digits: str, lang: str, style: str) -> str:
    return " ".join(digits) if lang == "en" else digits_to_words(digits, style)


def _spoken_slot(profile: Profile, day: date, m: int, lang: str) -> dict:
    return {"time": _hhmm(m), "spoken": profile.say_time(_hhmm(m), lang)}


def _nearest(starts: list[int], wanted: int, n: int = 3) -> list[int]:
    return sorted(sorted(starts, key=lambda s: (abs(s - wanted), s))[:n])


def _base(profile: Profile, day: date, lang: str) -> dict:
    return {"date": day.isoformat(), "weekday": day.strftime("%A"), "spoken_date": profile.say_day(day, lang)}


def _check_day(profile: Profile, day: date, lang: str) -> dict | None:
    today = profile.today()
    if day < today:
        return {**_base(profile, day, lang), "available": False, "reason": "date_in_past"}
    if (day - today).days > profile.booking_horizon_days:
        return {**_base(profile, day, lang), "available": False, "reason": "too_far_ahead",
                "max_days_ahead": profile.booking_horizon_days}
    if not profile.shifts_on(day):
        return {**_base(profile, day, lang), "available": False, "reason": "closed_that_day",
                "spoken_hours": profile.hours_text(lang)}
    return None


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

async def check_availability(ctx: ToolContext, date: str, time: str = "", party_size: int | str = 0,
                             area: str = "", service: str = "", doctor: str = "") -> dict:
    p = ctx.profile
    lang = ctx.language
    day = parse_date_arg(date, p)
    early = _check_day(p, day, lang)
    if early:
        return early
    res: dict[str, Any] = _base(p, day, lang)
    want = parse_time_arg(time, p) if time else None

    if p.kind == "restaurant":
        raw_party = normalize_digits(str(party_size or "")).strip()
        party = int(raw_party) if raw_party.isdigit() else (gulf.parse_party_size(raw_party, asked=True) or 0)
        if party <= 0:
            raise ToolError("party_size is required for a table booking")
        if party > p.max_party_size:
            return {**res, "available": False, "reason": "party_too_large", "max_party_size": p.max_party_size,
                    "note": "offer transfer_to_human for group bookings"}
        a = _area_key(p, area)
        res.update(party_size=party, spoken_party=people(party, p.style) if lang == "ar" else
                   f"{party} {'person' if party == 1 else 'people'}", area=a)
        starts = candidate_starts(p, day, p.dining_minutes, fit_whole=False)
        free = [s for s in starts if table_free_area(p, ctx.store, day, s, party, a)]

        def pick(s):
            return table_free_area(p, ctx.store, day, s, party, a)
    else:
        svc = p.find_service(service) if service else None
        if not svc:
            raise ToolError(f"service is required; one of: {', '.join(s.id for s in p.services)}")
        doc = p.find_doctor(doctor) if doctor else None
        if doctor and not doc:
            raise ToolError(f"unknown doctor {doctor!r}; doctors: {', '.join(d.id for d in p.doctors)}")
        res.update(service=svc.id, spoken_service=svc.name(lang), duration_min=svc.duration_min)
        if svc.price is not None:
            res.update(price=svc.price, spoken_price=p.say_price(svc.price, lang))
        if doc:
            res.update(doctor=doc.id, spoken_doctor=doc.name(lang))
        starts = candidate_starts(p, day, svc.duration_min, fit_whole=True)
        if p.doctors and not (([doc] if doc else p.doctors_for(svc.id, day)) and
                              (doc is None or doc in p.doctors_for(svc.id, day))):
            working = sorted({d for x in p.doctors_for(svc.id) for d in x.days},
                             key=lambda d: ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(d))
            return {**res, "available": False, "reason": "no_doctor_that_day", "doctor_days": working}
        free = [s for s in starts if clinic_free_doctor(p, ctx.store, day, s, svc, doc)]

        def pick(s):
            return clinic_free_doctor(p, ctx.store, day, s, svc, doc)

    if want is not None:
        w = _mins(want)
        key = _hhmm(w)
        res["requested_time"] = key
        res["spoken_requested_time"] = p.say_time(key, lang)
        if w in free:
            res["available"] = True
            chosen = pick(w)
            if p.kind == "restaurant":
                res["area"] = chosen
                res["spoken_area"] = SEATING_AR.get(chosen, chosen) if lang == "ar" else chosen
            elif chosen and chosen != "any":
                d = next(x for x in p.doctors if x.id == chosen)
                res.update(doctor=d.id, spoken_doctor=d.name(lang))
        else:
            res["available"] = False
            res["reason"] = "outside_opening_hours" if w not in starts and not p.is_open_at(day, want) \
                else ("too_soon" if w not in starts and p.is_open_at(day, want) and day == p.today()
                      else "taken")
            res["alternatives"] = [_spoken_slot(p, day, s, lang) for s in _nearest(free, w)]
            if not free:
                res["reason"] = "fully_booked"
    else:
        res["available"] = bool(free)
        res["slots"] = [_spoken_slot(p, day, s, lang) for s in free[:6]]
        res["total_free_slots"] = len(free)
        if not free:
            res["reason"] = "fully_booked"
    return res


def _new_code(store: Store) -> str:
    for _ in range(50):
        code = f"{random.randint(10000, 99999)}"
        if not store.get_booking(code):
            return code
    raise ToolError("could not allocate a booking code")


async def create_booking(ctx: ToolContext, date: str, time: str, customer_name: str,
                         confirmed_by_caller: bool = False, party_size: int | str = 0, area: str = "",
                         service: str = "", doctor: str = "", phone: str = "", notes: str = "") -> dict:
    p = ctx.profile
    if not customer_name or not str(customer_name).strip():
        raise ToolError("customer_name is required")
    if str(confirmed_by_caller).lower() not in ("true", "1", "yes"):
        return {"booked": False, "reason": "not_confirmed",
                "note": "read the details back to the caller and get a clear yes, then call again with "
                        "confirmed_by_caller=true"}
    check = await check_availability(ctx, date=date, time=time, party_size=party_size, area=area,
                                     service=service, doctor=doctor)
    if not check.get("available"):
        return {"booked": False, **{k: v for k, v in check.items() if k != "available"}}
    number = ""
    if phone:
        ph = parse_uae_phone(phone)
        if not ph:
            raise ToolError(f"invalid phone number {phone!r}")
        number = ph.e164
    else:
        ph = parse_uae_phone(ctx.caller_number)
        number = ph.e164 if ph else ctx.caller_number
    day = parse_date_arg(date, p)
    t = parse_time_arg(time, p)
    if p.kind == "restaurant":
        svc_id, svc_name, duration = "table", "table", p.dining_minutes
        doctor_id = None
        party, area_key = check["party_size"], check["area"]
    else:
        svc = p.find_service(service)
        svc_id, svc_name, duration = svc.id, svc.name("ar"), svc.duration_min
        doctor_id = check.get("doctor")
        party, area_key = 1, None
    code = _new_code(ctx.store)
    b = ctx.store.create_booking(
        call_id=ctx.call_id, code=code, kind=p.kind, service_id=svc_id, service_name=svc_name,
        doctor_id=doctor_id, party_size=party, area=area_key, date=day.isoformat(), time=_hhmm(_mins(t)),
        duration_min=duration, customer_name=str(customer_name).strip(), phone=number, notes=notes,
        language=ctx.language)
    ctx.webhooks.emit("booking.created", {**b, "business": p.name})
    out = {"booked": True, "booking_id": b["id"], "code": code, "spoken_code": spoken_digits(code, ctx.language, p.style),
           "date": b["date"], "time": b["time"], "customer_name": b["customer_name"],
           "spoken_date": check["spoken_date"], "spoken_time": p.say_time(b["time"], ctx.language),
           "phone": number, "spoken_last4": spoken_digits(last_digits(number), ctx.language, p.style) if number else "",
           "can_receive_whatsapp": bool(parse_uae_phone(number) and parse_uae_phone(number).whatsapp_capable)}
    for k in ("party_size", "spoken_party", "area", "spoken_area", "service", "spoken_service", "doctor",
              "spoken_doctor", "spoken_price"):
        if k in check:
            out[k] = check[k]
    return out


async def send_booking_summary(ctx: ToolContext, booking_code: str = "", to_number: str = "",
                               channel: str = "") -> dict:
    """Send the confirmation by WhatsApp template (or SMS)."""
    if ctx.notifier is None:
        return {"sent": False, "reason": "notifications_disabled"}
    b = ctx.store.get_booking(booking_code) if booking_code else None
    if b is None:
        mine = ctx.store.list_bookings(ctx.call_id)
        b = mine[-1] if mine else None
    if b is None:
        return {"sent": False, "reason": "no_booking", "note": "create the booking first"}
    ch = (channel or "").lower() or None
    if ch and ch not in ("whatsapp", "sms"):
        ch = gulf.channel_preference(ch)
    res = await ctx.notifier.send_booking_summary(b, to=to_number, channel=ch, lang=ctx.language,
                                                  call_id=ctx.call_id)
    if res.get("to"):
        res["spoken_last4"] = spoken_digits(last_digits(res["to"]), ctx.language, ctx.profile.style)
    ctx.store.add_event(ctx.call_id, "booking_summary", {k: v for k, v in res.items() if k != "attempts"})
    return res


async def cancel_booking(ctx: ToolContext, booking_code: str, confirmed_by_caller: bool = False) -> dict:
    code = normalize_digits(str(booking_code)).replace(" ", "")
    if not code.isdigit():
        code = "".join(str(gulf.word_value(w)) for w in str(booking_code).split()
                       if gulf.word_value(w) is not None and gulf.word_value(w) < 10)
    b = ctx.store.get_booking(code)
    if not b or b["status"] != "confirmed":
        return {"cancelled": False, "reason": "not_found"}
    caller = parse_uae_phone(ctx.caller_number)
    if not caller or caller.e164 != b["phone"]:
        return {"cancelled": False, "reason": "caller_mismatch",
                "note": "only the number that made the booking can cancel by phone; offer transfer_to_human"}
    if str(confirmed_by_caller).lower() not in ("true", "1", "yes"):
        return {"cancelled": False, "reason": "not_confirmed", "note": "read back the booking and ask to confirm"}
    ctx.store.update_booking(b["id"], status="cancelled")
    ctx.webhooks.emit("booking.cancelled", {"id": b["id"], "code": b["code"], "business": ctx.profile.name})
    return {"cancelled": True, "code": b["code"], "spoken_date": ctx.profile.say_day(date.fromisoformat(b["date"])),
            "spoken_time": ctx.profile.say_time(b["time"], ctx.language)}


async def get_business_info(ctx: ToolContext, topic: str = "") -> dict:
    p = ctx.profile
    out: dict[str, Any] = {"name": p.display_name(ctx.language), "spoken_hours": p.hours_text(ctx.language),
                           "address": p.address.get(ctx.language) or p.address.get("ar", "")}
    if p.kind == "clinic":
        out["services"] = [{"id": s.id, "name": s.name(ctx.language), "duration_min": s.duration_min,
                            "spoken_price": p.say_price(s.price, ctx.language) if s.price is not None else None}
                           for s in p.services]
    else:
        out["seating_areas"] = [SEATING_AR.get(a, a) if ctx.language == "ar" else a for a in p.areas]
    return out


async def transfer_to_human(ctx: ToolContext, reason: str = "") -> dict:
    number = ctx.profile.transfer_number
    if not number:
        return {"transferred": False, "reason": "no_staff_line",
                "note": "apologise and offer to take a message with take_message"}
    ctx.control.update(action="transfer", number=number, reason=reason)
    return {"transferred": True, "note": "tell the caller you are connecting them now"}


async def take_message(ctx: ToolContext, name: str = "", message: str = "", callback_number: str = "") -> dict:
    ph = parse_uae_phone(callback_number or ctx.caller_number)
    data = {"call_id": ctx.call_id, "name": name, "message": message,
            "callback_number": ph.e164 if ph else (callback_number or ctx.caller_number),
            "language": ctx.language, "business": ctx.profile.name}
    oid = ctx.webhooks.emit("message.created", data)
    ctx.store.add_event(ctx.call_id, "message", data)
    return {"saved": True, "message_id": f"msg_{oid}"}


async def end_call(ctx: ToolContext, reason: str = "") -> dict:
    ctx.control.update(action="hangup", reason=reason or "completed")
    return {"ok": True}


def default_tools(profile: Profile | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    s = {"type": "string"}
    clinic = profile is not None and profile.kind == "clinic"
    avail_props: dict[str, Any] = {"date": {**s, "description": "YYYY-MM-DD"},
                                   "time": {**s, "description": "HH:MM 24h, optional"}}
    book_props: dict[str, Any] = {
        "date": {**s, "description": "YYYY-MM-DD"}, "time": {**s, "description": "HH:MM 24h"},
        "customer_name": s,
        "confirmed_by_caller": {"type": "boolean",
                                "description": "true only after you read the details back and the caller said yes"},
        "phone": {**s, "description": "optional; defaults to caller ID"}, "notes": s}
    if clinic:
        extra = {"service": {**s, "description": "service id"},
                 "doctor": {**s, "description": "doctor id, optional"}}
        req = ["service", "date"]
    else:
        extra = {"party_size": {"type": "integer"},
                 "area": {**s, "description": "seating area, optional: " + ", ".join(profile.areas if profile else [])}}
        req = ["party_size", "date"]
    reg.add(Tool("check_availability", "Check free times for a date (optionally a specific time).",
                 {"type": "object", "properties": {**avail_props, **extra}, "required": req}, check_availability))
    reg.add(Tool("create_booking", "Create a confirmed booking after the caller confirmed the read-back.",
                 {"type": "object", "properties": {**book_props, **extra},
                  "required": req + ["time", "customer_name", "confirmed_by_caller"]}, create_booking))
    reg.add(Tool("send_booking_summary",
                 "Send the booking confirmation by WhatsApp template (default) or SMS.",
                 {"type": "object", "properties": {
                     "booking_code": s, "to_number": {**s, "description": "optional; defaults to the booking phone"},
                     "channel": {"type": "string", "enum": ["whatsapp", "sms"]}}}, send_booking_summary))
    reg.add(Tool("cancel_booking", "Cancel a booking by its code (caller ID must match).",
                 {"type": "object", "properties": {"booking_code": s, "confirmed_by_caller": {"type": "boolean"}},
                  "required": ["booking_code"]}, cancel_booking))
    reg.add(Tool("get_business_info", "Opening hours, address, services and prices.",
                 {"type": "object", "properties": {"topic": s}}, get_business_info))
    reg.add(Tool("transfer_to_human", "Transfer the call to staff.",
                 {"type": "object", "properties": {"reason": s}}, transfer_to_human))
    reg.add(Tool("take_message", "Leave a message for staff to call back.",
                 {"type": "object", "properties": {"name": s, "message": s, "callback_number": s}}, take_message))
    reg.add(Tool("end_call", "Hang up after saying goodbye.",
                 {"type": "object", "properties": {"reason": s}}, end_call))
    return reg
