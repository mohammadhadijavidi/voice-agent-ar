"""Business profile: a UAE restaurant or clinic, described in plain YAML.

Two fictional sample businesses ship with the package (``restaurant`` and
``clinic``). Point ``PROFILE`` at your own YAML to customise. See
``examples/my_restaurant.yaml`` for every field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from importlib import resources
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import yaml

from .arabic.normalize import normalize
from .arabic.numbers import WEEKDAYS_AR, date_to_words, price, time_to_words

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
SEATING_AR = {"indoor": "الصالة الداخلية", "outdoor": "الجلسات الخارجية", "family": "قسم العائلات"}


@dataclass
class Service:
    """A clinic service (consultation, cleaning ...) or a restaurant experience (e.g. iftar menu)."""
    id: str
    names: dict[str, str]
    duration_min: int = 30
    price: float | None = None
    aliases: list[str] = field(default_factory=list)

    def name(self, lang: str = "ar") -> str:
        return self.names.get(lang) or self.names.get("ar") or self.names.get("en") or self.id

    def keywords(self) -> list[str]:
        words = [self.id.replace("_", " ")] + list(self.names.values()) + list(self.aliases)
        return sorted({normalize(w) for w in words if w}, key=len, reverse=True)


@dataclass
class Doctor:
    id: str
    names: dict[str, str]
    services: list[str]
    days: list[str] = field(default_factory=lambda: list(WEEKDAYS))

    def name(self, lang: str = "ar") -> str:
        return self.names.get(lang) or self.names.get("ar") or self.id


Shift = tuple[time, time]


@dataclass
class Profile:
    name: str
    kind: str = "restaurant"                  # "restaurant" | "clinic"
    names: dict[str, str] = field(default_factory=dict)
    timezone: str = "Asia/Dubai"
    currency: str = "AED"
    languages: list[str] = field(default_factory=lambda: ["ar", "en"])
    default_language: str = "ar"
    dialect: str = "gulf"                     # "gulf" | "msa": how the agent talks and reads numbers
    hours: dict[str, list[Shift]] = field(default_factory=dict)
    slot_minutes: int = 30
    min_notice_minutes: int = 30
    booking_horizon_days: int = 60
    # restaurant
    dining_minutes: int = 90
    areas: dict[str, int] = field(default_factory=dict)       # area -> covers (seats) per sitting
    max_party_size: int = 12
    # clinic
    services: list[Service] = field(default_factory=list)
    doctors: list[Doctor] = field(default_factory=list)
    # conversation
    agent_name: str = "مريم"
    greeting: dict[str, str] = field(default_factory=dict)
    instructions: str = ""
    address: dict[str, str] = field(default_factory=dict)
    transfer_number: str = ""
    callback_number: str = ""                 # printed in the WhatsApp / SMS confirmation
    stt_hints: list[str] = field(default_factory=list)
    sample: bool = False
    now_fn: Callable[[], datetime] | None = None

    # ---- names & time ---------------------------------------------------
    def display_name(self, lang: str = "ar") -> str:
        return self.names.get(lang) or self.name

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def now(self) -> datetime:
        if self.now_fn:
            n = self.now_fn()
            return n if n.tzinfo else n.replace(tzinfo=self.tz)
        return datetime.now(self.tz)

    def today(self) -> date:
        return self.now().date()

    def shifts_on(self, d: date) -> list[Shift]:
        return self.hours.get(WEEKDAYS[d.weekday()]) or []

    def is_open_at(self, d: date, t: time, duration: int = 0) -> bool:
        m = t.hour * 60 + t.minute
        for a, b in self.shifts_on(d):
            start, end = a.hour * 60 + a.minute, _end_minutes(b)
            if start <= m and m + duration <= end:
                return True
        return False

    def opens_hour(self, h: int, m: int = 0) -> bool:
        """True if any day has a shift containing h:m (used to disambiguate "الساعة ثمان")."""
        mm = h * 60 + m
        return any(a.hour * 60 + a.minute <= mm < _end_minutes(b)
                   for shifts in self.hours.values() for a, b in (shifts or []))

    # ---- lookup -----------------------------------------------------------
    def find_service(self, query: str) -> Service | None:
        if not query:
            return None
        q = normalize(query)
        for s in self.services:
            if q == s.id or q in s.keywords():
                return s
        padded = f" {q} "
        best = None
        for s in self.services:
            for k in s.keywords():
                if (f" {k} " in padded or f" ال{k} " in padded) and (best is None or len(k) > best[0]):
                    best = (len(k), s)
        return best[1] if best else None

    def doctors_for(self, service_id: str, d: date | None = None) -> list[Doctor]:
        out = [doc for doc in self.doctors if service_id in doc.services]
        if d is not None:
            out = [doc for doc in out if WEEKDAYS[d.weekday()] in doc.days]
        return out

    def find_doctor(self, query: str) -> Doctor | None:
        q = normalize(query or "")
        if not q:
            return None
        for doc in self.doctors:
            if q == doc.id or any(normalize(n) in q or q in normalize(n) for n in doc.names.values()):
                return doc
        return None

    def greeting_for(self, lang: str) -> str:
        return self.greeting.get(lang) or self.greeting.get("ar") or f"أهلاً، معك {self.display_name()}."

    # ---- spoken formatting --------------------------------------------------
    @property
    def style(self) -> str:
        return "msa" if self.dialect == "msa" else "gulf"

    def say_day(self, d: date, lang: str = "ar") -> str:
        if lang == "en":
            return f"{WEEKDAYS_EN[d.weekday()]}, {MONTHS_EN[d.month - 1]} {d.day}"
        return date_to_words(d, self.style, today=self.today())

    def say_time(self, t: str | time, lang: str = "ar") -> str:
        if isinstance(t, str):
            t = time.fromisoformat(t)
        if lang == "en":
            h = t.hour % 12 or 12
            suffix = "AM" if t.hour < 12 else "PM"
            return f"{h}:{t.minute:02d} {suffix}" if t.minute else f"{h} {suffix}"
        return time_to_words(t, self.style)

    def say_price(self, amount: float, lang: str = "ar") -> str:
        if lang == "en":
            return f"{amount:g} dirhams"
        return price(amount, self.currency, self.style)

    def write_day(self, d: date, lang: str = "ar") -> str:
        """Written form for WhatsApp / SMS: "الخميس 5 نوفمبر"."""
        if lang == "en":
            return f"{WEEKDAYS_EN[d.weekday()][:3]} {d.day} {MONTHS_EN[d.month - 1][:3]}"
        from .arabic.numbers import MONTHS_AR
        return f"{WEEKDAYS_AR[d.weekday()]} {d.day} {MONTHS_AR[d.month - 1]}"

    def write_time(self, t: str | time, lang: str = "ar") -> str:
        if isinstance(t, str):
            t = time.fromisoformat(t)
        h = t.hour % 12 or 12
        if lang == "en":
            return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"
        return f"{h}:{t.minute:02d} {'صباحاً' if t.hour < 12 else 'مساءً'}"

    def hours_text(self, lang: str = "ar") -> str:
        """Opening hours as a spoken sentence."""
        groups: list[tuple[list[int], list[Shift]]] = []
        for i, day in enumerate(WEEKDAYS):
            h = self.hours.get(day) or []
            if groups and groups[-1][1] == h:
                groups[-1][0].append(i)
            else:
                groups.append(([i], h))
        names = WEEKDAYS_EN if lang == "en" else WEEKDAYS_AR
        parts = []
        for days, shifts in groups:
            if len(days) == 1:
                span = names[days[0]]
            else:
                span = f"{names[days[0]]} to {names[days[-1]]}" if lang == "en" else \
                    f"من {names[days[0]]} إلى {names[days[-1]]}"
            if not shifts:
                parts.append(f"{span}: closed" if lang == "en" else f"{span} مسكرين")
                continue
            if lang == "en":
                sh = " and ".join(f"{self.say_time(a, 'en')} to {self.say_time(b, 'en')}" for a, b in shifts)
            else:
                sh = " و".join(f"من {self.say_time(a)} إلى {self.say_time(_clip(b))}" for a, b in shifts)
            parts.append(f"{span}: {sh}" if lang == "en" else f"{span} {sh}")
        return "; ".join(parts) if lang == "en" else "، ".join(parts)

    def hours_machine(self) -> str:
        """Compact 24h hours line for the system prompt (exact, for the model and the mock)."""
        out = []
        for day in WEEKDAYS:
            shifts = self.hours.get(day) or []
            out.append(f"{day} " + (",".join(f"{a:%H:%M}-{_fmt_end(b)}" for a, b in shifts) or "closed"))
        return "; ".join(out)

    # ---- the prompt ---------------------------------------------------------
    def system_prompt(self, lang: str, caller: str = "") -> str:
        from .prompts import build_system_prompt
        return build_system_prompt(self, lang, caller)


def _end_minutes(t: time) -> int:
    # "24:00" is stored as 23:59:59 so it can be a datetime.time; count it as midnight
    return 24 * 60 if (t.hour, t.minute, t.second) == (23, 59, 59) else t.hour * 60 + t.minute


def _clip(t: time) -> time:
    return time(0, 0) if (t.hour, t.minute, t.second) == (23, 59, 59) else t


def _fmt_end(t: time) -> str:
    return "24:00" if (t.hour, t.minute, t.second) == (23, 59, 59) else f"{t:%H:%M}"


def _parse_clock(v: str) -> time:
    v = v.strip()
    if v in ("24:00", "00:00+"):
        return time(23, 59, 59)
    return time.fromisoformat(v)


def _parse_hours(raw: dict) -> dict[str, list[Shift]]:
    out: dict[str, list[Shift]] = {}
    for day in WEEKDAYS:
        v = (raw or {}).get(day)
        if not v or str(v).lower() == "closed":
            out[day] = []
            continue
        shifts = []
        for part in str(v).split(","):
            a, b = part.split("-")
            start, end = _parse_clock(a), _parse_clock(b)
            if _end_minutes(end) <= start.hour * 60 + start.minute:
                raise ValueError(f"hours for {day}: {part!r} ends before it starts (use 24:00 for midnight)")
            shifts.append((start, end))
        out[day] = shifts
    return out


def profile_from_dict(data: dict) -> Profile:
    services = [
        Service(id=str(s["id"]), names=dict(s.get("names") or {"ar": s["id"]}),
                duration_min=int(s.get("duration_min", 30)),
                price=float(s["price"]) if s.get("price") is not None else None,
                aliases=[str(a) for a in s.get("aliases") or []])
        for s in data.get("services") or []
    ]
    doctors = [Doctor(id=str(d["id"]), names=dict(d.get("names") or {}),
                      services=[str(x) for x in d.get("services") or []],
                      days=[str(x) for x in d.get("days") or WEEKDAYS])
               for d in data.get("doctors") or []]
    kind = data.get("kind", "restaurant")
    if kind not in ("restaurant", "clinic"):
        raise ValueError(f"kind must be 'restaurant' or 'clinic', got {kind!r}")
    if kind == "clinic" and not services:
        raise ValueError("a clinic profile needs at least one service")
    if kind == "restaurant" and not data.get("areas"):
        raise ValueError("a restaurant profile needs 'areas' (seating area -> number of seats)")
    return Profile(
        name=data["name"], kind=kind, names=dict(data.get("names") or {}),
        timezone=data.get("timezone", "Asia/Dubai"), currency=data.get("currency", "AED"),
        languages=list(data.get("languages") or ["ar", "en"]),
        default_language=data.get("default_language", "ar"),
        dialect=data.get("dialect", "gulf"),
        hours=_parse_hours(data.get("hours") or {}),
        slot_minutes=int(data.get("slot_minutes", 30)),
        min_notice_minutes=int(data.get("min_notice_minutes", 30)),
        booking_horizon_days=int(data.get("booking_horizon_days", 60)),
        dining_minutes=int(data.get("dining_minutes", 90)),
        areas={str(k): int(v) for k, v in (data.get("areas") or {}).items()},
        max_party_size=int(data.get("max_party_size", 12)),
        services=services, doctors=doctors,
        agent_name=data.get("agent_name", "مريم"),
        greeting=dict(data.get("greeting") or {}),
        instructions=data.get("instructions", ""),
        address=dict(data.get("address") or {}) if isinstance(data.get("address"), dict)
        else {"ar": str(data.get("address") or "")},
        transfer_number=str(data.get("transfer_number") or ""),
        callback_number=str(data.get("callback_number") or ""),
        stt_hints=[str(x) for x in data.get("stt_hints") or []],
        sample=bool(data.get("sample", False)),
    )


def bundled_profiles() -> list[str]:
    root = resources.files("voice_agent_ar") / "profiles"
    return sorted(p.name.removesuffix(".yaml") for p in root.iterdir() if p.name.endswith(".yaml"))


def load_profile(name_or_path: str) -> Profile:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        res = resources.files("voice_agent_ar") / "profiles" / f"{name_or_path}.yaml"
        if not res.is_file():
            raise FileNotFoundError(f"Profile {name_or_path!r} not found (bundled: {', '.join(bundled_profiles())})")
        text = res.read_text(encoding="utf-8")
    return profile_from_dict(yaml.safe_load(text))
