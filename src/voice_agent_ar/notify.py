"""Booking summary to the caller: WhatsApp template (Meta Cloud API) or SMS (Twilio).

Why a *template*: the caller phoned you and has not messaged your WhatsApp number,
so there is no open 24-hour customer-service window. WhatsApp only allows
business-initiated messages through a pre-approved template (category "utility").
``examples/whatsapp_template.json`` is a ready-to-submit Arabic template whose body
parameters match ``template_params()`` below.

SMS goes through Twilio's Messages API. Arabic SMS is encoded as UCS-2, so one
segment is only 70 characters (67 when split), and the body is kept short. In the
UAE, application-to-person SMS needs a sender ID registered for your business; set
``SMS_FROM`` to that sender ID or to your Twilio number.

Without credentials the notifier runs in dry-run mode: it builds the exact HTTP
request it WOULD send and stores it (visible in the dashboard and the call log), so
the whole flow is testable offline. Nothing is sent and nothing pretends it was.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import httpx

from .arabic.phone import parse_uae_phone
from .profile import Profile
from .storage import Store

log = logging.getLogger("voice_agent_ar.notify")

GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "ÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà")


def sms_segments(body: str) -> tuple[str, int]:
    """("GSM-7" | "UCS-2", number of segments) for an SMS body."""
    if all(c in GSM7 for c in body):
        n = len(body)
        return "GSM-7", 1 if n <= 160 else math.ceil(n / 153)
    n = len(body.encode("utf-16-le")) // 2
    return "UCS-2", 1 if n <= 70 else math.ceil(n / 67)


@dataclass
class NotifyConfig:
    channel: str = "whatsapp_then_sms"     # whatsapp | sms | whatsapp_then_sms | none
    dry_run: bool = True
    wa_token: str = ""
    wa_phone_number_id: str = ""
    wa_template: str = "booking_confirmation_ar"
    wa_template_lang: str = "ar"
    wa_api_version: str = "v21.0"
    wa_base_url: str = "https://graph.facebook.com"
    sms_from: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_base_url: str = "https://api.twilio.com"


def written_party(n: int) -> str:
    """Written (not spoken) Arabic: "طاولة لشخص واحد", "طاولة لشخصين", "طاولة لـ 5 أشخاص", "طاولة لـ 12 شخصاً"."""
    if n == 1:
        return "طاولة لشخص واحد"
    if n == 2:
        return "طاولة لشخصين"
    return f"طاولة لـ {n} " + ("أشخاص" if 3 <= n <= 10 else "شخصاً")


def booking_lines(profile: Profile, b: dict, lang: str = "ar") -> dict[str, str]:
    d = date.fromisoformat(b["date"])
    what = b.get("service_name") or ""
    if profile.kind == "restaurant":
        n = int(b.get("party_size") or 0)
        what = written_party(n) if lang == "ar" else f"Table for {n}"
    elif b.get("doctor_id"):
        doc = next((x for x in profile.doctors if x.id == b["doctor_id"]), None)
        if doc:
            what += f" - {doc.name(lang)}"
    return {"name": b.get("customer_name") or "", "business": profile.display_name(lang), "what": what,
            "day": profile.write_day(d, lang), "time": profile.write_time(b["time"], lang),
            "code": b.get("code") or "", "contact": _written_number(profile.callback_number)}


def _written_number(number: str) -> str:
    """Local format with no spaces ("040000000"). Inside right-to-left Arabic text a spaced
    "04 000 0000" is displayed as "0000 000 04", and "+971..." as "971...+" (bidi reordering).
    A single unbroken run of digits always displays in the right order."""
    p = parse_uae_phone(number or "")
    if p and p.kind != "international":
        return p.national.replace(" ", "")
    return number or ""


def template_params(profile: Profile, b: dict) -> list[str]:
    """Body parameters {{1}}..{{6}} of the bundled WhatsApp template (see examples/whatsapp_template.json)."""
    x = booking_lines(profile, b, "ar")
    return [x["name"], x["business"], x["what"], f"{x['day']} {x['time']}", x["code"], x["contact"] or "-"]


def sms_body(profile: Profile, b: dict, lang: str = "ar") -> str:
    x = booking_lines(profile, b, lang)
    if lang == "en":
        body = f"{x['business']}: {x['what']}, {x['day']} {x['time']}. Ref {x['code']}"
    else:
        body = f"{x['business']}: تم تأكيد حجزك، {x['what']}، {x['day']} {x['time']}. رقم الحجز {x['code']}"
    if x["contact"]:
        body += f" Tel {x['contact']}" if lang == "en" else f" للتواصل {x['contact']}"
    return body


class Notifier:
    def __init__(self, cfg: NotifyConfig, store: Store, profile: Profile,
                 client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.store = store
        self.profile = profile
        self.client = client

    @property
    def wa_configured(self) -> bool:
        return bool(self.cfg.wa_token and self.cfg.wa_phone_number_id)

    @property
    def sms_configured(self) -> bool:
        return bool(self.cfg.twilio_account_sid and self.cfg.twilio_auth_token and self.cfg.sms_from)

    def plan(self, preferred: str | None = None) -> list[str]:
        ch = preferred or self.cfg.channel
        return {"whatsapp": ["whatsapp"], "sms": ["sms"], "whatsapp_then_sms": ["whatsapp", "sms"],
                "none": []}.get(ch, ["whatsapp", "sms"])

    # ---- requests -------------------------------------------------------
    def whatsapp_request(self, to_e164: str, b: dict) -> dict:
        c = self.cfg
        return {
            "method": "POST",
            "url": f"{c.wa_base_url}/{c.wa_api_version}/{c.wa_phone_number_id or '<PHONE_NUMBER_ID>'}/messages",
            "json": {
                "messaging_product": "whatsapp", "recipient_type": "individual",
                "to": to_e164.lstrip("+"), "type": "template",
                "template": {"name": c.wa_template, "language": {"code": c.wa_template_lang},
                             "components": [{"type": "body", "parameters": [
                                 {"type": "text", "text": v} for v in template_params(self.profile, b)]}]},
            },
        }

    def sms_request(self, to_e164: str, b: dict, lang: str = "ar") -> dict:
        c = self.cfg
        body = sms_body(self.profile, b, lang)
        enc, seg = sms_segments(body)
        return {"method": "POST",
                "url": f"{c.twilio_base_url}/2010-04-01/Accounts/{c.twilio_account_sid or '<ACCOUNT_SID>'}"
                       "/Messages.json",
                "data": {"To": to_e164, "From": c.sms_from or "<SMS_FROM>", "Body": body},
                "encoding": enc, "segments": seg}

    async def _post(self, req: dict, auth=None, headers=None) -> httpx.Response:
        client = self.client or httpx.AsyncClient(timeout=15.0)
        try:
            if "json" in req:
                return await client.post(req["url"], json=req["json"], headers=headers)
            return await client.post(req["url"], data=req["data"], auth=auth)
        finally:
            if self.client is None:
                await client.aclose()

    # ---- send ---------------------------------------------------------
    async def send_booking_summary(self, booking: dict, to: str = "", channel: str | None = None,
                                   lang: str = "ar", call_id: str = "") -> dict:
        number = to or booking.get("phone") or ""
        phone = parse_uae_phone(number)
        if phone is None:
            return {"sent": False, "reason": "invalid_number", "detail": f"not a valid phone number: {number!r}"}
        if not phone.whatsapp_capable:
            return {"sent": False, "reason": "landline",
                    "detail": "landline numbers cannot receive WhatsApp or SMS; ask for a mobile number"}
        attempts = []
        for ch in self.plan(channel):
            req = self.whatsapp_request(phone.e164, booking) if ch == "whatsapp" else \
                self.sms_request(phone.e164, booking, lang)
            configured = self.wa_configured if ch == "whatsapp" else self.sms_configured
            base = {"booking_id": booking.get("id"), "call_id": call_id or booking.get("call_id"),
                    "channel": ch, "to_number": phone.e164,
                    "body": " | ".join(template_params(self.profile, booking)) if ch == "whatsapp"
                    else req["data"]["Body"], "request": req}
            if self.cfg.dry_run or not configured:
                self.store.add_notification(**base, status="dry_run",
                                            error="" if self.cfg.dry_run else f"{ch} not configured")
                attempts.append({"channel": ch, "status": "dry_run"})
                return {"sent": False, "dry_run": True, "channel": ch, "to": phone.e164,
                        "note": "dry-run: request built and logged, nothing was sent", "attempts": attempts}
            try:
                if ch == "whatsapp":
                    r = await self._post(req, headers={"Authorization": f"Bearer {self.cfg.wa_token}"})
                    ok = r.status_code < 400
                    pid = ((r.json().get("messages") or [{}])[0].get("id", "")) if ok else ""
                else:
                    r = await self._post(req, auth=(self.cfg.twilio_account_sid, self.cfg.twilio_auth_token))
                    ok = r.status_code < 400
                    pid = r.json().get("sid", "") if ok else ""
                err = "" if ok else f"HTTP {r.status_code}: {r.text[:300]}"
            except httpx.HTTPError as e:
                ok, pid, err = False, "", repr(e)[:300]
            self.store.add_notification(**base, status="sent" if ok else "failed", provider_id=pid, error=err)
            attempts.append({"channel": ch, "status": "sent" if ok else "failed", "error": err})
            if ok:
                return {"sent": True, "channel": ch, "to": phone.e164, "provider_id": pid, "attempts": attempts}
            log.warning("%s confirmation failed: %s", ch, err)
        return {"sent": False, "reason": "all_channels_failed" if attempts else "disabled", "attempts": attempts}
