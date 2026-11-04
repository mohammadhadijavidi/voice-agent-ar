"""Twilio helpers: TwiML, webhook signature validation, stream tokens, call control.

No Twilio SDK dependency: the request signature algorithm and the two REST calls we
need (update a live call with new TwiML) are small and documented.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from xml.sax.saxutils import escape

import httpx

log = logging.getLogger("voice_agent.twilio")


def _attr(v: str) -> str:
    return '"' + escape(str(v), {'"': "&quot;"}) + '"'


def compute_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """X-Twilio-Signature: base64(HMAC-SHA1(auth_token, url + sorted(k+v)))."""
    data = url + "".join(k + str(params[k]) for k in sorted(params))
    digest = hmac.new(auth_token.encode(), data.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def validate_signature(auth_token: str, url: str, params: dict[str, str], signature: str) -> bool:
    if not signature:
        return False
    candidates = {url}
    # Twilio signs the URL exactly as configured; tolerate default ports added/removed by proxies
    if url.startswith("https://") and ":443/" in url:
        candidates.add(url.replace(":443/", "/", 1))
    elif url.startswith("https://"):
        host_end = url.find("/", 8)
        host_end = len(url) if host_end == -1 else host_end
        candidates.add(url[:host_end] + ":443" + url[host_end:])
    return any(hmac.compare_digest(compute_signature(auth_token, u, params), signature) for u in candidates)


def stream_token(secret: str, call_sid: str) -> str:
    """Short HMAC passed as a <Parameter> so only streams we started are accepted."""
    return hmac.new(secret.encode(), ("stream:" + call_sid).encode(), hashlib.sha256).hexdigest()[:32]


def connect_stream_twiml(ws_url: str, params: dict[str, str]) -> str:
    inner = "".join(f"<Parameter name={_attr(k)} value={_attr(v)}/>" for k, v in params.items())
    return ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
            f"<Stream url={_attr(ws_url)}>{inner}</Stream></Connect></Response>")


def dial_twiml(number: str, say: str = "") -> str:
    s = f"<Say>{escape(say)}</Say>" if say else ""
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{s}<Dial>{escape(number)}</Dial></Response>'


def hangup_twiml() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'


class TwilioREST:
    """Minimal Twilio REST client for live call control."""

    def __init__(self, account_sid: str, auth_token: str, client: httpx.AsyncClient | None = None,
                 base_url: str = "https://api.twilio.com"):
        self.sid = account_sid
        self.token = auth_token
        self.base_url = base_url.rstrip("/")
        self.client = client

    @property
    def configured(self) -> bool:
        return bool(self.sid and self.token)

    async def update_call(self, call_sid: str, twiml: str) -> bool:
        if not (self.configured and call_sid):
            return False
        client = self.client or httpx.AsyncClient(timeout=10.0)
        try:
            r = await client.post(f"{self.base_url}/2010-04-01/Accounts/{self.sid}/Calls/{call_sid}.json",
                                  data={"Twiml": twiml}, auth=(self.sid, self.token))
            if r.status_code >= 400:
                log.error("Twilio call update failed: %s %s", r.status_code, r.text[:300])
            return r.status_code < 400
        finally:
            if self.client is None:
                await client.aclose()
