"""FastAPI app: Twilio webhooks, the Media Streams WebSocket, dashboard and JSON API."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import secrets
from contextlib import asynccontextmanager
from importlib import resources
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .agent import CallSession
from .config import Settings
from .profile import Profile, load_profile
from .providers.llm import make_llm
from .providers.stt import make_stt
from .providers.tts import make_tts
from .storage import Store
from .notify import Notifier, NotifyConfig
from .tools import ToolRegistry, default_tools
from .twilio import TwilioREST, connect_stream_twiml, dial_twiml, hangup_twiml, stream_token, validate_signature
from .vad import VADConfig
from .webhooks import WebhookDispatcher

log = logging.getLogger("voice_agent_ar.server")


def create_app(settings: Settings | None = None, *, profile: Profile | None = None,
               store: Store | None = None, tools: ToolRegistry | None = None,
               llm=None, stt=None, tts=None, twilio_rest: TwilioREST | None = None,
               webhook_client=None, notifier: Notifier | None = None, notify_client=None) -> FastAPI:
    settings = settings or Settings.from_env()
    profile = profile or load_profile(settings.profile)
    if settings.transfer_number:
        profile.transfer_number = settings.transfer_number
    if settings.dialect in ("gulf", "msa"):
        profile.dialect = settings.dialect
    store = store or Store(settings.db_path)
    tools = tools or default_tools(profile)
    llm = llm or make_llm(settings.llm)
    stt = stt or make_stt(settings.stt)
    tts = tts or make_tts(settings.tts)
    rest = twilio_rest or TwilioREST(settings.twilio_account_sid, settings.twilio_auth_token)
    webhooks = WebhookDispatcher(store, settings.crm_webhook_url, settings.crm_webhook_secret,
                                 client=webhook_client)
    notifier = notifier or Notifier(NotifyConfig(
        channel=settings.notify_channel, dry_run=settings.notify_dry_run, wa_token=settings.wa_token,
        wa_phone_number_id=settings.wa_phone_number_id, wa_template=settings.wa_template,
        wa_template_lang=settings.wa_template_lang, sms_from=settings.sms_from,
        twilio_account_sid=settings.twilio_account_sid, twilio_auth_token=settings.twilio_auth_token),
        store, profile, client=notify_client)
    # The stream token secret: Twilio auth token if present, else a per-process random key
    stream_secret = settings.twilio_auth_token or secrets.token_hex(16)
    sessions: dict[str, CallSession] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        webhooks.start()
        yield
        await webhooks.stop()
        for s in list(sessions.values()):
            await s.finalize(reason="server_shutdown")

    app = FastAPI(title="voice-agent-ar", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.profile = profile
    app.state.sessions = sessions
    app.state.webhooks = webhooks
    app.state.stream_secret = stream_secret
    app.state.notifier = notifier

    static_dir = resources.files("voice_agent_ar") / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------ helpers
    def public_base(request: Request) -> str:
        if settings.public_url:
            return settings.public_url
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
        return f"{proto}://{host}"

    async def twilio_form(request: Request) -> dict[str, str]:
        form = {k: str(v) for k, v in (await request.form()).items()}
        if settings.twilio_auth_token and settings.validate_twilio_signature:
            url = public_base(request) + request.url.path
            if request.url.query:
                url += "?" + request.url.query
            if not validate_signature(settings.twilio_auth_token, url, form,
                                      request.headers.get("x-twilio-signature", "")):
                raise HTTPException(403, "invalid Twilio signature")
        return form

    def require_dashboard(request: Request) -> None:
        tok = settings.dashboard_token
        if not tok:
            return
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() \
            or request.query_params.get("token", "") or request.cookies.get("va_token", "")
        if not hmac.compare_digest(given, tok):
            raise HTTPException(401, "dashboard token required")

    def check_stream(call_sid: str, params: dict) -> bool:
        return hmac.compare_digest(params.get("token", ""), stream_token(stream_secret, call_sid))

    # ------------------------------------------------------------- twilio
    @app.post("/twilio/voice")
    async def twilio_voice(request: Request, form: dict = Depends(twilio_form)):
        call_sid = form.get("CallSid", "")
        base = public_base(request)
        parts = urlsplit(base)
        ws_scheme = "wss" if parts.scheme == "https" else "ws"
        ws_url = f"{ws_scheme}://{parts.netloc}{parts.path.rstrip('/')}/twilio/stream"
        params = {"token": stream_token(stream_secret, call_sid), "from": form.get("From", ""),
                  "to": form.get("To", ""), "direction": form.get("Direction", "inbound")}
        return Response(connect_stream_twiml(ws_url, params), media_type="application/xml")

    @app.post("/twilio/status")
    async def twilio_status(form: dict = Depends(twilio_form)):
        sid = form.get("CallSid", "")
        for s in sessions.values():
            if s.call_sid == sid and s.call_id:
                store.add_event(s.call_id, "twilio_status", {"status": form.get("CallStatus")})
        return Response(status_code=204)

    async def apply_control(call_sid: str, control: dict) -> None:
        action = control.get("action")
        if action == "transfer" and control.get("number"):
            ok = await rest.update_call(call_sid, dial_twiml(control["number"]))
        elif action == "hangup":
            ok = await rest.update_call(call_sid, hangup_twiml())
        else:
            return
        if not ok:
            log.info("call control %s not sent (Twilio REST not configured or call is simulated)", action)

    @app.websocket("/twilio/stream")
    async def twilio_stream(ws: WebSocket):
        await ws.accept()
        pending: set[asyncio.Task] = set()

        def on_control(call_sid: str, control: dict) -> None:
            t = asyncio.create_task(apply_control(call_sid, control))
            pending.add(t)
            t.add_done_callback(pending.discard)

        async def close() -> None:
            with contextlib.suppress(Exception):
                await ws.close()

        session = CallSession(
            send=ws.send_text, close=close, store=store, profile=profile, llm=llm, stt=stt, tts=tts,
            tools=tools, webhooks=webhooks,
            vad=VADConfig(threshold=settings.vad_threshold, endpoint_ms=settings.vad_endpoint_ms,
                          barge_in_ms=settings.barge_in_ms),
            barge_in_ms=settings.barge_in_ms, summary_enabled=settings.summary_enabled,
            max_call_seconds=settings.max_call_seconds, stream_token_check=check_stream,
            on_control=on_control, notifier=notifier)
        key = secrets.token_hex(6)
        sessions[key] = session
        try:
            while not session.closed:
                raw = await ws.receive_text()
                await session.handle(raw)
                if session.finalized:
                    break
        except WebSocketDisconnect:
            pass
        except RuntimeError:  # receive after close
            pass
        finally:
            await session.finalize(reason="stream_closed")
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            sessions.pop(key, None)

    # ----------------------------------------------------------- dashboard
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        html = (static_dir / "dashboard.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/health")
    async def health():
        return {"ok": True, "version": __version__, "live_calls": len(sessions),
                "providers": {"llm": llm.name, "stt": stt.name, "tts": tts.name},
                "mock": any(getattr(p, "is_mock", False) for p in (llm, stt, tts))}

    @app.get("/api/info", dependencies=[Depends(require_dashboard)])
    async def info():
        return {"business": profile.display_name("ar"), "business_en": profile.name, "kind": profile.kind,
                "dialect": profile.dialect, "sample_profile": profile.sample, "languages": profile.languages,
                "notify": {"channel": settings.notify_channel,
                           "dry_run": settings.notify_dry_run or not (notifier.wa_configured or notifier.sms_configured)},
                "providers": {"llm": f"{llm.name}", "stt": stt.name, "tts": tts.name},
                "mock": any(getattr(p, "is_mock", False) for p in (llm, stt, tts)),
                "webhook": bool(settings.crm_webhook_url), "version": __version__}

    @app.get("/api/stats", dependencies=[Depends(require_dashboard)])
    async def stats():
        return store.stats()

    @app.get("/api/calls", dependencies=[Depends(require_dashboard)])
    async def calls(limit: int = 50, offset: int = 0):
        return store.list_calls(min(limit, 200), offset)

    @app.get("/api/calls/{call_id}", dependencies=[Depends(require_dashboard)])
    async def call_detail(call_id: str):
        c = store.get_call(call_id)
        if not c:
            raise HTTPException(404, "call not found")
        return {"call": c, "turns": store.turns(call_id), "events": store.events(call_id),
                "bookings": store.list_bookings(call_id), "notifications": store.notifications(call_id=call_id)}

    @app.get("/api/bookings", dependencies=[Depends(require_dashboard)])
    async def bookings(limit: int = 100):
        return store.list_bookings(limit=limit)

    @app.get("/api/notifications", dependencies=[Depends(require_dashboard)])
    async def notifications():
        return store.notifications()

    @app.get("/api/outbox", dependencies=[Depends(require_dashboard)])
    async def outbox():
        return store.outbox_all()[-100:]

    @app.exception_handler(HTTPException)
    async def http_error(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app
