"""Command line: ``voice-agent-ar serve | simulate | doctor | scenarios``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import socket
import sys
import tempfile
from importlib import resources
from pathlib import Path

from . import __version__
from .config import Settings, load_dotenv


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.asynccontextmanager
async def running_server(app, host: str = "127.0.0.1", port: int | None = None):
    """Run a uvicorn server for ``app`` in the current event loop (used by simulate + tests)."""
    import uvicorn
    port = port or _free_port()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", ws="websockets-sansio",
                            lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        await task


def bundled_scenarios() -> dict[str, Path]:
    root = resources.files("voice_agent_ar") / "scenarios"
    return {p.name.removesuffix(".yaml"): Path(str(p)) for p in root.iterdir() if p.name.endswith(".yaml")}


async def _simulate(args) -> int:
    from .server import create_app
    from .simulator import Scenario, TwilioSimulator
    from .storage import Store

    scen = bundled_scenarios()
    paths = []
    for name in args.scenario:
        if name == "all":
            paths += [scen[k] for k in sorted(scen)]
        elif name in scen:
            paths.append(scen[name])
        elif Path(name).exists():
            paths.append(Path(name))
        else:
            print(f"unknown scenario {name!r}; bundled: {', '.join(sorted(scen))}", file=sys.stderr)
            return 2
    db = args.db or os.path.join(tempfile.mkdtemp(prefix="voice-agent-ar-"), "sim.db")
    store = Store(db)
    groups: dict[str, list] = {}
    for p in paths:
        sc = Scenario.load(p)
        groups.setdefault(args.profile or sc.profile or "restaurant", []).append(sc)

    for profile_name, scenarios in groups.items():
        settings = Settings.from_env(profile=profile_name, db_path=db)
        # The simulator stands in for Twilio; it never has your real auth token, and never sends WhatsApp/SMS.
        settings.twilio_auth_token = ""
        settings.twilio_account_sid = ""
        settings.notify_dry_run = True
        if not (settings.stt.provider == "mock" and settings.tts.provider == "mock"):
            print("note: the simulator speaks via the FSK test modem, so STT_PROVIDER and TTS_PROVIDER "
                  "are forced to 'mock' here. LLM_PROVIDER is used as configured.", file=sys.stderr)
            settings.stt.provider = "mock"
            settings.tts.provider = "mock"
        app = create_app(settings, store=store)
        async with running_server(app) as base:
            sim = TwilioSimulator(base, speed=args.speed)
            for sc in scenarios:
                sc.seed(store, app.state.profile)
                result = await sim.run(sc)
                for _ in range(60):
                    calls = store.list_calls(1)
                    if calls and calls[0]["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)
                call = store.list_calls(1)[0] if store.list_calls(1) else {}
                notes = store.notifications(call_id=call.get("id")) if call else []
                if args.json:
                    print(json.dumps({"scenario": sc.name, "profile": profile_name, "transcript": result.transcript,
                                      "call": call, "bookings": store.list_bookings(call.get("id")),
                                      "notifications": notes}, ensure_ascii=False, indent=2, default=str))
                    continue
                label = "MOCK providers" if settings.llm.provider == "mock" else f"LLM: {settings.llm.provider}"
                print(f"\n=== {sc.name} · {profile_name} ({label}) ===")
                print(result.format())
                if call:
                    print(f"outcome: {call.get('outcome')} | language: {call.get('language')} | "
                          f"barge-ins: {call.get('barge_ins')} | bookings: {call.get('bookings')}")
                    for n in notes:
                        print(f"confirmation: {n['channel']} -> {n['to_number']} [{n['status']}] {n['body']}")
                    print(f"summary: {call.get('summary')}")
    if not args.json:
        print(f"\ncall log database: {db}\nview it: voice-agent-ar serve --db {db}")
    return 0


def _serve(args) -> int:
    import uvicorn

    from .server import create_app

    overrides = {}
    if args.profile:
        overrides["profile"] = args.profile
    if args.db:
        overrides["db_path"] = args.db
    settings = Settings.from_env(**overrides)
    app = create_app(settings)
    mock = [k for k in ("llm", "stt", "tts") if getattr(settings, k).provider == "mock"]
    if mock:
        print(f"warning: mock providers active for {', '.join(mock)}. Real callers will not be "
              f"understood. Configure .env (see .env.example).", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, ws="websockets-sansio",
                proxy_headers=True, forwarded_allow_ips="*")
    return 0


def _doctor(args) -> int:
    settings = Settings.from_env()
    print(json.dumps(settings.describe(), indent=2, ensure_ascii=False))
    problems = []
    try:
        from .profile import load_profile
        prof = load_profile(settings.profile)
        print(f"profile: {prof.display_name('ar')} ({prof.kind}, dialect {settings.dialect or prof.dialect})")
        if prof.sample:
            problems.append("PROFILE is a fictional sample business: copy examples/my_restaurant.yaml or "
                            "examples/my_clinic.yaml and point PROFILE at it.")
    except Exception as e:  # noqa: BLE001
        problems.append(f"PROFILE could not be loaded: {e}")
    if settings.notify_dry_run:
        problems.append("NOTIFY_DRY_RUN=true: WhatsApp/SMS confirmations are logged, not sent.")
    elif not (settings.wa_token and settings.wa_phone_number_id) and not settings.sms_from:
        problems.append("No WhatsApp (WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID) or SMS (SMS_FROM) configured.")
    if not settings.public_url:
        problems.append("PUBLIC_URL is not set: Twilio signature checks rely on the Host header behind proxies.")
    if not settings.twilio_auth_token:
        problems.append("TWILIO_AUTH_TOKEN is not set: incoming webhooks are NOT authenticated, and "
                        "hang-up/transfer cannot be sent to Twilio.")
    for k in ("llm", "stt", "tts"):
        c = getattr(settings, k)
        if c.provider == "mock":
            problems.append(f"{k.upper()}_PROVIDER=mock: fine for tests, not for real callers.")
        elif not c.api_key and c.provider not in ("ollama", "lmstudio"):
            problems.append(f"{k.upper()}: no API key set for provider {c.provider}.")
    for p in problems:
        print("- " + p)
    print("ok" if not problems else f"{len(problems)} item(s) to review")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="voice-agent-ar",
                                 description="Arabic (Gulf dialect) AI phone agent for UAE restaurants and clinics "
                                             "on Twilio Media Streams (STT -> LLM -> TTS).")
    ap.add_argument("--version", action="version", version=f"voice-agent-ar {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="run the Twilio webhook + Media Streams server and dashboard")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    s.add_argument("--profile", help="bundled profile (restaurant, clinic) or path to a YAML file")
    s.add_argument("--db", help="SQLite path (default DB_PATH or voice_agent.db)")
    s.add_argument("--log-level", default="info")

    m = sub.add_parser("simulate", help="place simulated Twilio calls against a local server")
    m.add_argument("scenario", nargs="*", default=["restaurant_booking"],
                   help="bundled scenario name(s), 'all', or a YAML path")
    m.add_argument("--profile")
    m.add_argument("--db", help="keep the call log in this SQLite file")
    m.add_argument("--speed", type=float, default=25.0, help="virtual clock speed-up (1 = real time)")
    m.add_argument("--json", action="store_true")

    sub.add_parser("doctor", help="print the effective configuration and warnings")
    sy = sub.add_parser("say", help="show how a sentence will be spoken (Arabic numbers, times, phones)")
    sy.add_argument("text")
    sy.add_argument("--dialect", choices=["gulf", "msa"], default="gulf")
    sub.add_parser("scenarios", help="list bundled simulator scenarios")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "simulate":
        return asyncio.run(_simulate(args))
    if args.cmd == "doctor":
        return _doctor(args)
    if args.cmd == "say":
        from .arabic.speak import to_speech
        print(to_speech(args.text, args.dialect))
        return 0
    if args.cmd == "scenarios":
        from .simulator import Scenario
        for name, p in sorted(bundled_scenarios().items()):
            sc = Scenario.load(p)
            print(f"{name:26s} [{sc.profile or 'restaurant':10s}] {sc.description}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
