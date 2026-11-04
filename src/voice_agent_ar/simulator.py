"""Call simulator: behaves like Twilio Media Streams on the other end of the WebSocket.

It does what Twilio does on a real call:
  1. POSTs the incoming-call webhook (``/twilio/voice``) and parses the returned TwiML,
  2. opens the ``<Stream>`` WebSocket and sends ``connected`` / ``start`` with the
     ``<Parameter>`` values from the TwiML,
  3. streams the caller's audio as 20 ms base64 mu-law ``media`` frames (silence
     between turns, like a real phone line),
  4. "plays" the agent's audio on a virtual clock and echoes each ``mark`` once the
     audio before it has played; on ``clear`` it drops buffered audio and returns
     the outstanding marks, as Twilio does,
  5. sends ``stop`` when the script hangs up.

Caller speech is produced with the FSK test modem (``fsk.py``), which the mock STT
decodes; agent audio from the mock TTS is decoded the same way so the transcript of
what the caller actually *heard* (including where a barge-in cut it off) is exact.
With real TTS the simulator still works, it just cannot decode the agent's words.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets
import yaml

from . import fsk
from .audio import FRAME_BYTES, FRAME_MS, silence
from .twilio import compute_signature


@dataclass
class Step:
    say: str
    barge_in_after_ms: int | None = None   # start talking this long into the agent's reply
    pause_ms: int = 300                    # silence before speaking (after the agent finished)


@dataclass
class Scenario:
    name: str
    steps: list[Step]
    caller: str = "+971500000000"
    to: str = "+97140000000"
    hangup: str = "agent"                  # "agent": wait for the agent to end the call; "caller": send stop
    description: str = ""
    profile: str = ""                      # bundled profile this scenario is written for
    setup_bookings: list[dict] = field(default_factory=list)   # pre-existing bookings ("date": "+1" = tomorrow)

    def seed(self, store, profile) -> int:
        """Insert ``setup_bookings`` so the scenario meets a realistic, partly booked day."""
        from datetime import date as _date, timedelta
        n = 0
        for i, b in enumerate(self.setup_bookings):
            d = str(b.get("date", "+0"))
            day = profile.today() + timedelta(days=int(d)) if d.startswith(("+", "-")) else _date.fromisoformat(d)
            store.create_booking(
                call_id="seed", code=f"S{i:04d}{uuid.uuid4().hex[:4]}", kind=profile.kind,
                service_id=b.get("service", "table"), service_name=b.get("service", "table"),
                doctor_id=b.get("doctor"), party_size=int(b.get("party_size", 1)), area=b.get("area"),
                date=day.isoformat(), time=b["time"],
                duration_min=int(b.get("duration_min", profile.dining_minutes if profile.kind == "restaurant"
                                       else 30)),
                customer_name=b.get("name", "حجز تجريبي"), phone="", notes="scenario seed", language="ar")
            n += 1
        return n

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        steps = []
        for s in d.get("steps", []):
            if isinstance(s, str):
                steps.append(Step(say=s))
            else:
                steps.append(Step(say=s["say"], barge_in_after_ms=s.get("barge_in_after_ms"),
                                  pause_ms=int(s.get("pause_ms", 300))))
        return cls(name=d.get("name", "scenario"), steps=steps, caller=str(d.get("caller", "+971500000000")),
                   to=str(d.get("to", "+97140000000")), hangup=d.get("hangup", "agent"),
                   description=d.get("description", ""), profile=str(d.get("profile", "")),
                   setup_bookings=list(d.get("setup_bookings") or []))

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class HeardSegment:
    mark: str
    text: str | None          # decoded agent sentence (mock TTS only)
    audio_ms: int
    played_ms: int
    interrupted: bool = False


@dataclass
class SimResult:
    call_sid: str
    transcript: list[dict] = field(default_factory=list)  # {"role", "text", "interrupted"}
    agent_segments: list[HeardSegment] = field(default_factory=list)
    clears: int = 0
    marks_echoed: int = 0
    media_in: int = 0
    media_out: int = 0
    closed_by_agent: bool = False
    virtual_ms: int = 0

    def agent_said(self) -> list[str]:
        return [t["text"] for t in self.transcript if t["role"] == "agent"]

    def format(self) -> str:
        lines = []
        for t in self.transcript:
            who = "المتصل " if t["role"] == "caller" else "المساعد"
            cut = "  [قاطعه المتصل]" if t.get("interrupted") else ""
            lines.append(f"{who}: {t['text']}{cut}")
        lines.append(f"-- {self.virtual_ms / 1000:.1f}s of call audio, {self.clears} barge-in clear(s), "
                     f"{'agent' if self.closed_by_agent else 'caller'} ended the call")
        return "\n".join(lines)


class _Player:
    """Virtual playback buffer for the agent's audio."""

    def __init__(self):
        self.buf = bytearray()
        self.items: list[tuple[str, int]] = []   # (mark name, byte offset in stream)
        self.written = 0
        self.played = 0
        self.seg_audio = bytearray()             # audio of the sentence currently being received
        self.seg_start_played = 0
        self.last_audio_at = 0.0


class TwilioSimulator:
    def __init__(self, base_url: str, *, speed: float = 25.0, auth_token: str = "",
                 decode_agent: bool = True, idle_ms: int = 1200, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.speed = speed
        self.auth_token = auth_token
        self.decode_agent = decode_agent
        self.idle_ms = idle_ms
        self.timeout_s = timeout_s

    async def _twiml(self, call_sid: str, sc: Scenario) -> tuple[str, dict]:
        url = f"{self.base_url}/twilio/voice"
        form = {"CallSid": call_sid, "AccountSid": "AC" + "0" * 32, "From": sc.caller, "To": sc.to,
                "Direction": "inbound", "CallStatus": "ringing"}
        headers = {}
        if self.auth_token:
            headers["X-Twilio-Signature"] = compute_signature(self.auth_token, url, form)
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, data=form, headers=headers)
            r.raise_for_status()
        xml = r.text
        m = re.search(r'<Stream url="([^"]+)"', xml)
        if not m:
            raise RuntimeError(f"no <Stream> in TwiML: {xml}")
        params = {html.unescape(k): html.unescape(v)
                  for k, v in re.findall(r'<Parameter name="([^"]+)" value="([^"]*)"', xml)}
        ws_url = html.unescape(m.group(1))
        return ws_url, params

    async def run(self, sc: Scenario) -> SimResult:
        call_sid = "CA" + uuid.uuid4().hex
        stream_sid = "MZ" + uuid.uuid4().hex
        ws_url, params = await self._twiml(call_sid, sc)
        params["simulated"] = "1"
        res = SimResult(call_sid=call_sid)
        player = _Player()
        frame_s = FRAME_MS / 1000 / self.speed
        vclock = {"ms": 0}

        closed = asyncio.Event()
        async with websockets.connect(ws_url, max_size=2**22) as ws:
            async def send(obj):
                try:
                    await ws.send(json.dumps(obj))
                except websockets.ConnectionClosed:
                    closed.set()

            await send({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            await send({"event": "start", "sequenceNumber": "1", "streamSid": stream_sid, "start": {
                "streamSid": stream_sid, "accountSid": "AC" + "0" * 32, "callSid": call_sid,
                "tracks": ["inbound"], "customParameters": params,
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}})

            seq = {"n": 2, "chunk": 0}

            def close_segment(name: str, interrupted: bool):
                audio = bytes(player.seg_audio)
                text = fsk.decode(audio) if (self.decode_agent and audio) else None
                audio_ms = len(audio) // 8
                played = min(audio_ms, max(0, (player.played - player.seg_start_played) // 8))
                res.agent_segments.append(HeardSegment(name, text, audio_ms, played, interrupted))
                player.seg_audio = bytearray()

            async def receiver():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        ev = msg.get("event")
                        if ev == "media":
                            data = base64.b64decode(msg["media"]["payload"])
                            if not player.seg_audio:
                                player.seg_start_played = player.written
                            player.buf.extend(data)
                            player.seg_audio.extend(data)
                            player.written += len(data)
                            player.last_audio_at = time.monotonic()
                            res.media_out += 1
                        elif ev == "mark":
                            name = msg["mark"]["name"]
                            player.items.append((name, player.written))
                            close_segment(name, False)
                        elif ev == "clear":
                            res.clears += 1
                            # audio received since the last mark is also dropped
                            if player.seg_audio:
                                close_segment("(unmarked)", True)
                            dropped = [n for n, _ in player.items]
                            for seg in res.agent_segments:
                                if seg.mark in dropped:
                                    seg.interrupted = True
                            player.buf.clear()
                            player.played = player.written
                            player.items.clear()
                            for n in dropped:   # Twilio returns outstanding marks on clear
                                await send({"event": "mark", "streamSid": stream_sid,
                                            "sequenceNumber": str(seq["n"]), "mark": {"name": n}})
                                seq["n"] += 1
                                res.marks_echoed += 1
                except websockets.ConnectionClosed:
                    pass
                closed.set()

            recv_task = asyncio.create_task(receiver())

            async def tick(caller_frame: bytes):
                """Advance the virtual clock by one frame: send caller audio, play agent audio."""
                await send({"event": "media", "sequenceNumber": str(seq["n"]), "streamSid": stream_sid,
                            "media": {"track": "inbound", "chunk": str(seq["chunk"]),
                                      "timestamp": str(vclock["ms"]),
                                      "payload": base64.b64encode(caller_frame).decode()}})
                seq["n"] += 1
                seq["chunk"] += 1
                res.media_in += 1
                vclock["ms"] += FRAME_MS
                n = min(FRAME_BYTES, len(player.buf))
                if n:
                    del player.buf[:n]
                    player.played += n
                while player.items and player.items[0][1] <= player.played:
                    name, _ = player.items.pop(0)
                    await send({"event": "mark", "streamSid": stream_sid, "sequenceNumber": str(seq["n"]),
                                "mark": {"name": name}})
                    seq["n"] += 1
                    res.marks_echoed += 1
                await asyncio.sleep(frame_s)

            sil = silence(FRAME_MS)
            deadline = time.monotonic() + self.timeout_s

            async def wait_agent_idle():
                """Keep the line open (silence) until the agent has finished talking."""
                idle = 0
                started = time.monotonic()
                while not closed.is_set() and time.monotonic() < deadline:
                    await tick(sil)
                    busy = player.buf or player.items or (time.monotonic() - player.last_audio_at) < 0.05
                    idle = 0 if busy else idle + FRAME_MS
                    if idle >= self.idle_ms and time.monotonic() - started > 0.15:
                        return

            async def wait_agent_speaking(after_ms: int):
                """Wait until the agent's next reply has played for ``after_ms``."""
                start_played = player.played
                while not closed.is_set() and time.monotonic() < deadline:
                    await tick(sil)
                    if (player.played - start_played) // 8 >= after_ms:
                        return

            def agent_lines_since(idx: int) -> int:
                for seg in res.agent_segments[idx:]:
                    if seg.text is None and not seg.audio_ms:
                        continue
                    text = seg.text if seg.text is not None else f"<{seg.audio_ms} ms of audio>"
                    if seg.interrupted:
                        text = _heard_prefix(seg)
                    if not text:
                        continue
                    if res.transcript and res.transcript[-1]["role"] == "agent" and \
                            not res.transcript[-1].get("interrupted"):
                        res.transcript[-1]["text"] += " " + text
                        res.transcript[-1]["interrupted"] = seg.interrupted
                    else:
                        res.transcript.append({"role": "agent", "text": text, "interrupted": seg.interrupted})
                return len(res.agent_segments)

            seg_idx = 0
            steps = sc.steps

            async def wait_for_turn(step: Step | None):
                """After the agent starts replying: either let it finish, or cut in mid-reply."""
                if step is not None and step.barge_in_after_ms is not None:
                    await wait_agent_speaking(step.barge_in_after_ms)
                else:
                    await wait_agent_idle()

            await wait_for_turn(steps[0] if steps else None)   # greeting
            for i, step in enumerate(steps):
                if closed.is_set():
                    break
                if step.barge_in_after_ms is None:
                    for _ in range(step.pause_ms // FRAME_MS):
                        await tick(sil)
                    seg_idx = agent_lines_since(seg_idx)
                for frame in _frames(fsk.encode(step.say)):
                    await tick(frame)
                if step.barge_in_after_ms is not None:
                    # a few frames for the clear to arrive, then record the cut-off reply first
                    for _ in range(10):
                        await tick(sil)
                    seg_idx = agent_lines_since(seg_idx)
                res.transcript.append({"role": "caller", "text": step.say})
                await wait_for_turn(steps[i + 1] if i + 1 < len(steps) else None)
                if i + 1 >= len(steps) or steps[i + 1].barge_in_after_ms is None:
                    seg_idx = agent_lines_since(seg_idx)

            if sc.hangup == "agent":
                # give the agent time to say goodbye and close the stream
                t_end = time.monotonic() + 5
                while not closed.is_set() and time.monotonic() < min(t_end, deadline):
                    await tick(sil)
                res.closed_by_agent = closed.is_set()
            if not closed.is_set():
                await send({"event": "stop", "sequenceNumber": str(seq["n"]), "streamSid": stream_sid,
                            "stop": {"accountSid": "AC" + "0" * 32, "callSid": call_sid}})
                await asyncio.sleep(0.2)
            agent_lines_since(seg_idx)
            await ws.close()
            recv_task.cancel()
        res.virtual_ms = vclock["ms"]
        return res


def _frames(data: bytes) -> list[bytes]:
    out = []
    for i in range(0, len(data), FRAME_BYTES):
        chunk = data[i:i + FRAME_BYTES]
        out.append(chunk + b"\xff" * (FRAME_BYTES - len(chunk)))
    return out


def _heard_prefix(seg: HeardSegment) -> str:
    """For a cut-off sentence, show roughly how much the caller heard."""
    if not seg.text:
        return ""
    if seg.played_ms <= 0:
        return ""
    frac = min(1.0, seg.played_ms / max(1, seg.audio_ms))
    words = seg.text.split()
    k = max(1, round(len(words) * frac))
    return " ".join(words[:k]) + (" —" if k < len(words) else "")
