"""One phone call: Twilio Media Streams <-> VAD <-> STT <-> LLM (+tools) <-> TTS.

Protocol (bidirectional ``<Connect><Stream>``):
  inbound  ``connected`` ``start`` ``media`` ``mark`` ``dtmf`` ``stop``
  outbound ``media`` (agent audio), ``mark`` (playback checkpoints), ``clear`` (flush on barge-in)

Barge-in: agent audio is sent one sentence at a time, each followed by a ``mark``.
Twilio echoes a mark once that audio has actually been played to the caller, so the
session always knows which sentences the caller heard. When the caller talks over
the agent for ``barge_in_ms`` the session cancels the LLM/TTS work, sends ``clear``
(Twilio drops buffered audio) and stores only the heard part of the reply, marked
with an em dash, in both the transcript and the LLM history.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .audio import FRAME_BYTES
from .arabic.normalize import detect_language
from .arabic.speak import to_speech
from .profile import Profile
from .providers.llm import LLMProvider, TextDelta, ToolCalls
from .providers.stt import STTProvider
from .providers.tts import TTSProvider
from .storage import Store
from .tools import ToolContext, ToolRegistry
from .vad import UtteranceSegmenter, VADConfig
from .webhooks import WebhookDispatcher

log = logging.getLogger("voice_agent_ar.call")

SendFn = Callable[[str], Awaitable[None]]

_SENTENCE_END = re.compile(r"([.!?؟。…]+|[\n])(\s+|$)")
_CLAUSE_END = re.compile(r"([,،;؛:])\s+")
MAX_TOOL_ROUNDS = 5
SEND_CHUNK = 1600  # bytes of mu-law per media message (200 ms)


def split_ready_text(buf: str, final: bool = False, min_clause: int = 90) -> tuple[list[str], str]:
    """Split streamed LLM text into speakable pieces; return (pieces, remainder)."""
    out = []
    while True:
        m = _SENTENCE_END.search(buf)
        if m and m.end(1) >= 2:
            out.append(buf[:m.end(1)].strip())
            buf = buf[m.end():]
            continue
        if len(buf) > min_clause:
            c = None
            for c in _CLAUSE_END.finditer(buf):
                if c.end(1) >= min_clause // 2:
                    break
            if c and c.end(1) >= min_clause // 2:
                out.append(buf[:c.end(1)].strip())
                buf = buf[c.end():]
                continue
        break
    if final and buf.strip():
        out.append(buf.strip())
        buf = ""
    return [p for p in out if p], buf


@dataclass
class Reply:
    """The agent's spoken output for one caller turn (may span several LLM rounds)."""
    gen: int
    started: float
    sentences: list[tuple[str, str]] = field(default_factory=list)  # (mark name, text)
    heard: set[str] = field(default_factory=set)                    # mark names played
    round_text: list[str] = field(default_factory=list)
    round_start: int = 0           # index into sentences where the current LLM round began
    msg: dict | None = None        # assistant history message of the current round
    done_rounds: list[tuple[dict, int | None, int, int]] = field(default_factory=list)
    first_audio_at: float | None = None
    tools_task: asyncio.Task | None = None


class CallSession:
    def __init__(self, *, send: SendFn, store: Store, profile: Profile, llm: LLMProvider,
                 stt: STTProvider, tts: TTSProvider, tools: ToolRegistry,
                 webhooks: WebhookDispatcher, vad: VADConfig | None = None,
                 barge_in_ms: int = 240, summary_enabled: bool = True,
                 max_call_seconds: int = 900, stream_token_check: Callable[[str, dict], bool] | None = None,
                 on_control: Callable[[str, dict], None] | None = None,
                 close: Callable[[], Awaitable[None]] | None = None, notifier=None):
        self._send = send
        self._close = close
        self.store = store
        self.profile = profile
        self.llm, self.stt, self.tts = llm, stt, tts
        self.tools = tools
        self.webhooks = webhooks
        self.segmenter = UtteranceSegmenter(vad or VADConfig())
        self.barge_in_ms = barge_in_ms
        self.summary_enabled = summary_enabled
        self.max_call_seconds = max_call_seconds
        self.stream_token_check = stream_token_check
        self.on_control = on_control
        self.notifier = notifier

        self.call_id: str | None = None
        self.call_sid = ""
        self.stream_sid = ""
        self.caller = ""
        self.language = profile.default_language
        self.messages: list[dict] = []
        self.control: dict = {}
        self.started_at = time.time()
        self.closed = False
        self.finalized = False

        self._gen = 0
        self._mark_seq = 0
        self._pending_marks: dict[str, str] = {}   # mark -> text, in send order
        self._reply: Reply | None = None
        self._respond_task: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._utter_task: asyncio.Task | None = None
        self._utterances: asyncio.Queue = asyncio.Queue()
        self._marks_drained = asyncio.Event()
        self._marks_drained.set()
        self._barge_armed = True
        self.metrics = {"turns": 0, "barge_ins": 0, "first_audio_ms": [], "llm_ms": []}

    # ------------------------------------------------------------------ io
    async def send_json(self, obj: dict) -> None:
        if self.closed:
            return
        try:
            await self._send(json.dumps(obj))
        except Exception:  # socket gone
            self.closed = True

    @property
    def speaking(self) -> bool:
        return bool(self._pending_marks) or (self._respond_task is not None and not self._respond_task.done())

    # ------------------------------------------------------------ dispatch
    async def handle(self, raw: str | dict) -> None:
        msg = json.loads(raw) if isinstance(raw, str) else raw
        ev = msg.get("event")
        if ev == "connected":
            return
        if ev == "start":
            await self._on_start(msg)
        elif ev == "media":
            await self._on_media(msg)
        elif ev == "mark":
            self._on_mark((msg.get("mark") or {}).get("name", ""))
        elif ev == "dtmf":
            digit = (msg.get("dtmf") or {}).get("digit", "")
            if self.call_id:
                self.store.add_event(self.call_id, "dtmf", {"digit": digit})
        elif ev == "stop":
            await self.finalize(reason="caller_hangup" if not self.control else self.control.get("action", ""))

    async def _on_start(self, msg: dict) -> None:
        start = msg.get("start") or {}
        self.stream_sid = msg.get("streamSid") or start.get("streamSid", "")
        self.call_sid = start.get("callSid", "")
        params = start.get("customParameters") or {}
        if self.stream_token_check and not self.stream_token_check(self.call_sid, params):
            log.warning("rejecting stream %s: bad or missing stream token", self.stream_sid)
            self.closed = True
            if self._close:
                await self._close()
            return
        fmt = start.get("mediaFormat") or {}
        if fmt and (fmt.get("encoding") != "audio/x-mulaw" or int(fmt.get("sampleRate", 8000)) != 8000):
            log.warning("unexpected media format %s", fmt)
        self.caller = params.get("from", "")
        self.call_id = self.store.create_call(
            call_sid=self.call_sid, stream_sid=self.stream_sid, from_number=self.caller,
            to_number=params.get("to", ""), direction=params.get("direction", "inbound"),
            language=self.language, simulated=params.get("simulated") == "1",
            providers={"llm": self.llm.name, "stt": self.stt.name, "tts": self.tts.name})
        self.store.add_event(self.call_id, "call_started", {"stream_sid": self.stream_sid,
                                                            "call_sid": self.call_sid})
        self.messages = [{"role": "system", "content": self.profile.system_prompt(self.language, self.caller)}]
        self._utter_task = asyncio.create_task(self._utterance_worker())
        self._watchdog = asyncio.create_task(self._max_duration())
        greeting = self.profile.greeting_for(self.language)
        self._start_reply(self._speak_fixed(greeting))

    async def _on_media(self, msg: dict) -> None:
        if not self.call_id or self.finalized:
            return
        media = msg.get("media") or {}
        if media.get("track", "inbound") not in ("inbound", "inbound_track"):
            return
        data = base64.b64decode(media.get("payload", ""))
        for i in range(0, len(data), FRAME_BYTES):
            frame = data[i:i + FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame = frame + b"\xff" * (FRAME_BYTES - len(frame))
            for ev in self.segmenter.feed(frame):
                if ev.kind == "speech_start":
                    self._barge_armed = True
                elif ev.kind == "utterance_end":
                    self._utterances.put_nowait((ev.audio, time.time()))
            if (self.segmenter.in_speech and self._barge_armed and self.speaking
                    and self.segmenter.voiced_total_ms() >= self.barge_in_ms):
                self._barge_armed = False
                await self.barge_in()

    def _on_mark(self, name: str) -> None:
        if name in self._pending_marks:
            # marks are played in order; anything before this one has played too
            for k in list(self._pending_marks):
                self._pending_marks.pop(k)
                if self._reply and k.startswith(f"g{self._reply.gen}-"):
                    self._reply.heard.add(k)
                if k == name:
                    break
        if not self._pending_marks:
            self._marks_drained.set()

    # ------------------------------------------------------------- replies
    def _start_reply(self, coro) -> None:
        self._gen += 1
        self._reply = Reply(gen=self._gen, started=time.time())
        self._respond_task = asyncio.create_task(coro)

    async def _send_audio(self, text: str) -> None:
        """Synthesize one sentence, stream it to Twilio, then send its mark."""
        reply = self._reply
        spoken = to_speech(text, self.profile.style) if self.language == "ar" else text
        async for chunk in self.tts.synthesize(spoken, self.language):
            for i in range(0, len(chunk), SEND_CHUNK):
                if reply and reply.first_audio_at is None:
                    reply.first_audio_at = time.time()
                await self.send_json({"event": "media", "streamSid": self.stream_sid,
                                      "media": {"payload": base64.b64encode(chunk[i:i + SEND_CHUNK]).decode()}})
        self._mark_seq += 1
        name = f"g{reply.gen if reply else 0}-{self._mark_seq}"
        self._pending_marks[name] = text
        self._marks_drained.clear()
        if reply:
            reply.sentences.append((name, text))
        await self.send_json({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}})

    async def _speak_fixed(self, text: str) -> None:
        reply = self._reply
        msg = {"role": "assistant", "content": text}
        self.messages.append(msg)
        turn_id = self.store.add_turn(self.call_id, "assistant", text, self.language)
        pieces, _ = split_ready_text(text, final=True)
        reply.done_rounds.append((msg, turn_id, 0, len(pieces)))
        reply.msg = msg
        for p in pieces:
            await self._send_audio(p)

    async def barge_in(self) -> None:
        reply = self._reply
        if reply and reply.tools_task and not reply.tools_task.done():
            # let running tools finish so tool results stay paired with their calls
            with contextlib.suppress(Exception):
                await asyncio.shield(reply.tools_task)
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._respond_task
        await self.send_json({"event": "clear", "streamSid": self.stream_sid})
        self._pending_marks.clear()
        self._marks_drained.set()
        self._gen += 1  # late marks echoed after clear belong to the old generation
        self.metrics["barge_ins"] += 1
        heard_all: list[str] = []
        unheard_all: list[str] = []
        if reply:
            def heard_in(start: int, end: int | None) -> tuple[list[str], bool]:
                sent = reply.sentences[start:end]
                unheard_all.extend(t for n, t in sent if n not in reply.heard)
                got = [t for n, t in sent if n in reply.heard]
                expected = (end - start) if end is not None else len(sent)
                return got, len(got) == expected

            # rounds already committed to history (e.g. "Let me check." before a tool call)
            for msg, turn_id, start, end in reply.done_rounds:
                got, complete = heard_in(start, end)
                heard_all += got
                if not complete and msg.get("content"):
                    msg["content"] = (" ".join(got) + " —").strip()
                    if turn_id:
                        self.store.mark_interrupted(turn_id, msg["content"])
            # the round being streamed right now
            if reply.msg is None and (reply.round_text or len(reply.sentences) > reply.round_start):
                got, _ = heard_in(reply.round_start, None)
                heard_all += got
                cut = (" ".join(got) + " —").strip()
                self.messages.append({"role": "assistant", "content": cut})
                self.store.add_turn(self.call_id, "assistant", cut, self.language, interrupted=True)
        self.store.add_event(self.call_id, "barge_in", {
            "heard": " ".join(heard_all),
            "unheard": " ".join(unheard_all),
            "heard_sentences": len(reply.heard) if reply else 0,
            "sent_sentences": len(reply.sentences) if reply else 0})
        self._reply = None

    # ---------------------------------------------------------- user turns
    async def _utterance_worker(self) -> None:
        while True:
            audio, ended_at = await self._utterances.get()
            try:
                await self._process_utterance(audio, ended_at)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the call alive on provider errors
                log.exception("turn failed")
                if self.call_id:
                    self.store.add_event(self.call_id, "error", {"stage": "turn", "error": repr(e)[:300]})

    async def _process_utterance(self, audio: bytes, ended_at: float) -> None:
        t0 = time.time()
        tr = await self.stt.transcribe(audio, None, self.profile.stt_hints)
        stt_ms = (time.time() - t0) * 1000
        text = (tr.text or "").strip()
        if not text:
            self.store.add_event(self.call_id, "noise", {"ms": len(audio) // 8})
            return
        lang = tr.language or detect_language(text, self.language)
        if lang in self.profile.languages and lang != self.language:
            self.store.add_event(self.call_id, "language_switch", {"from": self.language, "to": lang})
            self.language = lang
            self.store.update_call(self.call_id, language=lang)
        # A new caller turn supersedes anything still being prepared
        if self._respond_task and not self._respond_task.done():
            if self._reply and (self._reply.sentences or self._reply.round_text):
                await self.barge_in()
            else:
                self._respond_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._respond_task
        self.messages[0] = {"role": "system", "content": self.profile.system_prompt(self.language, self.caller)}
        self.messages.append({"role": "user", "content": text})
        self.store.add_turn(self.call_id, "user", text, self.language, latency_ms=stt_ms)
        self.metrics["turns"] += 1
        self._start_reply(self._respond(ended_at, stt_ms))

    async def _respond(self, ended_at: float, stt_ms: float) -> None:
        try:
            await self._respond_inner(ended_at, stt_ms)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("LLM/TTS turn failed")
            self.store.add_event(self.call_id, "error", {"stage": "respond", "error": repr(e)[:300]})
            # The caller must never hear dead air: apologise with a fixed line.
            sorry = {"en": "Sorry, I'm having a technical problem. Could you say that again?",
                     "ar": "سامحني، صار عندي خلل بسيط. ممكن تعيد كلامك؟"}
            with contextlib.suppress(Exception):
                await self._speak_fixed(sorry.get(self.language, sorry["en"]))

    async def _respond_inner(self, ended_at: float, stt_ms: float) -> None:
        reply = self._reply
        ctx = ToolContext(call_id=self.call_id, caller_number=self.caller, language=self.language,
                          profile=self.profile, store=self.store, webhooks=self.webhooks,
                          notifier=self.notifier, control=self.control)
        llm_first = None
        for _round in range(MAX_TOOL_ROUNDS):
            buf = ""
            reply.round_text = []
            reply.round_start = len(reply.sentences)
            reply.msg = None
            calls = []
            t_llm = time.time()
            async for ev in self.llm.stream(self.messages, self.tools.schemas()):
                if isinstance(ev, TextDelta):
                    if llm_first is None:
                        llm_first = (time.time() - t_llm) * 1000
                    buf += ev.text
                    pieces, buf = split_ready_text(buf)
                    for p in pieces:
                        reply.round_text.append(p)
                        await self._send_audio(p)
                elif isinstance(ev, ToolCalls):
                    calls = ev.calls
            pieces, _ = split_ready_text(buf, final=True)
            for p in pieces:
                reply.round_text.append(p)
                await self._send_audio(p)
            text = " ".join(reply.round_text)
            msg: dict = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls]
            if text or calls:
                reply.msg = msg
                self.messages.append(msg)
                turn_id = self.store.add_turn(self.call_id, "assistant", text, self.language) if text else None
                reply.done_rounds.append((msg, turn_id, reply.round_start, len(reply.sentences)))
            if not calls:
                break
            # tools run to completion even if the caller barges in meanwhile
            reply.tools_task = asyncio.create_task(self._run_tools(calls, ctx))
            await asyncio.shield(reply.tools_task)
            if ctx.control.get("action"):
                # one more LLM round only if it has not said anything yet (e.g. transfer note)
                if text:
                    break
        first_audio_ms = (reply.first_audio_at - ended_at) * 1000 if reply.first_audio_at else None
        if first_audio_ms is not None:
            self.metrics["first_audio_ms"].append(first_audio_ms)
        if llm_first is not None:
            self.metrics["llm_ms"].append(llm_first)
        self.store.add_event(self.call_id, "turn_metrics", {
            "stt_ms": round(stt_ms, 1), "llm_first_token_ms": round(llm_first, 1) if llm_first else None,
            "first_audio_ms": round(first_audio_ms, 1) if first_audio_ms is not None else None})
        if ctx.control.get("action"):
            asyncio.create_task(self._apply_control())

    async def _run_tools(self, calls, ctx: ToolContext) -> None:
        for c in calls:
            t = time.time()
            result = await self.tools.call(c.name, dict(c.arguments), ctx)
            self.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                  "content": json.dumps(result, ensure_ascii=False, default=str)})
            self.store.add_event(self.call_id, "tool_call", {
                "name": c.name, "arguments": c.arguments, "result": result,
                "ms": round((time.time() - t) * 1000, 1)})

    async def _apply_control(self) -> None:
        """Wait until the caller has heard the last sentence, then hang up / transfer."""
        playback_s = sum(len(t) for t in self._pending_marks.values()) * 0.12 + 5
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._marks_drained.wait(), timeout=playback_s)
        action = self.control.get("action")
        if self.on_control:
            self.on_control(self.call_sid, dict(self.control))
        self.store.add_event(self.call_id, action, dict(self.control))
        await self.finalize(reason=action)
        if self._close:
            await self._close()

    async def _max_duration(self) -> None:
        await asyncio.sleep(self.max_call_seconds)
        self.control.setdefault("action", "hangup")
        self.control.setdefault("reason", "max_call_seconds reached")
        await self._apply_control()

    # ------------------------------------------------------------ teardown
    async def finalize(self, reason: str = "") -> None:
        if self.finalized or not self.call_id:
            return
        self.finalized = True
        current = asyncio.current_task()
        for t in (self._respond_task, self._utter_task, self._watchdog):
            if t and not t.done() and t is not current:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        outcome = {"transfer": "transferred", "hangup": "completed"}.get(self.control.get("action", ""),
                                                                         "caller_hangup")
        bookings = self.store.list_bookings(self.call_id)
        if bookings and outcome != "transferred":
            outcome = "booked"
        summary = ""
        if self.summary_enabled:
            try:
                summary = await asyncio.wait_for(self._summarize(), timeout=20)
            except Exception as e:
                self.store.add_event(self.call_id, "error", {"stage": "summary", "error": repr(e)[:300]})

        def avg(xs):
            return round(sum(xs) / len(xs), 1) if xs else None
        self.store.update_call(
            self.call_id, status="completed", ended_at=time.time(), outcome=outcome, summary=summary,
            turns=self.metrics["turns"], barge_ins=self.metrics["barge_ins"],
            first_audio_ms_avg=avg(self.metrics["first_audio_ms"]), llm_ms_avg=avg(self.metrics["llm_ms"]),
            language=self.language)
        self.store.add_event(self.call_id, "call_ended", {"reason": reason, "outcome": outcome})
        call = self.store.get_call(self.call_id)
        self.webhooks.emit("call.completed", {
            "call_id": self.call_id, "call_sid": self.call_sid, "from": self.caller,
            "language": self.language, "outcome": outcome, "summary": summary,
            "duration_s": round(call["ended_at"] - call["started_at"], 1),
            "bookings": bookings, "business": self.profile.name,
            "notifications": [{k: n[k] for k in ("channel", "to_number", "status")}
                              for n in self.store.notifications(call_id=self.call_id)]})

    async def _summarize(self) -> str:
        lines = []
        tool_names = {}
        for m in self.messages[1:]:
            if m["role"] == "user":
                lines.append(f"CALLER: {m['content']}")
            elif m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    tool_names[c["id"]] = (c["name"], c["arguments"])
                if m.get("content"):
                    tag = " [interrupted]" if m["content"].endswith("—") else ""
                    lines.append(f"AGENT: {m['content']}{tag}")
            elif m["role"] == "tool":
                name, args = tool_names.get(m["tool_call_id"], (m.get("name", "?"), {}))
                lines.append(f"TOOL {name} {json.dumps(args, ensure_ascii=False)} -> {m['content']}")
        if not lines:
            return ""
        prompt = [
            {"role": "system", "content": "SUMMARIZE the phone call below for the business owner in Arabic (MSA), "
                                          "in at most 3 sentences: who called, what they wanted, what was booked "
                                          "(with booking code), whether a WhatsApp/SMS confirmation was sent, and "
                                          "any follow-up needed."},
            {"role": "user", "content": "\n".join(lines)},
        ]
        return await self.llm.complete(prompt)
