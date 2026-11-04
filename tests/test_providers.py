"""Provider layer: request shapes for OpenAI-compatible, Anthropic, Deepgram, ElevenLabs (mocked HTTP)."""

import json

import httpx
import pytest

from voice_agent_ar.config import Settings
from voice_agent_ar.providers.llm import AnthropicLLM, LLMError, OpenAICompatibleLLM, TextDelta, ToolCalls, make_llm
from voice_agent_ar.providers.stt import DeepgramSTT, OpenAICompatibleSTT
from voice_agent_ar.providers.tts import ElevenLabsTTS, make_tts


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_presets(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    s = Settings.from_env()
    assert s.llm.base_url == "https://api.groq.com/openai/v1" and s.llm.api_key == "g"
    assert s.stt.extra["language"] == "ar" and s.profile == "restaurant"
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        Settings.from_env()


async def test_openai_compatible_streams_text_and_tool_calls(monkeypatch):
    sse = "\n\n".join("data: " + json.dumps(c) for c in [
        {"choices": [{"delta": {"content": "أبشر، "}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t1", "function": {
            "name": "check_availability", "arguments": '{"date": "2026-11-05",'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "party_size": 5}'}}]}}]},
    ]) + "\n\ndata: [DONE]\n\n"
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["url"] = str(req.url)
        return httpx.Response(200, text=sse)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    llm = OpenAICompatibleLLM(Settings.from_env().llm, client(handler))
    evs = [e async for e in llm.stream([{"role": "user", "content": "مرحبا"}], [{"name": "x", "parameters": {}}])]
    assert isinstance(evs[0], TextDelta) and evs[0].text == "أبشر، "
    assert isinstance(evs[-1], ToolCalls) and evs[-1].calls[0].arguments == {"date": "2026-11-05", "party_size": 5}
    assert seen["url"] == "http://localhost:11434/v1/chat/completions" and seen["body"]["stream"] is True


async def test_anthropic_wire_format(monkeypatch):
    sse = "".join(f"event: {t}\ndata: {json.dumps(d)}\n\n" for t, d in [
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "هلا"}}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "tool_use", "id": "tu1", "name": "end_call"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
    ])
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["headers"] = req.headers
        return httpx.Response(200, text=sse)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    llm = AnthropicLLM(Settings.from_env().llm, client(handler))
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "name": "t", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "x", "content": "{}"}]
    evs = [e async for e in llm.stream(msgs, [])]
    assert evs[0].text == "هلا" and evs[-1].calls[0].name == "end_call"
    assert seen["body"]["system"] == "S" and seen["headers"]["x-api-key"] == "a"
    assert seen["body"]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_missing_keys_fail_fast(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(LLMError):
        make_llm(Settings.from_env().llm)


async def test_stt_sends_arabic_and_hints(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = req.content
        return httpx.Response(200, json={"text": "ابي احجز", "language": "arabic"})
    monkeypatch.setenv("STT_PROVIDER", "openai")
    monkeypatch.setenv("STT_API_KEY", "k")
    stt = OpenAICompatibleSTT(Settings.from_env().stt, client(handler))
    tr = await stt.transcribe(b"\xff" * 800, None, ["باجر", "الفنار"])
    assert tr.text == "ابي احجز" and tr.language == "ar"
    assert "باجر".encode() in seen["body"] and b'name="language"\r\n\r\nar' in seen["body"]


async def test_deepgram_mulaw_passthrough(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = req.url
        seen["n"] = len(req.content)
        return httpx.Response(200, json={"results": {"channels": [{"alternatives": [
            {"transcript": "مرحبا", "confidence": 0.9}]}]}})
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_API_KEY", "k")
    stt = DeepgramSTT(Settings.from_env().stt, client(handler))
    tr = await stt.transcribe(b"\xff" * 1600, None, ["باجر"])
    assert tr.text == "مرحبا" and seen["n"] == 1600
    assert seen["url"].params["encoding"] == "mulaw" and seen["url"].params["language"] == "ar"
    assert "keyterm" not in seen["url"].params   # opt-in via STT_SEND_HINTS for Deepgram


async def test_elevenlabs_requests_ulaw_8000(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = req.url
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=b"\x7f" * 320)
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("TTS_API_KEY", "k")
    monkeypatch.setenv("TTS_VOICE_AR", "voice123")
    tts = ElevenLabsTTS(Settings.from_env().tts, client(handler))
    audio = b"".join([c async for c in tts.synthesize("هلا والله", "ar")])
    assert len(audio) == 320 and seen["url"].params["output_format"] == "ulaw_8000"
    assert "/voice123/" in str(seen["url"]) and seen["body"]["language_code"] == "ar"
    monkeypatch.setenv("TTS_PROVIDER", "openai")
    monkeypatch.delenv("TTS_API_KEY")
    with pytest.raises(Exception):
        make_tts(Settings.from_env().tts)
