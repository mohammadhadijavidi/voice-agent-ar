"""Text-to-speech providers. Each yields 8 kHz mu-law chunks ready for Twilio."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from ..audio import SAMPLE_RATE, pcm16_bytes_to_samples, pcm16_to_ulaw, resample
from ..config import ProviderConfig


class TTSError(RuntimeError):
    pass


class TTSProvider:
    name = "base"
    is_mock = False

    def voice_for(self, language: str) -> str:
        voices = getattr(self, "cfg", None) and self.cfg.extra.get("voices") or {}
        return voices.get(language) or voices.get("en") or ""

    async def synthesize(self, text: str, language: str = "en") -> AsyncIterator[bytes]:
        raise NotImplementedError
        yield  # pragma: no cover


class OpenAITTS(TTSProvider):
    """OpenAI /audio/speech with response_format=pcm (24 kHz PCM16), resampled to 8 kHz mu-law."""

    SOURCE_RATE = 24000

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.name = cfg.provider
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def synthesize(self, text, language="en"):
        body = {"model": self.cfg.model, "voice": self.voice_for(language) or "alloy",
                "input": text, "response_format": "pcm"}
        if self.cfg.extra.get("instructions") and not self.cfg.model.startswith("tts-1"):
            body["instructions"] = self.cfg.extra["instructions"]
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        carry = b""
        async with self.client.stream("POST", f"{self.cfg.base_url}/audio/speech", json=body,
                                      headers=headers) as resp:
            if resp.status_code >= 400:
                raise TTSError(f"{self.name} TTS HTTP {resp.status_code}: {(await resp.aread())[:300]!r}")
            async for chunk in resp.aiter_bytes():
                data = carry + chunk
                usable = len(data) - (len(data) % 6)  # whole samples, multiple of the 3:1 ratio
                carry = data[usable:]
                if usable:
                    samples = pcm16_bytes_to_samples(data[:usable])
                    yield pcm16_to_ulaw(resample(samples, self.SOURCE_RATE, SAMPLE_RATE))
        if len(carry) >= 6:
            samples = pcm16_bytes_to_samples(carry[: len(carry) - len(carry) % 6])
            yield pcm16_to_ulaw(resample(samples, self.SOURCE_RATE, SAMPLE_RATE))


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs streaming TTS with output_format=ulaw_8000 (Twilio's native format)."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.name = "elevenlabs"
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def synthesize(self, text, language="en"):
        voice = self.voice_for(language)
        if not voice:
            raise TTSError("Set TTS_VOICE (an ElevenLabs voice_id) or TTS_VOICE_EN/AR/FA")
        body = {"text": text, "model_id": self.cfg.model}
        if language and self.cfg.model.startswith(("eleven_flash_v2_5", "eleven_turbo_v2_5")):
            body["language_code"] = language
        async with self.client.stream(
            "POST", f"{self.cfg.base_url}/text-to-speech/{voice}/stream",
            params={"output_format": "ulaw_8000"}, json=body,
            headers={"xi-api-key": self.cfg.api_key, "Content-Type": "application/json"},
        ) as resp:
            if resp.status_code >= 400:
                raise TTSError(f"elevenlabs HTTP {resp.status_code}: {(await resp.aread())[:300]!r}")
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


def make_tts(cfg: ProviderConfig, client: httpx.AsyncClient | None = None) -> TTSProvider:
    if cfg.provider == "mock":
        from .mock import MockTTS
        return MockTTS()
    if not cfg.api_key:
        raise TTSError(f"An API key is required for TTS_PROVIDER={cfg.provider} (set TTS_API_KEY)")
    if cfg.provider == "elevenlabs":
        return ElevenLabsTTS(cfg, client)
    return OpenAITTS(cfg, client)
