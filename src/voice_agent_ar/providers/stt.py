"""Speech-to-text providers.

The agent segments the caller's audio into utterances itself (VAD + endpointing),
so STT providers only need to transcribe one finished utterance. That keeps every
provider pluggable through the same small interface and works with any batch
transcription API. A streaming provider can implement the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..audio import pcm16_to_wav_bytes, ulaw_to_pcm16
from ..config import ProviderConfig

_LANG_NAMES = {"english": "en", "arabic": "ar"}


def normalize_lang(code: str | None) -> str | None:
    if not code:
        return None
    c = code.strip().lower()
    c = _LANG_NAMES.get(c, c)
    c = c.split("-")[0]
    return c if c in ("en", "ar") else None


@dataclass
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None


class STTError(RuntimeError):
    pass


class STTProvider:
    name = "base"
    is_mock = False

    async def transcribe(self, ulaw: bytes, language: str | None = None,
                         hints: list[str] | None = None) -> Transcript:
        """``hints``: business words (names, menu items) to bias recognition where the API supports it."""
        raise NotImplementedError


class OpenAICompatibleSTT(STTProvider):
    """OpenAI /audio/transcriptions (whisper-1, gpt-4o-transcribe) and Groq Whisper."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.name = cfg.provider
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def transcribe(self, ulaw, language=None, hints=None):
        wav = pcm16_to_wav_bytes(ulaw_to_pcm16(ulaw))
        fmt = self.cfg.extra.get("response_format") or (
            "json" if self.cfg.model.startswith("gpt-4o") else "verbose_json")
        data = {"model": self.cfg.model, "response_format": fmt}
        lang = self.cfg.extra.get("language") or language
        if lang:
            data["language"] = lang
        if hints and self.cfg.extra.get("send_hints", True):
            # Whisper uses the prompt as preceding context: Gulf words and business names in Arabic script
            data["prompt"] = "مكالمة حجز باللهجة الإماراتية: " + "، ".join(hints[:40])
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        resp = await self.client.post(f"{self.cfg.base_url}/audio/transcriptions", data=data,
                                      files={"file": ("utterance.wav", wav, "audio/wav")}, headers=headers)
        if resp.status_code >= 400:
            raise STTError(f"{self.name} STT HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        return Transcript(text=(body.get("text") or "").strip(), language=normalize_lang(body.get("language")))


class DeepgramSTT(STTProvider):
    """Deepgram pre-recorded API with raw mu-law input (no transcoding needed)."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.name = "deepgram"
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def transcribe(self, ulaw, language=None, hints=None):
        params = {"model": self.cfg.model, "encoding": "mulaw", "sample_rate": "8000",
                  "channels": "1", "smart_format": "true", "punctuate": "true"}
        lang = self.cfg.extra.get("language") or language
        if lang:
            params["language"] = lang
        else:
            params["detect_language"] = "true"
        if hints and self.cfg.extra.get("send_hints"):
            # nova-3 takes "keyterm", older models "keywords"; check Deepgram's docs for Arabic support
            params["keyterm" if self.cfg.model.startswith("nova-3") else "keywords"] = hints[:50]
        resp = await self.client.post(f"{self.cfg.base_url}/listen", params=params, content=ulaw,
                                      headers={"Authorization": f"Token {self.cfg.api_key}",
                                               "Content-Type": "application/octet-stream"})
        if resp.status_code >= 400:
            raise STTError(f"deepgram HTTP {resp.status_code}: {resp.text[:300]}")
        ch = (resp.json().get("results", {}).get("channels") or [{}])[0]
        alt = (ch.get("alternatives") or [{}])[0]
        return Transcript(text=(alt.get("transcript") or "").strip(),
                          language=normalize_lang(ch.get("detected_language")),
                          confidence=alt.get("confidence"))


def make_stt(cfg: ProviderConfig, client: httpx.AsyncClient | None = None) -> STTProvider:
    if cfg.provider == "mock":
        from .mock import MockSTT
        return MockSTT()
    if not cfg.api_key and cfg.provider != "mock":
        raise STTError(f"An API key is required for STT_PROVIDER={cfg.provider} (set STT_API_KEY)")
    if cfg.provider == "deepgram":
        return DeepgramSTT(cfg, client)
    return OpenAICompatibleSTT(cfg, client)
