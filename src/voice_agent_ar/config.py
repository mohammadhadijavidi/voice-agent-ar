"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Preset base URLs for OpenAI-compatible LLM / STT endpoints.
LLM_PRESETS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", ""),
    "lmstudio": ("http://localhost:1234/v1", "local-model", ""),
    "anthropic": ("https://api.anthropic.com", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "mock": ("", "mock-receptionist", ""),
}

STT_PRESETS = {
    "openai": ("https://api.openai.com/v1", "whisper-1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "whisper-large-v3-turbo", "GROQ_API_KEY"),
    "deepgram": ("https://api.deepgram.com/v1", "nova-3", "DEEPGRAM_API_KEY"),
    "mock": ("", "fsk-test-modem", ""),
}

TTS_PRESETS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini-tts", "OPENAI_API_KEY", "alloy"),
    "elevenlabs": ("https://api.elevenlabs.io/v1", "eleven_flash_v2_5", "ELEVENLABS_API_KEY", ""),
    "mock": ("", "fsk-test-modem", "", "mock"),
}


DEFAULT_TTS_INSTRUCTIONS = ("Speak warm, natural Gulf (Emirati) Arabic like a friendly receptionist in Dubai. "
                            "Moderate pace, clear numbers and times.")


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v else default


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    extra: dict = field(default_factory=dict)


@dataclass
class Settings:
    llm: ProviderConfig
    stt: ProviderConfig
    tts: ProviderConfig
    public_url: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    validate_twilio_signature: bool = True
    transfer_number: str = ""
    profile: str = "restaurant"
    dialect: str = ""                       # override the profile: gulf | msa
    db_path: str = "voice_agent_ar.db"
    crm_webhook_url: str = ""
    crm_webhook_secret: str = ""
    dashboard_token: str = ""
    vad_threshold: float = 600.0
    vad_endpoint_ms: int = 600
    barge_in_ms: int = 240
    summary_enabled: bool = True
    max_call_seconds: int = 900
    notify_channel: str = "whatsapp_then_sms"
    notify_dry_run: bool = True
    wa_token: str = ""
    wa_phone_number_id: str = ""
    wa_template: str = "booking_confirmation_ar"
    wa_template_lang: str = "ar"
    sms_from: str = ""

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        llm_p = _env("LLM_PROVIDER", "mock").lower()
        if llm_p not in LLM_PRESETS:
            raise ValueError(f"LLM_PROVIDER={llm_p!r}; choose one of {sorted(LLM_PRESETS)}")
        base, model, key_env = LLM_PRESETS[llm_p]
        llm = ProviderConfig(
            provider=llm_p,
            model=_env("LLM_MODEL", model),
            base_url=_env("LLM_BASE_URL", base).rstrip("/"),
            api_key=_env("LLM_API_KEY") or (_env(key_env) if key_env else ""),
            extra={"temperature": _float("LLM_TEMPERATURE", 0.3),
                   "max_tokens": _int("LLM_MAX_TOKENS", 300)},
        )
        stt_p = _env("STT_PROVIDER", "mock").lower()
        if stt_p not in STT_PRESETS:
            raise ValueError(f"STT_PROVIDER={stt_p!r}; choose one of {sorted(STT_PRESETS)}")
        base, model, key_env = STT_PRESETS[stt_p]
        stt = ProviderConfig(
            provider=stt_p,
            model=_env("STT_MODEL", model),
            base_url=_env("STT_BASE_URL", base).rstrip("/"),
            api_key=_env("STT_API_KEY") or (_env(key_env) if key_env else ""),
            extra={"language": _env("STT_LANGUAGE", "ar"),
                   "response_format": _env("STT_RESPONSE_FORMAT"),
                   "send_hints": _env("STT_SEND_HINTS", "true" if stt_p != "deepgram" else "false").lower()
                   == "true"},
        )
        tts_p = _env("TTS_PROVIDER", "mock").lower()
        if tts_p not in TTS_PRESETS:
            raise ValueError(f"TTS_PROVIDER={tts_p!r}; choose one of {sorted(TTS_PRESETS)}")
        base, model, key_env, voice = TTS_PRESETS[tts_p]
        default_voice = _env("TTS_VOICE", voice)
        tts = ProviderConfig(
            provider=tts_p,
            model=_env("TTS_MODEL", model),
            base_url=_env("TTS_BASE_URL", base).rstrip("/"),
            api_key=_env("TTS_API_KEY") or (_env(key_env) if key_env else ""),
            extra={"voices": {
                "en": _env("TTS_VOICE_EN", default_voice),
                "ar": _env("TTS_VOICE_AR", default_voice),
            }, "instructions": _env("TTS_INSTRUCTIONS", DEFAULT_TTS_INSTRUCTIONS)},
        )
        s = cls(
            llm=llm, stt=stt, tts=tts,
            public_url=_env("PUBLIC_URL").rstrip("/"),
            twilio_account_sid=_env("TWILIO_ACCOUNT_SID"),
            twilio_auth_token=_env("TWILIO_AUTH_TOKEN"),
            validate_twilio_signature=_env("TWILIO_VALIDATE_SIGNATURE", "true").lower() != "false",
            transfer_number=_env("TRANSFER_NUMBER"),
            profile=_env("PROFILE", "restaurant"),
            dialect=_env("DIALECT").lower(),
            db_path=_env("DB_PATH", "voice_agent_ar.db"),
            crm_webhook_url=_env("CRM_WEBHOOK_URL"),
            crm_webhook_secret=_env("CRM_WEBHOOK_SECRET"),
            dashboard_token=_env("DASHBOARD_TOKEN"),
            vad_threshold=_float("VAD_THRESHOLD", 600.0),
            vad_endpoint_ms=_int("VAD_ENDPOINT_MS", 600),
            barge_in_ms=_int("BARGE_IN_MS", 240),
            summary_enabled=_env("SUMMARY_ENABLED", "true").lower() != "false",
            max_call_seconds=_int("MAX_CALL_SECONDS", 900),
            notify_channel=_env("NOTIFY_CHANNEL", "whatsapp_then_sms").lower(),
            notify_dry_run=_env("NOTIFY_DRY_RUN", "true").lower() != "false",
            wa_token=_env("WHATSAPP_TOKEN"),
            wa_phone_number_id=_env("WHATSAPP_PHONE_NUMBER_ID"),
            wa_template=_env("WHATSAPP_TEMPLATE", "booking_confirmation_ar"),
            wa_template_lang=_env("WHATSAPP_TEMPLATE_LANG", "ar"),
            sms_from=_env("SMS_FROM"),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def describe(self) -> dict:
        """Config summary that is safe to print (no secrets)."""
        def p(c: ProviderConfig) -> dict:
            return {"provider": c.provider, "model": c.model, "base_url": c.base_url or "-",
                    "api_key": "set" if c.api_key else "not set"}
        return {
            "llm": p(self.llm), "stt": p(self.stt), "tts": p(self.tts),
            "public_url": self.public_url or "(not set: derived from request Host header)",
            "twilio_signature_validation": bool(self.twilio_auth_token and self.validate_twilio_signature),
            "transfer_number": self.transfer_number or "(not set)",
            "profile": self.profile, "db_path": self.db_path,
            "crm_webhook": self.crm_webhook_url or "(not set: leads stored locally only)",
            "dashboard_token": "set" if self.dashboard_token else "not set (open dashboard)",
            "dialect": self.dialect or "(from profile)",
            "notify": {"channel": self.notify_channel, "dry_run": self.notify_dry_run,
                       "whatsapp": "configured" if (self.wa_token and self.wa_phone_number_id) else "not configured",
                       "whatsapp_template": f"{self.wa_template} ({self.wa_template_lang})",
                       "sms": "configured" if (self.sms_from and self.twilio_auth_token) else "not configured"},
        }
