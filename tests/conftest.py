from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from voice_agent_ar.config import Settings
from voice_agent_ar.profile import load_profile
from voice_agent_ar.storage import Store

# Wednesday 4 November 2026, 09:00 Dubai time. Fixed so slot math is deterministic.
FIXED_NOW = datetime(2026, 11, 4, 9, 0, tzinfo=ZoneInfo("Asia/Dubai"))

ENV = ("LLM_PROVIDER", "STT_PROVIDER", "TTS_PROVIDER", "LLM_API_KEY", "STT_API_KEY", "TTS_API_KEY",
       "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "DEEPGRAM_API_KEY",
       "ELEVENLABS_API_KEY", "TWILIO_AUTH_TOKEN", "TWILIO_ACCOUNT_SID", "TRANSFER_NUMBER", "CRM_WEBHOOK_URL",
       "CRM_WEBHOOK_SECRET", "PUBLIC_URL", "DASHBOARD_TOKEN", "LLM_BASE_URL", "LLM_MODEL", "PROFILE", "DB_PATH",
       "DIALECT", "NOTIFY_CHANNEL", "NOTIFY_DRY_RUN", "WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "SMS_FROM",
       "STT_LANGUAGE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def fixed(name: str):
    p = load_profile(name)
    p.now_fn = lambda: FIXED_NOW
    return p


@pytest.fixture
def restaurant():
    return fixed("restaurant")


@pytest.fixture
def clinic():
    return fixed("clinic")


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env(db_path=str(tmp_path / "test.db"))
