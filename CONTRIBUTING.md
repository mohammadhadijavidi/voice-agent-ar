# Contributing

Thanks for helping. Issues and pull requests are welcome in Arabic or English.

## Setup

```bash
git clone https://github.com/mohammadhadijavidi/voice-agent-ar
cd voice-agent-ar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
voice-agent-ar simulate all
```

No API keys are needed: tests and the simulator use mock STT/LLM/TTS and a simulated Twilio Media Streams caller.

## Most useful contributions

- **Dialect coverage.** If a Gulf phrase is misunderstood (a day, a time, a party size, a yes/no), add it to
  `src/voice_agent_ar/arabic/gulf.py` together with a test case in `tests/test_gulf_dialect.py`.
  Real phrasing from Emirati, Saudi, Kuwaiti, Qatari, Bahraini and Omani callers is especially welcome.
- **Spoken numbers.** Corrections to `arabic/numbers.py` (Gulf and MSA forms) with tests.
- **Providers.** New STT/TTS providers that do well on Gulf Arabic, behind the existing small interfaces.
- **Scenarios.** New simulated calls in `src/voice_agent_ar/scenarios/` that reproduce a real failure.

## Rules

- Keep runtime dependencies minimal.
- Every behaviour change needs a test; `pytest -q` and `voice-agent-ar simulate all` must pass.
- Sample businesses, names and numbers must stay clearly fictional.
- Use only official APIs (Twilio, WhatsApp Cloud API).
