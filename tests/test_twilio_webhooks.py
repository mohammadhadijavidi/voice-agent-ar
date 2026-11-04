import httpx
from fastapi.testclient import TestClient

from voice_agent_ar.server import create_app
from voice_agent_ar.twilio import compute_signature, connect_stream_twiml, dial_twiml, stream_token, validate_signature
from voice_agent_ar.webhooks import WebhookDispatcher, sign, verify


def test_twilio_signature_reference_example():
    # Vector from Twilio's documentation of the X-Twilio-Signature algorithm
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+12349013030", "Digits": "1234",
              "From": "+12349013030", "To": "+18005551212"}
    assert compute_signature("12345", url, params) == "0/KCTR6DLpKmkAf8muzZqo1nDgQ="
    assert validate_signature("12345", url, params, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")
    assert not validate_signature("12345", url, {**params, "Digits": "9"}, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")


def test_twiml_is_escaped():
    xml = connect_stream_twiml("wss://x.test/twilio/stream", {"from": '+971"<&>'})
    assert 'value="+971&quot;&lt;&amp;&gt;"' in xml
    assert "<Dial>+971&lt;4&gt;</Dial>" in dial_twiml("+971<4>")


def test_voice_webhook_requires_valid_signature(settings, restaurant, store):
    settings.twilio_auth_token = "tok"
    settings.public_url = "https://agent.example.com"
    c = TestClient(create_app(settings, profile=restaurant, store=store))
    form = {"CallSid": "CA1", "From": "+971500000000", "To": "+97140000000"}
    assert c.post("/twilio/voice", data=form).status_code == 403
    sig = compute_signature("tok", "https://agent.example.com/twilio/voice", form)
    r = c.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": sig})
    assert r.status_code == 200 and '<Stream url="wss://agent.example.com/twilio/stream">' in r.text
    assert f'name="token" value="{stream_token("tok", "CA1")}"' in r.text


def test_stream_without_token_is_rejected(settings, restaurant, store):
    c = TestClient(create_app(settings, profile=restaurant, store=store))
    with c.websocket_connect("/twilio/stream") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {}}})
    assert store.list_calls() == []


def test_dashboard_token(settings, restaurant, store):
    settings.dashboard_token = "s3cret"
    c = TestClient(create_app(settings, profile=restaurant, store=store))
    assert c.get("/api/calls").status_code == 401
    assert c.get("/api/calls", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/health").json()["ok"] is True


async def test_crm_webhook_signed_and_retried(store):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(500 if len(calls) == 1 else 200)
    wh = WebhookDispatcher(store, "https://crm.test/hook", "k", client=httpx.AsyncClient(
        transport=httpx.MockTransport(handler)))
    wh.emit("booking.created", {"code": "12345"})
    assert await wh.deliver_pending() == 0
    assert await wh.deliver_pending() == 1
    assert verify(calls[-1].content, "k", calls[-1].headers["x-voice-agent-signature"])
    assert sign(b"x", "k").startswith("sha256=")
