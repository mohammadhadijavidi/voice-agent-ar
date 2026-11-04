# Connecting a real Twilio number

1. Buy or port a number in the Twilio console. UAE local numbers have regulatory requirements; many
   businesses forward their existing landline/mobile to a Twilio number instead.
2. Expose the server over HTTPS (`cloudflared tunnel --url http://localhost:8080` or `ngrok http 8080`)
   and set `PUBLIC_URL` to that address.
3. Phone number → Voice configuration → "A call comes in": Webhook, `POST https://<PUBLIC_URL>/twilio/voice`.
   Optional status callback: `POST https://<PUBLIC_URL>/twilio/status`.
4. Set `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`. With the token set, every webhook must carry a valid
   `X-Twilio-Signature` and the Media Stream must present the per-call token issued in the TwiML.
5. `voice-agent-ar doctor` prints the effective configuration and anything missing.

## WhatsApp confirmations

1. In Meta Business Manager create a WhatsApp Business account and phone number (Cloud API).
2. Submit `examples/whatsapp_template.json` (category UTILITY, language `ar`) in WhatsApp Manager →
   Message templates. Wait for approval.
3. Set `WHATSAPP_TOKEN` (a system-user token), `WHATSAPP_PHONE_NUMBER_ID`, then `NOTIFY_DRY_RUN=false`.

The caller has not messaged you on WhatsApp, so a free-form message is not allowed: only an approved
template can be sent. That is why the summary uses a template.

## SMS confirmations

Set `SMS_FROM` to a sender your Twilio account may use towards UAE numbers (an alphanumeric sender ID
registered for your business, or a Twilio number) and `NOTIFY_CHANNEL=sms` or `whatsapp_then_sms`.
Arabic SMS is UCS-2: 70 characters per segment; the summary is kept to two segments.
