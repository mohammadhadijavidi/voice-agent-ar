"""CRM / automation webhooks with a durable outbox.

Events (``booking.created``, ``lead.created``, ``call.completed``) are written to the
SQLite outbox first, then POSTed as JSON. Each request carries
``X-Voice-Agent-Signature: sha256=<hex HMAC of the raw body>`` when a secret is set,
so n8n, Zapier, Make, HubSpot workflows or your own backend can verify it.
Failed deliveries stay in the outbox and are retried with backoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time

import httpx

from .storage import Store

log = logging.getLogger("voice_agent.webhooks")


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify(body: bytes, secret: str, header: str) -> bool:
    return hmac.compare_digest(sign(body, secret), header or "")


class WebhookDispatcher:
    def __init__(self, store: Store, url: str = "", secret: str = "",
                 client: httpx.AsyncClient | None = None, max_attempts: int = 8):
        self.store = store
        self.url = url
        self.secret = secret
        self.client = client
        self.max_attempts = max_attempts
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def emit(self, event: str, data: dict) -> int | None:
        """Queue an event. Stored even when no URL is configured (visible in the dashboard)."""
        payload = {"event": event, "sent_at": time.time(), "data": data}
        oid = self.store.outbox_add(event, payload)
        self._wake.set()
        return oid

    async def deliver_pending(self) -> int:
        """Try to deliver everything pending once. Returns number delivered."""
        if not self.enabled:
            return 0
        client = self.client or httpx.AsyncClient(timeout=10.0)
        delivered = 0
        try:
            for row in self.store.outbox_pending(self.max_attempts):
                body = json.dumps(row["payload"], ensure_ascii=False).encode()
                headers = {"Content-Type": "application/json",
                           "X-Voice-Agent-Event": row["event"],
                           "X-Voice-Agent-Delivery": str(row["id"])}
                if self.secret:
                    headers["X-Voice-Agent-Signature"] = sign(body, self.secret)
                try:
                    r = await client.post(self.url, content=body, headers=headers)
                    ok = 200 <= r.status_code < 300
                    self.store.outbox_result(row["id"], ok, "" if ok else f"HTTP {r.status_code}")
                    delivered += ok
                except httpx.HTTPError as e:
                    self.store.outbox_result(row["id"], False, repr(e))
        finally:
            if self.client is None:
                await client.aclose()
        return delivered

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.deliver_pending()
                pending = self.store.outbox_pending(self.max_attempts) if self.enabled else []
                backoff = min(backoff * 2, 300.0) if pending else 30.0
            except Exception:  # pragma: no cover - never let the loop die
                log.exception("webhook delivery loop error")

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
