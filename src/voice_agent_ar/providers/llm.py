"""LLM providers with streaming and tool calling.

Internal message format is OpenAI-style chat messages:
  {"role": "system"|"user"|"assistant"|"tool", "content": str,
   "tool_calls": [{"id", "name", "arguments": dict}]   (assistant only)
   "tool_call_id": str, "name": str                    (tool only)}

Every provider exposes ``stream(messages, tools)`` which yields ``TextDelta`` events
while the model talks and finally one ``ToolCalls`` event if it wants tools.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from ..config import ProviderConfig


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolCalls:
    calls: list[ToolCall] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


class LLMProvider:
    name = "base"
    is_mock = False

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[TextDelta | ToolCalls]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def complete(self, messages: list[dict]) -> str:
        out = []
        async for ev in self.stream(messages, []):
            if isinstance(ev, TextDelta):
                out.append(ev.text)
        return "".join(out).strip()


async def _sse_lines(resp: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Yield (event, data) pairs from an SSE response."""
    event = ""
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = "", []
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "event":
            event = value
        elif key == "data":
            data_lines.append(value)
    if data_lines:
        yield event, "\n".join(data_lines)


def _parse_args(raw: str) -> dict:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except json.JSONDecodeError:
        return {"_raw": raw}


class OpenAICompatibleLLM(LLMProvider):
    """OpenAI, OpenRouter, Groq, Ollama, LM Studio, vLLM ... anything speaking /chat/completions."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.name = cfg.provider
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant", "content": m.get("content") or None,
                    "tool_calls": [{"id": c["id"], "type": "function", "function": {
                        "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                        for c in m["tool_calls"]],
                })
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    async def stream(self, messages, tools):
        body = {
            "model": self.cfg.model,
            "messages": self.to_wire(messages),
            "stream": True,
            "temperature": self.cfg.extra.get("temperature", 0.3),
            "max_tokens": self.cfg.extra.get("max_tokens", 300),
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        pending: dict[int, dict] = {}
        async with self.client.stream("POST", f"{self.cfg.base_url}/chat/completions",
                                      json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                raise LLMError(f"{self.name} HTTP {resp.status_code}: {(await resp.aread())[:300]!r}")
            async for _, data in _sse_lines(resp):
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield TextDelta(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = pending.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
        if pending:
            yield ToolCalls([ToolCall(s["id"] or f"call_{uuid.uuid4().hex[:8]}", s["name"], _parse_args(s["args"]))
                             for _, s in sorted(pending.items())])


class AnthropicLLM(LLMProvider):
    """Anthropic Messages API (native, streaming, tool use)."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.name = "anthropic"
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    @staticmethod
    def to_wire(messages: list[dict]) -> tuple[str, list[dict]]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        out: list[dict] = []

        def push(role: str, block: dict):
            if out and out[-1]["role"] == role:
                out[-1]["content"].append(block)
            else:
                out.append({"role": role, "content": [block]})

        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                push("user", {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]})
            elif m["role"] == "assistant":
                if m.get("content"):
                    push("assistant", {"type": "text", "text": m["content"]})
                for c in m.get("tool_calls") or []:
                    push("assistant", {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]})
            else:
                push("user", {"type": "text", "text": m["content"]})
        return system, out

    async def stream(self, messages, tools):
        system, wire = self.to_wire(messages)
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.extra.get("max_tokens", 300),
            "temperature": self.cfg.extra.get("temperature", 0.3),
            "system": system,
            "messages": wire,
            "stream": True,
        }
        if tools:
            body["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                              "input_schema": t["parameters"]} for t in tools]
        headers = {"x-api-key": self.cfg.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        blocks: dict[int, dict] = {}
        async with self.client.stream("POST", f"{self.cfg.base_url}/v1/messages", json=body,
                                      headers=headers) as resp:
            if resp.status_code >= 400:
                raise LLMError(f"anthropic HTTP {resp.status_code}: {(await resp.aread())[:300]!r}")
            async for event, data in _sse_lines(resp):
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("type") or event
                if kind == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        blocks[ev["index"]] = {"id": cb.get("id", ""), "name": cb.get("name", ""), "args": ""}
                elif kind == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta" and d.get("text"):
                        yield TextDelta(d["text"])
                    elif d.get("type") == "input_json_delta" and ev.get("index") in blocks:
                        blocks[ev["index"]]["args"] += d.get("partial_json", "")
                elif kind == "error":
                    raise LLMError(f"anthropic stream error: {ev.get('error')}")
        if blocks:
            yield ToolCalls([ToolCall(b["id"], b["name"], _parse_args(b["args"]))
                             for _, b in sorted(blocks.items())])


def make_llm(cfg: ProviderConfig, client: httpx.AsyncClient | None = None) -> LLMProvider:
    if cfg.provider == "mock":
        from .mock import MockLLM
        return MockLLM()
    if cfg.provider == "anthropic":
        if not cfg.api_key:
            raise LLMError("ANTHROPIC_API_KEY (or LLM_API_KEY) is required for LLM_PROVIDER=anthropic")
        return AnthropicLLM(cfg, client)
    if cfg.provider in ("openai", "openrouter", "groq") and not cfg.api_key:
        raise LLMError(f"An API key is required for LLM_PROVIDER={cfg.provider} (set LLM_API_KEY)")
    return OpenAICompatibleLLM(cfg, client)
