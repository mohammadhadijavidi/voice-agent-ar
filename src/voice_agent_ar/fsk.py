"""A tiny FSK modem used ONLY by the mock STT/TTS and the call simulator.

There is no API key in CI, so the mock providers cannot recognise or produce real
speech. Instead, the simulator "speaks" by modulating the caller's line as binary
FSK tones (Bell-202-like, 1200/2200 Hz) and the mock STT demodulates it. The mock
TTS does the same in reverse, so the simulator can tell what the agent said and
exactly where a barge-in cut it off.

The point: every test call travels through the same path real audio does
(base64 -> mu-law -> 20 ms frames -> VAD -> utterance segmentation -> barge-in),
with deterministic, checkable content. It is a test modem, not speech recognition.
"""

from __future__ import annotations

import binascii
import math
import struct
from array import array

from .audio import SAMPLE_RATE, pcm16_to_ulaw, ulaw_to_pcm16

MARK_HZ = 1200.0   # bit 1
SPACE_HZ = 2200.0  # bit 0
BIT_SAMPLES = 40   # 5 ms per bit at 8 kHz -> 200 bit/s
AMPLITUDE = 8000
SYNC = b"\x7e\x02"


def _bits(data: bytes):
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1


def encode(text: str) -> bytes:
    """Text -> mu-law audio (8 kHz) carrying an FSK frame: SYNC | len | utf8 | crc16."""
    payload = text.encode("utf-8")
    frame = SYNC + struct.pack(">H", len(payload)) + payload
    frame += struct.pack(">H", binascii.crc_hqx(frame, 0xFFFF))
    samples = array("h")
    phase = 0.0
    for bit in _bits(frame):
        step = 2 * math.pi * (MARK_HZ if bit else SPACE_HZ) / SAMPLE_RATE
        for _ in range(BIT_SAMPLES):
            samples.append(int(AMPLITUDE * math.sin(phase)))
            phase += step
    return pcm16_to_ulaw(samples)


def _power(samples, start: int, freq: float) -> float:
    # Goertzel algorithm over one bit window
    k = 2 * math.cos(2 * math.pi * freq / SAMPLE_RATE)
    s1 = s2 = 0.0
    for i in range(start, start + BIT_SAMPLES):
        s0 = samples[i] + k * s1 - s2
        s2, s1 = s1, s0
    return s1 * s1 + s2 * s2 - k * s1 * s2


def _read_bytes(samples, start: int, count: int) -> bytes | None:
    out = bytearray()
    pos = start
    for _ in range(count):
        if pos + 8 * BIT_SAMPLES > len(samples):
            return None
        byte = 0
        for _ in range(8):
            bit = 1 if _power(samples, pos, MARK_HZ) > _power(samples, pos, SPACE_HZ) else 0
            byte = (byte << 1) | bit
            pos += BIT_SAMPLES
        out.append(byte)
    return bytes(out)


def decode(ulaw: bytes) -> str | None:
    """mu-law audio -> text, or None if no valid FSK frame is present."""
    samples = ulaw_to_pcm16(ulaw)
    n = len(samples)
    # locate signal onset
    onset = None
    for i in range(0, n - BIT_SAMPLES, 8):
        if max(abs(s) for s in samples[i:i + 8]) > AMPLITUDE // 4:
            onset = i
            break
    if onset is None:
        return None
    for offset in range(max(0, onset - 8), min(onset + BIT_SAMPLES, n), 2):
        head = _read_bytes(samples, offset, 4)
        if head is None:
            return None
        if head[:2] != SYNC:
            continue
        length = struct.unpack(">H", head[2:4])[0]
        rest = _read_bytes(samples, offset + 32 * BIT_SAMPLES, length + 2)
        if rest is None:
            continue
        frame = head + rest[:-2]
        if struct.unpack(">H", rest[-2:])[0] != binascii.crc_hqx(frame, 0xFFFF):
            continue
        try:
            return frame[4:].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return None


def duration_ms(text: str) -> int:
    n_bits = 8 * (len(SYNC) + 4 + len(text.encode("utf-8")))
    return n_bits * BIT_SAMPLES * 1000 // SAMPLE_RATE
