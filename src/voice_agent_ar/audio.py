"""Audio helpers for 8 kHz telephone audio.

Twilio Media Streams carry G.711 mu-law, 8 kHz, mono, base64-encoded. Python 3.13
removed the stdlib ``audioop`` module, so the codec is implemented here in pure
Python with lookup tables (fast enough for many concurrent calls: one 20 ms frame
is 160 bytes).
"""

from __future__ import annotations

import io
import math
import struct
import wave
from array import array

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000  # 160 mu-law bytes per 20 ms frame
ULAW_SILENCE = 0xFF

_BIAS = 0x84
_CLIP = 32635


def _encode_sample(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    if sign:
        sample = -sample
    if sample > _CLIP:
        sample = _CLIP
    sample += _BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _decode_byte(u: int) -> int:
    u = ~u & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return -sample if sign else sample


_DECODE = [_decode_byte(i) for i in range(256)]
# Index with (sample & 0xFFFF) so negative int16 values map correctly.
_ENCODE = bytes(_encode_sample(i - 65536 if i >= 32768 else i) for i in range(65536))


def ulaw_to_pcm16(data: bytes) -> array:
    """Decode mu-law bytes to signed 16-bit samples."""
    dec = _DECODE
    return array("h", [dec[b] for b in data])


def pcm16_to_ulaw(samples: array | list[int]) -> bytes:
    """Encode signed 16-bit samples to mu-law bytes."""
    enc = _ENCODE
    return bytes([enc[s & 0xFFFF] for s in samples])


def pcm16_bytes_to_samples(data: bytes) -> array:
    """Little-endian PCM16 bytes -> array('h')."""
    if len(data) % 2:
        data = data[:-1]
    arr = array("h")
    arr.frombytes(data)
    if struct.pack("=h", 1) != struct.pack("<h", 1):  # big-endian host
        arr.byteswap()
    return arr


def rms(samples: array | list[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def resample(samples: array | list[int], src_rate: int, dst_rate: int) -> array:
    """Resample mono PCM16.

    Integer downsampling (e.g. 24 kHz -> 8 kHz) averages each block, which acts as
    a crude low-pass filter. Other ratios use linear interpolation. Good enough for
    narrow-band telephone audio; swap in a proper polyphase filter if you need more.
    """
    if src_rate == dst_rate or not samples:
        return array("h", samples)
    if src_rate > dst_rate and src_rate % dst_rate == 0:
        f = src_rate // dst_rate
        n = len(samples) // f
        return array("h", [int(sum(samples[i * f:(i + 1) * f]) / f) for i in range(n)])
    ratio = src_rate / dst_rate
    n = int(len(samples) / ratio)
    out = array("h")
    last = len(samples) - 1
    for i in range(n):
        pos = i * ratio
        j = int(pos)
        frac = pos - j
        a = samples[j]
        b = samples[j + 1] if j < last else a
        out.append(int(a + (b - a) * frac))
    return out


def frames(data: bytes, size: int = FRAME_BYTES) -> list[bytes]:
    """Split mu-law audio into fixed-size frames, padding the last one with silence."""
    out = []
    for i in range(0, len(data), size):
        chunk = data[i:i + size]
        if len(chunk) < size:
            chunk = chunk + bytes([ULAW_SILENCE]) * (size - len(chunk))
        out.append(chunk)
    return out


def silence(ms: int) -> bytes:
    return bytes([ULAW_SILENCE]) * (SAMPLE_RATE * ms // 1000)


def tone(freq: float, ms: int, amplitude: int = 6000, rate: int = SAMPLE_RATE) -> array:
    n = rate * ms // 1000
    fade = max(1, min(n // 8, rate // 200))  # short fade in/out to avoid clicks
    out = array("h")
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade) if n > 1 else 1.0
        out.append(int(amplitude * env * math.sin(2 * math.pi * freq * i / rate)))
    return out


def pcm16_to_wav_bytes(samples: array, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def read_wav_as_ulaw(path: str) -> bytes:
    """Read a mono/stereo PCM16 WAV file and return 8 kHz mu-law bytes."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    samples = pcm16_bytes_to_samples(raw)
    if channels > 1:
        samples = array("h", samples[::channels])
    return pcm16_to_ulaw(resample(samples, rate, SAMPLE_RATE))


def write_ulaw_as_wav(path: str, ulaw: bytes) -> None:
    with open(path, "wb") as f:
        f.write(pcm16_to_wav_bytes(ulaw_to_pcm16(ulaw)))
