import math
import wave
from array import array

from voice_agent_ar import audio, fsk
from voice_agent_ar.vad import UtteranceSegmenter, VADConfig


def test_ulaw_roundtrip_is_close():
    samples = array("h", [0, 100, -100, 1000, -1000, 8000, -8000, 32000, -32000])
    back = audio.ulaw_to_pcm16(audio.pcm16_to_ulaw(samples))
    for a, b in zip(samples, back):
        # G.711 is logarithmic: error grows with amplitude, max ~3% + small offset
        assert abs(a - b) <= max(16, abs(a) * 0.04)


def test_ulaw_known_values():
    # standard G.711 reference points
    assert audio.pcm16_to_ulaw(array("h", [0])) == b"\xff"
    assert audio._DECODE[0xFF] == 0
    assert audio._DECODE[0x00] == -32124
    assert audio._DECODE[0x80] == 32124


def test_resample_24k_to_8k_preserves_tone():
    src = audio.tone(440, 200, rate=24000)
    out = audio.resample(src, 24000, 8000)
    assert len(out) == len(src) // 3
    # zero crossings ~ 2 * 440 * 0.2s
    crossings = sum(1 for a, b in zip(out, out[1:]) if (a < 0) != (b < 0))
    assert 160 <= crossings <= 192


def test_wav_roundtrip(tmp_path):
    ulaw = audio.pcm16_to_ulaw(audio.tone(300, 500))
    p = tmp_path / "x.wav"
    audio.write_ulaw_as_wav(str(p), ulaw)
    with wave.open(str(p)) as w:
        assert w.getframerate() == 8000 and w.getnframes() == 4000
    assert len(audio.read_wav_as_ulaw(str(p))) == 4000


def test_frames_pad_last():
    fr = audio.frames(b"\x00" * 330)
    assert [len(f) for f in fr] == [160, 160, 160]
    assert fr[-1].endswith(b"\xff" * 150)


def test_fsk_roundtrip_multilingual():
    for text in ["Book a haircut at 4 pm", "مرحبا، أبي موعد بكرة", "سلام، نوبت می‌خوام ۱۱ صبح"]:
        enc = audio.silence(200) + fsk.encode(text) + audio.silence(400)
        assert fsk.decode(enc) == text


def test_fsk_rejects_noise_and_truncation():
    assert fsk.decode(audio.silence(500)) is None
    assert fsk.decode(audio.pcm16_to_ulaw(audio.tone(1000, 500))) is None
    assert fsk.decode(fsk.encode("hello there")[:1500]) is None


def _feed(seg, data):
    events = []
    for f in audio.frames(data):
        events += seg.feed(f)
    return events


def test_vad_segments_one_utterance():
    seg = UtteranceSegmenter(VADConfig(endpoint_ms=400))
    speech = audio.pcm16_to_ulaw(audio.tone(500, 800))
    events = _feed(seg, audio.silence(300) + speech + audio.silence(600))
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "utterance_end"]
    end = events[-1]
    # utterance contains the speech plus the trailing endpoint silence
    assert 800 <= end.duration_ms <= 800 + 400 + 20


def test_vad_ignores_short_blips_and_bridges_short_pauses():
    seg = UtteranceSegmenter(VADConfig(start_ms=100, endpoint_ms=500))
    blip = audio.pcm16_to_ulaw(audio.tone(500, 60))
    assert _feed(seg, blip + audio.silence(800)) == []
    word = audio.pcm16_to_ulaw(audio.tone(500, 300))
    events = _feed(seg, word + audio.silence(300) + word + audio.silence(600))
    assert [e.kind for e in events] == ["speech_start", "utterance_end"]
    assert events[-1].duration_ms >= 900


def test_vad_threshold_scale():
    seg = UtteranceSegmenter()
    quiet = audio.pcm16_to_ulaw(array("h", [int(100 * math.sin(i)) for i in range(160)]))
    loud = audio.pcm16_to_ulaw(audio.tone(500, 20, amplitude=4000))
    assert not seg.is_voiced(quiet)
    assert seg.is_voiced(loud)
