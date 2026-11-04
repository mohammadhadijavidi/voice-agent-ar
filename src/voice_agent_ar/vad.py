"""Frame-based voice activity detection and utterance segmentation.

Everything is counted in 20 ms frames rather than wall-clock time, so behaviour is
identical whether audio arrives in real time from Twilio or faster from the
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import FRAME_MS, rms, ulaw_to_pcm16


@dataclass
class VADConfig:
    threshold: float = 600.0      # RMS on the int16 scale; phone-line silence is ~0-50
    start_ms: int = 100           # speech needed to open an utterance
    endpoint_ms: int = 600        # trailing silence that closes an utterance
    barge_in_ms: int = 240        # speech needed while the agent talks to count as barge-in
    max_utterance_ms: int = 30000


@dataclass
class SegmentEvent:
    kind: str                     # "speech_start" | "utterance_end"
    audio: bytes = b""            # mu-law audio of the utterance (for utterance_end)
    duration_ms: int = 0


@dataclass
class UtteranceSegmenter:
    """Turns a stream of mu-law frames into speech_start / utterance_end events."""

    config: VADConfig = field(default_factory=VADConfig)
    in_speech: bool = False
    speech_run: int = 0           # consecutive voiced frames
    silence_run: int = 0          # consecutive unvoiced frames (inside an utterance)
    voiced_frames: int = 0        # voiced frames in the current utterance
    _pre: list = field(default_factory=list)
    _buf: bytearray = field(default_factory=bytearray)

    def is_voiced(self, frame: bytes) -> bool:
        return rms(ulaw_to_pcm16(frame)) >= self.config.threshold

    def feed(self, frame: bytes) -> list[SegmentEvent]:
        c = self.config
        voiced = self.is_voiced(frame)
        events: list[SegmentEvent] = []
        if not self.in_speech:
            if voiced:
                self.speech_run += 1
                self._pre.append(frame)
                if self.speech_run * FRAME_MS >= c.start_ms:
                    self.in_speech = True
                    self.silence_run = 0
                    self.voiced_frames = self.speech_run
                    self._buf = bytearray(b"".join(self._pre))
                    self._pre = []
                    events.append(SegmentEvent("speech_start"))
            else:
                self.speech_run = 0
                self._pre = []
            return events

        self._buf.extend(frame)
        if voiced:
            self.speech_run += 1
            self.voiced_frames += 1
            self.silence_run = 0
        else:
            self.speech_run = 0
            self.silence_run += 1
        dur = len(self._buf) // 8  # 8 bytes per ms at 8 kHz
        if self.silence_run * FRAME_MS >= c.endpoint_ms or dur >= c.max_utterance_ms:
            events.append(SegmentEvent("utterance_end", bytes(self._buf), dur))
            self.reset()
        return events

    def voiced_ms(self) -> int:
        """Current run of consecutive voiced audio, in ms."""
        return self.speech_run * FRAME_MS

    def voiced_total_ms(self) -> int:
        """Voiced audio in the current utterance so far, in ms (short pauses don't reset it)."""
        return self.voiced_frames * FRAME_MS

    def reset(self) -> None:
        self.voiced_frames = 0
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self._pre = []
        self._buf = bytearray()
