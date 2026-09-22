"""
audio_source.py — where the PCM comes from.

One interface, so the GPU loop never knows whether it is being fed by a test
file or by imood voice over the network. The file source exists so the
streaming pipeline can be diffed frame-by-frame against the offline baseline;
that regression check is the only way to tell whether a change to the
streaming path altered the picture.

Formats follow imood voice's TTS output (docs/tts-streaming-spec.md):
16kHz mono PCM16 little-endian, no container, 320ms per chunk.
"""

import threading
import time

import numpy as np

SAMPLE_RATE = 16000


class AudioSource(threading.Thread):
    """Runs on its own thread and pushes into an AudioRingBuffer. Closing the
    ring is what tells the consumer the utterance ended."""

    # Live sources drop when the consumer falls behind; a file source waits,
    # because losing part of the file makes any comparison meaningless.
    blocking_put = False

    def __init__(self, ring, log=None, name="audio-source"):
        threading.Thread.__init__(self, name=name, daemon=True)
        self.ring = ring
        self.log = log
        self._stop = threading.Event()
        self.chunks_sent = 0
        self.samples_sent = 0
        self.error = None

    def stop(self):
        self._stop.set()

    def _emit(self, samples):
        dropped = self.ring.put(samples, block=self.blocking_put)
        self.chunks_sent += 1
        self.samples_sent += samples.size
        if self.log is not None:
            self.log.frame("ingest.chunk", seq=self.chunks_sent,
                           ms=round(samples.size * 1000.0 / SAMPLE_RATE, 1),
                           level_ms=round(self.ring.level_ms, 1),
                           dropped=int(dropped))

    def run(self):
        try:
            self.produce()
        except Exception as exc:                      # noqa: BLE001
            self.error = exc
            if self.log is not None:
                self.log.event("ingest.error", error=repr(exc))
        finally:
            self.ring.close()
            if self.log is not None:
                self.log.event("ingest.done", chunks=self.chunks_sent,
                               ms=round(self.samples_sent * 1000.0 / SAMPLE_RATE, 1))

    def produce(self):
        raise NotImplementedError


class FileAudioSource(AudioSource):
    """Decode a file once, then hand it over in chunks.

    `pace`:
      fast     — push as fast as the consumer drains: put() blocks for room
                 rather than dropping, so the whole file gets through and the
                 GPU path is the only limit. Use this for the baseline diff.
                 (Without the back-pressure this is a trap: a 34s clip poured
                 into a 10s buffer loses two thirds of itself, the muxer sees
                 audio end early, and -shortest stops ffmpeg reading frames.)
      realtime — sleep chunk_ms between chunks, i.e. what a perfectly paced
                 TTS would do. Use this to see whether the pipeline keeps up.

    `jitter_ms` and `stall_at` reproduce the two ways a real TTS misbehaves:
    irregular arrival, and a mid-utterance pause while the model thinks.
    """

    blocking_put = True

    def __init__(self, ring, path, chunk_ms=320, pace="fast", jitter_ms=0.0,
                 stall_at=None, stall_ms=0, log=None):
        AudioSource.__init__(self, ring, log=log, name="file-source")
        self.path = path
        self.chunk_ms = chunk_ms
        self.pace = pace
        self.jitter_ms = jitter_ms
        self.stall_at = stall_at
        self.stall_ms = stall_ms
        self._rng = np.random.RandomState(0)

    def produce(self):
        from src.whisper.audio import load_audio

        audio = load_audio(self.path)
        chunk = int(SAMPLE_RATE * self.chunk_ms / 1000)
        if self.log is not None:
            self.log.event("ingest.start", source="file", path=self.path,
                           duration_ms=round(len(audio) * 1000.0 / SAMPLE_RATE, 1),
                           chunk_ms=self.chunk_ms, pace=self.pace)

        for i, start in enumerate(range(0, len(audio), chunk)):
            if self._stop.is_set():
                break

            if self.stall_at is not None and i == self.stall_at and self.stall_ms:
                if self.log is not None:
                    self.log.event("ingest.stall", after_chunk=i, ms=self.stall_ms)
                time.sleep(self.stall_ms / 1000.0)

            self._emit(audio[start:start + chunk])

            if self.pace == "realtime":
                delay = self.chunk_ms
                if self.jitter_ms:
                    delay += float(self._rng.uniform(-self.jitter_ms, self.jitter_ms))
                time.sleep(max(0.0, delay) / 1000.0)


class CallbackAudioSource(AudioSource):
    """Push PCM in from elsewhere — a WebSocket handler, a test, whatever.

    The M3 ws-push ingest is this plus a server loop: the handler calls feed()
    for each frame it receives and finish() when the utterance ends, so no
    JoyGen-side code has to know about the transport.
    """

    def __init__(self, ring, log=None):
        AudioSource.__init__(self, ring, log=log, name="callback-source")
        self._queue = []
        self._cond = threading.Condition()
        self._done = False

    def feed(self, samples):
        """Accepts bytes (PCM16 LE), int16 or float32, as the spec allows."""
        if isinstance(samples, (bytes, bytearray, memoryview)):
            samples = np.frombuffer(bytes(samples), dtype=np.int16)
        samples = np.asarray(samples)
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        with self._cond:
            self._queue.append(np.asarray(samples, dtype=np.float32).reshape(-1))
            self._cond.notify_all()

    def finish(self):
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def produce(self):
        if self.log is not None:
            self.log.event("ingest.start", source="callback")
        while True:
            with self._cond:
                while not self._queue and not self._done and not self._stop.is_set():
                    self._cond.wait(0.1)
                if not self._queue:
                    if self._done or self._stop.is_set():
                        return
                    continue
                pending = self._queue
                self._queue = []
            for samples in pending:
                self._emit(samples)


def build_source(cfg, ring, log=None, path=None):
    """Pick a source from configs/pipeline.yaml's joygen_input.source."""
    kind = cfg.joygen_input.source
    if kind == "file":
        return FileAudioSource(
            ring, path, chunk_ms=cfg.tts.chunk_ms,
            pace=cfg.joygen_input.get("file_pace", "fast"),
            jitter_ms=cfg.joygen_input.get("file_jitter_ms", 0.0), log=log)
    if kind == "ws-push":
        raise NotImplementedError(
            "ws-push ingest lands in M3; use source: file for now")
    raise ValueError("unknown joygen_input.source: {}".format(kind))
