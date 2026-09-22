"""
stream_feature.py — sliding-window replacement for Audio2Feature.audio2feat().

JoyGen's audio feature path is file-based: audio2feat() calls
model.transcribe(path), which mel-spectrograms the whole file up front
(src/audio2feature.py:114, src/whisper/transcribe.py:102). Nothing can start
until the last sample exists, which is exactly what input streaming has to
remove.

What that transcribe() actually does is encoder-only — it never decodes text.
It walks the mel in 30s windows and keeps every layer's activations
(transcribe.py:125-134). So the same features can be produced incrementally
from here, by running the same encoder over a window that slides with the
audio. No JoyGen source is modified; this module only imports it.

Three things could make streaming features differ from whole-file features.
M1 measured all three; the ranking was not what it looked like up front.

  1. Absolute position (dominant). The encoder adds a fixed positional
     embedding (model.py:172), so the same audio placed at a different offset
     inside the window comes out different. audio2feat() walks 30s windows
     from zero, so `window_mode="anchored"` starts every window on that same
     grid and each row keeps the position index it would have had offline.
     A sliding window loses this and degrades the moment it leaves zero: on a
     34s clip, sliding reached cos 0.88 against 0.97 for anchored, and the
     error profile collapsed after the window detached from the start.

  2. Missing right context. The encoder is not causal, so rows at the window's
     right edge are followed by zero padding rather than the audio that
     actually comes next. `right_context_ms` holds those rows back until real
     audio sits to their right: 0 -> cos 0.911, 320 -> 0.971, 400 -> 0.974,
     800 -> 0.982. This is latency traded for accuracy.

  3. log_mel normalisation. log_mel_spectrogram normalises against
     log_spec.max() of whatever it was handed (audio.py:140), which differs
     per window. mel_norm="running" keeps a session-wide reference instead.
     Measured difference: negligible either way, so the default is the simpler
     "window".

`left_context_ms` only applies to window_mode="sliding"; anchored ignores it,
and the sweep confirmed identical output for 500ms and 16000ms.

Numbers: tests/test_stream_feature_parity.py and
docs/local_records/M1 whisper 串流特徵對拍 - 260918.md
"""

import time

import numpy as np
import torch

from src.whisper.audio import HOP_LENGTH, N_FFT, N_MELS, mel_filters, pad_or_trim

SAMPLE_RATE = 16000

# The encoder halves the mel rate: conv2 has stride 2, so one output row
# covers two mel frames = 320 samples = 20ms. This is the "audio idx in
# 50FPS" that feature2chunks() indexes against.
MEL_PER_STEP = 2
SAMPLES_PER_STEP = HOP_LENGTH * MEL_PER_STEP  # 320
STEP_MS = SAMPLES_PER_STEP * 1000 // SAMPLE_RATE  # 20

# Whisper's fixed context: 3000 mel frames -> 1500 encoder rows -> 30s.
MAX_STEPS = 1500


def log_mel_raw(audio):
    """The first half of whisper.audio.log_mel_spectrogram: everything up to
    the normalisation. Split out so the reference maximum can be chosen by the
    caller without paying for a second STFT."""
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    filters = mel_filters(audio.device, N_MELS)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    return log_spec, float(log_spec.max())


def normalize_mel(log_spec, reference):
    """The second half: clamp to 80dB below `reference`, then scale. Upstream
    always passes log_spec.max(); streaming may pass a session-wide value."""
    floor = torch.tensor(reference - 8.0, device=log_spec.device,
                         dtype=log_spec.dtype)
    return (torch.maximum(log_spec, floor) + 4.0) / 4.0


def log_mel(audio, ref_max=None):
    """Drop-in for whisper.audio.log_mel_spectrogram. ref_max=None reproduces
    it exactly. Returns (mel, raw_max)."""
    log_spec, raw_max = log_mel_raw(audio)
    return normalize_mel(log_spec, raw_max if ref_max is None else ref_max), raw_max


class OfflineFeatures:
    """StreamingAudio2Feature's interface, backed by a whole-file audio2feat().

    Everything downstream — ring buffer, batching, blending, encoding — stays
    identical, so a run with this and a run with the real thing differ in
    exactly one variable. That is the only way to attribute a picture
    difference to the streaming features rather than to the rest of the
    pipeline, and it is how the long-audio comparison is done: there is no
    offline baseline for a 34s clip, because building one needs audio2motion
    and edit_expression, which are stages input streaming does not run.
    """

    def __init__(self, audio_processor, audio_path):
        self._rows = audio_processor.audio2feat(audio_path)
        self._emitted = 0
        self._samples = 0
        self.encode_calls = 0
        self.encode_ms = 0.0

    def push(self, samples):
        self._samples += np.asarray(samples).size
        upto = min(len(self._rows), self._samples // SAMPLES_PER_STEP)
        out = self._rows[self._emitted:upto]
        self._emitted = upto
        return out

    def flush(self):
        out = self._rows[self._emitted:]
        self._emitted = len(self._rows)
        return out

    @property
    def rows_emitted(self):
        return self._emitted

    def stats(self):
        return {"rows": self._emitted, "source": "offline_audio2feat",
                "encode_calls": 0, "encode_ms_mean": 0.0}


class StreamingAudio2Feature:
    """Feed PCM in, get whisper encoder rows out, at 50 rows/second.

    Rows are final once returned: push() only emits a step when every sample
    it covers has arrived, so the caller can append them to a growing feature
    array exactly as audio2feat() would have produced it.
    """

    def __init__(self, audio_processor, left_context_ms=2000,
                 right_context_ms=400, mel_norm="window", fp16=True,
                 device=None, max_window_ms=30000, window_mode="anchored"):
        if mel_norm not in ("window", "running"):
            raise ValueError("mel_norm must be 'window' or 'running'")
        if window_mode not in ("sliding", "anchored"):
            raise ValueError("window_mode must be 'sliding' or 'anchored'")
        self.window_mode = window_mode

        self.model = audio_processor.model
        self.device = device or next(self.model.parameters()).device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.mel_norm = mel_norm

        self.left_ctx_steps = max(0, int(left_context_ms) // STEP_MS)
        # Rows at the right edge of the window are followed by zero padding
        # rather than by the audio that actually comes next, and the encoder
        # is not causal, so those rows are the least like their whole-file
        # counterparts. Holding them back until real audio sits to their right
        # trades latency for accuracy.
        self.right_ctx_steps = max(0, int(right_context_ms) // STEP_MS)
        self.max_steps = min(MAX_STEPS, int(max_window_ms) // STEP_MS)
        if self.left_ctx_steps >= self.max_steps:
            raise ValueError("left_context_ms must stay under max_window_ms")

        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0     # absolute sample index of _buf[0]
        self._emitted = 0       # absolute steps already returned
        self._running_max = None

        self.encode_calls = 0
        self.encode_ms = 0.0

    # -- input ------------------------------------------------------

    @staticmethod
    def to_float32(samples):
        """Accept bytes / int16 / float32 and normalise to what whisper's
        load_audio() produces, so streaming and file input agree bit for bit."""
        if isinstance(samples, (bytes, bytearray, memoryview)):
            samples = np.frombuffer(bytes(samples), dtype=np.int16)
        samples = np.asarray(samples)
        if samples.dtype == np.int16:
            return samples.astype(np.float32) / 32768.0
        return np.ascontiguousarray(samples, dtype=np.float32)

    def push(self, samples):
        """Append PCM and return whatever rows became final, shape
        (n_new, n_layers, 384). Returns an empty array when the new audio did
        not complete another 20ms step."""
        chunk = self.to_float32(samples)
        if chunk.size:
            self._buf = np.concatenate([self._buf, chunk])

        total_steps = (self._buf_start + self._buf.size) // SAMPLES_PER_STEP
        emit_limit = total_steps - self.right_ctx_steps
        if emit_limit <= self._emitted:
            return self._empty()

        if self.window_mode == "anchored":
            # The encoder adds an absolute positional embedding, so the same
            # audio at a different offset inside the window produces different
            # activations. Whole-file extraction walks 30s windows from zero;
            # anchoring to that same grid puts every row at the same position
            # index it would have had offline, which is what parity depends on
            # — far more than how much context the window holds.
            win_start_step = (self._emitted // self.max_steps) * self.max_steps
            win_end_step = min(total_steps, win_start_step + self.max_steps)
            emit_limit = min(emit_limit, win_start_step + self.max_steps)
        else:
            win_start_step = max(0, self._emitted - self.left_ctx_steps)
            if total_steps - win_start_step > self.max_steps:
                win_start_step = total_steps - self.max_steps
            win_end_step = total_steps

        if emit_limit <= self._emitted:
            return self._empty()

        win_start_sample = win_start_step * SAMPLES_PER_STEP
        if win_start_sample < self._buf_start:
            win_start_sample = self._buf_start
            win_start_step = win_start_sample // SAMPLES_PER_STEP

        lo = win_start_sample - self._buf_start
        hi = win_end_step * SAMPLES_PER_STEP - self._buf_start
        rows = self._encode(self._buf[lo:hi])

        out = rows[self._emitted - win_start_step: emit_limit - win_start_step]
        self._emitted = emit_limit
        self._trim()
        return out

    def flush(self):
        """End of stream: release the rows held back for right context, and
        zero-pad a trailing partial step so the last fragment still produces a
        row. audio2feat() drops that fragment, so parity tests ignore it."""
        remainder = (self._buf_start + self._buf.size) % SAMPLES_PER_STEP
        pad = np.zeros(SAMPLES_PER_STEP - remainder, dtype=np.float32) \
            if remainder else np.zeros(0, dtype=np.float32)

        held = self.right_ctx_steps
        self.right_ctx_steps = 0
        try:
            return self.push(pad)
        finally:
            self.right_ctx_steps = held

    # -- internals --------------------------------------------------

    def _empty(self):
        return np.zeros((0, self.model.dims.n_audio_layer + 1, 384), np.float32)

    def _encode(self, window):
        t0 = time.perf_counter()
        audio = torch.from_numpy(np.ascontiguousarray(window)).to(self.device)

        log_spec, raw_max = log_mel_raw(audio)
        if self.mel_norm == "running":
            # Fold this window's max into the session reference before using
            # it, so a window that contains the loudest audio so far still
            # normalises against itself, as the whole-file path would.
            self._running_max = raw_max if self._running_max is None \
                else max(self._running_max, raw_max)
            mel = normalize_mel(log_spec, self._running_max)
        else:
            mel = normalize_mel(log_spec, raw_max)

        segment = pad_or_trim(mel, MAX_STEPS * MEL_PER_STEP)
        segment = segment.unsqueeze(0).to(self.dtype)

        with torch.no_grad():
            _, embeddings = self.model.encoder(segment, include_embeddings=True)

        # (1, n_layers, 1500, 384) -> (1500, n_layers, 384), matching
        # Audio2Feature.audio2feat()'s layout.
        rows = embeddings.transpose(0, 2, 1, 3)[0]

        self.encode_calls += 1
        self.encode_ms += (time.perf_counter() - t0) * 1000.0
        return rows[: window.size // SAMPLES_PER_STEP]

    def _trim(self):
        """Drop audio no future window can reach back to."""
        if self.window_mode == "anchored":
            keep_step = (self._emitted // self.max_steps) * self.max_steps
        else:
            keep_step = max(0, self._emitted - self.left_ctx_steps)
        keep_from = keep_step * SAMPLES_PER_STEP
        drop = keep_from - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = keep_from

    # -- introspection ----------------------------------------------

    @property
    def rows_emitted(self):
        return self._emitted

    def stats(self):
        return {
            "rows": self._emitted,
            "encode_calls": self.encode_calls,
            "encode_ms_total": round(self.encode_ms, 1),
            "encode_ms_mean": round(self.encode_ms / self.encode_calls, 2)
            if self.encode_calls else 0.0,
            "window_mode": self.window_mode,
            "left_context_ms": self.left_ctx_steps * STEP_MS,
            "right_context_ms": self.right_ctx_steps * STEP_MS,
            "mel_norm": self.mel_norm,
            "fp16": self.dtype == torch.float16,
        }
