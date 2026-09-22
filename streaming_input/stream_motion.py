"""
stream_motion.py — expression coefficients from audio, incrementally.

JoyGen's stage 1 (`inference_audio2motion.py`) is file-shaped: it writes a
16kHz wav next to the source audio, runs HuBERT over the whole thing, runs
parselmouth over the whole thing, and forwards the entire sequence through the
VAE in one go. Nothing can start until the last sample exists — the same
problem `stream_feature.py` solved for the whisper branch.

Why windowing works here where whisper needed anchoring
-------------------------------------------------------
The whisper encoder adds an absolute positional embedding, so the same audio
at a different offset inside the window comes out different, and M1 had to pin
every window to a fixed 30s grid. Nothing in this path does that:

  * HuBERT uses a convolutional (relative) positional embedding, and upstream
    already chops long audio into independent 20s clips
    (extract_hubert.py:66) — so a window boundary is a cost it already pays.
  * `PitchContourVAEModel` is convolutional end to end: Conv1d conditioning
    stacks, an FVAE with kernel 5 and stride 4, and a glow prior. The one
    piece of global attention in FVAE is behind `sqz_prior`, which this
    checkpoint leaves off (vae.py:289, load_audio2secc()).

So the receptive field is finite and a window with enough context on both
sides reproduces the whole-file answer.

`window_ms` is the whole span handed to the model and `hop_ms` is how often it
runs, so the left context is window_ms - hop_ms and it costs nothing in
latency. Only `hop_ms` and `right_context_ms` delay a frame. An earlier
version waited for a full window before emitting anything, which put
first-frame latency at 4.8s on a 5s clip — the window has to reach backwards,
never forwards.

Two things still make windows disagree
--------------------------------------
1. The VAE samples. FVAE draws `z_p` from the prior at T/4 resolution
   (vae.py:261) and pushes it through the flow, so two windows covering the
   same moment draw different noise. Expression coefficients are a smooth
   low-dimensional signal, so overlapping windows are crossfaded rather than
   butt-joined; `fade_ms` sets the ramp.
2. Wav2Vec2's feature extractor normalises each input to zero mean and unit
   variance, over whatever window it is handed. That is the same hazard as
   whisper's per-window `log_spec.max()`, and `audio_norm="running"` keeps a
   session-wide mean/variance instead so the reference stops moving.

Calibration: on demo/xinwen_5s.mp3, seed 0 reproduces the shipped
results/smoke_test/a2m/xinwen_5s.npy exactly, and two different seeds differ
by mean 0.061 / max 0.442 against a signal whose std is 0.347. Streaming error
below the seed-to-seed spread is, by definition, indistinguishable from having
picked a different random seed.
"""

import random
import time

import numpy as np
import torch

SAMPLE_RATE = 16000
MOTION_FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // MOTION_FPS      # 640
HUBERT_STRIDE = 320                                # HuBERT runs at 50fps
HUBERT_KERNEL = 400
X_MULTIPLY = 8                                     # hubert length padding


def hubert_frames_for(n_samples):
    """What get_hubert_from_16k_speech() will return for this many samples."""
    return max(0, (n_samples - (HUBERT_KERNEL - HUBERT_STRIDE)) // HUBERT_STRIDE)


class StreamingAudio2Motion:
    """Feed PCM in, get 64-dim expression coefficients out at 25fps.

    Frames are final once returned: a frame is only emitted when the window
    that produced it had `right_context_ms` of real audio after it, and when
    any overlap with the next window has already been faded in.
    """

    def __init__(self, a2m_ckpt, hubert_path, window_ms=4000, hop_ms=320,
                 right_context_ms=400, fade_ms=400, temperature=0.2,
                 mouth_amp=0.45, seed=0, audio_norm="window", device=None,
                 model=None):
        if audio_norm not in ("window", "running"):
            raise ValueError("audio_norm must be 'window' or 'running'")

        self.hubert_path = hubert_path
        # window_ms is the whole span handed to the model; hop_ms is how often
        # it runs. Left context is therefore window_ms - hop_ms, and it costs
        # nothing in latency — only the hop and the right context do. Waiting
        # for a full window before producing anything, which an earlier
        # version did, pushed first-frame latency to 4.8s on a 5s clip.
        self.window_samples = int(SAMPLE_RATE * window_ms / 1000)
        self.hop_samples = int(SAMPLE_RATE * hop_ms / 1000)
        self.right_frames = int(right_context_ms / 1000 * MOTION_FPS)
        self.fade_frames = max(1, int(fade_ms / 1000 * MOTION_FPS))
        self.temperature = temperature
        self.mouth_amp = mouth_amp
        self.seed = seed
        self.audio_norm = audio_norm

        # Round to whole output frames so window edges land on frame
        # boundaries; otherwise every window is offset by a fraction of a
        # frame and the crossfade smears.
        self.window_samples -= self.window_samples % SAMPLES_PER_FRAME
        self.hop_samples = max(SAMPLES_PER_FRAME,
                               self.hop_samples - self.hop_samples % SAMPLES_PER_FRAME)

        self.model = model
        if self.model is None:
            from inference_audio2motion import Audio2Motion
            holder = Audio2Motion(a2m_ckpt, inp={"a2m_ckpt": a2m_ckpt})
            self.model = holder.audio2secc_model
            self.hparams = holder.audio2secc_hparams
        self.device = device or next(self.model.parameters()).device

        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0            # absolute sample index of _buf[0]
        self._emitted = 0              # absolute frames already returned
        self._pending = np.zeros((0, 64), np.float32)   # frames held for fade
        self._pending_start = 0
        self._next_window = 0          # absolute sample where the next window ends
        self._run_sum = 0.0
        self._run_sqsum = 0.0
        self._run_n = 0

        self.windows = 0
        self.window_ms_total = 0.0

    # -- input ------------------------------------------------------

    @staticmethod
    def to_float32(samples):
        if isinstance(samples, (bytes, bytearray, memoryview)):
            samples = np.frombuffer(bytes(samples), dtype=np.int16)
        samples = np.asarray(samples)
        if samples.dtype == np.int16:
            return samples.astype(np.float32) / 32768.0
        return np.ascontiguousarray(samples, dtype=np.float32)

    def push(self, samples, final=False):
        """Append PCM, run any window that is now complete, and return the
        frames that became final. Shape (n_new, 64)."""
        chunk = self.to_float32(samples)
        if chunk.size:
            self._buf = np.concatenate([self._buf, chunk])
            self._run_sum += float(chunk.sum())
            self._run_sqsum += float(np.square(chunk, dtype=np.float64).sum())
            self._run_n += chunk.size

        out = []
        total = self._buf_start + self._buf.size
        while True:
            # Run on whatever has arrived, as soon as a hop's worth is new.
            # The window reaches back for context but never waits for it.
            if final:
                if self._next_window >= total:
                    break
                end = total
            else:
                if total - self._next_window < self.hop_samples:
                    break
                end = total - (total % SAMPLES_PER_FRAME)
                if end <= self._next_window:
                    break

            if end - self._window_start(end) < HUBERT_KERNEL:
                break
            produced = self._run_window(end, final)
            self._next_window = end
            if produced is not None and len(produced):
                out.append(produced)
            if final:
                break

        if final:
            tail = self._drain_pending(everything=True)
            if len(tail):
                out.append(tail)

        return np.concatenate(out) if out else np.zeros((0, 64), np.float32)

    def flush(self):
        return self.push(np.zeros(0, np.float32), final=True)

    # -- windows ----------------------------------------------------

    def _window_start(self, end):
        start = max(0, end - self.window_samples)
        return start - start % SAMPLES_PER_FRAME

    def _normalise(self, audio):
        """Wav2Vec2's extractor would normalise per window; doing it here with
        session statistics keeps the reference still across windows."""
        if self.audio_norm == "window" or self._run_n == 0:
            return audio
        mean = self._run_sum / self._run_n
        var = max(self._run_sqsum / self._run_n - mean * mean, 1e-10)
        return (audio - mean) / np.sqrt(var)

    def _run_window(self, end, final):
        from audio2motion.data_gen.utils.process_audio.extract_hubert import (
            get_hubert_from_16k_speech,
        )
        from audio2motion.data_gen.utils.process_audio.extract_mel_f0 import (
            extract_f0_from_wav_and_mel,
            extract_mel_from_fname,
        )

        start = self._window_start(end)
        lo = start - self._buf_start
        hi = end - self._buf_start
        audio = np.ascontiguousarray(self._buf[lo:hi], dtype=np.float32)
        if audio.size < HUBERT_KERNEL:
            return None

        t0 = time.perf_counter()
        hubert = get_hubert_from_16k_speech(
            self._normalise(audio), self.hubert_path,
            device=str(self.device)).detach().cpu().numpy()
        n_valid = hubert.shape[0]
        pad = (-n_valid) % X_MULTIPLY
        if pad:
            hubert = np.pad(hubert, ((0, pad), (0, 0)))

        # Returns (wav, mel) — the wav comes back padded to line up with the
        # mel frames, and passing the unpadded audio to the pitch tracker
        # instead makes the two disagree by more than its 8-frame tolerance.
        wav_padded, mel = extract_mel_from_fname(audio)
        f0, _ = extract_f0_from_wav_and_mel(wav_padded, mel)
        f0 = f0.reshape([-1, 1])
        if f0.shape[0] > len(hubert):
            f0 = f0[:len(hubert)]
        elif f0.shape[0] < len(hubert):
            f0 = np.pad(f0, pad_width=((0, len(hubert) - f0.shape[0]), (0, 0)))

        t_x = hubert.shape[0]
        batch = {
            "hubert": torch.from_numpy(hubert).float().unsqueeze(0).to(self.device),
            "f0": torch.from_numpy(f0).float().reshape([1, -1]).to(self.device),
            "x_mask": torch.ones([1, t_x]).float().to(self.device),
            "y_mask": torch.ones([1, t_x // 2]).float().to(self.device),
            "blink": torch.zeros([1, t_x, 1]).long().to(self.device),
            "eye_amp": torch.ones([1, 1]).to(self.device),
            "mouth_amp": torch.ones([1, 1]).to(self.device) * self.mouth_amp,
        }
        batch["audio"] = batch["hubert"]

        # Seeded per window rather than per session: a rerun has to reproduce
        # itself, and the alternative — one stream of RNG state — would make
        # every window depend on how many came before it.
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        ret = {}
        with torch.no_grad():
            self.model.forward(batch, ret=ret, train=False,
                               temperature=self.temperature)
        exp = ret["pred"][0].detach().cpu().numpy()

        self.windows += 1
        self.window_ms_total += (time.perf_counter() - t0) * 1000.0

        # Frames the padding invented are not backed by audio, so they are
        # dropped mid-stream. At the end they are kept, because that is what
        # the offline path emits and the parity target includes them.
        keep = len(exp) if final else n_valid // 2
        exp = exp[:keep]
        return self._merge(start // SAMPLES_PER_FRAME, exp, end, final)

    # -- stitching --------------------------------------------------

    def _merge(self, frame0, exp, end_sample, final):
        """Crossfade `exp` into the pending buffer, then release whatever now
        has enough audio to its right."""
        if not len(exp):
            return np.zeros((0, 64), np.float32)

        if not len(self._pending):
            self._pending_start = max(frame0, self._emitted)
            self._pending = np.zeros((0, 64), np.float32)

        pend_end = self._pending_start + len(self._pending)
        new_end = frame0 + len(exp)

        grown = np.zeros((max(pend_end, new_end) - self._pending_start, 64),
                         np.float32)
        grown[:len(self._pending)] = self._pending

        for i in range(len(exp)):
            abs_f = frame0 + i
            if abs_f < self._pending_start:
                continue           # already emitted, cannot be revised
            j = abs_f - self._pending_start
            if j < len(self._pending):
                # overlap: ramp the new window in over fade_frames
                overlap_pos = abs_f - max(frame0, self._pending_start)
                w = min(1.0, (overlap_pos + 1) / float(self.fade_frames))
                grown[j] = (1.0 - w) * grown[j] + w * exp[i]
            else:
                grown[j] = exp[i]

        self._pending = grown
        return self._drain_pending(everything=final, end_sample=end_sample)

    def _drain_pending(self, everything=False, end_sample=None):
        if not len(self._pending):
            return np.zeros((0, 64), np.float32)

        if everything:
            release = len(self._pending)
        else:
            # Hold back the right context plus the fade ramp: both can still
            # be revised by the window that has not run yet.
            hold = self.right_frames + self.fade_frames
            release = max(0, len(self._pending) - hold)
        if release <= 0:
            return np.zeros((0, 64), np.float32)

        out = self._pending[:release].copy()
        self._pending = self._pending[release:]
        self._pending_start += release
        self._emitted = self._pending_start
        self._trim()
        return out

    def _trim(self):
        keep_from = max(0, self._next_window - self.window_samples
                        - SAMPLES_PER_FRAME)
        drop = keep_from - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = keep_from

    # -- introspection ----------------------------------------------

    @property
    def frames_emitted(self):
        return self._emitted

    def stats(self):
        return {
            "frames": self._emitted,
            "windows": self.windows,
            "window_ms_mean": round(self.window_ms_total / self.windows, 1)
            if self.windows else 0.0,
            "window_ms": int(self.window_samples * 1000 / SAMPLE_RATE),
            "hop_ms": int(self.hop_samples * 1000 / SAMPLE_RATE),
            "right_context_ms": int(self.right_frames * 1000 / MOTION_FPS),
            "fade_ms": int(self.fade_frames * 1000 / MOTION_FPS),
            "audio_norm": self.audio_norm,
            "temperature": self.temperature,
        }
