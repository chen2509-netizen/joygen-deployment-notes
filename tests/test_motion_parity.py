"""
test_motion_parity.py — M4 stage 1 gate.

Compares StreamingAudio2Motion against the expression coefficients
`inference_audio2motion.py` produces for the whole file.

The target is exact: seed 0 reproduces the shipped
`results/smoke_test/a2m/xinwen_5s.npy` bit for bit, so there is a fixed thing
to be wrong about. The threshold is not arbitrary either — two different
random seeds move the coefficients by mean 0.061 / max 0.442 against a signal
whose std is 0.347, so any streaming error below that is indistinguishable
from having picked a different seed, which the model does anyway.

`--control` runs one window over the whole clip. If that does not land on top
of the reference, the harness is broken and the sweep means nothing.

Run:
    bash scripts/run_input.sh motion-parity <audio> <ref.npy>
    bash scripts/run_input.sh motion-parity <audio> <ref.npy> --control
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

from streaming_input.stream_motion import (  # noqa: E402
    SAMPLE_RATE,
    StreamingAudio2Motion,
)

A2M_CKPT = "./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae"
HUBERT_PATH = "pretrained_models/audio2motion/hubert"

WINDOW_SWEEP = [2000, 4000, 8000]
HOP_SWEEP = [320, 640, 1280]
NORM_SWEEP = ["window", "running"]

# Measured in M4 Spike B on demo/xinwen_5s.mp3 — the model's own spread.
SEED_SPREAD_MEAN = 0.061
SEED_SPREAD_MAX = 0.442

_failures = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        _failures.append(name)


def decode_pcm(path):
    """Decode without going through Audio2Motion.save_wav16k, which writes a
    _16k.wav next to the source — and the sources live under JoyGen, which is
    read-only."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
           "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def compare(ref, got):
    n = min(len(ref), len(got))
    if n == 0:
        return {"frames": 0, "max": float("nan"), "mean": float("nan"),
                "p95": float("nan"), "len_ref": len(ref), "len_got": len(got)}
    d = np.abs(ref[:n].astype(np.float64) - got[:n].astype(np.float64))
    return {
        "frames": int(n),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p95": float(np.percentile(d, 95)),
        "len_ref": int(len(ref)),
        "len_got": int(len(got)),
    }


def run_streaming(audio, chunk_samples, **kwargs):
    feeder = StreamingAudio2Motion(A2M_CKPT, HUBERT_PATH, **kwargs)
    out = []
    t0 = time.perf_counter()
    for start in range(0, len(audio), chunk_samples):
        got = feeder.push(audio[start:start + chunk_samples])
        if len(got):
            out.append(got)
    tail = feeder.flush()
    if len(tail):
        out.append(tail)
    wall = (time.perf_counter() - t0) * 1000.0

    exp = np.concatenate(out) if out else np.zeros((0, 64), np.float32)
    stats = feeder.stats()
    stats["wall_ms"] = round(wall, 1)
    return exp, stats


def main(argv=None):
    p = argparse.ArgumentParser(description="M4 stage 1: streaming audio2motion")
    p.add_argument("audio")
    p.add_argument("ref_npy")
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--window", type=int, nargs="*", default=None)
    p.add_argument("--hop", type=int, nargs="*", default=None)
    p.add_argument("--norm", nargs="*", default=None, choices=NORM_SWEEP)
    p.add_argument("--fade-ms", type=int, default=200)
    p.add_argument("--right-ms", type=int, default=400)
    p.add_argument("--control", action="store_true")
    p.add_argument("--report", default=None)
    args = p.parse_args(argv)

    ref = np.load(args.ref_npy)
    audio = decode_pcm(args.audio)
    dur = len(audio) / SAMPLE_RATE
    chunk = int(SAMPLE_RATE * args.chunk_ms / 1000)
    print("[info] audio {:.2f}s  reference {}  chunk {}ms".format(
        dur, ref.shape, args.chunk_ms))
    print("[info] 判準：seed 之間的差距 mean {} / max {}".format(
        SEED_SPREAD_MEAN, SEED_SPREAD_MAX))

    if args.control:
        combos = [(int(dur * 1000) + 2000, int(dur * 1000) + 2000, "window")]
        chunk = len(audio)
        print("[info] CONTROL：單一視窗涵蓋整段、一次 push")
    else:
        windows = args.window or WINDOW_SWEEP
        hops = args.hop or HOP_SWEEP
        norms = args.norm or NORM_SWEEP
        combos = [(w, h, n) for n in norms for w in windows for h in hops]

    header = ("  {:<9}{:>9}{:>8}{:>8}{:>10}{:>10}{:>10}{:>9}"
              .format("norm", "window", "hop", "frames", "max", "mean",
                      "p95", "win_ms"))
    print("\n[ 串流 vs 離線 audio2motion ]")
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for window_ms, hop_ms, norm in combos:
        exp, stats = run_streaming(
            audio, chunk, window_ms=window_ms, hop_ms=hop_ms,
            right_context_ms=args.right_ms, fade_ms=args.fade_ms,
            audio_norm=norm, seed=0)
        row = compare(ref, exp)
        results.append({"window_ms": window_ms, "hop_ms": hop_ms,
                        "audio_norm": norm, "diff": row, "timing": stats})
        print("  {:<9}{:>9}{:>8}{:>8}{:>10.4f}{:>10.4f}{:>10.4f}{:>9.1f}"
              .format(norm, window_ms, hop_ms, row["frames"], row["max"],
                      row["mean"], row["p95"], stats["window_ms_mean"]))

    best = min(results, key=lambda r: r["diff"]["mean"])
    d = best["diff"]
    print("\n[ 判斷點 ]")
    print("  最佳組合: norm={} window={}ms hop={}ms".format(
        best["audio_norm"], best["window_ms"], best["hop_ms"]))
    print("  max={:.4f}  mean={:.4f}  p95={:.4f}  （幀數 {} vs 參考 {}）".format(
        d["max"], d["mean"], d["p95"], d["len_got"], d["len_ref"]))
    print("  每個視窗 {:.1f}ms".format(best["timing"]["window_ms_mean"]))

    if args.control:
        check("control 貼近離線（mean < 0.01）", d["mean"] < 0.01, round(d["mean"], 5))
        check("control 幀數一致", d["len_got"] == d["len_ref"],
              (d["len_got"], d["len_ref"]))
    else:
        check("mean 低於 seed 間差距", d["mean"] < SEED_SPREAD_MEAN,
              round(d["mean"], 4))
        check("max 低於 seed 間差距", d["max"] < SEED_SPREAD_MAX,
              round(d["max"], 4))
        check("幀數與離線一致", abs(d["len_got"] - d["len_ref"]) <= 2,
              (d["len_got"], d["len_ref"]))

    path = Path(args.report) if args.report else (
        NOTES_ROOT / "timing" / "motion_parity_{}{}.json".format(
            "control_" if args.control else "",
            datetime.now().strftime("%m%d_%H%M")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"audio": args.audio, "ref": args.ref_npy,
                   "duration_s": round(dur, 3), "control": bool(args.control),
                   "seed_spread": {"mean": SEED_SPREAD_MEAN,
                                   "max": SEED_SPREAD_MAX},
                   "results": results}, f, indent=2)
    print("\n  json -> {}".format(path))
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
