"""
test_stream_feature_parity.py — M1 gate.

Compares StreamingAudio2Feature against JoyGen's whole-file
Audio2Feature.audio2feat(), sweeping the three knobs that should explain the
difference:

  left context  — how much already-emitted audio each encoder window carries
  right context — how long a row is held back so real audio, not zero padding,
                  follows it (the encoder is not causal, so rows at the window
                  edge are the least trustworthy); this is added latency
  mel_norm      — which maximum log_mel normalises against

Input streaming only works if some setting is close enough that the UNet
cannot tell the difference. The numbers here decide whisper_left_context_ms,
whisper_right_context_ms and whisper_mel_norm in configs/pipeline.yaml.

The `control` mode is the sanity check for the harness itself: one push with
context wider than the clip makes the streaming window identical to the
whole-file window, so the residual is pure fp16 noise. If control is not
near-perfect, the implementation is wrong and the sweep means nothing.

Run:
    bash scripts/run_input.sh parity [audio] [flags]
    bash scripts/run_input.sh parity --control

Reads JoyGen, writes nothing outside joygen-deployment-notes.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

from streaming_input.stream_feature import (  # noqa: E402
    SAMPLE_RATE,
    StreamingAudio2Feature,
)

DEFAULT_AUDIO = "demo/xinwen_5s.mp3"
DEFAULT_WHISPER = "pretrained_models/whisper/tiny.pt"
LEFT_SWEEP = [500, 1000, 2000, 4000]
RIGHT_SWEEP = [0, 200, 400, 800]
MEL_NORM_SWEEP = ["window", "running"]
WINDOW_MODE_SWEEP = ["anchored", "sliding"]


def compare(a, b):
    """Row-wise agreement between two (T, L, 384) feature arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return {"rows": 0, "max_abs": float("nan"), "mean_abs": float("nan"),
                "max_rel_pct": float("nan"), "cos_min": float("nan"),
                "cos_mean": float("nan"), "rows_below_0999": 0}

    a = a[:n].astype(np.float64).reshape(n, -1)
    b = b[:n].astype(np.float64).reshape(n, -1)

    diff = np.abs(a - b)
    scale = np.abs(a).max() or 1.0

    dot = (a * b).sum(axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide(dot, norms, out=np.ones_like(dot), where=norms > 0)

    return {
        "rows": int(n),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel_pct": float(diff.max() / scale * 100.0),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "rows_below_0999": int((cos < 0.999).sum()),
    }


def profile_rows(offline, streamed, buckets=10):
    """Where the disagreement sits. A few catastrophic rows and a uniformly
    mediocre stream need completely different fixes, and the aggregate cosine
    cannot tell them apart. Row norm is reported alongside because cosine is
    meaningless on near-silent rows, whose vectors are tiny."""
    n = min(len(offline), len(streamed))
    a = offline[:n].astype(np.float64).reshape(n, -1)
    b = streamed[:n].astype(np.float64).reshape(n, -1)

    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    denom = norms_a * norms_b
    cos = np.divide((a * b).sum(axis=1), denom,
                    out=np.ones(n), where=denom > 0)

    edges = np.linspace(0, n, buckets + 1).astype(int)
    by_bucket = []
    for i in range(buckets):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        by_bucket.append({
            "from_s": round(lo * 0.02, 2),
            "to_s": round(hi * 0.02, 2),
            "cos_mean": float(cos[lo:hi].mean()),
            "cos_min": float(cos[lo:hi].min()),
            "norm_mean": float(norms_a[lo:hi].mean()),
        })

    worst = np.argsort(cos)[:10]
    return {
        "buckets": by_bucket,
        "worst_rows": [{"row": int(i), "t_s": round(i * 0.02, 2),
                        "cos": float(cos[i]), "norm": float(norms_a[i])}
                       for i in worst],
        "cos_p50": float(np.percentile(cos, 50)),
        "cos_p05": float(np.percentile(cos, 5)),
        "rows_below_09": int((cos < 0.9).sum()),
        "quiet_rows_below_09": int(((cos < 0.9) &
                                    (norms_a < np.median(norms_a))).sum()),
    }


def build_chunks(processor, features, fps):
    """What actually reaches the UNet: the 50x384 slices feature2chunks()
    would hand the model, built over the rows we have."""
    chunks = []
    i = 0
    while True:
        if int(i * 50.0 / fps) > len(features):
            break
        sliced, _ = processor.get_sliced_feature(
            feature_array=features, vid_idx=i, audio_feat_length=[2, 2], fps=fps)
        chunks.append(sliced)
        i += 1
    return np.asarray(chunks)


def run_streaming(processor, audio, left_ms, right_ms, mel_norm, fp16,
                  chunk_samples, window_mode):
    feeder = StreamingAudio2Feature(
        processor, left_context_ms=left_ms, right_context_ms=right_ms,
        mel_norm=mel_norm, fp16=fp16, window_mode=window_mode)

    collected = []
    t0 = time.perf_counter()
    for start in range(0, len(audio), chunk_samples):
        rows = feeder.push(audio[start:start + chunk_samples])
        if len(rows):
            collected.append(rows)
    tail = feeder.flush()
    if len(tail):
        collected.append(tail)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    features = np.concatenate(collected) if collected else feeder._empty()
    stats = feeder.stats()
    stats["wall_ms"] = round(wall_ms, 1)
    return features, stats


def print_table(title, results, key):
    header = ("  {:<10}{:<10}{:>8}{:>8}{:>7}{:>11}{:>11}{:>10}{:>10}{:>9}"
              .format("window", "mel_norm", "left", "right", "rows", "max_abs",
                      "max_rel%", "cos_min", "cos_mean", "enc_ms"))
    print("\n[ {} ]".format(title))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        m = r[key]
        print("  {:<10}{:<10}{:>8}{:>8}{:>7}{:>11.4f}{:>11.3f}{:>10.5f}{:>10.5f}{:>9.1f}"
              .format(r["window_mode"], r["mel_norm"], r["left_context_ms"],
                      r["right_context_ms"], m["rows"], m["max_abs"],
                      m["max_rel_pct"], m["cos_min"], m["cos_mean"],
                      r["timing"]["encode_ms_mean"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="M1: streaming vs whole-file whisper features")
    parser.add_argument("audio", nargs="?", default=DEFAULT_AUDIO,
                        help="audio path, relative to the JoyGen checkout")
    parser.add_argument("--whisper_model_path", default=DEFAULT_WHISPER)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--chunk-ms", type=int, default=320,
                        help="simulated TTS chunk size; 320ms = one UNet batch")
    parser.add_argument("--fp32", action="store_true",
                        help="run the encoder in fp32; audio2feat() uses fp16")
    parser.add_argument("--left-context", type=int, nargs="*", default=None)
    parser.add_argument("--right-context", type=int, nargs="*", default=None)
    parser.add_argument("--mel-norm", nargs="*", default=None,
                        choices=["window", "running"])
    parser.add_argument("--window-mode", nargs="*", default=None,
                        choices=["anchored", "sliding"],
                        help="anchored: window starts on the same 30s grid "
                             "audio2feat() uses, so rows keep their absolute "
                             "position in the encoder. sliding: window follows "
                             "the write head.")
    parser.add_argument("--profile", action="store_true",
                        help="per-position error profile, to tell a few bad "
                             "rows apart from a uniformly bad stream")
    parser.add_argument("--control", action="store_true",
                        help="harness sanity check: one push, context wider "
                             "than the clip, so the window matches the "
                             "whole-file window exactly")
    parser.add_argument("--report", default=None,
                        help="output json; defaults to <notes>/timing/parity_<ts>.json")
    args = parser.parse_args(argv)

    from src.audio2feature import Audio2Feature
    from src.whisper.audio import load_audio

    if not os.path.exists(args.audio):
        sys.exit("[error] audio not found: {} (cwd={})".format(args.audio, os.getcwd()))

    print("[info] loading whisper: {}".format(args.whisper_model_path))
    processor = Audio2Feature(model_path=args.whisper_model_path)

    audio = load_audio(args.audio)
    duration_s = len(audio) / SAMPLE_RATE
    print("[info] audio: {} ({:.2f}s, {} samples)".format(
        args.audio, duration_s, len(audio)))

    t0 = time.perf_counter()
    offline = processor.audio2feat(args.audio)
    offline_ms = (time.perf_counter() - t0) * 1000.0
    offline_chunks = build_chunks(processor, offline, args.fps)
    print("[info] offline: {} rows, {} chunks, {:.0f}ms".format(
        len(offline), len(offline_chunks), offline_ms))

    if args.control:
        lefts = [int(duration_s * 1000) + 5000]
        rights = [0]
        norms = ["window"]
        chunk_ms = int(duration_s * 1000) + 1000
        print("[info] CONTROL: single push, left_context={}ms".format(lefts[0]))
    else:
        lefts = args.left_context if args.left_context is not None else LEFT_SWEEP
        rights = args.right_context if args.right_context is not None else RIGHT_SWEEP
        norms = args.mel_norm if args.mel_norm is not None else MEL_NORM_SWEEP
        chunk_ms = args.chunk_ms

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    print("[info] chunk = {}ms, {} combination(s)\n".format(
        chunk_ms, len(norms) * len(lefts) * len(rights)))

    modes = args.window_mode if args.window_mode is not None else WINDOW_MODE_SWEEP
    if args.control:
        modes = ["sliding"]

    combos = [(w, m, l, r)
              for w in modes for m in norms for l in lefts for r in rights]

    results = []
    for window_mode, mel_norm, left_ms, right_ms in combos:
        features, stats = run_streaming(
            processor, audio, left_ms, right_ms, mel_norm,
            not args.fp32, chunk_samples, window_mode)
        entry = {
            "window_mode": window_mode,
            "mel_norm": mel_norm,
            "left_context_ms": left_ms,
            "right_context_ms": right_ms,
            "features": compare(offline, features),
            "unet_chunks": compare(offline_chunks,
                                   build_chunks(processor, features, args.fps)),
            "timing": stats,
        }
        if args.profile:
            entry["profile"] = profile_rows(offline, features)
        results.append(entry)

    print_table("feature rows vs audio2feat()", results, "features")
    print_table("same settings, on the 50x384 chunks the UNet sees",
                results, "unet_chunks")

    best = max(results, key=lambda r: r["unet_chunks"]["cos_mean"])
    enc = best["timing"]["encode_ms_mean"]
    print("\n[ 判斷點 ]")
    print("  最佳組合: window={} mel_norm={} left={}ms right={}ms".format(
        best["window_mode"], best["mel_norm"], best["left_context_ms"],
        best["right_context_ms"]))
    print("  UNet chunk: cos_mean={:.5f} cos_min={:.5f} 最大相對誤差={:.3f}%".format(
        best["unet_chunks"]["cos_mean"], best["unet_chunks"]["cos_min"],
        best["unet_chunks"]["max_rel_pct"]))
    print("  encoder 每次呼叫 {:.1f}ms / {}ms 音訊 ({:.1f}% 佔用)".format(
        enc, args.chunk_ms, enc / args.chunk_ms * 100.0))
    print("  額外延遲（右側 context）: {}ms".format(best["right_context_ms"]))

    if args.profile and "profile" in best:
        p = best["profile"]
        print("\n[ 最佳組合的逐位置誤差 ]")
        print("  {:>8}{:>10}{:>10}{:>12}".format("區段(s)", "cos_mean", "cos_min", "norm_mean"))
        for b in p["buckets"]:
            print("  {:>8}{:>10.5f}{:>10.5f}{:>12.1f}".format(
                "{:.0f}-{:.0f}".format(b["from_s"], b["to_s"]),
                b["cos_mean"], b["cos_min"], b["norm_mean"]))
        print("  中位數 cos={:.5f}  第5百分位={:.5f}".format(p["cos_p50"], p["cos_p05"]))
        print("  cos<0.9 的 row: {} / {}（其中 {} 是低能量 row）".format(
            p["rows_below_09"], best["features"]["rows"], p["quiet_rows_below_09"]))
        print("  最差 10 個 row: " + ", ".join(
            "{:.1f}s(cos={:.2f},norm={:.0f})".format(w["t_s"], w["cos"], w["norm"])
            for w in p["worst_rows"][:5]))
    if enc > args.chunk_ms:
        print("  ⚠ encoder 比音訊到達還慢，串流跟不上即時")
    if args.control and best["unet_chunks"]["cos_mean"] < 0.999:
        print("  ⚠ control 沒有貼近 1.0 —— 實作有問題，sweep 的數字不能採信")

    report_path = Path(args.report) if args.report else (
        NOTES_ROOT / "timing" /
        "parity_{}{}.json".format("control_" if args.control else "",
                                  datetime.now().strftime("%m%d_%H%M")))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "audio": args.audio,
            "duration_s": round(duration_s, 3),
            "chunk_ms": chunk_ms,
            "fps": args.fps,
            "fp16": not args.fp32,
            "control": bool(args.control),
            "offline_rows": int(len(offline)),
            "offline_ms": round(offline_ms, 1),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print("\n  json -> {}\n".format(report_path))
    return results


if __name__ == "__main__":
    main()
