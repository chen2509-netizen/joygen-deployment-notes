"""
compare_frames.py — perceptual comparison between two frame directories.

streaming/diff_frames.py answers "is the blend path bit-identical", which is
the right question when only the plumbing changed. Input streaming changes the
features themselves, so the frames are expected to differ and that tool can
only ever say FAIL.

What matters instead is how much they differ, and where. Almost the whole
frame is copied from the source video untouched, so a full-frame average is
diluted to meaninglessness by the static background — JoyGen only rewrites the
face crop, so the numbers are reported over the crop box too.

Usage:
    python tests/compare_frames.py <dir_a> <dir_b> [--boxes <pose_path>] [--json out]

`pose_path` is the intermediate directory holding <i>_box.npy, i.e.
<intermediate_dir>/<video>/<audio>/<video>.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

NOTES_ROOT = Path(__file__).resolve().parent.parent


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def frame_index(name):
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else None


def source_index(i, n_frames, loop_mode):
    """Same mapping AvatarCache uses. Without it, any frame past the end of the
    avatar gets measured against the wrong crop box — and with a 5s avatar
    under 34s of audio that is most of the run."""
    if n_frames <= 0 or i < n_frames:
        return i
    if loop_mode == "restart":
        return i % n_frames
    if n_frames == 1:
        return 0
    period = 2 * n_frames - 2
    j = i % period
    return j if j < n_frames else period - j


def load_boxes(pose_path):
    if not pose_path:
        return None
    boxes = {}
    for f in os.listdir(pose_path):
        if f.endswith("_box.npy"):
            boxes[int(f.split("_")[0])] = np.load(os.path.join(pose_path, f))
    return boxes or None


def compare_dirs(dir_a, dir_b, boxes=None, loop_frames=0,
                 loop_mode="pingpong"):
    names_a = {f for f in os.listdir(dir_a) if f.endswith(".png")}
    names_b = {f for f in os.listdir(dir_b) if f.endswith(".png")}
    shared = sorted(names_a & names_b, key=lambda n: frame_index(n) or 0)
    if not shared:
        raise SystemExit("[error] no PNG filenames in common")

    rows = []
    for name in shared:
        a = cv2.imread(os.path.join(dir_a, name))
        b = cv2.imread(os.path.join(dir_b, name))
        if a is None or b is None or a.shape != b.shape:
            raise SystemExit("[error] unreadable or mismatched: {}".format(name))

        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        row = {
            "name": name,
            "psnr": psnr(a, b),
            "mean_abs": float(diff.mean()),
            "max_abs": int(diff.max()),
            "pixels_over_8": int((diff.max(axis=2) > 8).sum()),
        }

        # The PNGs are 1-based (i+1_edit.png); the boxes are 0-based.
        idx = frame_index(name)
        if boxes and idx is not None:
            box_idx = source_index(idx - 1, loop_frames, loop_mode)
        if boxes and idx is not None and box_idx in boxes:
            x1, y1, x2, y2 = [int(v) for v in boxes[box_idx]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(a.shape[1], x2), min(a.shape[0], y2)
            if x2 > x1 and y2 > y1:
                ca, cb = a[y1:y2, x1:x2], b[y1:y2, x1:x2]
                cd = np.abs(ca.astype(np.int16) - cb.astype(np.int16))
                row["face_psnr"] = psnr(ca, cb)
                row["face_mean_abs"] = float(cd.mean())
                row["face_max_abs"] = int(cd.max())
        rows.append(row)
    return rows


def summarise(rows, key):
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    if not vals:
        return None
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p05": float(np.percentile(vals, 5)),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="perceptual frame comparison")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--boxes", default=None,
                   help="intermediate pose_path holding <i>_box.npy")
    p.add_argument("--loop_frames", type=int, default=0,
                   help="avatar frame count, so looped frames map back to the "
                        "right crop box (0 = no looping)")
    p.add_argument("--loop_mode", default="pingpong",
                   choices=["pingpong", "restart"])
    p.add_argument("--json", default=None)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)

    boxes = load_boxes(args.boxes)
    rows = compare_dirs(args.dir_a, args.dir_b, boxes,
                        args.loop_frames, args.loop_mode)

    print("\n=== {} vs {} ===".format(args.dir_a, args.dir_b))
    print("  比對張數：{}{}".format(
        len(rows), "" if boxes else "（沒給 --boxes，只有全畫面數字）"))

    for label, key in (("全畫面 PSNR (dB)", "psnr"),
                       ("臉部框 PSNR (dB)", "face_psnr"),
                       ("全畫面 mean|diff|", "mean_abs"),
                       ("臉部框 mean|diff|", "face_mean_abs")):
        s = summarise(rows, key)
        if s:
            print("  {:<20} mean={:8.3f}  worst={:8.3f}  p05={:8.3f}".format(
                label, s["mean"], s["min"] if "PSNR" in label else s["max"],
                s["p05"]))

    order = sorted(rows, key=lambda r: r.get("face_psnr", r["psnr"]))
    print("\n  最差 {} 張（依臉部 PSNR）：".format(args.top))
    print("    {:<18}{:>10}{:>10}{:>11}{:>10}".format(
        "name", "psnr", "face", "face_mean", "max"))
    for r in order[:args.top]:
        print("    {:<18}{:>10.2f}{:>10.2f}{:>11.3f}{:>10}".format(
            r["name"], r["psnr"], r.get("face_psnr", float("nan")),
            r.get("face_mean_abs", float("nan")), r["max_abs"]))

    identical = sum(1 for r in rows if r["max_abs"] == 0)
    print("\n  完全相同：{} / {}".format(identical, len(rows)))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({"dir_a": args.dir_a, "dir_b": args.dir_b,
                       "frames": rows,
                       "summary": {k: summarise(rows, k) for k in
                                   ("psnr", "face_psnr", "mean_abs",
                                    "face_mean_abs")}}, f, indent=2)
        print("  json -> {}".format(out))
    return rows


if __name__ == "__main__":
    main()
