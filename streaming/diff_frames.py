"""
diff_frames.py — 逐張 PNG 比對，驗證 Step 2 輸出跟 baseline 一致

用法：
    python -m streaming.diff_frames <baseline_dir> <step2_dir>

判定：所有對應檔案 pixel-exact 相同 → PASS，否則列出差異最大的前幾張。

位置：JoyGen/streaming/diff_frames.py
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def compare(baseline_dir, step2_dir, tol=0):
    b = Path(baseline_dir)
    s = Path(step2_dir)

    b_files = sorted(b.glob("*_edit.png"),
                     key=lambda p: int(p.stem.split("_")[0]))
    s_files = sorted(s.glob("*_edit.png"),
                     key=lambda p: int(p.stem.split("_")[0]))

    if not b_files:
        sys.exit(f"[error] baseline_dir 沒有 *_edit.png：{b}")
    if not s_files:
        sys.exit(f"[error] step2_dir 沒有 *_edit.png：{s}")

    n = min(len(b_files), len(s_files))
    if len(b_files) != len(s_files):
        print(f"[warn] frame 數不同：baseline={len(b_files)} step2={len(s_files)}，"
              f"只比對前 {n} 張")

    max_diffs = []   # (idx, max_pixel_diff, mean_pixel_diff)
    n_exact = 0

    for i in range(n):
        bp, sp = b_files[i], s_files[i]
        # 檔名應該一致（都是 {idx}_edit.png）
        if bp.name != sp.name:
            print(f"[warn] 檔名對不上：{bp.name} vs {sp.name}（照順序繼續比）")

        bi = cv2.imread(str(bp))
        si = cv2.imread(str(sp))
        if bi is None or si is None:
            print(f"[error] 讀不到：{bp} 或 {sp}")
            continue
        if bi.shape != si.shape:
            print(f"[error] shape 不同：{bp.name} baseline={bi.shape} step2={si.shape}")
            continue

        diff = cv2.absdiff(bi, si)
        mx, mn = int(diff.max()), float(diff.mean())
        if mx <= tol:
            n_exact += 1
        max_diffs.append((bp.name, mx, mn))

    max_diffs.sort(key=lambda x: -x[1])
    print(f"\n=== 比對結果 ===")
    print(f"  比對張數：{n}")
    print(f"  完全相同（tol={tol}）：{n_exact} / {n}")

    if n_exact == n:
        print("  ✅ PASS：所有 frame pixel-exact 相同")
        return 0

    print("\n  ❌ FAIL：差異最大的前 5 張")
    print(f"    {'name':<20}{'max_diff':>10}{'mean_diff':>12}")
    for name, mx, mn in max_diffs[:5]:
        print(f"    {name:<20}{mx:>10}{mn:>12.3f}")
    print("\n  提示：max_diff=1~2 通常是 cv2.imwrite JPEG/PNG 壓縮或 float→uint8 rounding，")
    print("        可用 --tol 2 放寬容忍度；差異很大則代表 blend 邏輯真的動到了。")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline_dir")
    p.add_argument("step2_dir")
    p.add_argument("--tol", type=int, default=0,
                   help="每個像素允許的最大差異值（預設 0，pixel-exact）")
    args = p.parse_args()
    sys.exit(compare(args.baseline_dir, args.step2_dir, tol=args.tol))


if __name__ == "__main__":
    main()
