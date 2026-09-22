"""
test_depth_cache_parity.py — M4 gate for the pose-driven half.

Renders every frame's depth map from the cached 3DMM fit plus the offline
expression coefficients, and compares against the `<i>_depth_edit_exp.jpg`
that `inference_edit_expression.py` produced for the same audio. If the two
agree, the fit really can be hoisted out of the per-frame loop, which is the
whole premise of streaming pose-driven.

The reference is a JPEG, so exact equality is not on offer: lossy compression
puts noise everywhere, and the depth map has a hard silhouette where a single
step of ringing costs 150+ grey levels. What the test therefore checks is the
shape of the disagreement — tiny away from the outline, concentrated on it —
rather than a single max.

Run:
    bash scripts/run_input.sh depth-parity
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

_failures = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        _failures.append(name)


def main(argv=None):
    p = argparse.ArgumentParser(description="M4: cached-fit depth vs offline")
    p.add_argument("--cache", default=str(NOTES_ROOT / "cache" / "avatar3d" /
                                          "example_5s.npz"))
    p.add_argument("--pose_dir", required=True,
                   help="intermediate dir with <i>_depth_edit_exp.jpg/_box.npy")
    p.add_argument("--exp_npy", required=True,
                   help="offline expression coefficients for the same audio")
    p.add_argument("--frames", type=int, default=0, help="0 = all")
    p.add_argument("--json", default=str(NOTES_ROOT / "timing" /
                                         "m4_depth_parity.json"))
    args = p.parse_args(argv)

    from streaming_input.avatar3d import Depth3DRenderer

    exp_all = np.load(args.exp_npy)
    renderer = Depth3DRenderer(args.cache)
    n = args.frames or min(renderer.n_frames, len(exp_all))
    print("[info] cache={} frames  exp={}  comparing {}".format(
        renderer.n_frames, exp_all.shape, n))

    rows = []
    render_ms = []
    for i in range(n):
        ref_path = os.path.join(args.pose_dir, "{}_depth_edit_exp.jpg".format(i))
        if not os.path.exists(ref_path):
            continue
        box = np.load(os.path.join(args.pose_dir, "{}_box.npy".format(i)))

        t0 = time.perf_counter()
        full, lm_full = renderer.render(i, exp_all[i])
        render_ms.append((time.perf_counter() - t0) * 1000.0)

        x1, y1, x2, y2 = [int(v) for v in box]
        got = np.array(full.crop((x1, y1, x2, y2)).convert("L"), np.int16)
        ref = np.array(Image.open(ref_path).convert("L"), np.int16)
        if got.shape != ref.shape:
            check("frame {} shape".format(i), False,
                  "{} vs {}".format(got.shape, ref.shape))
            continue

        d = np.abs(got - ref)
        gy, gx = np.gradient(ref.astype(np.float32))
        edge = (np.abs(gy) + np.abs(gx)) > 20
        big = d > 32

        lm_ref = np.load(os.path.join(args.pose_dir, "{}_lm.npy".format(i)))
        lm_got = lm_full - np.expand_dims(np.tile(np.array([x1, y1]), (468, 1)),
                                          axis=0)

        rows.append({
            "frame": i,
            "mean": float(d.mean()),
            "max": int(d.max()),
            "pct_over_8": float((d > 8).sum() / d.size * 100.0),
            "pct_over_32": float(big.sum() / d.size * 100.0),
            "edge_share_of_big": float(100.0 * (big & edge).sum() / max(big.sum(), 1)),
            "mean_off_edge": float(d[~edge].mean()),
            "lm_max": float(np.abs(lm_got[0] - lm_ref).max()),
        })

    if not rows:
        sys.exit("[error] nothing compared — check --pose_dir")

    def agg(key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)), float(np.max(vals))

    mean_mean, mean_worst = agg("mean")
    off_mean, off_worst = agg("mean_off_edge")
    o8_mean, o8_worst = agg("pct_over_8")
    o32_mean, o32_worst = agg("pct_over_32")
    edge_share = float(np.mean([r["edge_share_of_big"] for r in rows]))
    lm_mean, lm_worst = agg("lm_max")

    print("\n[ {} 幀對照 ]".format(len(rows)))
    print("  {:<28}{:>10}{:>10}".format("指標", "平均", "最差"))
    print("  " + "-" * 46)
    print("  {:<28}{:>10.4f}{:>10.4f}".format("mean|diff| (全圖)", mean_mean, mean_worst))
    print("  {:<28}{:>10.4f}{:>10.4f}".format("mean|diff| (非輪廓)", off_mean, off_worst))
    print("  {:<28}{:>10.3f}{:>10.3f}".format("|diff|>8  的像素 %", o8_mean, o8_worst))
    print("  {:<28}{:>10.3f}{:>10.3f}".format("|diff|>32 的像素 %", o32_mean, o32_worst))
    print("  {:<28}{:>10.4f}{:>10.4f}".format("lm468 max|diff| (px)", lm_mean, lm_worst))
    print("\n  |diff|>32 之中落在輪廓上的比例: {:.1f}%".format(edge_share))
    print("  render: 平均 {:.1f}ms  最差 {:.1f}ms  (每幀預算 40ms @25fps)".format(
        float(np.mean(render_ms)), float(np.max(render_ms))))

    # 自我一致性才是真正的不變量：實際系統裡快取就是唯一真相，不存在「當初
    # 那一次擬合」可以對照。上面那些數字是跟一次歷史擬合比，會受 MTCNN 在 GPU
    # 上的浮點非決定性影響 —— crop_params 會被 int() 取整，次像素差異翻過整數
    # 邊界就是整整 1px 位移，而深度圖是硬邊，因此貢獻大量 >32 的像素。
    # 實測 corr(lm_max, pct>32) = 0.55 就是這件事。
    print("\n[ 自我一致性：同一份快取渲染兩次 ]")
    a1, _ = renderer.render(0, exp_all[0])
    a2, _ = renderer.render(0, exp_all[0])
    repeat = int(np.abs(np.array(a1, np.int16) - np.array(a2, np.int16)).max())
    check("重複渲染位元相同", repeat == 0, repeat)

    print("\n[ 判斷 ]")
    check("非輪廓區域幾乎一致（mean < 2.0）", off_mean < 2.0, round(off_mean, 4))
    check("大誤差集中在輪廓（> 50%）", edge_share > 50.0, round(edge_share, 1))
    check("landmark 位移 < 5px", lm_mean < 5.0, round(lm_mean, 3))
    check("render 快過即時預算（< 40ms）", float(np.mean(render_ms)) < 40.0,
          round(float(np.mean(render_ms)), 1))
    print("  （|diff|>32 平均 {:.3f}% 僅供參考，不作為判準：見上方說明）"
          .format(o32_mean))

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"cache": args.cache, "pose_dir": args.pose_dir,
                   "frames": rows,
                   "render_ms_mean": float(np.mean(render_ms)),
                   "edge_share_of_big": edge_share}, f, indent=2)
    print("\n  json -> {}".format(out))

    if _failures:
        print("\nFAILED: {}".format(", ".join(_failures)))
        return 1
    print("\ndepth cache parity 通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
