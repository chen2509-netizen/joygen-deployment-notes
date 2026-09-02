"""
frame_timer.py — 時間量測工具（階段 + 逐 frame）

Step 1 的 baseline 與之後 streaming 版本共用同一套計時邏輯，
確保「原版 vs 改版」的數字是同一個基準量出來的，可以直接比較。

位置：JoyGen/streaming/frame_timer.py
"""

import csv
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseTimer:
    """量測粗粒度階段：模型載入、前處理、生成、貼圖、ffmpeg。

    重點是看清楚「第一張畫面出現之前」有多少固定成本躲不掉——
    模型載入 + 前處理就是首幀延遲的下限，輸出端再怎麼改都省不掉這段。
    """

    def __init__(self):
        self.phases = {}

    @contextmanager
    def phase(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.phases[name] = round(dt, 1)
            print(f"[phase] {name:<24} {dt:12,.1f} ms", flush=True)

    def as_dict(self):
        return dict(self.phases)


class FrameTimer:
    """逐 frame 計時。每完成一個處理單位呼叫一次 mark()。"""

    def __init__(self, tag, verbose=False):
        self.tag = tag
        self.verbose = verbose      # 逐 frame 印出會拖慢速度，預設關閉
        self._t0 = None
        self._prev = None
        self.records = []           # [(idx, elapsed_ms, delta_ms), ...]

    def start(self):
        self._t0 = time.perf_counter()
        self._prev = self._t0

    def mark(self):
        if self._t0 is None:
            raise RuntimeError(f"[{self.tag}] start() must be called before mark()")

        now = time.perf_counter()
        idx = len(self.records) + 1
        elapsed = (now - self._t0) * 1000.0
        delta = (now - self._prev) * 1000.0
        self._prev = now
        self.records.append((idx, elapsed, delta))

        if self.verbose:
            print(f"[{self.tag}] #{idx:05d} elapsed={elapsed:10,.1f}ms "
                  f"delta={delta:8.1f}ms", flush=True)

    def summary(self):
        if not self.records:
            return {"tag": self.tag, "frames": 0}

        deltas = [d for _, _, d in self.records]
        return {
            "tag": self.tag,
            "frames": len(self.records),
            "first_ms": round(self.records[0][2], 1),
            "total_ms": round(self.records[-1][1], 1),
            "mean_ms": round(statistics.mean(deltas), 1),
            "median_ms": round(statistics.median(deltas), 1),
            "min_ms": round(min(deltas), 1),
            "max_ms": round(max(deltas), 1),
            # stdev 小 → 每幀耗時一致，瓶頸是 GPU 吞吐（硬體上限）
            # stdev 大 → 有東西間歇性卡住，代表有優化空間
            "stdev_ms": round(statistics.stdev(deltas), 1) if len(deltas) > 1 else 0.0,
        }


def write_report(out_prefix, phases, timers, meta=None, mode="baseline"):
    """輸出逐 frame CSV + 摘要 JSON，並印出對照表與判斷點。

    mode 決定標題與要顯示哪些判斷點：
      baseline — 顯示「串流化理論可省」（貼圖 + ffmpeg，這兩段還存在）
      step2/3  — 那兩段已經被合併/取代，改顯示串流實際表現
    """
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "idx", "elapsed_ms", "delta_ms"])
        for t in timers:
            for idx, elapsed, delta in t.records:
                w.writerow([t.tag, idx, f"{elapsed:.2f}", f"{delta:.2f}"])

    stats = [t.summary() for t in timers]
    phase_ms = phases.as_dict()

    report = {"meta": meta or {}, "phases_ms": phase_ms, "frame_stats": stats}
    json_path = prefix.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    titles = {
        "baseline": "BASELINE SUMMARY (原始兩段式流程)",
        "step2": "STEP 2 SUMMARY (貼圖已併入生成迴圈)",
        "step3": "STEP 3 SUMMARY (frame 即時送進常駐 ffmpeg)",
    }
    print("\n" + "=" * 74)
    print(titles.get(mode, f"{mode.upper()} SUMMARY"))
    print("=" * 74)

    print("\n[ 各階段耗時 ]")
    total = sum(phase_ms.values())
    for name, ms in phase_ms.items():
        pct = (ms / total * 100) if total else 0
        print(f"  {name:<24} {ms:12,.1f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<24} {total:12,.1f} ms")

    print("\n[ 逐 frame 統計 (ms) ]")
    head = (f"  {'stage':<14}{'frames':>8}{'mean':>11}{'median':>11}"
            f"{'stdev':>10}{'min':>10}{'max':>11}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for s in stats:
        if not s.get("frames"):
            continue
        print(f"  {s['tag']:<14}{s['frames']:>8}{s['mean_ms']:>11,.1f}"
              f"{s['median_ms']:>11,.1f}{s['stdev_ms']:>10,.1f}"
              f"{s['min_ms']:>10,.1f}{s['max_ms']:>11,.1f}")

    print("\n[ 判斷點 ]")

    # 首幀延遲下限：模型載入 + warmup + 前處理，全部 mode 都適用
    fixed = (phase_ms.get("1_model_load", 0)
             + phase_ms.get("1b_warmup", 0)
             + phase_ms.get("2_preprocess", 0))
    print(f"  首幀延遲下限（載入 + warmup + 前處理）: {fixed:12,.1f} ms")
    print("     └ 這段在生成開始前就得跑完，串流化省不掉")

    if mode == "baseline":
        # 貼圖與一次性 ffmpeg 是串流化要拿掉的兩段，只有 baseline 才有
        saveable = phase_ms.get("4_blend_write", 0) + phase_ms.get("5_ffmpeg_video", 0)
        print(f"  串流化理論可省（貼圖 + ffmpeg 轉檔） : {saveable:12,.1f} ms")
    else:
        # 串流版：這兩段已經不存在，改看實際的首幀與吞吐表現
        ffm = (meta or {}).get("first_frame_after_gen_start_ms")
        if ffm:
            print(f"  生成開始後第一張送出                 : {ffm:12,.1f} ms")
            print(f"  → 端到端首幀延遲（固定成本 + 首張）  : {fixed + ffm:12,.1f} ms")

    # 每 frame 節奏穩定度：找主要的逐 frame 階段（各 mode 名稱不同）
    main_tag = next((s for s in stats
                     if s.get("frames") and s["tag"] != "preprocess"), None)
    if main_tag and main_tag["mean_ms"]:
        ratio = main_tag["stdev_ms"] / main_tag["mean_ms"]
        note = ("穩定" if ratio < 0.25 else "跳動（含 batch 邊界效應，非必然是問題）")
        print(f"  {main_tag['tag']} stdev/mean{'':<14}: {ratio:12.2f}   {note}")

    fps_actual = (1000.0 / main_tag["mean_ms"]) if main_tag and main_tag["mean_ms"] else 0
    if fps_actual:
        print(f"  實際輸出速率                         : {fps_actual:12.1f} fps"
              f"   (25fps 為即時門檻)")

    print(f"\n  csv  -> {csv_path}")
    print(f"  json -> {json_path}\n")
    return report
