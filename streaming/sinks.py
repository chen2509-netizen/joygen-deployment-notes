"""
sinks.py — frame 輸出目的地

Step 2 的 callback 只會寫 PNG；Step 3 起改成把 frame 直接送進常駐 ffmpeg，
即時編成 H.264 / MPEG-TS 經 UDP 送出，不再等整支影片跑完才轉檔。

位置：JoyGen/streaming/sinks.py
"""

import os
import subprocess
import sys

import cv2


class FFmpegSink:
    """常駐 ffmpeg subprocess：frame 一產生就寫進 stdin，即時編碼送出。

    取代原本「全部 frame 寫成 PNG → 一次性 ffmpeg 轉檔」的收尾流程。

    注意畫面尺寸要用**原始影片解析度**（貼回原圖後的 combine_img），
    不是 img_size（256 只是臉部裁切的大小）。
    """

    def __init__(self, width, height, fps, target,
                 preset="ultrafast", tune="zerolatency", verbose=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.target = target
        self.frames_written = 0

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning" if verbose else "error",
            # 輸入：從 stdin 讀 raw BGR frame（cv2 的原生格式，不用轉換）
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            # 輸出：H.264，低延遲設定
            "-c:v", "libx264",
            "-preset", preset,
            "-tune", tune,
            "-pix_fmt", "yuv420p",
        ]

        # target 決定封裝與去向：udp://... 走 MPEG-TS，其餘當檔案路徑
        if target.startswith("udp://") or target.startswith("rtp://"):
            cmd += ["-f", "mpegts", target]
        else:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            cmd += [target]

        print(f"[sink] ffmpeg: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        """frame 是 numpy BGR uint8 array，shape 必須是 (height, width, 3)。

        .tobytes() 只是把 numpy 的記憶體內容當成連續 bytes 交出去，
        幾乎零成本——pipe 只吃 byte stream，不吃 Python 物件。
        """
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames_written += 1
        except BrokenPipeError:
            sys.exit("[error] ffmpeg 已結束（BrokenPipe）。"
                     "如果推 UDP，請確認接收端還在；或看 ffmpeg 的錯誤訊息。")

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        print(f"[sink] ffmpeg exited({code}), frames written: {self.frames_written}",
              flush=True)


class PngSink:
    """把 frame 寫成 PNG，檔名跟 baseline 一致（{idx}_edit.png）。

    Step 3 起預設關閉（--debug 才啟用）：寫檔是阻塞的磁碟 I/O，
    開著會拖慢主迴圈、污染計時數據。只在需要用 diff_frames 驗證正確性時才開。
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def write(self, frame, idx):
        cv2.imwrite(f"{self.save_dir}/{idx}_edit.png", frame)

    def close(self):
        pass
