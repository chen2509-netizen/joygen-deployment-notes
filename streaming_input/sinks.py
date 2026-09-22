"""Frame output sinks: persistent ffmpeg encoder, optional PNG dump.

Copy of streaming/sinks.py with one addition: audio can arrive in pieces.

The output-streaming version decodes the whole source file into
`self._audio_pcm` in __init__ and slices from it. Input streaming has no file
— the PCM shows up 320ms at a time — so `audio_stream=True` starts that buffer
empty and `append_audio()` grows it. Everything downstream (the fifo, the
per-frame slicing in write_audio_upto) is unchanged, because slicing a
bytearray that is still growing already does the right thing: it writes only
what has arrived.

Deliberately a copy rather than a shared module. Output streaming is the
measured baseline going into the A100 test, and it should not move because
input streaming needed something.
"""

import os
import subprocess
import sys

import cv2


class FFmpegSink:
    """Feeds raw BGR frames to a long-running ffmpeg process.

    Supported targets:
        rtp://host:port   RTP/H.264 (RFC 6184), writes an SDP file for the receiver
        udp://host:port   MPEG-TS over UDP
        <path>            plain file (mp4 etc.)

    Passing audio_path muxes that file alongside the frames. JoyGen never
    modifies the audio — it only reads it to condition the model — so the
    original file can go straight through to the output.

    audio_sync controls *how* that audio reaches ffmpeg (Step 6):
        "fifo"  (方向 B, 預設) — the source audio is decoded once into 16kHz
                mono PCM up front, then written out one frame's worth at a
                time, right after each video frame, over a named pipe. Audio
                is paced by generation the same way video already is (video
                has always been paced this way, since it arrives over a
                stdin pipe that ffmpeg only reads as fast as we write it).
                This is what keeps A/V from drifting when generation is
                slower than realtime.
        "file"  (方向 A) — ffmpeg reads/decodes the source file directly at
                full speed. Only useful for validating that the mapping
                itself is correct (does audio line up with the mouth) against
                a file target; muxing live will drift once generation falls
                behind, since ffmpeg's muxer eventually stops waiting on the
                slow video input (`max_interleave_delta`) and just flushes
                the audio ahead.

    Note that '-f rtp' carries a single stream, so an rtp:// target with audio
    is switched to rtp_mpegts, which puts both tracks on one port. That variant
    does not use an SDP file; the receiver reads the rtp:// URL directly.
    """

    AUDIO_SAMPLE_RATE = 16000  # 16kHz mono s16le, per imood-ai-architecture.md

    def __init__(self, width, height, fps, target,
                 preset="ultrafast", tune="zerolatency",
                 bitrate=None, gop=None, pkt_size=1200,
                 sdp_path="stream.sdp", audio_path=None, audio_sync="fifo",
                 audio_stream=False, verbose=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.target = target
        self.sdp_path = None
        self.recv_hint = None
        self.frames_written = 0

        # ---- audio-sync state ----
        # audio_stream=True means "audio is coming, but not from a file":
        # the buffer starts empty and append_audio() fills it as the stream
        # arrives. Everything else behaves as if a file had been decoded.
        self.audio_stream = audio_stream
        has_audio = bool(audio_path) or audio_stream
        self.audio_sync = ("fifo" if audio_stream else audio_sync) if has_audio else None
        self._audio_fifo_path = None
        self._audio_fifo_fd = None
        self._audio_pcm = bytearray() if audio_stream else b""
        self._audio_bytes_per_frame = 0
        self._audio_bytes_written = 0

        if has_audio and self.audio_sync == "fifo":
            if not audio_stream:
                self._audio_pcm = _decode_pcm_s16le(audio_path, self.AUDIO_SAMPLE_RATE)
            # 16-bit mono -> 2 bytes/sample
            self._audio_bytes_per_frame = int(round(self.AUDIO_SAMPLE_RATE / fps)) * 2
            self._audio_fifo_path = f"/tmp/jg_audio_{os.getpid()}.pcm"
            if os.path.exists(self._audio_fifo_path):
                os.remove(self._audio_fifo_path)
            os.mkfifo(self._audio_fifo_path)
            # O_RDWR on a FIFO never blocks (POSIX) — returns immediately
            # without waiting for a reader. Writes buffer in the kernel pipe
            # until ffmpeg opens its read side during startup. We open this
            # *before* Popen so ffmpeg finds the write side ready and doesn't
            # have to race against Python.
            self._audio_fifo_fd = os.open(self._audio_fifo_path, os.O_RDWR)

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning" if verbose else "error",
            # input 0: frames over stdin
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]

        if has_audio and self.audio_sync == "fifo":
            # input 1: raw PCM, paced one frame's worth at a time from Python
            # (see write_audio_upto). ffmpeg just reads whatever is written to
            # the pipe, so audio and video advance in lockstep by construction.
            # -probesize 32: format is already declared (-f s16le -ar -ac), so
            # there is nothing to detect; without this ffmpeg blocks waiting
            # for probe data before it starts reading stdin (input 0), which
            # deadlocks because Python doesn't write audio until after writing
            # the first video frame.
            cmd += ["-f", "s16le", "-ar", str(self.AUDIO_SAMPLE_RATE), "-ac", "1",
                    "-probesize", "32", "-i", self._audio_fifo_path]
        elif has_audio:
            # input 1 (方向 A): the source audio, read/decoded as fast as
            # ffmpeg likes. Frame PTS come from -r above, so alignment is
            # correct on paper, but the muxer will flush audio ahead if video
            # falls far enough behind (see class docstring).
            cmd += ["-i", audio_path]

        cmd += [
            "-c:v", "libx264",
            "-preset", preset,
            "-tune", tune,
            "-pix_fmt", "yuv420p",
            "-x264opts", "repeat-headers=1",  # 加在此處，全域生效
        ]


        # Capping the bitrate keeps keyframes from blowing past the receiver's
        # socket buffer, which is what corrupts playback over UDP transports.
        if bitrate:
            cmd += ["-b:v", bitrate, "-maxrate", bitrate,
                    "-bufsize", _double_bitrate(bitrate)]
        if gop:
            cmd += ["-g", str(gop), "-keyint_min", str(gop)]

        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k",
                    # Video arrives far slower than audio decodes; without this
                    # the muxer stalls trying to interleave them in step.
                    "-max_interleave_delta", "0",
                    # Stop at whichever input ends first.
                    "-shortest"]

        if target.startswith("rtp://"):
            if has_audio:
                cmd += ["-f", "rtp_mpegts",
                        _with_query(target, f"pkt_size={pkt_size}")]
                self.recv_hint = f"ffplay -i {target}"
            else:
                # Repeat SPS/PPS with every keyframe so a receiver that joins
                # late, or reuses an older SDP, can still sync on its own.
                self.sdp_path = sdp_path
                cmd += ["-x264opts", "repeat-headers=1",
                        "-f", "rtp", "-sdp_file", sdp_path,
                        _with_query(target, f"pkt_size={pkt_size}")]
                self.recv_hint = f"bash scripts/run_input.sh recv {sdp_path}"
        elif target.startswith("udp://"):
            cmd += ["-f", "mpegts", "-flush_packets", "1",
                    _with_query(target, "pkt_size=1316&buffer_size=2000000")]
            self.recv_hint = f"bash scripts/run_input.sh recv {target}"
        else:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            cmd += [target]

        print(f"[sink] {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        if self.sdp_path:
            print(f"[sink] SDP will be written to {sdp_path}; "
                  f"start the receiver once it appears", flush=True)
        elif self.recv_hint and not os.path.exists(str(target)):
            print(f"[sink] receiver: {self.recv_hint}", flush=True)

    def write(self, frame):
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames_written += 1
        except BrokenPipeError:
            sys.exit("[error] ffmpeg exited early. If streaming, check that the "
                     "receiver is still running, and re-run with --verbose to "
                     "see ffmpeg's own output.")

    def append_audio(self, pcm_bytes):
        """Add newly arrived PCM (16kHz mono s16le). Only meaningful with
        audio_stream=True. write_audio_upto() will not run ahead of what has
        been appended, so a late chunk delays its own frames rather than
        desynchronising the ones already sent."""
        if not self.audio_stream or not pcm_bytes:
            return
        self._audio_pcm.extend(pcm_bytes)

    def audio_ms_buffered(self):
        if not self._audio_bytes_per_frame:
            return 0.0
        return (len(self._audio_pcm) - self._audio_bytes_written) * 1000.0 \
            / (self.AUDIO_SAMPLE_RATE * 2)

    def write_audio_upto(self, frame_idx):
        """方向 B：把 frame_idx（0-based）對應的那段 PCM 補到 fifo 裡。

        跟畫面用同一個時間軸換算（frame_idx/fps <-> sample_count/sample_rate），
        呼叫時機是「這張畫面剛寫進 video pipe 之後」，所以音訊永遠不會跑到比
        目前這張畫面更晚的時間點。no-op when audio_sync 不是 "fifo"。
        """
        if self.audio_sync != "fifo":
            return
        end_byte = min((frame_idx + 1) * self._audio_bytes_per_frame,
                       len(self._audio_pcm))
        chunk = self._audio_pcm[self._audio_bytes_written:end_byte]
        if chunk:
            os.write(self._audio_fifo_fd, chunk)
            self._audio_bytes_written = end_byte

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        if self._audio_fifo_fd is not None:
            # Flush whatever's left (e.g. rounding, or audio slightly longer
            # than frame_count * bytes_per_frame covers) before closing.
            tail = self._audio_pcm[self._audio_bytes_written:]
            if tail:
                os.write(self._audio_fifo_fd, tail)
            os.close(self._audio_fifo_fd)
        code = self.proc.wait()
        if self._audio_fifo_path and os.path.exists(self._audio_fifo_path):
            os.remove(self._audio_fifo_path)
        print(f"[sink] ffmpeg exited({code}), frames written: {self.frames_written}",
              flush=True)


class PngSink:
    """Writes frames as {idx}_edit.png, matching the baseline's naming.

    Blocking disk I/O — only enable when comparing output, not when timing.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def write(self, frame, idx):
        cv2.imwrite(f"{self.save_dir}/{idx}_edit.png", frame)

    def close(self):
        pass


def _decode_pcm_s16le(audio_path, sample_rate):
    """一次性把來源音訊解成 16-bit mono PCM，存記憶體供逐 frame 切片用。"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path,
           "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    return result.stdout


def _with_query(url, query):
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{query}"


def _double_bitrate(bitrate):
    """'4M' -> '8M', '2000k' -> '4000k'."""
    unit = bitrate[-1]
    if unit.isdigit():
        return str(int(bitrate) * 2)
    return f"{int(bitrate[:-1]) * 2}{unit}"
