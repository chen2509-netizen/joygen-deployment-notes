"""Frame output sinks: persistent ffmpeg encoder, optional PNG dump."""

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

    Note that '-f rtp' carries a single stream, so an rtp:// target with audio
    is switched to rtp_mpegts, which puts both tracks on one port. That variant
    does not use an SDP file; the receiver reads the rtp:// URL directly.
    """

    def __init__(self, width, height, fps, target,
                 preset="ultrafast", tune="zerolatency",
                 bitrate=None, gop=None, pkt_size=1200,
                 sdp_path="stream.sdp", audio_path=None, verbose=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.target = target
        self.sdp_path = None
        self.recv_hint = None
        self.frames_written = 0

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

        if audio_path:
            # input 1: the source audio, read as fast as it can be decoded.
            # Frame PTS come from -r above, so alignment does not depend on how
            # slowly frames arrive over stdin.
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

        if audio_path:
            cmd += ["-c:a", "aac", "-b:a", "128k",
                    # Video arrives far slower than audio decodes; without this
                    # the muxer stalls trying to interleave them in step.
                    "-max_interleave_delta", "0",
                    # Stop at whichever input ends first.
                    "-shortest"]

        if target.startswith("rtp://"):
            if audio_path:
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
                self.recv_hint = f"bash scripts/run.sh recv {sdp_path}"
        elif target.startswith("udp://"):
            cmd += ["-f", "mpegts", "-flush_packets", "1",
                    _with_query(target, "pkt_size=1316&buffer_size=2000000")]
            self.recv_hint = f"bash scripts/run.sh recv {target}"
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

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
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


def _with_query(url, query):
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{query}"


def _double_bitrate(bitrate):
    """'4M' -> '8M', '2000k' -> '4000k'."""
    unit = bitrate[-1]
    if unit.isdigit():
        return str(int(bitrate) * 2)
    return f"{int(bitrate[:-1]) * 2}{unit}"
