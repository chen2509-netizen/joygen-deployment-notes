"""
push_audio.py — push an audio file at a running JoyGen service.

Stands in for imood voice while the TTS stack is not up, and doubles as the
manual smoke test for the ingest path:

    bash scripts/run_input.sh serve demo/example_5s.mp4 \\
        results/smoke_test/edit_exp xinwen_5s /tmp/live.mp4 --no_idle
    python tests/push_audio.py demo/xinwen_5s.mp3 --voice female

Chunking follows docs/tts-streaming-spec.md: 16kHz mono PCM16, 320ms per
chunk, last chunk short rather than padded. --pace realtime sleeps between
chunks the way a perfectly paced TTS would; the default sends as fast as the
service accepts, which is what measures the GPU rather than the sleep.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

from streaming_input.ingest import IngestClient, SAMPLE_RATE  # noqa: E402


def decode_pcm16(path, sample_rate=SAMPLE_RATE):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
           "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout


def main(argv=None):
    p = argparse.ArgumentParser(description="push audio to the JoyGen service")
    p.add_argument("audio", nargs="+", help="one file per utterance")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--voice", default="female", choices=["female", "male"])
    p.add_argument("--emotion", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--session", default="cli")
    p.add_argument("--chunk_ms", type=int, default=320)
    p.add_argument("--pace", choices=["fast", "realtime"], default="fast")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="socket timeout; END 會等整句畫完才回，所以這要蓋過"
                        "生成時間而不是網路往返時間")
    p.add_argument("--gap_ms", type=int, default=0,
                   help="pause between utterances, so idle frames are exercised")
    args = p.parse_args(argv)

    chunk_bytes = int(SAMPLE_RATE * args.chunk_ms / 1000) * 2
    client = IngestClient(args.host, args.port, timeout=args.timeout)
    print("[push] connected to {}:{}".format(args.host, args.port))

    try:
        for i, path in enumerate(args.audio):
            pcm = decode_pcm16(path)
            tag = "u{}".format(i + 1)
            ack = client.begin(args.session, tag, voice=args.voice,
                               emotion=args.emotion, text=args.text)
            if not ack.get("ok"):
                sys.exit("[push] service refused the utterance: {}".format(ack))

            t0 = time.perf_counter()
            sent = 0
            for start in range(0, len(pcm), chunk_bytes):
                client.audio(pcm[start:start + chunk_bytes])
                sent += 1
                if args.pace == "realtime":
                    time.sleep(args.chunk_ms / 1000.0)
            result = client.end()
            wall = (time.perf_counter() - t0) * 1000.0

            print("[push] {} {}  {:.1f}s audio in {} chunks -> {} frames, "
                  "first frame {}ms, wall {:.0f}ms".format(
                      tag, Path(path).name,
                      len(pcm) / 2.0 / SAMPLE_RATE, sent,
                      result.get("frames"),
                      result.get("ingest_to_first_frame_ms"), wall))

            if args.gap_ms and i + 1 < len(args.audio):
                time.sleep(args.gap_ms / 1000.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
