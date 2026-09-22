"""
test_ring_buffer.py — AudioRingBuffer, no GPU required.

Covers what actually bites in the pipeline: wraparound (the buffer is a ring,
and a 320ms chunk routinely straddles the end), blocking reads (a short read
would desynchronise audio from frames), overrun accounting (the A100 test
needs to know whether audio was dropped) and clean shutdown (the consumer
must be able to tell "nothing yet" from "nothing ever again").
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

from streaming_input.ring_buffer import AudioRingBuffer  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        _failures.append(name)


def test_roundtrip():
    ring = AudioRingBuffer(1000)
    data = np.arange(100, dtype=np.float32)
    ring.put(data)
    out = ring.read(100)
    check("roundtrip preserves samples", np.array_equal(out, data))
    check("level empties after read", ring.level == 0, ring.level)


def test_wraparound():
    """A chunk that straddles the end must come back contiguous and in order."""
    ring = AudioRingBuffer(100)
    ring.put(np.arange(80, dtype=np.float32))
    ring.read(70)                      # head now at 70
    ring.put(np.arange(80, 140, dtype=np.float32))   # writes across the seam
    out = ring.read(70)
    check("wraparound keeps order",
          np.array_equal(out, np.arange(70, 140, dtype=np.float32)),
          out[:5])


def test_overrun():
    ring = AudioRingBuffer(100)
    ring.put(np.ones(80, dtype=np.float32))
    dropped = ring.put(np.full(40, 2.0, dtype=np.float32))
    check("overrun drops the excess", dropped == 20, dropped)
    check("overrun counted", ring.overruns == 1, ring.overruns)
    check("buffer stays at capacity", ring.level == 100, ring.level)
    out = ring.read(100)
    check("oldest audio is what got dropped",
          out[-1] == 2.0 and out[0] == 1.0, (out[0], out[-1]))


def test_chunk_larger_than_capacity():
    ring = AudioRingBuffer(50)
    dropped = ring.put(np.arange(120, dtype=np.float32))
    check("oversized chunk keeps the newest", dropped == 70, dropped)
    out = ring.read(50)
    check("oversized chunk tail is correct",
          np.array_equal(out, np.arange(70, 120, dtype=np.float32)), out[:3])


def test_blocking_read():
    ring = AudioRingBuffer(1000)
    got = {}

    def consumer():
        t0 = time.perf_counter()
        got["data"] = ring.read(500)
        got["waited_ms"] = (time.perf_counter() - t0) * 1000.0

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.15)
    ring.put(np.ones(500, dtype=np.float32))
    t.join(timeout=5)

    check("blocking read returns full length", got["data"].size == 500,
          got["data"].size)
    check("blocking read actually waited", got["waited_ms"] > 100,
          round(got["waited_ms"], 1))
    check("underrun counted", ring.underruns == 1, ring.underruns)


def test_blocking_put_waits_instead_of_dropping():
    """The file source must not lose audio when the GPU falls behind. A 34s
    clip poured into a 10s buffer used to silently drop two thirds of itself,
    which ended the muxer's audio early and deadlocked ffmpeg."""
    ring = AudioRingBuffer(100)
    ring.put(np.ones(100, dtype=np.float32))        # full
    done = {}

    def producer():
        t0 = time.perf_counter()
        done["dropped"] = ring.put(np.full(50, 7.0, dtype=np.float32), block=True)
        done["waited_ms"] = (time.perf_counter() - t0) * 1000.0

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.15)
    ring.read(50)                                   # make room
    t.join(timeout=5)

    check("blocking put drops nothing", done["dropped"] == 0, done["dropped"])
    check("blocking put waited for room", done["waited_ms"] > 100,
          round(done["waited_ms"], 1))
    check("no overrun recorded", ring.overruns == 0, ring.overruns)


def test_blocking_put_gives_up_when_closed():
    ring = AudioRingBuffer(100)
    ring.put(np.ones(100, dtype=np.float32))
    done = {}

    def producer():
        done["dropped"] = ring.put(np.ones(50, dtype=np.float32), block=True)

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.1)
    ring.close()
    t.join(timeout=5)
    check("blocking put returns after close", not t.is_alive())


def test_close_wakes_reader():
    ring = AudioRingBuffer(1000)
    ring.put(np.ones(100, dtype=np.float32))
    got = {}

    def consumer():
        got["a"] = ring.read(500)      # only 100 available, then closed
        got["b"] = ring.read(500)      # nothing left

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.1)
    ring.close()
    t.join(timeout=5)

    check("close releases a partial read", got["a"].size == 100, got["a"].size)
    check("reads after drain return empty", got["b"].size == 0, got["b"].size)
    check("drained reported", ring.drained())


def test_timeout():
    ring = AudioRingBuffer(1000)
    t0 = time.perf_counter()
    out = ring.read(100, timeout=0.2)
    elapsed = (time.perf_counter() - t0) * 1000.0
    check("timeout returns empty", out.size == 0, out.size)
    check("timeout respected", 150 < elapsed < 900, round(elapsed, 1))


def test_stats():
    ring = AudioRingBuffer(16000, sample_rate=16000)
    ring.put(np.zeros(8000, dtype=np.float32))
    s = ring.stats()
    check("level_ms in milliseconds", abs(s["level_ms"] - 500.0) < 0.1, s["level_ms"])
    check("capacity_ms in milliseconds", abs(s["capacity_ms"] - 1000.0) < 0.1,
          s["capacity_ms"])


def main():
    for fn in (test_roundtrip, test_wraparound, test_overrun,
               test_chunk_larger_than_capacity, test_blocking_read,
               test_blocking_put_waits_instead_of_dropping,
               test_blocking_put_gives_up_when_closed,
               test_close_wakes_reader, test_timeout, test_stats):
        print("[{}]".format(fn.__name__))
        fn()
    print("")
    if _failures:
        print("FAILED: {}".format(", ".join(_failures)))
        return 1
    print("all ring buffer tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
