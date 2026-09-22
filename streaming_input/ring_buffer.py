"""
ring_buffer.py — PCM between the audio source and the GPU loop.

The two sides run at unrelated rates. TTS emits when the model finishes a
segment (CosyVoice2 RTF ~0.24x, so bursty and faster than realtime); the GPU
loop consumes exactly one 320ms batch at a time, at whatever speed the UNet
manages. The buffer absorbs that difference and makes the mismatch visible:
an underrun means the pipeline starved, an overrun means audio was dropped
because the consumer could not keep up. Both are counted, because on the A100
those counts are the first thing to look at when playback stutters.

Fixed capacity, so a stalled consumer cannot grow memory without bound.
read() blocks rather than returning short, since a partial batch would
silently desynchronise audio from frames.
"""

import threading

import numpy as np


class AudioRingBuffer:
    def __init__(self, capacity_samples, sample_rate=16000):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity = int(capacity_samples)
        self.sample_rate = sample_rate

        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._head = 0          # next sample to read
        self._level = 0         # samples currently held
        self._closed = False

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        self.underruns = 0
        self.overruns = 0
        self.dropped_samples = 0
        self.total_in = 0
        self.total_out = 0
        self.high_water = 0

    # -- producer ---------------------------------------------------

    def put(self, samples, block=False, timeout=None):
        """Append samples. Returns how many had to be dropped because the
        buffer was full — the oldest audio goes first, since stale audio is
        worse than a gap.

        block=True waits for room instead of dropping. That is what a file
        source wants: the file is not going anywhere, and silently discarding
        part of it turns a regression test into a comparison against the wrong
        audio. A live source wants the default, because it cannot stall the
        socket reader to wait for the GPU — there, dropping is the honest
        outcome and the overrun counter records it.

        A chunk larger than the whole buffer can never fit, so it falls back
        to dropping even under block=True rather than waiting forever."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return 0

        with self._lock:
            if self._closed:
                raise RuntimeError("put() on a closed ring buffer")

            if block and samples.size <= self.capacity:
                while (self.capacity - self._level) < samples.size \
                        and not self._closed:
                    if not self._not_full.wait(timeout):
                        break

            dropped = 0
            if samples.size > self.capacity:
                # A single chunk larger than the whole buffer: keep the newest.
                dropped = samples.size - self.capacity
                samples = samples[-self.capacity:]

            free = self.capacity - self._level
            if samples.size > free:
                shortfall = samples.size - free
                self._head = (self._head + shortfall) % self.capacity
                self._level -= shortfall
                dropped += shortfall

            tail = (self._head + self._level) % self.capacity
            first = min(samples.size, self.capacity - tail)
            self._buf[tail:tail + first] = samples[:first]
            if first < samples.size:
                self._buf[:samples.size - first] = samples[first:]

            self._level += samples.size
            self.total_in += samples.size
            self.high_water = max(self.high_water, self._level)
            if dropped:
                self.overruns += 1
                self.dropped_samples += dropped

            self._not_empty.notify_all()
            return dropped

    def close(self):
        """No more audio is coming. Wakes readers so they can drain, and any
        producer still waiting for room so it does not hang."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    # -- consumer ---------------------------------------------------

    def read(self, n, timeout=None):
        """Block until n samples are available, then return exactly n.

        Returns fewer (possibly zero) only once the buffer is closed and
        drained, or when `timeout` expires — the caller distinguishes the two
        by checking `closed`."""
        n = int(n)
        with self._lock:
            waited = False
            while self._level < n and not self._closed:
                waited = True
                if not self._not_empty.wait(timeout):
                    break
            if waited:
                # One underrun per starved read, not per wakeup — spurious
                # wakeups would otherwise inflate the count.
                self.underruns += 1

            take = min(n, self._level)
            if take == 0:
                return np.zeros(0, dtype=np.float32)

            first = min(take, self.capacity - self._head)
            out = np.empty(take, dtype=np.float32)
            out[:first] = self._buf[self._head:self._head + first]
            if first < take:
                out[first:] = self._buf[:take - first]

            self._head = (self._head + take) % self.capacity
            self._level -= take
            self.total_out += take
            self._not_full.notify_all()
            return out

    # -- introspection ----------------------------------------------

    @property
    def level(self):
        with self._lock:
            return self._level

    @property
    def level_ms(self):
        return self.level * 1000.0 / self.sample_rate

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def drained(self):
        with self._lock:
            return self._closed and self._level == 0

    def stats(self):
        with self._lock:
            return {
                "capacity_ms": round(self.capacity * 1000.0 / self.sample_rate, 1),
                "level_ms": round(self._level * 1000.0 / self.sample_rate, 1),
                "high_water_ms": round(self.high_water * 1000.0 / self.sample_rate, 1),
                "underruns": self.underruns,
                "overruns": self.overruns,
                "dropped_ms": round(self.dropped_samples * 1000.0 / self.sample_rate, 1),
                "total_in_ms": round(self.total_in * 1000.0 / self.sample_rate, 1),
                "total_out_ms": round(self.total_out * 1000.0 / self.sample_rate, 1),
            }
