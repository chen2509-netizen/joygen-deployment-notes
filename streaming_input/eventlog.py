"""
eventlog.py — one jsonl event stream per session.

Every hop in the pipeline emits a row here (VAD, ASR, LLM, TTS chunk, ingest,
whisper encode, UNet batch, blend, frame out). The A100 test needs to answer
"where did the time go" from the log alone, without reproducing the run, so
rows carry absolute wall-clock plus a monotonic offset from session start.

Writes are line-buffered and flushed per event: a run that dies mid-stream
still leaves a usable log.
"""

import json
import os
import threading
import time
from datetime import datetime


class EventLog:
    def __init__(self, path=None, session_id=None, stdout=True, per_frame=True):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stdout = stdout
        self.per_frame = per_frame
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._counts = {}
        self._fh = None

        if path:
            path = str(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
        self.path = path

    def event(self, stage, **fields):
        """Record one event. `stage` is a dotted name such as 'tts.chunk'."""
        row = {
            "t": round(time.time(), 6),
            "dt_ms": round((time.perf_counter() - self._t0) * 1000.0, 3),
            "session": self.session_id,
            "stage": stage,
        }
        row.update(fields)

        with self._lock:
            self._counts[stage] = self._counts.get(stage, 0) + 1
            if self._fh is not None:
                self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._fh.flush()
            if self.stdout:
                extra = " ".join(
                    "{}={}".format(k, v) for k, v in fields.items()
                )
                print("[{:>10.1f}ms] {:<22} {}".format(row["dt_ms"], stage, extra),
                      flush=True)

    def frame(self, stage, **fields):
        """Per-frame event. Suppressed when per_frame is off, because at 25fps
        this is the only stage that can dominate the log's own cost."""
        if self.per_frame:
            self.event(stage, **fields)

    def counts(self):
        with self._lock:
            return dict(self._counts)

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class NullEventLog(EventLog):
    """Drop-in for tests and for code paths that should not log."""

    def __init__(self):
        EventLog.__init__(self, path=None, stdout=False, per_frame=False)

    def event(self, stage, **fields):
        pass
