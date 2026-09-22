"""
test_tts_ingest_mock.py — the ingest protocol, without models or a GPU.

Stands a real IngestServer up on a loopback port and drives it with the
reference client plus a mock TTS that reproduces how CosyVoice actually
behaves: 320ms chunks arriving at the model's pace rather than real time, a
short last chunk that is not zero-padded, and pauses while the model thinks
(docs/tts-streaming-spec.md).

The cases here are the ones that would otherwise only show up on the A100:
several utterances over one connection, a client that dies mid-sentence, a
wrong sample rate, and the emotion field that the BERT integration will start
filling in later.
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

from streaming_input.ingest import (  # noqa: E402
    AUDIO,
    BEGIN,
    END,
    IngestClient,
    IngestServer,
    SAMPLE_RATE,
    send_frame,
    send_json,
)

_failures = []
CHUNK_BYTES = 10240          # 320ms of PCM16 at 16kHz, per the TTS spec


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        _failures.append(name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Collector:
    """Stands in for the GPU pipeline: drains the chunk generator and records
    what arrived."""

    def __init__(self, delay_per_chunk=0.0):
        self.calls = []
        self.delay = delay_per_chunk

    def __call__(self, meta, chunks):
        audio = bytearray()
        n = 0
        for payload in chunks:
            audio.extend(payload)
            n += 1
            if self.delay:
                time.sleep(self.delay)
        self.calls.append({"meta": meta, "bytes": len(audio), "chunks": n,
                           "audio": bytes(audio)})
        return {"frames": len(audio) // (SAMPLE_RATE * 2 // 25)}


def serve(handler):
    port = free_port()
    server = IngestServer("127.0.0.1", port, handler)
    server.start_listening()
    server.start()
    return server, port


def tts_chunks(total_ms, chunk_ms=320):
    """PCM shaped like CosyVoice's output: fixed-size chunks and a short,
    un-padded tail."""
    total = int(SAMPLE_RATE * total_ms / 1000)
    step = int(SAMPLE_RATE * chunk_ms / 1000)
    tone = (np.sin(np.arange(total) * 0.05) * 8000).astype(np.int16)
    for start in range(0, total, step):
        yield tone[start:start + step].tobytes()


def test_single_utterance():
    collector = Collector()
    server, port = serve(collector)
    with IngestClient("127.0.0.1", port) as client:
        ack = client.begin("s1", "u1", voice="female")
        check("begin acknowledged", ack.get("ok") is True, ack)
        sent = 0
        for chunk in tts_chunks(2000):
            client.audio(chunk)
            sent += len(chunk)
        done = client.end()

    check("end acknowledged", done.get("ok") is True, done)
    call = collector.calls[0]
    check("all audio arrived", call["bytes"] == sent,
          (call["bytes"], sent))
    check("chunk count preserved", call["chunks"] == 7, call["chunks"])
    check("voice carried through", call["meta"]["voice"] == "female",
          call["meta"])
    server.stop()


def test_short_tail_chunk():
    """The TTS does not pad its last chunk; the bytes must still arrive whole."""
    collector = Collector()
    server, port = serve(collector)
    with IngestClient("127.0.0.1", port) as client:
        client.begin("s1", "u1")
        chunks = list(tts_chunks(1000))       # 1000ms -> 3x320 + 40ms tail
        for chunk in chunks:
            client.audio(chunk)
        client.end()

    call = collector.calls[0]
    check("tail chunk is short", len(chunks[-1]) < CHUNK_BYTES, len(chunks[-1]))
    check("tail chunk not lost",
          call["bytes"] == sum(len(c) for c in chunks), call["bytes"])
    server.stop()


def test_several_utterances_one_connection():
    collector = Collector()
    server, port = serve(collector)
    with IngestClient("127.0.0.1", port) as client:
        for i in range(3):
            client.begin("s1", "u{}".format(i), voice="male" if i else "female")
            for chunk in tts_chunks(640):
                client.audio(chunk)
            client.end()

    check("three utterances handled", len(collector.calls) == 3,
          len(collector.calls))
    check("utterance ids in order",
          [c["meta"]["utterance"] for c in collector.calls] == ["u0", "u1", "u2"],
          [c["meta"]["utterance"] for c in collector.calls])
    check("voice switches per utterance",
          [c["meta"]["voice"] for c in collector.calls] ==
          ["female", "male", "male"],
          [c["meta"]["voice"] for c in collector.calls])
    server.stop()


def test_begin_without_end_starts_next_utterance():
    """A client that forgets END before the next BEGIN must not lose the
    following utterance."""
    collector = Collector()
    server, port = serve(collector)
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    send_json(sock, BEGIN, {"session": "s", "utterance": "a",
                            "sample_rate": SAMPLE_RATE})
    sock.recv(4096)                                   # status
    send_frame(sock, AUDIO, b"\x01\x02" * 100)
    send_json(sock, BEGIN, {"session": "s", "utterance": "b",
                            "sample_rate": SAMPLE_RATE})
    time.sleep(0.3)
    send_frame(sock, AUDIO, b"\x03\x04" * 100)
    send_frame(sock, END)
    time.sleep(0.5)
    sock.close()
    time.sleep(0.3)

    ids = [c["meta"]["utterance"] for c in collector.calls]
    check("stashed BEGIN is not dropped", ids == ["a", "b"], ids)
    server.stop()


def test_client_disconnect_is_implicit_end():
    collector = Collector()
    server, port = serve(collector)
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    send_json(sock, BEGIN, {"session": "s", "utterance": "dead",
                            "sample_rate": SAMPLE_RATE})
    sock.recv(4096)
    send_frame(sock, AUDIO, b"\x01\x02" * 500)
    sock.close()                                      # die mid-utterance
    time.sleep(0.5)

    check("disconnect ends the utterance", len(collector.calls) == 1,
          len(collector.calls))
    check("audio before the disconnect is kept",
          collector.calls and collector.calls[0]["bytes"] == 1000,
          collector.calls[0]["bytes"] if collector.calls else None)
    server.stop()


def test_wrong_sample_rate_rejected():
    collector = Collector()
    server, port = serve(collector)
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    send_json(sock, BEGIN, {"session": "s", "utterance": "bad",
                            "sample_rate": 44100})
    status = json.loads(sock.recv(4096)[5:].decode("utf-8"))
    sock.close()
    time.sleep(0.2)

    check("wrong sample rate refused", status.get("ok") is False, status)
    check("refusal names the rate", "16000" in status.get("error", ""), status)
    check("handler never ran", len(collector.calls) == 0, len(collector.calls))
    server.stop()


def test_emotion_field_passthrough():
    """Nothing consumes emotion yet, but the field has to survive the wire so
    the BERT integration does not need a protocol change."""
    collector = Collector()
    server, port = serve(collector)
    with IngestClient("127.0.0.1", port) as client:
        client.begin("s1", "u1", emotion="joy", text="太好了")
        client.audio(next(tts_chunks(320)))
        client.end()

    meta = collector.calls[0]["meta"]
    check("emotion carried", meta.get("emotion") == "joy", meta)
    check("text carried", meta.get("text") == "太好了", meta)
    server.stop()


def test_backpressure_reaches_the_sender():
    """A slow consumer must slow the sender down, not silently drop audio.
    With the socket unread, TCP's window closes and the client blocks."""
    collector = Collector(delay_per_chunk=0.05)
    server, port = serve(collector)

    elapsed = {}

    def sender():
        with IngestClient("127.0.0.1", port) as client:
            client.begin("s1", "slow")
            t0 = time.perf_counter()
            for chunk in tts_chunks(6400):            # 20 chunks
                client.audio(chunk)
            client.end()                              # waits for the handler
            elapsed["ms"] = (time.perf_counter() - t0) * 1000.0

    t = threading.Thread(target=sender)
    t.start()
    t.join(timeout=30)

    check("sender waited for the slow consumer", elapsed.get("ms", 0) > 900,
          round(elapsed.get("ms", 0), 1))
    check("no audio lost under backpressure",
          collector.calls[0]["bytes"] == 20 * CHUNK_BYTES,
          collector.calls[0]["bytes"])
    server.stop()


def main():
    for fn in (test_single_utterance, test_short_tail_chunk,
               test_several_utterances_one_connection,
               test_begin_without_end_starts_next_utterance,
               test_client_disconnect_is_implicit_end,
               test_wrong_sample_rate_rejected,
               test_emotion_field_passthrough,
               test_backpressure_reaches_the_sender):
        print("[{}]".format(fn.__name__))
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            print("  ERROR {}: {!r}".format(fn.__name__, exc))
            _failures.append(fn.__name__)
    print("")
    if _failures:
        print("FAILED: {}".format(", ".join(_failures)))
        return 1
    print("all ingest tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
