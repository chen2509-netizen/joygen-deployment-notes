"""
ingest.py — how audio reaches JoyGen from imood voice.

Transport is a plain TCP stream with length-prefixed frames. Not WebSocket:
this link is server to server, so the handshake and masking buy nothing, and
the joygen conda env is pinned to Python 3.8 + CUDA 11.7 for the A100 deploy —
adding a dependency to it days before that is a worse trade than 60 lines of
framing. TCP gives what the feature extractor actually needs, which is every
sample, in order: a dropped packet would shift the whole mel timeline.

Video still goes out over RTP, where loss is survivable and a jitter buffer is
the right tool.

Frame format:

    | type: 1 byte | length: 4 bytes big-endian | payload: length bytes |

    type 1  BEGIN  JSON. Starts an utterance.
                   {"session": str, "utterance": str, "voice": "female|male",
                    "sample_rate": 16000, "emotion": int|null, "text": str|null}
                   `emotion` is the BERT class code (0 neutral, 1 joy, 2 anger,
                   3 sorrow, 4 happy; defined in imood-emotion's labels.py) and
                   is carried through to the log only — nothing here acts on it
                   yet. Older clients that send a string, or nothing, still work.
    type 2  AUDIO  raw PCM16 little-endian mono at sample_rate.
    type 3  END    empty. The utterance is complete.
    type 4  STATUS JSON, server to client. Sent on accept and at END.

A connection may carry several utterances in sequence. Closing the socket
mid-utterance is treated as an implicit END, so a crashed client cannot wedge
the service.
"""

import json
import socket
import struct
import threading

BEGIN, AUDIO, END, STATUS = 1, 2, 3, 4
HEADER = struct.Struct(">BI")
MAX_PAYLOAD = 1 << 22          # 4MB; a 320ms chunk is 10KB
SAMPLE_RATE = 16000


def send_frame(sock, ftype, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    sock.sendall(HEADER.pack(ftype, len(payload)) + payload)


def send_json(sock, ftype, obj):
    send_frame(sock, ftype, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def _recv_exactly(sock, n):
    chunks = []
    got = 0
    while got < n:
        block = sock.recv(min(65536, n - got))
        if not block:
            return None
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def recv_frame(sock):
    """Returns (type, payload) or None when the peer closed cleanly."""
    head = _recv_exactly(sock, HEADER.size)
    if head is None:
        return None
    ftype, length = HEADER.unpack(head)
    if length > MAX_PAYLOAD:
        raise ValueError("frame too large: {} bytes".format(length))
    payload = _recv_exactly(sock, length) if length else b""
    if payload is None:
        return None
    return ftype, payload


class IngestServer(threading.Thread):
    """Accepts connections and drives one utterance at a time.

    `handler` is called as handler(meta, chunks) where `chunks` is a generator
    yielding PCM bytes until END. Running the handler on the connection thread
    keeps backpressure honest: while the GPU is behind, this stops reading the
    socket, TCP's window closes, and imood voice feels it — which is what we
    want it to feel, rather than discovering the backlog as dropped audio.
    """

    def __init__(self, host, port, handler, log=None, backlog=4):
        threading.Thread.__init__(self, name="ingest-server", daemon=True)
        self.host = host
        self.port = port
        self.handler = handler
        self.log = log
        self.backlog = backlog
        self._sock = None
        self._stop = threading.Event()
        self.connections = 0
        self.utterances = 0

    def start_listening(self):
        """Bind before start() so a port clash fails loudly in the caller."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(self.backlog)
        if self.log is not None:
            self.log.event("ingest.listening", host=self.host, port=self.port)
        return self

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def run(self):
        if self._sock is None:
            self.start_listening()
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            self.connections += 1
            try:
                self._serve(conn, addr)
            except Exception as exc:                  # noqa: BLE001
                if self.log is not None:
                    self.log.event("ingest.conn_error", peer=str(addr),
                                   error=repr(exc))
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _serve(self, conn, addr):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.log is not None:
            self.log.event("ingest.connected", peer="{}:{}".format(*addr))

        pending = {"frame": None}

        def chunks():
            """Yields PCM until END. Stashes a BEGIN that arrives without an
            END first, so the next utterance on this connection is not lost."""
            while True:
                frame = recv_frame(conn)
                if frame is None:
                    return                       # peer closed == implicit END
                ftype, payload = frame
                if ftype == AUDIO:
                    yield payload
                elif ftype == END:
                    return
                elif ftype == BEGIN:
                    pending["frame"] = (ftype, payload)
                    return
                # anything else is ignored rather than fatal: an older client
                # sending an unknown control frame should not kill the stream

        while not self._stop.is_set():
            stashed = pending["frame"]
            pending["frame"] = None
            frame = stashed or recv_frame(conn)
            if frame is None:
                break
            ftype, payload = frame
            if ftype != BEGIN:
                continue

            meta = json.loads(payload.decode("utf-8")) if payload else {}
            rate = int(meta.get("sample_rate", SAMPLE_RATE))
            if rate != SAMPLE_RATE:
                send_json(conn, STATUS,
                          {"ok": False,
                           "error": "sample_rate must be {}, got {}".format(
                               SAMPLE_RATE, rate)})
                continue

            self.utterances += 1
            send_json(conn, STATUS, {"ok": True,
                                     "utterance": meta.get("utterance")})
            result = self.handler(meta, chunks())
            send_json(conn, STATUS, dict(result or {}, ok=True,
                                         utterance=meta.get("utterance")))


class IngestClient:
    """Reference client. imood voice carries its own copy of this (different
    repo, different venv) — this one is what the tests drive, and it is the
    definition the two implementations have to agree on."""

    # END 之後要等服務端把整句畫完才回 STATUS，所以逾時要蓋過一句話的生成
    # 時間，不是網路往返時間。pose-driven 在 4090 上一句 5 秒的話要 30 秒以上，
    # 預設 30 秒會在服務端一切正常的情況下讓 client 誤判失敗。
    def __init__(self, host, port, timeout=300.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def begin(self, session, utterance, voice="female", emotion=None, text=None):
        send_json(self.sock, BEGIN, {
            "session": session, "utterance": utterance, "voice": voice,
            "sample_rate": SAMPLE_RATE, "emotion": emotion, "text": text})
        return self._status()

    def audio(self, pcm_bytes):
        send_frame(self.sock, AUDIO, pcm_bytes)

    def end(self):
        send_frame(self.sock, END)
        return self._status()

    def _status(self):
        frame = recv_frame(self.sock)
        if frame is None:
            raise ConnectionError("server closed before sending status")
        ftype, payload = frame
        if ftype != STATUS:
            raise ValueError("expected STATUS, got type {}".format(ftype))
        return json.loads(payload.decode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
