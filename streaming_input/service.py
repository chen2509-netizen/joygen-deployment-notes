"""
service.py — JoyGen as a resident service.

Loads the models and the avatar once, then waits for imood voice to push
utterances over the ingest socket (see ingest.py). Video goes out over a
single continuous ffmpeg process, so the RTP receiver connects once and stays
connected across the whole conversation.

Between utterances the stream does not stop: idle frames keep flowing so the
receiver has something to decode and the avatar stays on screen. Without them
a viewer sees the picture freeze after every sentence, and an RTP receiver
that loses its keyframe cadence can take seconds to recover.

Utterances are serialised. A second one arriving while the first is still
generating waits for the sink; nothing interleaves onto the stream. Barge-in
(cancel the current utterance when the user speaks again) is not implemented —
it needs a decision about what the avatar should do mid-sentence, which is a
product question, not a plumbing one.
"""

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime

import numpy as np

from streaming_input.config import load_config, resolve_dir
from streaming_input.eventlog import EventLog
from streaming_input.ingest import IngestServer
from streaming_input.pipeline import SAMPLE_RATE, FramePipeline
from streaming_input.ring_buffer import AudioRingBuffer

# cwd is always the JoyGen checkout (utils/blending.py loads its weights from
# relative paths at import time), so a bare "stream.sdp" lands inside a repo
# that must stay untouched. run_input.sh always passes --sdp_file, but calling
# the module directly should not be a trap.
DEFAULT_SDP = str(resolve_dir("", "results") / "stream_input.sdp")


class JoyGenService:
    def __init__(self, cfg, args, log):
        self.cfg = cfg
        self.args = args
        self.log = log
        self.pipe = FramePipeline(cfg, args, log)
        self.fps = self.pipe.fps

        # One utterance on the sink at a time. Idle takes the same lock, so a
        # frame is never half written by one path while the other appends.
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._stop = threading.Event()

        jin = cfg.joygen_input
        self.prebuffer_samples = int(SAMPLE_RATE * jin.prebuffer_ms / 1000)
        self.ring_capacity = int(SAMPLE_RATE * jin.ring_buffer_s)
        self.idle_enabled = not args.no_idle
        # Only worth running when something consumes the coefficients. Without
        # the depth renderer it would cost a HuBERT pass per hop and also
        # delay every batch waiting for frames nothing reads.
        self.stream_motion = bool(args.stream_motion) and \
            self.pipe.depth_renderer is not None

        jout = cfg.joygen_output
        self.utterance_dir = resolve_dir(jout.get("utterance_dir", ""),
                                         "results/utterances")
        self.keep_utterances = int(jout.get("keep_utterances", 20))
        # Idle frames only make sense when something is watching a continuous
        # stream. In utterance_file mode each clip is its own file.
        if self.pipe.sink_mode == "utterance_file":
            self.idle_enabled = False

        self.server = IngestServer(jin.ingest_host, jin.ingest_port,
                                   self.on_utterance, log=log)
        self.server.start_listening()

        self.total_utterances = 0
        self.total_frames = 0

    # -- ingest -----------------------------------------------------

    def on_utterance(self, meta, chunks):
        """Runs on the connection thread. Reading `chunks` blocks on the
        socket, and the feeder thread below blocks on the ring when it is
        full, so backpressure reaches imood voice through TCP instead of
        showing up here as dropped audio."""
        tag = meta.get("utterance") or "u{}".format(self.total_utterances + 1)
        voice = meta.get("voice")
        emotion = meta.get("emotion")     # reserved for the BERT integration
        self.log.event("utterance.begin", utterance=tag, voice=voice,
                       emotion=emotion, session=meta.get("session"),
                       text_len=len(meta.get("text") or ""))

        ring = AudioRingBuffer(self.ring_capacity, SAMPLE_RATE)
        t0 = time.perf_counter()
        counters = {"bytes": 0, "chunks": 0, "first_chunk_ms": None}

        def pump():
            try:
                for payload in chunks:
                    if not payload:
                        continue
                    if counters["first_chunk_ms"] is None:
                        counters["first_chunk_ms"] = \
                            (time.perf_counter() - t0) * 1000.0
                    counters["bytes"] += len(payload)
                    counters["chunks"] += 1
                    samples = np.frombuffer(payload, dtype=np.int16) \
                        .astype(np.float32) / 32768.0
                    ring.put(samples, block=True)
                    self.log.frame("ingest.chunk", utterance=tag,
                                   seq=counters["chunks"],
                                   ms=round(samples.size * 1000.0 / SAMPLE_RATE, 1),
                                   level_ms=round(ring.level_ms, 1))
            except Exception as exc:                  # noqa: BLE001
                self.log.event("ingest.pump_error", utterance=tag,
                               error=repr(exc))
            finally:
                ring.close()

        pump_thread = threading.Thread(target=pump, name="ingest-pump",
                                       daemon=True)
        pump_thread.start()

        # Hold off until there is something to chew on, so the first UNet batch
        # is not built from a sliver of audio and then starved.
        deadline = time.perf_counter() + self.args.prebuffer_timeout
        while (ring.level < self.prebuffer_samples and not ring.closed
               and time.perf_counter() < deadline):
            time.sleep(0.005)
        prebuffer_ms = (time.perf_counter() - t0) * 1000.0

        # The face the user picked. `avatar` is the explicit field; `voice`
        # is accepted as a fallback because picking a face in the UI is what
        # picks the voice, so one value often serves for both.
        avatar_id = meta.get("avatar") or voice

        out_path = None
        if self.pipe.sink_mode == "utterance_file":
            out_path = str(self.utterance_dir / "{}.mp4".format(tag))

        self._busy.set()
        with self._lock:
            # Both are per-utterance: the whisper anchor grid and the motion
            # window/crossfade state must restart with the sentence, not run
            # from service start. The weights behind them are shared.
            feeder = self.pipe.make_feeder()
            motion = self.pipe.make_motion() if self.stream_motion else None
            stats = self.pipe.run_utterance(
                feeder, ring, read_timeout=self.args.read_timeout, tag=tag,
                motion=motion, avatar_id=avatar_id, output_path=out_path)
        self._busy.clear()
        self._prune_utterances()
        pump_thread.join(timeout=5)

        self.total_utterances += 1
        self.total_frames += stats["frames"]

        result = {
            "utterance": tag,
            "frames": stats["frames"],
            "audio_ms": round(counters["bytes"] / 2.0 / SAMPLE_RATE * 1000.0, 1),
            "chunks": counters["chunks"],
            "first_chunk_ms": round(counters["first_chunk_ms"] or 0, 1),
            "prebuffer_ms": round(prebuffer_ms, 1),
            # The number the A100 test is about: first PCM byte in to first
            # frame out, measured on this side of the wire.
            "ingest_to_first_frame_ms": round(
                (counters["first_chunk_ms"] or 0) + stats["first_frame_ms"], 1),
            "utterance_ms": stats["utterance_ms"],
            "avatar": stats.get("avatar"),
            # The web UI swaps its still photo for this once it exists.
            # A bare filename, not a path: the two sides see the directory
            # under different mount points.
            "video": os.path.basename(stats["video_path"])
            if stats.get("video_path") else None,
            "ring": stats["ring"],
            "feeder": stats["feeder"],
            "motion": stats.get("motion"),
            "exp_frames": stats.get("exp_frames"),
        }
        self.log.event("utterance.done", **result)
        return result

    def _prune_utterances(self):
        """Keep only the most recent clips. A long conversation would
        otherwise fill the disk with mp4s nobody will watch again."""
        if self.keep_utterances <= 0:
            return
        try:
            clips = sorted(self.utterance_dir.glob("*.mp4"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        for old in clips[self.keep_utterances:]:
            try:
                old.unlink()
            except OSError:
                pass

    # -- idle -------------------------------------------------------

    def idle_loop(self):
        """Paced at fps so the stream keeps its cadence when nobody speaks."""
        period = 1.0 / self.fps
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            if self._busy.is_set() or not self.idle_enabled:
                time.sleep(0.02)
                next_tick = time.perf_counter()
                continue
            if self._lock.acquire(timeout=0.05):
                try:
                    self.pipe.idle(1)
                finally:
                    self._lock.release()
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    # -- lifecycle --------------------------------------------------

    def serve_forever(self):
        self.server.start()
        self.log.event("service.ready",
                       ingest="{}:{}".format(self.server.host, self.server.port),
                       target=self.args.target, fps=self.fps,
                       pose_driven=bool(self.pipe.pose_driven),
                       depth_renderer=self.pipe.depth_renderer is not None,
                       stream_motion=self.stream_motion,
                       sink_mode=self.pipe.sink_mode,
                       avatars=self.pipe.avatar_ids,
                       avatar_frames=self.pipe.avatar.n_frames,
                       idle=self.idle_enabled)
        where = (str(self.utterance_dir) if self.pipe.sink_mode == "utterance_file"
                 else self.args.target)
        print("[service] ingest on {}:{}  ->  {}  (avatars: {})".format(
            self.server.host, self.server.port, where,
            ", ".join(self.pipe.avatar_ids)), flush=True)
        try:
            self.idle_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self.server.stop()
        with self._lock:
            self.pipe.close()
        self.log.event("service.stopped", utterances=self.total_utterances,
                       frames=self.pipe.out_frame)
        self.log.close()
        print("[service] stopped after {} utterance(s), {} frames".format(
            self.total_utterances, self.pipe.out_frame), flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="JoyGen resident streaming service")
    # 這三個給單一 avatar 用。要多張臉就別給，改在 configs/pipeline.yaml 的
    # joygen_input.avatars 列出來。
    p.add_argument("--video_path", default=None)
    p.add_argument("--intermediate_dir", default=None)
    p.add_argument("--intermediate_audio_key", default=None,
                   help="which audio's intermediate dir holds the avatar; "
                        "with pose_driven off its contents are audio-independent")
    p.add_argument("--config", default=None)
    p.add_argument("--vae_model_path", default="pretrained_models/sd-vae-ft-mse")
    p.add_argument("--whisper_model_path", default="pretrained_models/whisper/tiny.pt")
    p.add_argument("--unet_model_path", default="pretrained_models/joygen")
    p.add_argument("--target", default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--sdp_file", default=DEFAULT_SDP,
                   help="RTP 接收端要的 SDP。預設寫在 notes 底下："
                        "cwd 是 JoyGen，裸檔名會直接寫進唯讀的 repo")
    p.add_argument("--enable_pose_driven", action="store_true",
                   help="needs --avatar3d_cache and --stream_motion: the depth "
                        "is rendered per frame from the audio, not read from "
                        "maps baked for one specific audio file")
    p.add_argument("--no_pose_driven", action="store_true")
    p.add_argument("--avatar3d_cache", default=None,
                   help="npz from `run_input.sh cache3d` for this avatar")
    p.add_argument("--stream_motion", action="store_true",
                   help="derive expression coefficients from the incoming "
                        "audio (audio2motion); required for pose-driven")
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--no_idle", action="store_true",
                   help="do not emit frames between utterances; use for a file "
                        "target, where idle frames would just pad the file")
    p.add_argument("--png_dir", default=None)
    p.add_argument("--prebuffer_timeout", type=float, default=5.0)
    p.add_argument("--read_timeout", type=float, default=30.0)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.audio_path = None            # the service has no single audio file
    args.offline_features = False
    cfg = load_config(args.config)
    if args.target is None:
        args.target = cfg.joygen_output.target

    # The avatar is shared across utterances, so pose-driven cannot read the
    # offline depth maps — those belong to one audio file. It works only with
    # the M4 path: a cached 3DMM fit, re-rendered each frame from expression
    # coefficients derived from the audio that just arrived.
    if args.enable_pose_driven:
        has_cache = bool(args.avatar3d_cache) or all(
            a.get("avatar3d_cache") for a in (cfg.joygen_input.get("avatars") or []))
        missing = []
        if not has_cache:
            missing.append("--avatar3d_cache（或在 config 的每張 avatar 填 avatar3d_cache）")
        if not args.stream_motion:
            missing.append("--stream_motion")
        if missing:
            sys.exit("[error] --enable_pose_driven needs {} (build the cache "
                     "with `run_input.sh cache3d`)".format(" and ".join(missing)))
    elif args.stream_motion:
        print("[service] --stream_motion without pose-driven has no effect: "
              "the expression coefficients only feed the depth renderer",
              flush=True)

    session = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = resolve_dir(cfg.logging.dir, "logs")
    log = EventLog(path=str(log_dir / "service_{}.jsonl".format(session)),
                   session_id=session, stdout=cfg.logging.stdout,
                   per_frame=cfg.logging.per_frame)

    service = JoyGenService(cfg, args, log)

    def on_signal(signum, frame):
        service.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    service.serve_forever()


if __name__ == "__main__":
    main()
