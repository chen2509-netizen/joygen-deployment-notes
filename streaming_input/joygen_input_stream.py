"""
joygen_input_stream.py — one-shot input streaming from a file.

The regression harness: it drives FramePipeline with a file source so the
output can be compared frame by frame against the offline baseline. The live
path is service.py, which drives the same pipeline from the network.

Offline JoyGen needs the whole audio file before anything starts —
audio2feat() mel-spectrograms it end to end (src/audio2feature.py:114). Here
the audio arrives 320ms at a time and the only up-front work is the avatar,
which does not depend on the audio at all.

No JoyGen source is modified; this module imports it. Everything it writes
goes under joygen-deployment-notes.
"""

import argparse
import os
import sys
import time
from datetime import datetime

from streaming_input.audio_source import FileAudioSource
from streaming_input.config import load_config, resolve_dir
from streaming_input.eventlog import EventLog
from streaming_input.frame_timer import FrameTimer, PhaseTimer, write_report
from streaming_input.pipeline import SAMPLE_RATE, FramePipeline
from streaming_input.ring_buffer import AudioRingBuffer

# cwd is always the JoyGen checkout, so a bare "stream.sdp" would be written
# into a repo that must stay untouched. See service.py for the same note.
DEFAULT_SDP = str(resolve_dir("", "results") / "stream_input.sdp")


def main(args):
    cfg = load_config(args.config)
    jin = cfg.joygen_input

    session = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = resolve_dir(cfg.logging.dir, "logs")
    log = EventLog(path=str(log_dir / "input_{}.jsonl".format(session)),
                   session_id=session, stdout=cfg.logging.stdout,
                   per_frame=cfg.logging.per_frame)
    log.event("run.start", audio=args.audio_path, video=args.video_path,
              target=args.target, config=cfg.config_path)

    audio_basename = os.path.basename(args.audio_path).split(".")[0]
    # The intermediate dir is keyed by (video, audio) because edit_expression
    # bakes the audio-driven expression into the depth maps. With pose_driven
    # off none of that is read — only _ori/_face/_box, which depend on the
    # video alone — so a directory built for other audio is a valid avatar
    # source, and the only way to test audio that has none of its own.
    if args.intermediate_audio_key and args.enable_pose_driven \
            and not args.avatar3d_cache:
        sys.exit("[error] --intermediate_audio_key needs pose_driven off, or "
                 "an --avatar3d_cache to render the depth instead of reading "
                 "maps that belong to different audio")

    if args.debug:
        args.png_dir = str(
            resolve_dir(args.result_dir, "results/input_stream") /
            "{}#{}".format(os.path.basename(args.video_path).split(".")[0],
                           audio_basename))
    else:
        args.png_dir = None

    phases = PhaseTimer()
    pipe = FramePipeline(cfg, args, log, phases=phases)

    feeder = pipe.make_feeder(
        offline_audio_path=args.audio_path if args.offline_features else None)
    log.event("features.mode",
              mode="offline" if args.offline_features else "streaming")

    ring = AudioRingBuffer(int(SAMPLE_RATE * jin.ring_buffer_s), SAMPLE_RATE)
    source = FileAudioSource(ring, args.audio_path, chunk_ms=cfg.tts.chunk_ms,
                             pace=args.pace, jitter_ms=args.jitter_ms,
                             stall_at=args.stall_at, stall_ms=args.stall_ms,
                             log=log)

    motion = pipe.make_motion() if args.stream_motion else None
    if motion is not None:
        log.event("motion.mode", mode="streaming", **motion.stats())

    exp_source = None
    if args.exp_npy and motion is None:
        import numpy as np
        exp_track = np.load(args.exp_npy)
        log.event("exp.loaded", path=args.exp_npy, frames=int(len(exp_track)))

        def exp_source(local_frame):
            # No wrap-around: past the end there is no expression to show, and
            # looping back to frame 0 would restart the mouth mid-sentence.
            if local_frame < len(exp_track):
                return exp_track[local_frame]
            return None

    t_frame = FrameTimer("gen+blend+send", verbose=args.verbose)
    t_frame.start()
    source.start()
    with phases.phase("3_stream"):
        stats = pipe.run_utterance(feeder, ring, read_timeout=args.read_timeout,
                                   max_frames=args.max_frames, tag="file",
                                   exp_source=exp_source, motion=motion)
    pipe.close()

    if source.error is not None:
        log.event("run.source_error", error=repr(source.error))
    log.event("run.done", **stats)
    print("[info] frames streamed: {} -> {}".format(stats["frames"], args.target))
    print("[info] ring: {}".format(stats["ring"]))
    print("[info] feeder: {}".format(stats["feeder"]))

    meta = dict(stats, mode="input_stream", audio=args.audio_path,
                video=args.video_path, fps=pipe.fps,
                batch_frames=pipe.batch_frames, img_size=pipe.img_size,
                pose_driven=bool(pipe.pose_driven),
                avatar_frames=pipe.avatar.n_frames,
                loop_mode=pipe.avatar.loop_mode,
                stream_frame_size="{}x{}".format(pipe.avatar.width,
                                                 pipe.avatar.height),
                target=args.target, pace=args.pace, log=log.path,
                first_frame_after_start_ms=stats["first_frame_ms"])
    report = args.report or str(
        resolve_dir("", "timing") / "input_stream_{}".format(session))
    write_report(report, phases, [t_frame], meta=meta, mode="input_stream")
    log.close()
    return stats["frames"]


def build_parser():
    p = argparse.ArgumentParser(description="JoyGen input streaming (file)")
    p.add_argument("--audio_path", required=True)
    p.add_argument("--video_path", required=True)
    p.add_argument("--intermediate_dir", required=True)
    p.add_argument("--config", default=None, help="configs/pipeline.yaml")
    p.add_argument("--vae_model_path", default="pretrained_models/sd-vae-ft-mse")
    p.add_argument("--whisper_model_path", default="pretrained_models/whisper/tiny.pt")
    p.add_argument("--unet_model_path", default="pretrained_models/joygen")
    p.add_argument("--target", default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--result_dir", default="",
                   help="--debug PNG output; absolute, or relative to the notes root")
    p.add_argument("--report", default=None)
    p.add_argument("--sdp_file", default=DEFAULT_SDP,
                   help="RTP 接收端要的 SDP。預設寫在 notes 底下："
                        "cwd 是 JoyGen，裸檔名會直接寫進唯讀的 repo")
    p.add_argument("--enable_pose_driven", action="store_true")
    p.add_argument("--no_pose_driven", action="store_true")
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="also write PNGs for compare_frames (slows the loop)")
    p.add_argument("--pace", choices=["fast", "realtime"], default="fast")
    p.add_argument("--jitter_ms", type=float, default=0.0)
    p.add_argument("--stall_at", type=int, default=None)
    p.add_argument("--stall_ms", type=int, default=0)
    p.add_argument("--intermediate_audio_key", default=None,
                   help="read the avatar from the intermediate dir built for "
                        "this audio instead of --audio_path's; pose_driven "
                        "must be off")
    p.add_argument("--avatar3d_cache", default=None,
                   help="npz from streaming_input.avatar3d; enables pose-driven "
                        "depth rendered per frame instead of read from JPEGs")
    p.add_argument("--exp_npy", default=None,
                   help="expression coefficients for the depth renderer. Until "
                        "audio2motion is streamed this is the offline .npy, "
                        "which is what makes the pose-driven path testable")
    p.add_argument("--stream_motion", action="store_true",
                   help="derive the expression coefficients from the incoming "
                        "audio instead of --exp_npy; completes the pose-driven "
                        "path end to end")
    p.add_argument("--offline_features", action="store_true",
                   help="control run: whole-file audio2feat() through the "
                        "same pipeline, to isolate the streaming features")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--read_timeout", type=float, default=30.0)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.target is None:
        args.target = load_config(args.config).joygen_output.target
    t0 = time.time()
    main(args)
    print("[info] wall clock total: {:.1f}s".format(time.time() - t0))
