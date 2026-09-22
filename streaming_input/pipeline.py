"""
pipeline.py — models, avatar and sink, held open across utterances.

M2 ran this as a script: load, generate, exit. A live system cannot, because
loading costs ~2s of models plus ~6s of avatar preprocessing, and the RTP
receiver must not see the stream stop between sentences. So the expensive
parts live here and are reused, while each utterance gets its own feature
extractor.

The feature extractor has to be per-utterance: it anchors its encoder window
to an absolute 30s grid (see stream_feature.py), and that grid has to restart
with each utterance, not run from service start. Rebuilding it also drops the
accumulated rows, which is what keeps a long-running service from growing an
ever-larger feature array.

Two frame counters, deliberately:
  out_frame   — since service start. Drives the avatar and the audio the sink
                has been fed, both of which are continuous across utterances.
  local frame — within the utterance. Drives feature slicing, because the
                rows restart at zero each time.
"""

import os
import time

import cv2
import numpy as np
import torch
from diffusers import UNet2DConditionModel

from src.audio2feature import Audio2Feature
from src.modules.vae import VAE
from src.pe import PositionalEncoding
from utils.blending import get_image

from streaming_input.avatar3d import Depth3DRenderer
from streaming_input.avatar_cache import AvatarCache
from streaming_input.config import resolve_file
from streaming_input.sinks import FFmpegSink, PngSink
from streaming_input.stream_feature import OfflineFeatures, StreamingAudio2Feature

SAMPLE_RATE = 16000


def to_pcm16(samples):
    """float32 [-1,1] back to the wire format the sink and TTS both use."""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def rows_needed_for(frame_idx, fps):
    """Highest feature row get_sliced_feature() touches for this frame, plus
    one. Mirrors its index arithmetic rather than guessing."""
    return int(frame_idx * 50.0 / fps) + 6


def total_frames_for(rows, fps):
    """How many frames feature2chunks() would produce from `rows` rows.

    Its loop appends a chunk and only then tests `start_idx > len(features)`,
    so it overshoots by two, not zero. Matching it exactly matters: the
    baseline PNGs this gets compared against came out of that loop."""
    return int(rows * fps / 50.0) + 2


class FramePipeline:
    def __init__(self, cfg, args, log, phases=None):
        self.cfg = cfg
        self.log = log
        jin = cfg.joygen_input
        jout = cfg.joygen_output

        self.fps = args.fps or jin.fps
        self.batch_frames = jin.batch_frames
        self.img_size = jin.img_size
        self.samples_per_frame = int(round(SAMPLE_RATE / self.fps))
        self.read_samples = self.samples_per_frame * self.batch_frames

        self.pose_driven = jin.enable_pose_driven
        if getattr(args, "enable_pose_driven", False):
            self.pose_driven = True
        if getattr(args, "no_pose_driven", False):
            self.pose_driven = False

        device = torch.device("cuda", jin.gpu_id)
        self.device = device

        with self._phase(phases, "1_model_load"):
            self.audio_processor = Audio2Feature(model_path=args.whisper_model_path)
            self.vae = VAE(model_path=args.vae_model_path,
                           resized_img=self.img_size, device=device)
            self.unet = UNet2DConditionModel.from_pretrained(
                args.unet_model_path).to(device=device)
            self.pe = PositionalEncoding(d_model=384)
            self.timesteps = torch.tensor([0], device=device)

        with self._phase(phases, "1b_warmup"):
            dummy_latent = torch.zeros(
                (1, self.unet.config.in_channels,
                 self.img_size // 8, self.img_size // 8),
                device=device, dtype=self.unet.dtype)
            dummy_audio = self.pe(torch.zeros((1, 50, 384), device=device))
            with torch.no_grad():
                self.unet(dummy_latent, self.timesteps,
                          encoder_hidden_states=dummy_audio).sample
            torch.cuda.synchronize(device)

        # One set of model weights, several faces. Each face costs an avatar
        # cache (decoded frames + latents, a few hundred MB) and optionally a
        # 3DMM cache; the UNet/VAE/whisper/HuBERT are shared, which is the
        # whole reason not to run one service per face.
        self._avatars = {}
        self._renderers = {}
        self.avatar = None
        self.depth_renderer = None

        with self._phase(phases, "2_avatar_cache"):
            for spec in self._avatar_specs(args, jin):
                self._load_avatar(spec, phases, log)
        self.use_avatar(self.default_avatar)

        # audio2motion weights are loaded once and shared; only the windowing
        # state is per-utterance (see make_motion). A resident service must not
        # pay a checkpoint load on the first sentence someone speaks.
        self._a2m_model = None
        if getattr(args, "stream_motion", False):
            with self._phase(phases, "2c_audio2motion"):
                self._load_a2m()
            log.event("audio2motion.ready", ckpt=jin.a2m_ckpt)

        # Two output shapes, because they answer different questions.
        #
        # "persistent" keeps one ffmpeg alive for the whole service and pushes
        # RTP continuously — what a live avatar needs, and what M2-M4 measured.
        #
        # "utterance_file" writes one mp4 per sentence and hands back its path.
        # A browser cannot play RTP, so this is what the web UI uses: it shows
        # a still until the file is ready, then plays it. It also drops the
        # idle frames, since nothing is watching between sentences.
        self.sink_mode = jout.get("mode", "persistent")
        if self.sink_mode not in ("persistent", "utterance_file"):
            raise ValueError("joygen_output.mode must be 'persistent' or "
                             "'utterance_file'")
        self.sink = None
        self._sink_args = dict(
            fps=self.fps, preset=jout.preset, tune=jout.tune,
            bitrate=jout.bitrate, gop=jout.gop, pkt_size=jout.pkt_size,
            sdp_path=args.sdp_file,
            audio_stream=jout.mux_audio and not args.no_audio,
            verbose=args.verbose)
        if self.sink_mode == "persistent":
            self.open_sink(args.target)

        self.png_sink = None
        if getattr(args, "png_dir", None):
            self.png_sink = PngSink(args.png_dir)
            log.event("debug.png", dir=args.png_dir)

        self.out_frame = 0
        self.utterances = 0

    # -- avatars ----------------------------------------------------

    def _avatar_specs(self, args, jin):
        """Faces to load. `joygen_input.avatars` is the multi-face form; the
        single-avatar CLI flags stay supported so the one-shot regression
        harness and the older serve invocations keep working unchanged."""
        listed = jin.get("avatars") or []
        if listed and not getattr(args, "video_path", None):
            self.default_avatar = jin.get("default_avatar") or listed[0]["id"]
            return listed
        if listed and getattr(args, "avatars_from_config", False):
            self.default_avatar = jin.get("default_avatar") or listed[0]["id"]
            return listed

        self.default_avatar = "default"
        return [{
            "id": "default",
            "video_path": args.video_path,
            "intermediate_dir": args.intermediate_dir,
            "audio_key": args.intermediate_audio_key or
                         os.path.basename(args.audio_path or "").split(".")[0],
            "avatar3d_cache": getattr(args, "avatar3d_cache", None) or
                              jin.get("avatar3d_cache", ""),
        }]

    def _load_avatar(self, spec, phases, log):
        aid = spec["id"]
        # Relative paths in the yaml resolve against the notes root, so the
        # config carries no machine-specific absolute path. CLI --avatar3d_cache
        # goes through the same call: a relative arg typed from the JoyGen cwd
        # would otherwise land inside the JoyGen tree.
        cache3d = resolve_file(spec.get("avatar3d_cache") or "")

        # With a 3DMM cache the depth is rendered per frame, so the avatar
        # cache must not read the offline <i>_depth_edit_exp.jpg — those
        # belong to one audio file, which is the coupling M4 removed.
        self._avatars[aid] = AvatarCache(
            self.vae, spec["intermediate_dir"], spec["video_path"],
            spec["audio_key"], img_size=self.img_size,
            enable_pose_driven=self.pose_driven and not cache3d,
            loop_mode=self.cfg.joygen_input.loop_mode,
            start_frame=spec.get('still_frame', 0), log=log)
        log.event("avatar.loaded", id=aid, video=spec["video_path"],
                  frames=self._avatars[aid].n_frames)

        if self.pose_driven and cache3d:
            with self._phase(phases, "2b_depth_renderer_{}".format(aid)):
                self._renderers[aid] = Depth3DRenderer(
                    cache3d, img_size=self.img_size,
                    gpu_id=self.cfg.joygen_input.gpu_id)
            log.event("avatar3d.ready", id=aid, cache=cache3d,
                      frames=self._renderers[aid].n_frames)

    def use_avatar(self, avatar_id):
        """Pick the face for the next utterance. Unknown ids fall back to the
        default rather than failing — a typo in the frontend should not take
        the avatar off screen."""
        if avatar_id not in self._avatars:
            if avatar_id:
                self.log.event("avatar.unknown", requested=avatar_id,
                               using=self.default_avatar)
            avatar_id = self.default_avatar
        self.active_avatar = avatar_id
        self.avatar = self._avatars[avatar_id]
        self.depth_renderer = self._renderers.get(avatar_id)
        return avatar_id

    @property
    def avatar_ids(self):
        return sorted(self._avatars)

    # -- sink lifecycle ---------------------------------------------

    def open_sink(self, target):
        """Frame size follows the active avatar, so it has to be picked before
        the encoder starts."""
        self.sink = FFmpegSink(width=self.avatar.width,
                               height=self.avatar.height,
                               target=target, **self._sink_args)
        return self.sink

    def close_sink(self):
        if self.sink is not None:
            self.sink.close()
            self.sink = None

    @staticmethod
    def _phase(phases, name):
        if phases is None:
            from contextlib import contextmanager

            @contextmanager
            def noop():
                yield
            return noop()
        return phases.phase(name)

    # -- per utterance ----------------------------------------------

    def make_feeder(self, offline_audio_path=None):
        """A fresh extractor per utterance: the anchor grid restarts, and the
        rows accumulated by the previous utterance are released."""
        if offline_audio_path:
            return OfflineFeatures(self.audio_processor, offline_audio_path)
        jin = self.cfg.joygen_input
        return StreamingAudio2Feature(
            self.audio_processor,
            left_context_ms=jin.whisper_left_context_ms,
            right_context_ms=jin.whisper_right_context_ms,
            mel_norm=jin.whisper_mel_norm,
            fp16=jin.whisper_fp16,
            window_mode=jin.whisper_window_mode)

    def _load_a2m(self):
        if self._a2m_model is None:
            from inference_audio2motion import Audio2Motion

            ckpt = self.cfg.joygen_input.a2m_ckpt
            holder = Audio2Motion(ckpt, inp={"a2m_ckpt": ckpt})
            self._a2m_model = holder.audio2secc_model
        return self._a2m_model

    def make_motion(self):
        """A fresh expression extractor per utterance, on shared weights.

        Same reason the whisper feeder is rebuilt: the sliding window, the
        crossfade buffer and the running audio statistics all have to restart
        with the utterance rather than carry over from the previous sentence.
        """
        from streaming_input.stream_motion import StreamingAudio2Motion

        jin = self.cfg.joygen_input
        return StreamingAudio2Motion(
            jin.a2m_ckpt, jin.hubert_path,
            window_ms=jin.motion_window_ms,
            hop_ms=jin.motion_hop_ms,
            right_context_ms=jin.motion_right_context_ms,
            fade_ms=jin.motion_fade_ms,
            temperature=jin.motion_temperature,
            mouth_amp=jin.motion_mouth_amp,
            seed=jin.motion_seed,
            audio_norm=jin.motion_audio_norm,
            model=self._load_a2m())

    def run_utterance(self, feeder, ring, read_timeout=30.0, max_frames=0,
                      tag=None, exp_source=None, motion=None,
                      avatar_id=None, output_path=None):
        """Drain `ring` into frames until it closes. Returns per-utterance
        stats, including the latency from the first audio byte to the first
        frame going out — the number the A100 test is really about."""
        self.utterances += 1
        if avatar_id is not None:
            self.use_avatar(avatar_id)

        own_sink = False
        if self.sink_mode == "utterance_file":
            if not output_path:
                raise ValueError("utterance_file mode needs an output_path")
            # Restart at avatar frame 0 so every clip opens on the frame the
            # web UI is already showing as a still — the swap from photo to
            # video then lands on the same head pose and background.
            self.out_frame = 0
            self.open_sink(output_path)
            own_sink = True
        # exp_source(local_frame) -> 64 expression coefficients. Pose-driven
        # streaming needs one per frame; without it the depth renderer has
        # nothing to render and the run falls back to the cached latent.
        exp_frames = []
        if motion is not None:
            # Expression coefficients arrive with the audio, like the whisper
            # rows do. Frames then need both before they can be generated —
            # see the gate in drain().
            def exp_source(local_frame):
                if local_frame < len(exp_frames):
                    return exp_frames[local_frame]
                return None

        self._exp_source = exp_source
        rows_list = []
        local_frame = 0
        total_rows = 0
        t_start = time.perf_counter()
        first_audio_ms = None
        first_frame_ms = None
        start_out_frame = self.out_frame

        def feats():
            return np.concatenate(rows_list) if rows_list else \
                np.zeros((0, 5, 384), np.float32)

        def drain(final):
            nonlocal local_frame, first_frame_ms
            array = feats()
            rows = len(array)
            limit = total_frames_for(rows, self.fps) if final else None

            while True:
                if max_frames and local_frame >= max_frames:
                    return
                if final:
                    if local_frame >= limit:
                        return
                    count = min(self.batch_frames, limit - local_frame)
                else:
                    if rows_needed_for(local_frame + self.batch_frames - 1,
                                       self.fps) > rows:
                        return
                    if motion is not None and \
                            len(exp_frames) < local_frame + self.batch_frames:
                        # Whisper rows are ready but the expression track is
                        # not. Generating now would render a neutral mouth for
                        # frames whose coefficients are still in flight.
                        return
                    count = self.batch_frames
                ffm = self._emit_batch(array, local_frame, count)
                if ffm is not None and first_frame_ms is None:
                    first_frame_ms = (time.perf_counter() - t_start) * 1000.0
                    self.log.event("frame.first", utterance=tag,
                                   ms=round(first_frame_ms, 1))
                local_frame += count

        while True:
            samples = ring.read(self.read_samples, timeout=read_timeout)
            if samples.size == 0:
                if ring.drained():
                    break
                self.log.event("ring.starved", utterance=tag,
                               level_ms=round(ring.level_ms, 1))
                continue
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - t_start) * 1000.0

            t_enc = time.perf_counter()
            rows = feeder.push(samples)
            if len(rows):
                rows_list.append(rows)
                total_rows += len(rows)
            if motion is not None:
                for frame in motion.push(samples):
                    exp_frames.append(frame)
            self.log.frame("feature.push", utterance=tag,
                           samples=int(samples.size), rows=int(len(rows)),
                           total_rows=total_rows,
                           encode_ms=round((time.perf_counter() - t_enc) * 1000.0, 1),
                           level_ms=round(ring.level_ms, 1))

            self.sink.append_audio(to_pcm16(samples))
            drain(final=False)

        tail = feeder.flush()
        if len(tail):
            rows_list.append(tail)
            total_rows += len(tail)
        if motion is not None:
            for frame in motion.flush():
                exp_frames.append(frame)
        drain(final=True)

        if own_sink:
            # ffmpeg only writes the moov atom on exit, so the file is not
            # playable until the encoder has actually finished.
            self.close_sink()

        return {
            "frames": self.out_frame - start_out_frame,
            "rows": total_rows,
            "first_frame_ms": round(first_frame_ms or 0, 1),
            "first_audio_ms": round(first_audio_ms or 0, 1),
            "utterance_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
            "avatar": self.active_avatar,
            "video_path": output_path if own_sink else None,
            "ring": ring.stats(),
            "feeder": feeder.stats(),
            "motion": motion.stats() if motion is not None else None,
            "exp_frames": len(exp_frames),
        }

    def _emit_batch(self, feats, start_frame, count):
        """One UNet batch. Returns True once at least one frame went out."""
        t_b = time.perf_counter()
        whisper_batch, latent_batch = [], []
        exp_source = getattr(self, "_exp_source", None)
        depth_ms = 0.0
        for f in range(start_frame, start_frame + count):
            sliced, _ = self.audio_processor.get_sliced_feature(
                feature_array=feats, vid_idx=f, audio_feat_length=[2, 2],
                fps=self.fps)
            whisper_batch.append(sliced)

            out_idx = self.out_frame + (f - start_frame)
            if self.depth_renderer is not None and exp_source is not None:
                exp = exp_source(f)
                if exp is None:
                    # Past the end of the expression track: fall back to the
                    # cached latent rather than rendering a neutral mouth that
                    # would visibly snap shut mid-sentence.
                    latent_batch.append(self.avatar.latent(out_idx))
                else:
                    t_d = time.perf_counter()
                    src = self.avatar.source_index(out_idx)
                    box = [int(v) for v in self.avatar.boxes[src]]
                    depth_latent = self.depth_renderer.depth_latent(
                        src, exp, box, self.vae)
                    depth_ms += (time.perf_counter() - t_d) * 1000.0
                    latent_batch.append(
                        self.avatar.compose_latent(out_idx, depth_latent))
            else:
                latent_batch.append(self.avatar.latent(out_idx))

        audio_feature_batch = self.pe(torch.stack(
            [torch.FloatTensor(a) for a in whisper_batch]).to(self.unet.device))
        latents = torch.cat(latent_batch, dim=0).to(dtype=self.unet.dtype)

        t_unet = time.perf_counter()
        with torch.no_grad():
            pred = self.unet(latents, self.timesteps,
                             encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred)
        unet_ms = (time.perf_counter() - t_unet) * 1000.0

        t_blend = time.perf_counter()
        wrote = None
        for res_frame in recon:
            ori_img, box = self.avatar.blend_inputs(self.out_frame)
            x1, y1, x2, y2 = box
            try:
                crop = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                self.out_frame += 1
                continue
            combined = get_image(ori_img, crop, box)
            self.sink.write(combined)
            self.sink.write_audio_upto(self.out_frame)
            if self.png_sink is not None:
                self.png_sink.write(combined, self.out_frame + 1)
            self.out_frame += 1
            wrote = True

        self.log.frame("batch.done", start=start_frame, n=count,
                       depth_ms=round(depth_ms, 1),
                       unet_ms=round(unet_ms, 1),
                       blend_ms=round((time.perf_counter() - t_blend) * 1000.0, 1),
                       total_ms=round((time.perf_counter() - t_b) * 1000.0, 1))
        return wrote

    # -- between utterances -----------------------------------------

    def idle(self, n_frames):
        """Keep the stream alive while nobody is speaking.

        Emits the untouched source frames with silent audio. No GPU: the mouth
        is simply whatever the reference video does, which is also what the
        avatar should look like when it has nothing to say. There is a visible
        seam where generated mouths hand over to original ones; running
        silence through the UNet instead would remove it at the cost of
        occupying the GPU permanently. Left as a known issue.
        """
        silence = b"\x00" * (self.samples_per_frame * 2)
        for _ in range(n_frames):
            ori_img, _ = self.avatar.blend_inputs(self.out_frame)
            self.sink.write(ori_img)
            self.sink.append_audio(silence)
            self.sink.write_audio_upto(self.out_frame)
            self.out_frame += 1

    def close(self):
        self.close_sink()
        if self.png_sink is not None:
            self.png_sink.close()
