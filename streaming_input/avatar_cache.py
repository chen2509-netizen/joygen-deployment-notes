"""
avatar_cache.py — the reference video, preprocessed once.

Offline JoyGen redoes this for every (video, audio) pair. In a live system
the avatar is fixed and the audio is whatever the user said, so everything
that depends only on the video — the original frames, the crop boxes and
their VAE latents — is computed at startup and reused for every utterance.

With enable_pose_driven=False that is the whole intermediate directory the
pipeline needs: `_depth_edit_exp.jpg` and `_lm.npy` are read only on the
pose-driven path (inference_joygen.py:158,181), and those two are the
audio-dependent ones, produced by audio2motion + edit_expression. Dropping
them is what makes Path A streamable without touching stages 1 and 2.

The audio outlasts the clip, so frames loop. pingpong avoids the visible jump
that restarting at frame 0 produces when the last and first frames differ.
"""

import glob
import os

import cv2
import numpy as np
import torch

from inference_joygen import create_mouth_mask, mouth_region_indices


class AvatarCache:
    def __init__(self, vae, intermediate_dir, video_path, audio_basename,
                 img_size=256, enable_pose_driven=False, loop_mode="pingpong",
                 start_frame=0, log=None):
        if loop_mode not in ("pingpong", "restart"):
            raise ValueError("loop_mode must be 'pingpong' or 'restart'")

        self.loop_mode = loop_mode
        self.enable_pose_driven = enable_pose_driven
        # Which source frame a clip opens on. The web UI shows that same frame
        # as a still while it waits, so starting elsewhere would make the swap
        # from photo to video jump. Frame 0 of a talking-head clip is usually
        # mid-syllable; a frame with the eyes open makes a better portrait.
        self.start_frame = int(start_frame)
        video_basename = os.path.basename(video_path).split(".")[0]
        pose_path = os.path.join(intermediate_dir, video_basename,
                                 audio_basename, video_basename)
        if not os.path.isdir(pose_path):
            raise FileNotFoundError(
                "intermediate dir not found: {}".format(pose_path))

        ori_files = glob.glob(os.path.join(pose_path, "*_ori.jpg"))
        if not ori_files:
            raise FileNotFoundError("no *_ori.jpg under {}".format(pose_path))

        indices = [int(os.path.basename(p).split("_")[0]) for p in ori_files]
        n = int(np.max(indices)) + 1
        if n != len(ori_files):
            raise RuntimeError(
                "face detection failed on some frames: {} files but max index {}"
                .format(len(ori_files), n - 1))

        self.pose_path = pose_path
        self.n_frames = n
        self.ori_imgs = []
        self.boxes = []
        self.latents = []
        # Kept apart from the combined latent so pose-driven streaming can
        # swap the depth half per utterance while the face half stays cached.
        self.face_latents = []

        blank_depth_latent = None
        if not enable_pose_driven:
            blank = np.zeros((img_size, img_size, 3), np.uint8)
            blank_depth_latent = vae.get_latents_for_nomask(blank)

        for i in range(n):
            ori = cv2.imread(os.path.join(pose_path, "{}_ori.jpg".format(i)))
            crop = cv2.imread(os.path.join(pose_path, "{}_face.jpg".format(i)))
            box = np.load(os.path.join(pose_path, "{}_box.npy".format(i)))

            crop = cv2.resize(crop, (img_size, img_size),
                              interpolation=cv2.INTER_LANCZOS4)
            latent, _ = vae.get_latents_for_unet(crop)

            if enable_pose_driven:
                depth = cv2.imread(
                    os.path.join(pose_path, "{}_depth_edit_exp.jpg".format(i)))
                lmk = np.array(
                    np.load(os.path.join(pose_path, "{}_lm.npy".format(i))),
                    np.int32)
                lip_mask = create_mouth_mask(depth, lmk[mouth_region_indices, :])
                depth = depth * lip_mask
                depth = cv2.resize(depth, (img_size, img_size),
                                   interpolation=cv2.INTER_LANCZOS4)
                depth[:img_size // 2, ...] = 0
                depth_latent = vae.get_latents_for_nomask(depth)
            else:
                depth_latent = blank_depth_latent

            self.ori_imgs.append(ori)
            self.boxes.append(box)
            self.face_latents.append(latent)
            self.latents.append(torch.cat([latent, depth_latent], axis=1))

        self.height, self.width = self.ori_imgs[0].shape[:2]
        if log is not None:
            log.event("avatar.ready", frames=n, size="{}x{}".format(
                self.width, self.height), pose_driven=bool(enable_pose_driven),
                loop=loop_mode)

    def source_index(self, i):
        """Map an output frame index onto a cached frame."""
        n = self.n_frames
        i = i + self.start_frame
        if i < n:
            return i
        if self.loop_mode == "restart":
            return i % n
        if n == 1:
            return 0
        period = 2 * n - 2
        j = i % period
        return j if j < n else period - j

    def latent(self, i):
        return self.latents[self.source_index(i)]

    def face_latent(self, i):
        return self.face_latents[self.source_index(i)]

    def compose_latent(self, i, depth_latent):
        """Pose-driven streaming: this frame's cached face latent joined to a
        depth latent rendered from the audio just received."""
        return torch.cat([self.face_latents[self.source_index(i)],
                          depth_latent], axis=1)

    def blend_inputs(self, i):
        idx = self.source_index(i)
        return self.ori_imgs[idx], [int(v) for v in self.boxes[idx]]
