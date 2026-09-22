"""
avatar3d.py — pose-driven depth, split into what the video decides and what
the audio decides.

Offline, `inference_edit_expression.py` does both at once for a fixed (video,
audio) pair: it detects the face, fits a 3DMM, drops the audio-driven
expression into the coefficient vector, renders a depth map, and writes
`<i>_depth_edit_exp.jpg`. That is why `intermediate_dir` is keyed by both the
video and the audio, and why M2 had to run with pose_driven off — the depth
maps belong to one specific audio file.

The split that makes it streamable:

    per video, once     MTCNN + 3DMM fit -> id/tex/angle/gamma/trans (193 of
                        the 257 coefficients), plus the crop geometry
    per frame, live     drop in this frame's 64 expression coefficients and
                        re-render

`facerecon_model.get_depth_lm468_edit_exp()` reads nothing but the coefficient
vector (facerecon_model.py:279) — no per-frame state survives from the fit —
so the two halves are exactly separable. Measured on a 4090: the fit is the
expensive part, the re-render is 5-8.5ms, against a 40ms/frame budget.

Verified in M4 Spike A: refitting the same frame twice gives coefficients and
crop parameters that are bit-identical, and re-rendering the same coefficients
twice gives an identical image, so the cache is not an approximation of the
offline path — it is the offline path with the fit hoisted out of the loop.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

from inference_joygen import create_mouth_mask, mouth_region_indices

CACHE_VERSION = 1


def _make_opt(bfm_folder, checkpoints_dir, name, epoch, gpu_id):
    """deep3d's options are argparse-based and read sys.argv, so build them
    the only way the upstream code allows: by handing it an argv."""
    from deep3d_facerecon.options.test_options import TestOptions

    saved = sys.argv
    sys.argv = [
        "avatar3d",
        "--name", name,
        "--epoch", str(epoch),
        "--use_opengl", "False",
        "--checkpoints_dir", checkpoints_dir,
        "--bfm_folder", bfm_folder,
        "--gpu_ids", str(gpu_id),
    ]
    try:
        return TestOptions().parse()
    finally:
        sys.argv = saved


def load_facerecon(bfm_folder="./pretrained_models/BFM",
                   checkpoints_dir="./pretrained_models",
                   name="face_recon_feat0.2_augment", epoch=20, gpu_id=0):
    from deep3d_facerecon.models import create_model

    opt = _make_opt(bfm_folder, checkpoints_dir, name, epoch, gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    return model, opt


# --------------------------------------------------------------- build

def build_cache(video_frames_dir, n_frames, out_path, bfm_folder,
                checkpoints_dir, name="face_recon_feat0.2_augment", epoch=20,
                gpu_id=0, log=None):
    """Fit every frame of the avatar once and store what the live path needs.

    Reads `<i>_ori.jpg` from an existing intermediate directory rather than
    decoding the video, so the frames are exactly the ones the rest of the
    pipeline uses. Writes only to `out_path`.
    """
    from deep3d_facerecon.util.load_mats import load_lm3d
    from facenet_pytorch import MTCNN
    from inference_edit_expression import read_data_beta

    model, opt = load_facerecon(bfm_folder, checkpoints_dir, name, epoch, gpu_id)
    device = model.device
    lm3d_std = load_lm3d(opt.bfm_folder)
    mtcnn = MTCNN(image_size=224, margin=0, min_face_size=50,
                  selection_method="largest", keep_all=False, device=device)

    coeffs = np.zeros((n_frames, 257), np.float32)
    crop_params = np.zeros((n_frames, 7), np.float32)
    trans_s = np.zeros((n_frames,), np.float32)
    sizes = np.zeros((n_frames, 2), np.int32)
    ok = np.zeros((n_frames,), np.bool_)

    for i in range(n_frames):
        img_path = os.path.join(video_frames_dir, "{}_ori.jpg".format(i))
        img = Image.open(img_path)
        try:
            boxes, _, points = mtcnn.detect(img, landmarks=True)
            if boxes is None or len(boxes) == 0:
                raise RuntimeError("no face detected")
            p5 = np.array(points[0], np.float32)
            im_t, lm_t, tp, cp = read_data_beta(img_path, p5, lm3d_std)
        except Exception as exc:                      # noqa: BLE001
            print("[avatar3d] frame {} failed: {}".format(i, exc), flush=True)
            continue

        model.set_input({"imgs": im_t, "lms": lm_t})
        model.test()
        pred = model.get_coeff()

        # exp (80:144) stays zero — it is what the audio fills in later
        coeffs[i, :80] = pred["id"]
        coeffs[i, 144:224] = pred["tex"]
        coeffs[i, 224:227] = pred["angle"]
        coeffs[i, 227:254] = pred["gamma"]
        coeffs[i, 254:] = pred["trans"]
        crop_params[i] = np.asarray(cp, np.float32)
        trans_s[i] = float(tp[2])
        sizes[i] = img.size            # (w0, h0)
        ok[i] = True

        if log is not None and i % 20 == 0:
            log.event("avatar3d.fit", frame=i, total=n_frames)

    if not ok.all():
        missing = int((~ok).sum())
        raise RuntimeError(
            "3DMM fit failed on {} of {} frames; the avatar video needs a "
            "detectable face in every frame".format(missing, n_frames))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, version=CACHE_VERSION, coeffs=coeffs,
                        crop_params=crop_params, trans_s=trans_s, sizes=sizes,
                        n_frames=n_frames)
    print("[avatar3d] cached {} frames -> {}".format(n_frames, out_path),
          flush=True)
    return out_path


# ------------------------------------------------------------- runtime

class Depth3DRenderer:
    """Turns (frame index, expression coefficients) into the depth latent the
    UNet expects, at live speed."""

    def __init__(self, cache_path, img_size=256, gpu_id=0,
                 bfm_folder="./pretrained_models/BFM",
                 checkpoints_dir="./pretrained_models",
                 name="face_recon_feat0.2_augment", epoch=20, model=None):
        data = np.load(cache_path)
        if int(data["version"]) != CACHE_VERSION:
            raise ValueError("cache version {} != {}".format(
                int(data["version"]), CACHE_VERSION))
        self.coeffs = data["coeffs"]
        self.crop_params = data["crop_params"]
        self.trans_s = data["trans_s"]
        self.sizes = data["sizes"]
        self.n_frames = int(data["n_frames"])
        self.img_size = img_size

        self.model = model
        if self.model is None:
            self.model, _ = load_facerecon(bfm_folder, checkpoints_dir, name,
                                           epoch, gpu_id)
        self._prime()

    def _prime(self):
        """get_depth_lm468_edit_exp reads self.input_img.shape[2] to flip the
        landmarks (facerecon_model.py:293). That is just the 224px fit
        resolution, but the attribute has to exist, and after build_cache the
        model may be a fresh instance that has never seen an image."""
        if getattr(self.model, "input_img", None) is None:
            dummy = torch.zeros((1, 3, 224, 224), device=self.model.device)
            self.model.input_img = dummy

    @torch.no_grad()
    def render(self, frame_idx, exp_coeff):
        """Depth image (uint8, HxW) and 468 landmarks, both in the crop's
        coordinate frame, for one frame."""
        i = int(frame_idx)
        coeff = self.coeffs[i:i + 1].copy()
        coeff[:, 80:144] = np.asarray(exp_coeff, np.float32).reshape(-1)[:64]

        depth, lm468 = self.model.get_depth_lm468_edit_exp(coeff)

        cp = self.crop_params[i]
        s = float(self.trans_s[i])
        w0, h0 = int(self.sizes[i][0]), int(self.sizes[i][1])
        w_scaled, h_scaled = int(w0 * s), int(h0 * s)
        w_crop = int(cp[5]) - int(cp[3])
        h_crop = int(cp[6]) - int(cp[4])

        full = depth.crop((-int(cp[3]), -int(cp[4]),
                           w_crop + w_scaled - int(cp[5]),
                           h_crop + h_scaled - int(cp[6]))).resize((w0, h0))

        tt = np.expand_dims(np.tile(np.array([cp[3], cp[4]]), (468, 1)), axis=0)
        lm_full = (lm468 + tt) / s
        return full, lm_full

    @torch.no_grad()
    def depth_latent(self, frame_idx, exp_coeff, box, vae):
        """The UNet's second condition: depth cropped to the face box, masked
        to the lips, top half blanked, VAE-encoded.

        Mirrors inference_joygen.py:181-187 exactly, except the depth arrives
        from the renderer instead of a JPEG on disk — so it skips a lossy
        round trip the offline path pays for.
        """
        full, lm_full = self.render(frame_idx, exp_coeff)
        x1, y1, x2, y2 = [int(v) for v in box]
        cropped = full.crop((x1, y1, x2, y2))

        depth_img = np.array(cropped.convert("L"), np.uint8)
        # cv2.imread() of the offline greyscale JPEG yields 3 equal channels;
        # create_mouth_mask and the VAE both expect that shape.
        depth_img = np.stack([depth_img] * 3, axis=-1)

        lmk = lm_full - np.expand_dims(np.tile(np.array([x1, y1]), (468, 1)),
                                       axis=0)
        lmk = np.array(lmk[0], np.int32)

        import cv2
        lip_mask = create_mouth_mask(depth_img, lmk[mouth_region_indices, :])
        depth_img = depth_img * lip_mask
        depth_img = cv2.resize(depth_img, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LANCZOS4)
        depth_img[:self.img_size // 2, ...] = 0
        return vae.get_latents_for_nomask(depth_img)


def _cli(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="build the per-video 3DMM cache used by pose-driven "
                    "input streaming")
    p.add_argument("--frames_dir", required=True,
                   help="intermediate dir holding <i>_ori.jpg")
    p.add_argument("--n_frames", type=int, required=True)
    p.add_argument("--out", required=True, help="absolute .npz path")
    p.add_argument("--bfm_folder", default="./pretrained_models/BFM")
    p.add_argument("--checkpoints_dir", default="./pretrained_models")
    p.add_argument("--name", default="face_recon_feat0.2_augment")
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--gpu_id", type=int, default=0)
    args = p.parse_args(argv)

    build_cache(args.frames_dir, args.n_frames, args.out, args.bfm_folder,
                args.checkpoints_dir, args.name, args.epoch, args.gpu_id)


if __name__ == "__main__":
    _cli()
