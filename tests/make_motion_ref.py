"""
make_motion_ref.py — offline expression coefficients, for a clip that has none.

`test_motion_parity.py` needs something exact to be wrong about. The repo
ships one such reference (`results/smoke_test/a2m/xinwen_5s.npy`) but 5s is
too short to test windowing with: a window wider than the clip is just the
control run wearing a disguise, which is how M1's first sweep produced a
falsely perfect answer.

This makes a reference for any clip. It copies the audio into
joygen-deployment-notes first, on purpose: `Audio2Motion.save_wav16k()` writes
`<name>_16k.wav` next to whatever path it is given
(inference_audio2motion.py:180), and the demo audio lives under JoyGen, which
is read-only. Point it at the original and it will write there.

Run:
    python tests/make_motion_ref.py <audio> --out <name>.npy [--seed 0]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

NOTES_ROOT = Path(__file__).resolve().parent.parent
if str(NOTES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTES_ROOT))

A2M_CKPT = "./pretrained_models/audio2motion/240210_real3dportrait_orig/audio2secc_vae"
HUBERT_PATH = "pretrained_models/audio2motion/hubert"


def main(argv=None):
    p = argparse.ArgumentParser(description="offline audio2motion reference")
    p.add_argument("audio")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--mouth_amp", type=float, default=0.45)
    p.add_argument("--work_dir", default=str(NOTES_ROOT / "cache" / "audio"))
    args = p.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    local = work / Path(args.audio).name
    if Path(args.audio).resolve() != local.resolve():
        shutil.copy(args.audio, local)
        print("[ref] copied audio -> {} (so the _16k.wav lands here, not in "
              "JoyGen)".format(local))

    from inference_audio2motion import Audio2Motion

    inp = {
        "a2m_ckpt": A2M_CKPT,
        "hubert_path": HUBERT_PATH,
        "drv_audio_name": str(local),
        "blink_mode": "period",
        "temperature": args.temperature,
        "mouth_amp": args.mouth_amp,
        "seed": args.seed,
    }
    infer = Audio2Motion(inp["a2m_ckpt"], inp=inp)
    exp = infer.infer_once(inp)["exp"].detach().cpu().numpy()

    out = Path(args.out)
    if not out.is_absolute():
        out = NOTES_ROOT / "cache" / "a2m" / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), exp)
    print("[ref] {} frames (seed {}) -> {}".format(len(exp), args.seed, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
