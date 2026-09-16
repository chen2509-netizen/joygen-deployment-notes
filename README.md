# JoyGen Deployment Notes

Internal deployment record for [JoyGen](https://github.com/JOY-MM/JoyGen) (Apache 2.0), an audio-driven 3D depth-aware talking-face video editing model.

This repo does **not** contain JoyGen's source code. It documents how to reproduce a working environment and run inference on our hardware, plus our own streaming modules (`streaming/`), which import JoyGen rather than copying it.

## If you just want to run the model

### 1. Prerequisites

- NVIDIA driver installed (`nvidia-smi` works)
- Miniconda, `git`
- Ubuntu 20.04 or 22.04

### 2. Download the pretrained weights (~5GB)

Download in a browser: https://drive.google.com/file/d/1kvGsljFRnXKUK_ETdd49jJy8DbdgZKkE

### 3. Run the installer

```bash
git clone https://github.com/chen2509-netizen/joygen-deployment-notes.git
cd joygen-deployment-notes
./scripts/setup.sh --weights /path/to/pretrained_models.tar.gz
```

Automates the whole of [`SETUP.md`](./SETUP.md): conda env, CUDA toolkit, PyTorch, JoyGen, nvdiffrast, weights, then a 5s smoke test. Idempotent — on failure, fix the cause and re-run the same command; finished steps are skipped. Full output goes to `setup.log`.

`--dry-run` prints the detected GPU and the install plan without changing anything. Other flags: `--profile cu117|cu118`, `--env-name`, `--projects-dir`, `--skip-weights`, `--skip-smoke-test`.

> **Status:** tested end-to-end on RTX 3060 / Ubuntu 20.04 (WSL2) and RTX 4090 / Ubuntu 22.04 (WSL2). Use `--profile cu118` on the 4090. The `gcc-9` fallback on Ubuntu 22.04 is not needed — gcc 11 (the distro default) works. If setup fails, [`SETUP.md`](./SETUP.md) has the manual steps.

### 4. Run inference

```bash
./scripts/run_inference.sh <audio> <video> <output_dir>
```

## What's here

```
.
├── SETUP.md                 # ← Start here: install instructions only
├── docs/
│   ├── DEPLOYMENT_LOG.md    # All issues hit and their fixes
│   ├── BENCHMARKS.md        # Inference timing on test clips
│   ├── STREAMING.md         # Streaming implementation: design, usage, results
│   └── buffering_reserach_20260819.md  # Feasibility research behind STREAMING.md
├── env/
│   ├── constraints.txt      # Pinned dependency versions
│   ├── activate_env_vars.sh # Conda env variables (persistent)
│   └── deactivate_env_vars.sh
├── streaming/               # Frame-by-frame output modules (our code, not JoyGen's)
│   ├── frame_timer.py
│   ├── sinks.py
│   ├── baseline_timing.py
│   ├── joygen_stream.py
│   └── diff_frames.py
└── scripts/
    ├── setup.sh             # One-shot installer
    ├── run_inference.sh     # Run JoyGen (file-to-file)
    └── run.sh               # Run streaming modules
```

## Environment

| | RTX 3060 (dev) | RTX 4090 (deployment) |
|---|---|---|
| OS | Ubuntu 20.04.6 LTS (WSL2) | Ubuntu 22.04 LTS (WSL2) |
| GPU | RTX 3060 (12GB) | RTX 4090 |
| Python | 3.8.19 | 3.8.19 |
| CUDA toolkit | 11.7 (conda) | 11.8 (conda, `--profile cu118`) |
| PyTorch | 2.0.1+cu117 | 2.0.1+cu118 |

`streaming/` and `scripts/run.sh` live in this repo and are **not** copied into the JoyGen checkout. `run.sh` sets `PYTHONPATH` and `cd`s into JoyGen automatically. Both repos should be cloned as siblings under the same parent directory (e.g. `~/imood_project/`).

## Going further

- [`docs/STREAMING.md`](./docs/STREAMING.md) — streaming output: what it changes, how to run it, measured results
- [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) — file-to-file inference timing on 5s/15s/34s clips
- [`docs/DEPLOYMENT_LOG.md`](./docs/DEPLOYMENT_LOG.md) — every issue hit during setup and how it was resolved
- [`docs/buffering_reserach_20260819.md`](./docs/buffering_reserach_20260819.md) — feasibility research the streaming work is based on

## Original project

- JoyGen: https://github.com/JOY-MM/JoyGen (Apache 2.0 License — retained as-is, no source copied here)
- nvdiffrast: https://github.com/NVlabs/nvdiffrast (separate dependency, own license)
