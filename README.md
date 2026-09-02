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

> **Status:** the install flow is the one verified on RTX 3060 / Ubuntu 20.04 and recorded in [`SETUP.md`](./SETUP.md), but `setup.sh` itself has not yet been run end-to-end. CUDA 11.8 for sm_89 (RTX 4090) and the `gcc-9` fallback on Ubuntu 22.04 are inferred, not tested. If it fails, [`SETUP.md`](./SETUP.md) has the manual steps.

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

| | |
|---|---|
| OS | Ubuntu 20.04.6 LTS (tested via WSL2) |
| GPU | RTX 3060 (12GB) |
| Python | 3.8.19 |
| CUDA toolkit | 11.7 (installed via conda, not system-wide) |
| PyTorch | 2.0.1+cu117 |

## Going further

- [`docs/STREAMING.md`](./docs/STREAMING.md) — streaming output: what it changes, how to run it, measured results
- [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) — file-to-file inference timing on 5s/15s/34s clips
- [`docs/DEPLOYMENT_LOG.md`](./docs/DEPLOYMENT_LOG.md) — every issue hit during setup and how it was resolved
- [`docs/buffering_reserach_20260819.md`](./docs/buffering_reserach_20260819.md) — feasibility research the streaming work is based on

## Original project

- JoyGen: https://github.com/JOY-MM/JoyGen (Apache 2.0 License — retained as-is, no source copied here)
- nvdiffrast: https://github.com/NVlabs/nvdiffrast (separate dependency, own license)
