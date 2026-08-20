# JoyGen Deployment Notes

Internal deployment record for [JoyGen](https://github.com/JOY-MM/JoyGen) (Apache 2.0), an audio-driven 3D depth-aware talking-face video editing model.

This repo does **not** contain JoyGen's source code. It documents how to reproduce a working environment and run inference on our hardware.

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

## What's here

```
.
├── SETUP.md                 # ← Start here: install instructions only
├── docs/
│   ├── DEPLOYMENT_LOG.md    # All issues hit and their fixes
│   └── BENCHMARKS.md        # Inference timing on test clips
│   └── buffering_reserach_20260819.md  # Streaming buffering feasibility research
└── env/
    ├── constraints.txt      # Pinned dependency versions
    ├── activate_env_vars.sh # Conda env variables (persistent)
    └── deactivate_env_vars.sh
└── scripts/
    └── run_inference.sh     # Thin wrapper to run JoyGen
```

## If you want to reproduce the exact debugging process

See [`docs/DEPLOYMENT_LOG.md`](./docs/DEPLOYMENT_LOG.md) for every issue hit during setup (dependency conflicts, ABI mismatches, etc.) and how each was resolved.

## Benchmarks

See [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) for inference timing on 5s/15s/34s test clips.

## Streaming research

See [`docs/buffering_reserach_20260819.md`](./docs/buffering_reserach_20260819.md) for the feasibility analysis of JoyGen's input/output buffering design toward real-time streaming, including proposed implementation approach and open items.

## Environment

| | |
|---|---|
| OS | Ubuntu 20.04.6 LTS (tested via WSL2) |
| GPU | RTX 3060 (12GB) |
| Python | 3.8.19 |
| CUDA toolkit | 11.7 (installed via conda, not system-wide) |
| PyTorch | 2.0.1+cu117 |

## Original project

- JoyGen: https://github.com/JOY-MM/JoyGen (Apache 2.0 License — retained as-is, no source copied here)
- nvdiffrast: https://github.com/NVlabs/nvdiffrast (separate dependency, own license)
