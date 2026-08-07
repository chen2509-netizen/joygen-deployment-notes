# JoyGen Deployment Notes

Internal deployment record for [JoyGen](https://github.com/JOY-MM/JoyGen) (Apache 2.0),
an audio-driven 3D depth-aware talking-face video editing model.

This repo does **not** contain JoyGen's source code. It documents how to reproduce
a working environment and run inference on our hardware.

## If you just want to run the model

Go to [`SETUP.md`](./SETUP.md). It has the minimal, working install path only —
no debugging history, no dead ends.

## If you want to reproduce the exact debugging process

See [`docs/DEPLOYMENT_LOG.md`](./docs/DEPLOYMENT_LOG.md) for every issue hit during
setup (dependency conflicts, ABI mismatches, etc.) and how each was resolved.

## Benchmarks

See [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) for inference timing on 5s/15s/34s
test clips.

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
