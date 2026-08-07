# Benchmarks

`bash scripts/inference_pipeline.sh <audio> <video> <result_dir>` on RTX 3060,
using JoyGen's bundled demo clips.

| Run | Audio / Video | Generated frames | Depth-map render | JoyGen generation | Total | Notes |
|---|---|---|---|---|---|---|
| 1 | 5s / 5s | 128 | 251.2s (0.51 fps) | 40.7s | ~494s | First run: includes one-time download of the `s3fd` face-detection checkpoint (~198s at 453kB/s). No `accelerate`. |
| 2 | 34s / 15s | 856 | 273.7s (3.13 fps) | 264.7s | ~538s | `s3fd` cached, GPU warm, `accelerate` installed. |
| 3 | 5s / 5s (same input as run 1) | 128 | 48.0s (2.67 fps) | 39.5s | ~87.5s | Re-run after a full `conda deactivate` / `conda activate` cycle, to confirm the environment survives a restart. Same output as run 1. |

## Reading these numbers

- The large gap between run 1 and run 3 (same input, ~494s vs ~87.5s) is almost
  entirely the one-time `s3fd` checkpoint download and GPU/model cold-start, not
  environment instability. Once cached, re-running the same clip is ~5-6x faster.
- Depth-map render time scales roughly linearly with frame count once warm
  (run 2 has 6.7x the frames of run 1/3 but only ~9% more render time than run 1,
  because run 1's total included the one-off download).
- `accelerate` affects model-load memory usage, not the numbers above — see
  `DEPLOYMENT_LOG.md` §10.
- Run 3 confirms the persisted `activate.d`/`deactivate.d` environment variable
  scripts (see `env/`) survive a full environment restart with no manual
  re-exporting required.
