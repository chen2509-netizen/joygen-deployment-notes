# Streaming Output

Frame-by-frame output for JoyGen, so playback can start before the whole clip is generated.

## What this changes

JoyGen's stock inference is a five-stage offline batch flow:

```
model load → preprocess (read all frames, VAE encode)
           → generate (UNet, batch of 8, accumulate into a list)
           → blend (paste every generated face back, write PNGs)
           → ffmpeg (one blocking call: PNG sequence → MP4, then mux audio)
```

Generation, blending, and encoding are three separate passes, so nothing is visible until the last one finishes.

`streaming/` merges them into one: each frame is blended and written into a persistent `ffmpeg` process the moment it is decoded. The one-shot `ffmpeg` call at the end is gone.

The modules import JoyGen's `data_generator`, `get_image`, `VAE`, and friends directly — no JoyGen source is copied into this repo.

## Running it

`streaming/` must run from a JoyGen checkout with its conda env active, since it imports JoyGen's modules and uses its relative model paths. Copy `streaming/` and `scripts/run.sh` into the JoyGen repo:

```bash
cd ~/projects/JoyGen
conda activate joygen
```

### Baseline (unmodified flow, with timing)

```bash
bash scripts/run.sh baseline <audio> <video> <intermediate_dir> [result_dir]
```

Reproduces stock behaviour and reports per-stage and per-frame timings. Use it as the comparison point for any change.

### Streaming

```bash
bash scripts/run.sh step3 <audio> <video> <intermediate_dir> [target] [--debug]
```

`target` decides where the encoded stream goes:

- a file path (default behaviour) — writes an MP4 as frames arrive
- `udp://host:port` — pushes MPEG-TS

Frames are fed in one at a time either way; only the destination differs.

`--debug` additionally writes PNGs so `diff` can check them. It costs roughly 15 ms/frame, so leave it off when measuring.

### Correctness check

```bash
bash scripts/run.sh diff <baseline_frames_dir> <stream_frames_dir> 2
```

Compares PNGs frame by frame. A tolerance of 2 covers float→uint8 rounding differences; anything larger means the blend path actually changed.

### Receiving a UDP stream

```bash
bash scripts/run.sh recv [port]
```

Starts `ffplay` listening on the port. Run this **before** starting the stream.

## Inputs

Three things are needed:

| Input | Notes |
|---|---|
| Audio | `.mp3` / `.wav` — what the face should appear to say |
| Video | `.mp4` — the clip whose mouth gets rewritten; the rest of the frame is kept as-is |
| `intermediate_dir` | Output of `inference_edit_expression.py`: per-frame `_ori.jpg`, `_face.jpg`, `_depth_edit_exp.jpg`, `_box.npy`, `_lm.npy` |

The streaming stage does no face detection of its own; it reads what `inference_edit_expression.py` already produced.

**Audio and video should be the same length.** Audio chunks and video frames are paired 1:1 in order and processing stops when either runs out, so extra video length is preprocessed and then discarded.

Output resolution is the source video's resolution, not `--img_size` — that only controls the face crop fed to the model.

## Results

5s clip, 127 frames, 900×900 source, RTX 3060.

| | Baseline | Streaming |
|---|---:|---:|
| Latency to first frame | 42,659 ms | **18,691 ms** |
| Total runtime | 42,659 ms | 37,182 ms |
| Output rate | — | 6.6 fps |

Correctness: 127/127 frames match the baseline.

Per-stage breakdown of the baseline, for reference:

| Stage | Time | Share |
|---|---:|---:|
| model_load | 2.7s | 6% |
| audio_feature | 2.0s | 5% |
| preprocess | 15.4s | 36% |
| generate | 13.9s | 33% |
| blend + write | 7.4s | 17% |
| ffmpeg | 1.2s | 3% |

Streaming removes the blend/write and one-shot ffmpeg stages from the critical path. What remains before the first frame can appear is model load, warmup, and preprocess — 17.7s of which 15.5s is preprocess, now the dominant cost.

6.6 fps is a throughput limit of this dev box. These are RTX 3060 numbers and should not be extrapolated to other hardware.

## Not done yet

- **Preprocessing is still a single upfront pass.** Pipelining it against generation would cut the wait before the first frame substantially, since only the first batch is needed to start.
- **Audio is not streamed.** Video and audio would need a shared PTS basis (`frame_index / fps` against `sample_count / sample_rate`); frames arrive in bursts of 8 rather than at a steady 40 ms cadence, so ffmpeg cannot infer the alignment on its own.
- **UDP delivery is unreliable here.** `ffplay` recognises the stream (h264, 900×900, 25 fps), but packet loss corrupts decoding on this setup. Delivery is expected to move to WebRTC, so this was not pursued further.
