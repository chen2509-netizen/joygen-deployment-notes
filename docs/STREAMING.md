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

## Repo layout and setup

`streaming/` and `scripts/run.sh` live in `joygen-deployment-notes`, not inside the JoyGen checkout. `run.sh` sets `PYTHONPATH` and `cd`s into JoyGen automatically, so you never need to copy files or set paths by hand.

Expected directory structure (both repos cloned as siblings):

```
~/imood_project/
├── JoyGen/                      # upstream model repo
└── joygen-deployment-notes/     # this repo
```

Run from anywhere:

```bash
conda activate joygen
bash ~/imood_project/joygen-deployment-notes/scripts/run.sh <mode> <args...>
```

## Running it

### Baseline (unmodified flow, with timing)

```bash
bash scripts/run.sh baseline <audio> <video> <intermediate_dir> [result_dir]
```

Reproduces stock behaviour and reports per-stage and per-frame timings. Use it as the comparison point for any change.

### Streaming (file output or RTP)

```bash
bash scripts/run.sh step3 <audio> <video> <intermediate_dir> [target] [flags]
```

`target` decides where the encoded stream goes:

| target | format | notes |
|---|---|---|
| `out.mp4` (or any path) | MP4 file | default if omitted |
| `rtp://host:port` | RTP/MPEG-TS | requires receiver running first; add `--bitrate 4M --gop 25` |
| `udp://host:port` | MPEG-TS over UDP | same |

Audio is muxed in by default. Pass `--no_audio` to suppress.

`--debug` additionally writes PNGs so `diff` can check them. It costs roughly 15 ms/frame, so leave it off when measuring.

#### Audio sync mode (`--audio-sync`)

| flag | behaviour |
|---|---|
| `--audio-sync fifo` | **Default.** Audio is decoded to PCM once up front, then written frame-by-frame alongside the video into a named pipe. Generation speed drives both tracks — no drift, even when fps is far below realtime. |
| `--audio-sync file` | ffmpeg reads the source audio file directly. Alignment is correct by PTS on paper, but the muxer will flush audio ahead if video generation falls too far behind. Only useful for quick file-output validation. |

Direction B (`fifo`) is the correct mode for any live or RTP target. Direction A (`file`) is retained for fast sanity checks against a file target only.

### Receiving an RTP stream

Start the receiver **before** the sender:

```bash
# Terminal 1 — receiver
bash scripts/run.sh recv rtp://127.0.0.1:23000

# Terminal 2 — sender
bash scripts/run.sh step3 demo/xinwen_5s.mp3 demo/example_5s.mp4 \
  results/smoke_test/edit_exp rtp://127.0.0.1:23000 --bitrate 4M --gop 25
```

`--bitrate 4M --gop 25` are required for RTP: without `--gop`, libx264 defaults to keyint=250, which means only one keyframe across a 5-second clip — any receiver that joins after the first packet will never decode a frame.

### Correctness check

```bash
bash scripts/run.sh diff <baseline_frames_dir> <stream_frames_dir> 2
```

Compares PNGs frame by frame. A tolerance of 2 covers float→uint8 rounding differences; anything larger means the blend path actually changed.

## Inputs

Three things are needed:

| Input | Notes |
|---|---|
| Audio | `.mp3` / `.wav` — what the face should appear to say |
| Video | `.mp4` — the clip whose mouth gets rewritten; the rest of the frame is kept as-is |
| `intermediate_dir` | Output of `inference_edit_expression.py`: per-frame `_ori.jpg`, `_face.jpg`, `_depth_edit_exp.jpg`, `_box.npy`, `_lm.npy` |

The streaming stage does no face detection of its own; it reads what `inference_edit_expression.py` already produced.

**Audio and video should be the same length.** Audio chunks and video frames are paired 1:1 in order and processing stops when either runs out.

Output resolution is the source video's resolution, not `--img_size` — that only controls the face crop fed to the model.

## Results

5s clip, 127 frames, 900×900 source.

### RTX 3060 (dev box)

| | Baseline | Streaming |
|---|---:|---:|
| Latency to first frame | 42,659 ms | **18,691 ms** |
| Total runtime | 42,659 ms | 37,182 ms |
| Output rate | — | 6.6 fps |

Correctness: 127/127 frames match the baseline.

### RTX 4090 (deployment target)

| | Baseline | Streaming (with audio, direction B) |
|---|---:|---:|
| Total runtime | 28,114 ms | 22,300–23,625 ms |
| Latency to first frame (TTFF) | 28,114 ms | **11,458–11,722 ms** |
| First frame after gen start | 28,114 ms | 522–631 ms |
| Output rate | — | 10.2–10.7 fps |
| UNet inference speed | 38.5 fps | 38.5 fps |

The gap between UNet speed (38.5 fps) and actual output rate (~10.5 fps) is dominated by preprocess and blend overhead, not GPU compute. 25 fps realtime is not yet reached on either machine.

### RTP audio/video sync (RTX 4090, direction B)

Measured with `ffplay` receiver on the same machine:

| metric | value |
|---|---|
| A-V offset | **−0.071 s** (audio ~71 ms ahead) |
| Frames received | 112/127 (~88%) |
| Frames lost | ~15, all at stream start (receiver missed first keyframe) |

The ~15 dropped frames at the start are a property of bare RTP with no signaling: the receiver is already listening when the sender starts, but it must wait for the next keyframe (every ~2.3s at gop=25, ~11fps) before it can begin decoding. This is resolved when moving to WebRTC, where signaling ensures both sides are ready before any media flows.

Per-stage breakdown of the baseline (RTX 3060), for reference:

| Stage | Time | Share |
|---|---:|---:|
| model_load | 2.7s | 6% |
| audio_feature | 2.0s | 5% |
| preprocess | 15.4s | 36% |
| generate | 13.9s | 33% |
| blend + write | 7.4s | 17% |
| ffmpeg | 1.2s | 3% |

## Implementation notes

### Audio sync (direction B) — how it works

`FFmpegSink` decodes the source audio file to 16 kHz mono PCM once at startup (`_decode_pcm_s16le`), then keeps it in memory. A named pipe (`/tmp/jg_audio_<pid>.pcm`) is opened with `O_RDWR` before ffmpeg launches — this avoids the blocking open that would deadlock if done after `Popen`. ffmpeg is given `-f s16le -ar 16000 -ac 1 -probesize 32 -i <fifo>` as its second input.

Each time a video frame is written to ffmpeg's stdin, `write_audio_upto(frame_idx)` immediately writes the corresponding PCM slice to the audio pipe. The slice boundary is `frame_idx * round(16000/fps) * 2` bytes. Both tracks advance in lockstep: ffmpeg's muxer sees audio and video arrive together and has nothing to buffer.

### Known limitations

- **Preprocessing is still a single upfront pass.** Pipelining it against generation would cut TTFF substantially — only the first batch is needed to start.
- **Output rate is ~10.5 fps on 4090**, well below the 25 fps realtime target. The bottleneck is preprocess and blend, not UNet inference (38.5 fps). Streaming input (avoiding full preprocess upfront) is the next step.
- **RTP delivery loses the first ~15 frames** due to the keyframe timing issue described above. Resolved by WebRTC signaling.
