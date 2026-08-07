# Setup

Tested on Ubuntu 20.04.6 (WSL2), RTX 3060, driver supporting CUDA 13.1 (backward
compatible with the CUDA 11.7 toolkit installed below).

## 0. Check GPU driver

```bash
nvidia-smi
```

If this fails, install the driver first:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

## 1. Conda environment + CUDA toolkit (scoped to the env, not system-wide)

```bash
conda create -n joygen python=3.8.19 ffmpeg -y
conda activate joygen
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit -y
```

Verify:

```bash
nvcc --version   # should show release 11.7
```

Host compiler (Ubuntu 20.04's default works):

```bash
sudo apt update
sudo apt install -y build-essential gcc-9 g++-9
```

## 2. PyTorch (cu117)

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.0.1+cu117 True
```

## 3. Clone JoyGen and lock dependency versions

```bash
cd ~/projects
git clone https://github.com/JOY-MM/JoyGen.git
cd JoyGen
```

Create `constraints.txt` (see [`env/constraints.txt`](../env/constraints.txt) in this
repo for the version we settled on) to stop later installs from silently upgrading
torch/mediapipe:

```bash
cp /path/to/this/repo/env/constraints.txt .
```

## 4. Core Python dependencies

```bash
pip install --no-cache-dir -U openmim
mim install "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0" -c constraints.txt

pip install -r requirements.txt -c constraints.txt
```

`requirements.txt` needs two version overrides for this environment (already applied
if you copy the patched file — see `docs/DEPLOYMENT_LOG.md` for why):

- `mediapipe==0.9.3.0` → `0.10.5` (original version pulled from PyPI)
- `torch==1.13.1` / `torchaudio==0.13.1` / `torchvision==0.14.1` → aligned to `constraints.txt`

If `av` fails to import with `undefined symbol: avcodec_close`, reinstall it via
conda-forge so it matches the bundled ffmpeg's ABI:

```bash
pip uninstall av -y
conda install -c conda-forge av=10.0.0 -y
```

## 5. nvdiffrast

```bash
cd ~/projects
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install --upgrade "setuptools>=64"   # older setuptools breaks pyproject.toml metadata
pip install . --no-build-isolation -c ../JoyGen/constraints.txt
```

Verify:

```bash
python -c "import nvdiffrast.torch as dr; print('nvdiffrast import OK')"
```

## 6. Persist environment variables (avoid manual export every session)

Copy the scripts in [`env/`](../env) into the conda env's activation hooks:

```bash
cp env/activate_env_vars.sh   $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cp env/deactivate_env_vars.sh $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

Reload and verify:

```bash
conda deactivate && conda activate joygen
echo $LD_LIBRARY_PATH   # should show $CONDA_PREFIX/lib:...
```

## 7. Download pretrained weights

```bash
cd ~/projects/JoyGen
mkdir -p pretrained_models
```

Combined archive (~5GB): https://drive.google.com/file/d/1kvGsljFRnXKUK_ETdd49jJy8DbdgZKkE

`gdown` frequently fails on this file (Google Drive quota on large files) — download
via browser instead and copy in:

```bash
cp /path/to/downloaded/pretrained_models.tar.gz pretrained_models/
cd pretrained_models
tar -xzvf pretrained_models.tar.gz
# archive extracts into a nested pretrained_models/ folder — flatten it:
mv pretrained_models/* .
rmdir pretrained_models
```

Verify the structure matches JoyGen's README (`BFM/`, `audio2motion/`, `joygen/`,
`dwpose/`, `face-parse-bisent/`, `face_recon_feat0.2_augment/`, `sd-vae-ft-mse/`,
`whisper/`).

## 8. Optional: accelerate (faster/lighter model loading)

```bash
pip install accelerate -c constraints.txt
```

## 9. Smoke test

```bash
cd ~/projects/JoyGen
bash scripts/inference_pipeline.sh demo/xinwen_5s.mp3 demo/example_5s.mp4 results/smoke_test
```

Output lands in `results/smoke_test/talk/`. See `docs/BENCHMARKS.md` for expected
timing.
