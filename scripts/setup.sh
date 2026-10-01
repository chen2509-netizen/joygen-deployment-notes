#!/usr/bin/env bash
#
# One-shot JoyGen installer. Automates SETUP.md; applies the fixes in
# docs/DEPLOYMENT_LOG.md inline. Idempotent — finished steps are skipped on re-run.
#
# Usage: ./scripts/setup.sh [options]
#
#   --profile NAME       cu117 | cu118 | auto           (default: auto, from GPU arch)
#   --projects-dir DIR   clone target for JoyGen/nvdiffrast  (default: $HOME/projects)
#   --env-name NAME      conda env name                      (default: joygen)
#   --weights PATH       pretrained_models.tar.gz, or an already-extracted directory
#   --skip-weights       leave pretrained_models/ alone
#   --skip-smoke-test    don't run the 5s demo at the end
#   --smoke-test         run the smoke test even when weights were skipped
#   --dry-run            detect GPU, print the plan, change nothing
#   -h, --help           this text
#
# Exit: 0 ok / 1 setup failure / 2 bad usage / 3 unsupported host
#
set -Eeuo pipefail

PROFILE="auto"
ENV_NAME="joygen"
PROJECTS_DIR="${HOME}/projects"
WEIGHTS_SRC=""
SKIP_WEIGHTS=0
SKIP_SMOKE=0
FORCE_SMOKE=0
DRY_RUN=0

JOYGEN_REPO="https://github.com/JOY-MM/JoyGen.git"
NVDIFFRAST_REPO="https://github.com/NVlabs/nvdiffrast.git"
PY_VERSION="3.8.19"
WEIGHTS_URL="https://drive.google.com/file/d/1kvGsljFRnXKUK_ETdd49jJy8DbdgZKkE"

# script lives in scripts/, repo root is one level up
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${REPO_DIR}/setup.log"

# Only the CUDA toolkit and the torch wheel index vary by GPU; everything else in
# this script is the RTX 3060 flow from SETUP.md verbatim.
#   cu117  cc 8.0-8.6  CUDA 11.7  verified on RTX 3060
#   cu118  cc 8.9      CUDA 11.8  RTX 4090 (nvcc 11.7 has no compute_89 target)
profile_config() {
  case "$1" in
    cu117) CUDA_VER=11.7; CUDA_LABEL="nvidia/label/cuda-11.7.0"
           TORCH=2.0.1 TVISION=0.15.2 TAUDIO=2.0.2
           TORCH_INDEX="https://download.pytorch.org/whl/cu117"; GCC_MAX=10 ;;
    cu118) CUDA_VER=11.8; CUDA_LABEL="nvidia/label/cuda-11.8.0"
           TORCH=2.0.1 TVISION=0.15.2 TAUDIO=2.0.2
           TORCH_INDEX="https://download.pytorch.org/whl/cu118"; GCC_MAX=11 ;;
    *) return 1 ;;
  esac
}

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BLUE=$'\033[1;34m'; C_GREEN=$'\033[1;32m'
  C_YELLOW=$'\033[1;33m'; C_RED=$'\033[1;31m'; C_DIM=$'\033[2m'
else
  C_RESET=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""
fi
_ts()  { date +'%H:%M:%S'; }
step() { echo "${C_BLUE}[$(_ts)] ==> $*${C_RESET}" | tee -a "$LOG_FILE"; }
info() { echo "${C_DIM}[$(_ts)]     $*${C_RESET}" | tee -a "$LOG_FILE"; }
ok()   { echo "${C_GREEN}[$(_ts)]  ok  $*${C_RESET}" | tee -a "$LOG_FILE"; }
warn() { echo "${C_YELLOW}[$(_ts)] warn $*${C_RESET}" | tee -a "$LOG_FILE" >&2; }
die()  { echo "${C_RED}[$(_ts)] FAIL ${1}${C_RESET}" | tee -a "$LOG_FILE" >&2; exit "${2:-1}"; }
trap 'echo "${C_RED}
Failed at line $LINENO. Log: ${LOG_FILE}
Fix the cause and re-run the same command.
Known failure modes: docs/DEPLOYMENT_LOG.md${C_RESET}" >&2' ERR

# env/activate_env_vars.sh dereferences $LD_LIBRARY_PATH unguarded, which trips set -u
conda_do() { set +u; conda "$@"; set -u; }

# the activated env's libffi breaks git over HTTPS — DEPLOYMENT_LOG.md #6
git_clean() { env -u LD_LIBRARY_PATH git "$@"; }

cap_ge() { awk "BEGIN{exit !(${1:-0} >= $2)}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)         PROFILE="${2:?}"; shift 2 ;;
    --env-name)        ENV_NAME="${2:?}"; shift 2 ;;
    --projects-dir)    PROJECTS_DIR="${2:?}"; shift 2 ;;
    --weights)         WEIGHTS_SRC="${2:?}"; shift 2 ;;
    --skip-weights)    SKIP_WEIGHTS=1; shift ;;
    --skip-smoke-test) SKIP_SMOKE=1; shift ;;
    --smoke-test)      FORCE_SMOKE=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

JOYGEN_DIR="${PROJECTS_DIR}/JoyGen"
NVDIFFRAST_DIR="${PROJECTS_DIR}/nvdiffrast"
WEIGHTS_DIR="${JOYGEN_DIR}/pretrained_models"
EXPECTED_WEIGHTS=(BFM audio2motion joygen dwpose face-parse-bisent
                  face_recon_feat0.2_augment sd-vae-ft-mse whisper)

: > "$LOG_FILE"
step "JoyGen setup — repo: ${REPO_DIR}  log: ${LOG_FILE}"

step "0/9  Host and GPU"

for c in git curl tar awk sed; do
  command -v "$c" >/dev/null 2>&1 || die "'$c' not found — sudo apt install -y $c"
done
for f in env/constraints.txt env/activate_env_vars.sh env/deactivate_env_vars.sh scripts/run_inference.sh; do
  [[ -f "${REPO_DIR}/${f}" ]] || die "${f} missing — run this from a full checkout of the repo" 2
done

GPU_NAME=""; COMPUTE_CAP=""; DRIVER_VER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
elif [[ $DRY_RUN -eq 0 ]]; then
  if grep -qi microsoft /proc/version 2>/dev/null; then
    die "nvidia-smi not found. Under WSL2 the driver is installed on the Windows side —
   install the NVIDIA Windows driver, then reopen the WSL shell." 3
  else
    die "nvidia-smi not found. Install the driver first:
     sudo ubuntu-drivers autoinstall && sudo reboot" 3
  fi
else
  warn "no nvidia-smi (dry run) — GPU detection skipped"
fi

# compute_cap query needs a recent driver; fall back to the product name
if [[ -z "$COMPUTE_CAP" || "$COMPUTE_CAP" == *"N/A"* ]] && [[ -n "$GPU_NAME" ]]; then
  case "$GPU_NAME" in
    *"RTX 40"*|*"L40"*|*" L4"*|*"6000 Ada"*)        COMPUTE_CAP="8.9" ;;
    *"RTX 30"*|*"A10"*|*"A40"*|*"A100"*|*"A6000"*)  COMPUTE_CAP="8.6" ;;
    *"H100"*|*"H200"*)                              COMPUTE_CAP="9.0" ;;
    *"RTX 50"*|*"B100"*|*"B200"*)                   COMPUTE_CAP="12.0" ;;
    *) COMPUTE_CAP="" ;;
  esac
  [[ -n "$COMPUTE_CAP" ]] && info "compute_cap inferred from GPU name: ${COMPUTE_CAP}"
fi
info "GPU: ${GPU_NAME:-unknown}   compute_cap: ${COMPUTE_CAP:-unknown}   driver: ${DRIVER_VER:-unknown}"

if [[ "$PROFILE" == "auto" ]]; then
  if [[ -z "$COMPUTE_CAP" ]]; then
    [[ $DRY_RUN -eq 1 ]] || warn "compute capability unknown — defaulting to cu117, override with --profile"
    PROFILE="cu117"
  elif cap_ge "$COMPUTE_CAP" 8.9; then PROFILE="cu118"
  else PROFILE="cu117"
  fi
fi

profile_config "$PROFILE" || die "unknown profile '${PROFILE}' (cu117 | cu118)" 2

# only sm_86 (RTX 3060) is verified end to end; anything else follows the same flow
if [[ -n "$COMPUTE_CAP" ]] && ! cap_ge "$COMPUTE_CAP" 8.0; then
  warn "compute_cap ${COMPUTE_CAP} is older than anything this flow was verified on (sm_86)"
elif [[ "$COMPUTE_CAP" != "8.6" && -n "$COMPUTE_CAP" ]]; then
  info "flow verified on sm_86; running it on sm_${COMPUTE_CAP//./} with profile ${PROFILE}"
fi

# nvcc 11.7 has no compute_89 target; torch still runs (sm_86 cubins are forward
# compatible within 8.x) but nvdiffrast's plugin build fails at first use
if [[ -n "$COMPUTE_CAP" ]] && [[ "$PROFILE" == "cu117" ]] && cap_ge "$COMPUTE_CAP" 8.9; then
  warn "compute_cap ${COMPUTE_CAP} with CUDA 11.7: nvdiffrast will fail to compile
     ('Unsupported gpu architecture compute_89'). Use --profile cu118."
fi

cat <<EOF
  ${C_BLUE}Plan${C_RESET}
    profile        ${PROFILE}
    cuda toolkit   ${CUDA_VER}  (conda env only, never system-wide)
    torch          ${TORCH} / torchvision ${TVISION} / torchaudio ${TAUDIO}
    conda env      ${ENV_NAME}  (python ${PY_VERSION})
    clone into     ${PROJECTS_DIR}
    weights        $( [[ $SKIP_WEIGHTS -eq 1 ]] && echo "skipped" || echo "${WEIGHTS_SRC:-<none given>}" )
EOF

if [[ $DRY_RUN -eq 1 ]]; then
  info "dry run — nothing was changed"
  exit 0
fi

step "1/9  Conda env '${ENV_NAME}' + CUDA ${CUDA_VER} toolkit"

command -v conda >/dev/null 2>&1 || die \
  "conda not found. Install Miniconda, then re-run in a fresh shell:
     curl -fsSLo /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash /tmp/mc.sh -b -p \$HOME/miniconda3 && \$HOME/miniconda3/bin/conda init bash"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "env '${ENV_NAME}' exists — reusing"
else
  conda create -n "$ENV_NAME" "python=${PY_VERSION}" ffmpeg -y
fi
conda_do activate "$ENV_NAME"
[[ -n "${CONDA_PREFIX:-}" ]] || die "conda activate failed"
info "CONDA_PREFIX=${CONDA_PREFIX}"

if nvcc --version 2>/dev/null | grep -q "release ${CUDA_VER}"; then
  info "cuda-toolkit ${CUDA_VER} already present"
else
  info "installing cuda-toolkit ${CUDA_VER} (~2-3GB)"
  conda install -c "$CUDA_LABEL" cuda-toolkit -y
fi
nvcc --version | grep -q "release ${CUDA_VER}" \
  || die "nvcc is not ${CUDA_VER} — a system-wide CUDA is shadowing the env's bin/"

# nvcc 11.7 accepts gcc <= 10, nvcc 11.8 accepts gcc <= 11. Ubuntu 20.04's default
# gcc-9 (what we tested on) is fine for both; 22.04 defaults to gcc-11 and needs a
# fallback for cu117.
GCC_MAJOR=0
command -v gcc >/dev/null 2>&1 && GCC_MAJOR="$(gcc -dumpversion | cut -d. -f1)"

if [[ "$GCC_MAJOR" -ge 1 && "$GCC_MAJOR" -le "$GCC_MAX" ]]; then
  info "host gcc-${GCC_MAJOR} accepted by nvcc ${CUDA_VER}"
elif command -v g++-9 >/dev/null 2>&1; then
  export CC=gcc-9 CXX=g++-9
  info "host gcc-${GCC_MAJOR:-none} unusable with nvcc ${CUDA_VER} — using gcc-9/g++-9"
else
  info "installing gcc-9/g++-9 (host gcc-${GCC_MAJOR:-none} exceeds nvcc ${CUDA_VER}'s limit of ${GCC_MAX})"
  if sudo -n true 2>/dev/null || sudo -v 2>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y build-essential gcc-9 g++-9
  fi
  command -v g++-9 >/dev/null 2>&1 \
    || die "gcc-9 is required (host has gcc-${GCC_MAJOR:-none}, nvcc ${CUDA_VER} accepts <= ${GCC_MAX}).
   Install it, then re-run:  sudo apt install -y build-essential gcc-9 g++-9"
  export CC=gcc-9 CXX=g++-9
fi
ok "env ready"

step "2/9  PyTorch ${TORCH}+${PROFILE}"

torch_ok() {
  python - "${TORCH}+${PROFILE}" <<'PY' 2>/dev/null
import sys
try: import torch
except Exception: sys.exit(1)
sys.exit(0 if torch.__version__ == sys.argv[1] else 1)
PY
}

if torch_ok; then
  info "torch ${TORCH}+${PROFILE} already installed"
else
  # --no-deps --force-reinstall: a plain install re-resolves and pulls a generic
  # build over the CUDA-matched one — DEPLOYMENT_LOG.md #1
  python -m pip install --no-cache-dir --force-reinstall --no-deps \
    "torch==${TORCH}" "torchvision==${TVISION}" "torchaudio==${TAUDIO}" \
    --index-url "$TORCH_INDEX"
  python -m pip install --no-cache-dir \
    numpy filelock typing-extensions sympy networkx jinja2 pillow requests
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"    torch={torch.__version__}  device={torch.cuda.get_device_name(0)}  "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
PY
ok "torch ready"

step "3/9  Clone JoyGen, install constraints.txt"

mkdir -p "$PROJECTS_DIR"
if [[ -d "${JOYGEN_DIR}/.git" ]]; then
  info "JoyGen already cloned at ${JOYGEN_DIR}"
else
  git_clean clone "$JOYGEN_REPO" "$JOYGEN_DIR"
fi
cd "$JOYGEN_DIR"

# env/constraints.txt pins torch without a +cuXXX local label, so one file covers
# cu117 and cu118, unchanged from the repo.
cp "${REPO_DIR}/env/constraints.txt" constraints.txt
info "$(tr '\n' ' ' < constraints.txt)"
ok "repo ready"

step "4/9  Patch upstream requirements.txt"

if [[ ! -f requirements.txt ]]; then
  warn "no requirements.txt upstream — layout changed, skipping"
else
  [[ -f requirements.txt.orig ]] || cp requirements.txt requirements.txt.orig
  # mediapipe 0.9.3.0 is off PyPI and 1.0.0 needs py>=3.9 (#3); upstream also pins
  # torch 1.13.1, which fights constraints.txt (#4)
  TORCH="$TORCH" TVISION="$TVISION" TAUDIO="$TAUDIO" python - <<'PY'
import os, re, pathlib
pins = {"mediapipe": "0.10.5", "torch": os.environ["TORCH"],
        "torchvision": os.environ["TVISION"], "torchaudio": os.environ["TAUDIO"]}
p = pathlib.Path("requirements.txt")
out, changed = [], []
for line in p.read_text().splitlines():
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line)
    if m and m.group(1).lower() in pins:
        name, old = m.group(1).lower(), m.group(2)
        new = pins[name]
        if old != new: changed.append(f"{name} {old}->{new}")
        out.append(f"{name}=={new}")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
print("    " + (", ".join(changed) if changed else "already aligned"))
PY
fi
ok "requirements.txt aligned"

step "5/9  Python dependencies (longest step, ~10-25 min)"

# av from conda-forge before requirements.txt: the pip sdist won't build under
# Cython 3 (#2) and links an FFmpeg ABI without avcodec_close (#5). Installing it
# first means pip sees the pin satisfied and never attempts the build.
if python -c "import av" >/dev/null 2>&1; then
  info "av already importable"
else
  info "installing av=10.0.0 from conda-forge"
  conda install -c conda-forge "av=10.0.0" -y
  python -c "import av; print('    av', av.__version__)" || die "av still broken — DEPLOYMENT_LOG.md #5"
fi

python -m pip install --no-cache-dir "cython<3" -c constraints.txt

if python -c "import mmcv, mmdet, mmpose" >/dev/null 2>&1; then
  info "mmcv/mmdet/mmpose already installed"
else
  python -m pip install --no-cache-dir -U openmim
  mim install "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0" -c constraints.txt
fi

[[ -f requirements.txt ]] && python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt

python -m pip install --no-cache-dir accelerate -c constraints.txt

torch_ok || die "torch was swapped out during dependency install (want ${TORCH}+${PROFILE}).
   Repair: python -m pip install torch==${TORCH} torchvision==${TVISION} torchaudio==${TAUDIO} \\
     --index-url ${TORCH_INDEX} --force-reinstall --no-deps"
ok "dependencies installed"

step "6/9  Persist env vars via conda activate/deactivate hooks"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cp "${REPO_DIR}/env/activate_env_vars.sh"   "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
cp "${REPO_DIR}/env/deactivate_env_vars.sh" "${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"

conda_do deactivate
conda_do activate "$ENV_NAME"
info "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
info "CUDA_HOME=${CUDA_HOME:-<unset>}"
[[ "${LD_LIBRARY_PATH:-}" == "${CONDA_PREFIX}/lib"* ]] || warn "activate hook did not take effect"
ok "hooks installed"

step "7/9  nvdiffrast"

if ! python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  [[ -d "${NVDIFFRAST_DIR}/.git" ]] || git_clean clone "$NVDIFFRAST_REPO" "$NVDIFFRAST_DIR"
  # setuptools < 64 can't parse the PEP 621 [project] table and installs it as
  # UNKNOWN-0.0.0 — DEPLOYMENT_LOG.md #7
  python -m pip install --no-cache-dir --upgrade "setuptools>=64"
  ( cd "$NVDIFFRAST_DIR" && python -m pip install . --no-build-isolation -c "${JOYGEN_DIR}/constraints.txt" )
  python -c "import nvdiffrast.torch" || die "nvdiffrast still not importable (#7)"
fi

# the CUDA plugin is JIT-built on first use, not on import — force it here so an
# arch/toolkit mismatch fails at setup time instead of mid-inference
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${COMPUTE_CAP:-8.6}}"
info "building nvdiffrast CUDA plugin (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})"
if ! python - <<'PY'
import sys, torch, nvdiffrast.torch as dr
try:
    dr.RasterizeCudaContext()
except Exception as e:
    print(f"    {type(e).__name__}: {e}"); sys.exit(1)
print("    nvdiffrast CUDA plugin OK")
PY
then
  die "nvdiffrast CUDA plugin build failed.
   'Unsupported gpu architecture compute_${COMPUTE_CAP//./}' means the toolkit predates the
   GPU — re-run with --profile cu118 (RTX 40xx).
   Otherwise check gcc version and CUDA_HOME (${CUDA_HOME:-unset})."
fi
ok "nvdiffrast ready"

step "8/9  Pretrained weights"

weights_complete() {
  local missing=()
  for d in "${EXPECTED_WEIGHTS[@]}"; do [[ -d "${WEIGHTS_DIR}/${d}" ]] || missing+=("$d"); done
  [[ ${#missing[@]} -eq 0 ]] || { echo "    missing: ${missing[*]}"; return 1; }
}

mkdir -p "$WEIGHTS_DIR"
if [[ $SKIP_WEIGHTS -eq 1 ]]; then
  info "--skip-weights"
elif weights_complete; then
  info "weights already in place"
else
  if [[ -d "$WEIGHTS_SRC" ]]; then
    info "copying from ${WEIGHTS_SRC}"
    cp -rn "${WEIGHTS_SRC}/." "$WEIGHTS_DIR/"
  elif [[ -f "$WEIGHTS_SRC" ]]; then
    info "extracting ${WEIGHTS_SRC} (~5GB)"
    tar -xzf "$WEIGHTS_SRC" -C "$WEIGHTS_DIR"
  else
    # gdown gets Drive's quota page (75KB of HTML) instead of the archive — #8
    warn "no weights, and --weights not given. Download in a browser:
       ${WEIGHTS_URL}
     then: ./scripts/setup.sh --weights /path/to/pretrained_models.tar.gz
     Everything else is installed; only this step is outstanding."
    SKIP_WEIGHTS=1
  fi
  # archive unpacks one level too deep — #9
  if [[ -d "${WEIGHTS_DIR}/pretrained_models" ]]; then
    info "flattening nested pretrained_models/"
    mv "${WEIGHTS_DIR}/pretrained_models/"* "${WEIGHTS_DIR}/"
    rmdir "${WEIGHTS_DIR}/pretrained_models"
  fi
  [[ $SKIP_WEIGHTS -eq 1 ]] || weights_complete || warn "weight tree incomplete — compare with JoyGen's README"
fi

step "9/9  Verification"

python - <<'PY'
import importlib, sys
mods = ["torch", "torchvision", "torchaudio", "av", "mediapipe", "mmcv", "mmdet",
        "mmpose", "nvdiffrast.torch", "diffusers", "transformers", "accelerate"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print(f"    ok   {m}")
    except Exception as e:
        bad.append(m); print(f"    FAIL {m}: {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
PY
ok "all imports clean"

if [[ $SKIP_SMOKE -eq 0 ]] && { [[ $SKIP_WEIGHTS -eq 0 ]] || [[ $FORCE_SMOKE -eq 1 ]]; }; then
  step "Smoke test: bundled 5s demo"
  info "first run also downloads the s3fd checkpoint (~200s) — see docs/BENCHMARKS.md"
  if JOYGEN_DIR="$JOYGEN_DIR" bash "${REPO_DIR}/scripts/run_inference.sh" \
       demo/xinwen_5s.mp3 demo/example_5s.mp4 results/smoke_test; then
    ok "smoke test passed — ${JOYGEN_DIR}/results/smoke_test/talk/"
  else
    warn "smoke test failed, but the environment is installed. Debug the run itself."
  fi
else
  info "smoke test skipped"
fi

cat <<EOF

${C_GREEN}JoyGen setup complete.${C_RESET}
  gpu        ${GPU_NAME:-unknown} (sm_${COMPUTE_CAP//./})   profile ${PROFILE}
  env        conda activate ${ENV_NAME}
  repo       ${JOYGEN_DIR}
  weights    $( [[ $SKIP_WEIGHTS -eq 1 ]] && echo "NOT installed — re-run with --weights <archive>" || echo "$WEIGHTS_DIR" )
  log        ${LOG_FILE}

Run inference:
  conda activate ${ENV_NAME}
  JOYGEN_DIR=${JOYGEN_DIR} ${REPO_DIR}/scripts/run_inference.sh <audio> <video> results/<name>

This env's libffi breaks git over HTTPS — clone with conda deactivated, or use
  env -u LD_LIBRARY_PATH git clone ...
EOF
