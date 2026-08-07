# Deployment Log

Detailed record of every issue hit while setting up JoyGen, kept separate from
`SETUP.md` so first-time users aren't forced to read the debugging history.
Read this if you're reproducing the setup on different hardware/OS and hit similar
errors, or if you want to understand *why* `constraints.txt` / patched
`requirements.txt` look the way they do.

## 1. `pip install <deps>` silently upgraded torch

Installing the general dependency list (numpy, diffusers, transformers, etc. per
JoyGen's issue #6) pulled in `torch==2.2.2`, breaking the `cu117` build. pip's
resolver doesn't treat already-installed packages as fixed — any new install can
trigger a re-resolve.

Fix: reinstall the pinned torch build with `--no-deps --force-reinstall`, then create
`constraints.txt` and pass `-c constraints.txt` to every subsequent `pip`/`mim` call.

## 2. `av==10.0.0` fails to build (Cython incompatibility)

`requirements.txt` pins `av==10.0.0`, which has no prebuilt wheel for this
Python/Cython combination and fails to compile from source — newer Cython (3.x)
rejects the package's older exception-handling syntax (`noexcept`).

Fix: `pip install "cython<3"` then rebuild with `--no-build-isolation`.

## 3. `mediapipe==0.9.3.0` no longer on PyPI

Removed from PyPI (~Nov 2024). Oldest available version is `0.10.5`. `1.0.0`
(the version pip installs by default when unpinned) uses `list[X]` type-hint syntax
that requires Python 3.9+, so it fails to import on 3.8.

Fix: pin `mediapipe==0.10.5` and patch `requirements.txt`'s line to match, otherwise
later `pip install -r requirements.txt` runs will conflict with `constraints.txt`
and abort with `ResolutionImpossible`.

## 4. `requirements.txt` pins an unrelated torch version

Line ~93-95 of the upstream `requirements.txt` pins `torch==1.13.1` /
`torchaudio==0.13.1` / `torchvision==0.14.1`, conflicting with `constraints.txt`.

Fix: edit those three lines in-place to match `constraints.txt`.

## 5. `av` imports but crashes: `undefined symbol: avcodec_close`

The pip-built `av==10.0.0` wheel links against an FFmpeg ABI that doesn't match the
newer ffmpeg conda installed as a `joygen` env dependency (`avcodec_close` was
removed from newer FFmpeg releases).

Fix: reinstall via conda-forge, which resolves a compatible ffmpeg/av pair together:

```bash
pip uninstall av -y
conda install -c conda-forge av=10.0.0 -y
```

## 6. `GLIBCXX_3.4.29 not found` after the conda-forge av install

conda-forge's `av` install pulled in newer C++ dependencies (e.g. `libLerc.so.4` via
Pillow) requiring a newer `libstdc++` than the one the process was actually loading
at runtime. The conda env *does* ship a new enough `libstdc++`
(`$CONDA_PREFIX/lib/libstdc++.so.6`, confirmed to have `GLIBCXX_3.4.29`) — it just
wasn't first on the linker search path.

Fix: prepend the conda env's lib directory:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Made persistent via a conda `activate.d` hook — see `env/activate_env_vars.sh`.

**Caveat:** setting `LD_LIBRARY_PATH` globally in the shell also affects system
tools. It broke `git clone` over HTTPS (`libp11-kit.so.0: undefined symbol:
ffi_type_pointer`) because git picked up the conda env's `libffi` instead of the
system one. Workaround: run `git clone` outside the conda env (`conda deactivate`
first), then `conda activate joygen` again for anything Python-related. A matching
`deactivate.d` hook (`env/deactivate_env_vars.sh`) unsets the variable on
`conda deactivate` to avoid this leaking into other shells.

## 7. nvdiffrast installs as package `UNKNOWN-0.0.0`

`pip install .` succeeded but produced a package with no name/version, causing
`import nvdiffrast.torch` to fail with `PackageNotFoundError`. Root cause:
`setuptools==60.2.0` (pulled in earlier by `openxlab`, a `mim`/`openmim` dependency)
is too old to parse the PEP 621 `[project]` table in nvdiffrast's `pyproject.toml`
(`setuptools>=64` required).

Fix:

```bash
pip install --upgrade "setuptools>=64"
```

This produces a `setuptools~=60.2.0` conflict warning from `openxlab` — safe to
ignore, `openxlab` is an unused transitive dependency of `mim`, not part of the
JoyGen inference path.

## 8. Google Drive `gdown` download fails silently

`gdown` returned a 75KB HTML page instead of the ~5GB pretrained weights archive —
Google Drive's virus-scan bypass / quota page for large files, which `gdown` doesn't
always handle even at the latest version.

Fix: download via browser manually, then copy into the WSL filesystem:

```bash
cp "/mnt/c/Users/<user>/Downloads/pretrained_models.tar.gz" pretrained_models/
```

## 9. Extracted archive nests an extra `pretrained_models/` folder

`tar -xzvf` produced `pretrained_models/pretrained_models/<subfolders>` instead of
`pretrained_models/<subfolders>` — one level too deep for the paths JoyGen's scripts
expect.

Fix: `mv pretrained_models/* .` then `rmdir pretrained_models`.

## 10. `accelerate` not installed — model loads without low-memory mode

Not an error, just a warning (`Cannot initialize model with low cpu memory usage`).
`accelerate` only affects *how* weights are loaded into memory (avoids a temporary
duplicate copy during load) — it does not affect inference speed. On this GPU
(12GB VRAM, sufficient system RAM) the difference was not measurable; on
memory-constrained hosts or with much larger models it can be the difference between
loading successfully and OOM.

```bash
pip install accelerate -c constraints.txt
```
