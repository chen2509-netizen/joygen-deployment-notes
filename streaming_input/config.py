"""
config.py — loads configs/pipeline.yaml.

Every tunable lives in that one file. Code reads it through `load_config()`
and nothing else hardcodes a default that the yaml also defines, so there is
exactly one place to change a parameter.

Paths resolve against the joygen-deployment-notes root, not the cwd: input
streaming runs with cwd set to the JoyGen checkout (utils/blending.py loads
its weights from relative paths at import time), so a relative path here
would land inside JoyGen.
"""

import os
from pathlib import Path

import yaml

NOTES_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = NOTES_ROOT / "configs" / "pipeline.yaml"


class Section(dict):
    """dict with attribute access, so cfg.vad.end_silence_ms reads cleanly."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                "no '{}' in config section (keys: {})".format(
                    name, ", ".join(sorted(self)))
            )

    def __setattr__(self, name, value):
        self[name] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Section({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path=None, overrides=None):
    """Read the yaml and apply `overrides` (dotted keys, e.g.
    {"joygen_input.fps": 30}) on top. Overrides come from CLI flags so a
    single run can be varied without editing the file."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError("config not found: {}".format(path))

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for dotted, value in (overrides or {}).items():
        node = raw
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value

    cfg = _wrap(raw)
    cfg.config_path = str(path)
    cfg.notes_root = str(NOTES_ROOT)
    return cfg


def resolve_dir(value, fallback_name):
    """Absolute output directory. Empty value falls back to
    <notes_root>/<fallback_name>, which keeps writes out of the JoyGen tree."""
    path = Path(value) if value else NOTES_ROOT / fallback_name
    if not path.is_absolute():
        path = NOTES_ROOT / path
    os.makedirs(str(path), exist_ok=True)
    return path


def resolve_file(value):
    """Same rule as resolve_dir but for a file that must already exist, so it
    creates nothing. Relative paths in the yaml resolve against the notes root
    rather than the cwd, which lets the config be machine-independent — the
    A100 will not have /home/cgmhaha in it.

    Empty stays empty: callers use "" to mean "this avatar has no 3DMM cache".
    """
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = NOTES_ROOT / path
    return str(path)
