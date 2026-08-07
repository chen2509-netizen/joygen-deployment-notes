#!/usr/bin/env bash
# Thin wrapper around JoyGen's own inference_pipeline.sh.
# Usage: ./run_inference.sh <audio> <video> <result_dir>
set -euo pipefail

JOYGEN_DIR="${JOYGEN_DIR:-$HOME/projects/JoyGen}"

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <audio_file> <video_file> <result_dir>" >&2
  exit 1
fi

cd "$JOYGEN_DIR"
bash scripts/inference_pipeline.sh "$1" "$2" "$3"
