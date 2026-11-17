#!/usr/bin/env bash
# [TRAIN] Create the pinned training environment (x86_64 Linux + NVIDIA GPU).
#
#   bash scripts/train/setup.sh            # create .venv and install the lock
#   source .venv/bin/activate
#   python scripts/check_env.py --role train
#
# Python 3.12 is required: LeRobot >= 0.5.0 dropped Python 3.10/3.11.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
LOCK="$REPO_ROOT/scripts/train/requirements-train.txt"
MIN_FREE_GB="${MIN_FREE_GB:-15}"   # torch + CUDA wheels + lerobot need ~10 GB

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "error: this is the [TRAIN] setup (x86_64). On the Jetson use scripts/jetson/." >&2
  exit 1
fi

command -v uv >/dev/null || { echo "error: uv not found — https://docs.astral.sh/uv/" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi not found — NVIDIA driver missing?" >&2; exit 1; }

free_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  echo "error: only ${free_gb} GB free, need ${MIN_FREE_GB} GB (override with MIN_FREE_GB=...)" >&2
  exit 1
fi

# LeRobot decodes dataset videos with torchcodec, which needs system ffmpeg libraries.
if ! command -v ffmpeg >/dev/null; then
  echo "warning: ffmpeg not found; dataset video decoding may fail (sudo apt install ffmpeg)" >&2
fi

echo "==> creating $VENV (Python 3.12)"
uv venv --python 3.12 "$VENV"
echo "==> installing pinned deps from $LOCK"
VIRTUAL_ENV="$VENV" uv pip install -r "$LOCK"
VIRTUAL_ENV="$VENV" uv pip install -e "$REPO_ROOT" --no-deps

echo
echo "done. Next:"
echo "  source $VENV/bin/activate"
echo "  hf auth login            # needed for gated models/datasets"
echo "  python scripts/check_env.py --role train"
