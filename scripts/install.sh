#!/usr/bin/env bash
# Create a uv venv, install TallyNet, and compile native kernels.
#
# Usage:
#   ./scripts/install.sh
#   ./scripts/install.sh --train          # also torchvision
#   ./scripts/install.sh --cpu-torch      # PyTorch CPU wheels (CI / no NVIDIA)
#   PYTHON=3.12 ./scripts/install.sh
#   ./scripts/install.sh --clear          # replace an existing .venv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

PYTHON="${PYTHON:-3.12}"
WITH_TRAIN=0
CPU_TORCH=0
CLEAR=0
UV_EXTRAS=(--extra dev)
EDITABLE_EXTRAS="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) WITH_TRAIN=1; shift ;;
    --cpu-torch) CPU_TORCH=1; shift ;;
    --clear) CLEAR=1; shift ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$WITH_TRAIN" -eq 1 ]]; then
  UV_EXTRAS+=(--extra train)
  EDITABLE_EXTRAS="dev,train"
fi

if [[ "$CLEAR" -eq 1 || ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "uv venv --python $PYTHON"
  uv venv --python "$PYTHON" --clear
else
  echo "reusing existing $ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

if [[ "$CPU_TORCH" -eq 1 ]]; then
  echo "installing CPU PyTorch, then project extras"
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -e ".[${EDITABLE_EXTRAS}]"
else
  uv sync "${UV_EXTRAS[@]}"
fi

export TALLYNET_MARCH="${TALLYNET_MARCH:-native}"
"$ROOT/scripts/build_native.sh" --out "$ROOT/.kernel_cache" --march "$TALLYNET_MARCH"

# Use the venv directly. `uv run` would re-sync the lockfile and can
# replace a CPU torch wheel installed with --cpu-torch.
"$ROOT/.venv/bin/python" -m tallynet.native info
"$ROOT/.venv/bin/python" - <<'PY'
from tallynet.native import status
s = status()
assert s["cpu"], f"native CPU kernel did not load: {s}"
print("install ok: cpu kernel at", s["cpu_path"])
PY
