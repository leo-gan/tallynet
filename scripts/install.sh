#!/usr/bin/env bash
# Create a venv, install TallyNet, and compile native kernels.
#
# Usage:
#   ./scripts/install.sh
#   ./scripts/install.sh --train          # also torchvision
#   ./scripts/install.sh --cpu-torch      # PyTorch CPU wheels (CI / no NVIDIA)
#   PYTHON=python3.12 ./scripts/install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
WITH_TRAIN=0
CPU_TORCH=0
EXTRAS="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) WITH_TRAIN=1; shift ;;
    --cpu-torch) CPU_TORCH=1; shift ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$WITH_TRAIN" -eq 1 ]]; then
  EXTRAS="dev,train"
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "creating $ROOT/.venv"
  "$PYTHON" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip

if [[ "$CPU_TORCH" -eq 1 ]]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install -e ".[${EXTRAS}]"

export TALLYNET_MARCH="${TALLYNET_MARCH:-native}"
"$ROOT/scripts/build_native.sh" --out "$ROOT/.kernel_cache" --march "$TALLYNET_MARCH"

python -m tallynet.native info
python - <<'PY'
from tallynet.native import status
s = status()
assert s["cpu"], f"native CPU kernel did not load: {s}"
print("install ok: cpu kernel at", s["cpu_path"])
PY
