#!/usr/bin/env bash
# Compile CPU (and optional CUDA) popcount kernels.
#
# Usage:
#   ./scripts/build_native.sh
#   TALLYNET_MARCH=x86-64-v3 ./scripts/build_native.sh --out tallynet/lib
#   ./scripts/build_native.sh --cuda --force
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

OUT="${TALLYNET_KERNEL_DIR:-$ROOT/.kernel_cache}"
MARCH="${TALLYNET_MARCH:-native}"
CUDA=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --march) MARCH="$2"; shift 2 ;;
    --cuda) CUDA=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT"
ARGS=( -m tallynet.native build --out "$OUT" --march "$MARCH" )
if [[ "$CUDA" -eq 1 ]]; then
  ARGS+=(--cuda)
fi
if [[ "$FORCE" -eq 1 ]]; then
  ARGS+=(--force)
fi

echo "building kernels with $PYTHON  out=$OUT  march=$MARCH  cuda=$CUDA"
"$PYTHON" "${ARGS[@]}"
echo "ok: $OUT"
