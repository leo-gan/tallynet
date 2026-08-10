#!/usr/bin/env bash
# Build a deployable wheel (and sdist) with a portable CPU kernel inside.
#
# Usage:
#   ./scripts/package.sh
#   TALLYNET_MARCH=x86-64-v3 ./scripts/package.sh
#
# Outputs:
#   dist/*.whl  dist/*.tar.gz
#   artifacts/native/libtallynet_popcount_cpu.so
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

# Portable SIMD for CI artifacts (AVX2 + POPCNT + BMI2). Override if needed.
MARCH="${TALLYNET_MARCH:-x86-64-v3}"
LIBDIR="$ROOT/tallynet/lib"
ART="$ROOT/artifacts/native"

mkdir -p "$LIBDIR" "$ART"
"$ROOT/scripts/build_native.sh" --out "$LIBDIR" --march "$MARCH" --force

cp -a "$LIBDIR"/"libtallynet_popcount_"*.so "$ART/" 2>/dev/null || true

rm -rf "$ROOT/dist" "$ROOT/build"
uv build

echo "wheels:"
ls -la "$ROOT/dist"
echo "kernels:"
ls -la "$ART"
