#!/usr/bin/env bash
# After --scale finishes: expand H until the gate says stop, then expand S.
# Prints one line at the end: DONE or FAILED.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RES="experiments/cifar_scale_gap/results"
LOG="$RES/campaign.log"
mkdir -p "$RES"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

wait_pid() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
}

if [[ "${1:-}" == "--wait-pid" ]]; then
  log "waiting for pid $2"
  wait_pid "$2"
  log "pid $2 exited"
fi

run_expand() {
  local flag=$1
  local tmp
  tmp="$(mktemp)"
  log "starting $flag"
  set +e
  PYTHONUNBUFFERED=1 uv run --extra train python -u \
    experiments/cifar_scale_gap/train.py "$flag" >"$tmp" 2>&1
  local rc=$?
  set -e
  tee -a "$LOG" <"$tmp"
  if [[ $rc -ne 0 ]]; then
    log "$flag failed rc=$rc"
    rm -f "$tmp"
    echo FAILED
    exit "$rc"
  fi
  if grep -qE "no H rung to run|no S rung to run" "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  if grep -q "^>> " "$tmp"; then
    rm -f "$tmp"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

while run_expand --expand-h; do
  log "expand-h finished a rung"
done

while run_expand --expand-s; do
  log "expand-s finished a rung"
done

log "campaign gate complete"
echo DONE
