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

decision() {
  PYTHONUNBUFFERED=1 uv run --extra train python -u \
    experiments/cifar_scale_gap/train.py --print-decision
}

run_cell() {
  local h=$1 s=$2 seed=$3
  log "train H=$h S=$s seed=$seed"
  set +e
  PYTHONUNBUFFERED=1 uv run --extra train python -u \
    experiments/cifar_scale_gap/train.py \
    --h-values "$h" --s-values "$s" --seeds "$seed" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "train H=$h S=$s seed=$seed failed rc=$rc"
    echo FAILED
    exit "$rc"
  fi
}

# One seed per process so a 10 h host cap cannot wipe a whole rung.
while true; do
  dec="$(decision | tee -a "$LOG")"
  next_h="$(echo "$dec" | sed -n 's/^EXPAND H: next H=\([0-9]*\).*/\1/p')"
  if [[ -z "$next_h" ]]; then
    break
  fi
  log "width rung H=$next_h"
  for seed in 0 1 2; do
    run_cell "$next_h" 32 "$seed"
  done
done

while true; do
  dec="$(decision | tee -a "$LOG")"
  s_line="$(echo "$dec" | sed -n 's/^EXPAND S: at H=\([0-9]*\) run S=\(.*\)/\1 \2/p')"
  if [[ -z "$s_line" ]]; then
    break
  fi
  s_h="${s_line%% *}"
  s_list="${s_line#* }"
  s_list="${s_list//,/ }"
  log "S rung at H=$s_h S=$s_list"
  for S in $s_list; do
    for seed in 0 1 2; do
      run_cell "$s_h" "$S" "$seed"
    done
  done
done

log "campaign gate complete"
echo DONE
