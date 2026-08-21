#!/bin/bash
# The int32 K-split ladder, ONE CELL PER PROCESS, ascending in submits.
#
# A cell in this class can take the device down across processes, so: nothing shares a
# process, the log is flushed to disk before the next cell starts, and the walk STOPS at
# the first cell that puts a WARNING in dmesg. Everything after such a cell in the same
# boot is contaminated (rule 85) and this script refuses to produce it.
#
# THE STOP CONDITION IS NOT SUFFICIENT. Run 2026-08-06 on the H96 MAX M9 at the shipped
# 1.5.0 module, this ladder took the WHOLE BOARD down — unreachable on the network, not
# merely the NPU wedged — somewhere in its first three cells, and the per-cell logs on
# disk were the only record. Expect to need a physical power cycle, run it where you can
# reach one, and read $OUT/*.log afterwards rather than this script's own stdout.
set -u
OUT=${OUT:-$HOME/plan/ladder}
BIN=${BIN:-$HOME/plan/i32cell}
mkdir -p "$OUT"

cells=(
  "64 8192 32     control-4submits-ocprog128"
  "128 8192 256   geometry-8submits-ocprog1024"
  "512 8192 256   geometry-32submits"
  "512 8192 2048  THE-WEDGE-256submits"
)
reps=${REPS:-3}

base=$(sudo dmesg | grep -c WARNING)
echo "baseline dmesg WARNING count = $base"

for spec in "${cells[@]}"; do
  set -- $spec
  M=$1; K=$2; N=$3; tag=$4
  # The wedge cell is run ONCE whatever REPS says: each occurrence costs a reboot.
  n=$reps
  case "$tag" in THE-WEDGE*) n=1;; esac
  for r in $(seq 1 $n); do
    f="$OUT/${tag}_r${r}.log"
    echo "=== $tag rep $r  (M=$M K=$K N=$N) -> $f"
    sudo -E env ROCKET_LOG_LEVEL=debug "$BIN" "$M" "$K" "$N" "$r" > "$f" 2>&1
    rc=$?
    sync
    now=$(sudo dmesg | grep -c WARNING)
    echo "    rc=$rc  WARNING count $base -> $now"
    grep -E "^(CELL|RC=|WRONG=)" "$f"
    if [ "$now" != "$base" ]; then
      echo "!!! STOP: this cell put a WARNING in dmesg. Everything after it in this boot"
      echo "!!! is contaminated. Last 30 dmesg lines:"
      sudo dmesg | tail -30
      exit 9
    fi
    if [ $rc -ne 0 ]; then
      echo "!!! cell non-clean (rc=$rc) with NO dmesg WARNING — continuing, this is a"
      echo "!!! different class from the wedge"
    fi
  done
done
echo "LADDER COMPLETE, no WARNING at any cell"
