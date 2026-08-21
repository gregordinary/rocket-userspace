#!/bin/bash
# How far does the RK3576 drain-deadline exposure reach on the int8 matmul entry?
#
# The recorded map — one shape wrong in 12 of 14 reps at the shipped dpu_grace_us=500,
# six others 0 of 5 — cannot tell a safe shape from an undersampled one, because the
# hazard is intermittent and 0 of 5 bounds nothing at that rate. The instrument is the
# module parameter, not the repetition count: on the SHIPPED module dpu_grace_us=0
# retires the job on the next poll tick (the cause dialled UP) and a large value waits
# (the cure). Read `dmesg | grep "Initialized rocket"` first — on a kernel carrying the
# drain-deadline patch 0 means WAIT instead and this script measures the opposite thing.
#
# Interleaved by construction: two passes over every (grace, shape) cell, because a
# correctness A/B over an intermittent hazard sampled in one stretch means nothing. The
# control that earns the claim is the failure coming back when the parameter goes back.
#
# One shape per PROCESS, pinned, so a cell's rate is not a function of what ran beside
# it. Restores dpu_grace_us on any exit.
#
# WHAT IT DID TO A BOARD. The first run of this script, with GRACES starting at 0, took the
# H96 MAX M9 off the network inside four minutes and it did not come back — on `ondemand`,
# pinned to one core, with no K-split shape in the grid, from a board that had been up for
# nearly two hours with zero dmesg WARNINGs and had just completed a clean device run. Not
# one grid cell was returned. Whether grace 0 caused it is UNMEASURED (the conv path's
# `rowmap` has swept grace to 0 at ten reps a cell without incident), but budget a power
# cycle before including 0, and prefer starting one notch in:
#
#   GRACES="250 500 4000" setsid nohup ./rk3576-mm-drain-sweep.sh > ~/mmdrain.log 2>&1 < /dev/null &
#
# The log goes to $HOME on purpose: a power cycle clears /tmp, and how far this got is the
# only thing that says which cell to suspect.
#
#   setsid nohup ./rk3576-mm-drain-sweep.sh > ~/mmdrain.log 2>&1 < /dev/null &
#
# AND THAT LOG HAS NEVER SURVIVED EITHER. Relaunched with GRACES="500 4000 250 100" -- so 0
# was never written, on `ondemand`, one pinned core, no K-split shape, on a board that had
# just returned 58 of 58 gate rows -- the board went unreachable in about FIVE SECONDS and
# ~/mmdrain.log was empty again. Three launches, three deaths, and each death's named
# parameter has since been excluded by the next run setting it safely and dying anyway.
# Redirection is not enough: stdbuf -oL, or a per-cell `sync`, or accept that the first cell
# is all you learn. Before running this script at all, run its first cell ALONE in the
# foreground, and read the SHAPES comment below.
set -u

REPS=${REPS:-5}
BIN=${BIN:-$HOME/npu/rocket-userspace/build/rk3576_mm_corr}
# Ordered SAFE-FIRST on purpose. The shipped default comes out first so a run that dies
# has already returned something, then the cure end, then one notch in from the extreme.
# 0 is NOT in the default list: it is the one value this script has taken to a board that
# then had to be power-cycled, and it is uncharged on this path. Add it deliberately, last
# and alone, with a power cycle budgeted -- 100 and 250 may carry the arm without it.
GRACES=${GRACES:-"500 4000 250 100"}
PASSES=${PASSES:-2}

# Ordered KNOWN-FIRST, which is not the same as small-first and is the correction the
# fourth board death bought. The old order led with `128 2048 2048` because it was the
# smallest; that shape had never been run on this part in any session -- not a gate row,
# not an sgemm A/B cell, not one of the six "0 of 5" cells -- and it is therefore the first
# device work of BOTH detached deaths, the second of which happened inside ~5 s. A cell
# being cheap is not a reason to believe it is safe. So the characterised shape goes first,
# and every never-run shape below is marked. Run a marked shape ALONE, in the foreground,
# before it appears in any loop.
SHAPES=(
  "256 2048 2048"   # characterised: 10 of 10 wrong at grace 500 over two sessions
  "512 2048 2048"   # run: sgemm A/B, 0 of 5
  "1024 2048 2048"  # run: sgemm A/B, 0 of 5
  "256 2048 512"    # run: sgemm A/B (as M512 N512), 0 of 5
  "256 4096 2048"   # run: sgemm A/B (as K4096), 0 of 5
  "128 2048 2048"   # NEVER RUN -- the grid's only 1-tile cell, and the prime suspect
  "384 2048 2048"   # NEVER RUN -- 3 tiles
  "256 2048 4096"   # NEVER RUN
)

restore() {
  echo 500 | sudo tee /sys/module/rocket/parameters/dpu_grace_us >/dev/null
  echo "RESTORED dpu_grace_us=$(cat /sys/module/rocket/parameters/dpu_grace_us)"
}
trap restore EXIT INT TERM

echo "module: $(sudo dmesg | grep -o 'Initialized rocket [0-9.]*' | tail -1)"
echo "blind_us=$(cat /sys/module/rocket/parameters/dpu_blind_us)" \
     "poll_us=$(cat /sys/module/rocket/parameters/poll_interval_us)" \
     "gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
W0=$(sudo dmesg | grep -c WARNING)
echo "WARNING baseline $W0"

for p in $(seq 1 "$PASSES"); do
  for g in $GRACES; do
    echo "$g" | sudo tee /sys/module/rocket/parameters/dpu_grace_us >/dev/null
    got=$(cat /sys/module/rocket/parameters/dpu_grace_us)
    if [ "$got" != "$g" ]; then
      echo "PASS $p GRACE $g NOT ACCEPTED (reads $got) -- skipping this grace"
      continue
    fi
    for s in "${SHAPES[@]}"; do
      out=$(sudo -E taskset -c 4 "$BIN" $s "$REPS" 2>&1)
      sum=$(echo "$out" | grep '^MMCORR')
      # every wrong rep's bounding box, on one line, so the signature is readable
      sig=$(echo "$out" | grep '^rep .*wrong' | sed 's/^rep [0-9]*: //' | tr '\n' ';')
      echo "PASS $p GRACE $g ${sum:-NO_SUMMARY}"
      [ -n "$sig" ] && echo "PASS $p GRACE $g   sig: $sig"
      # Rule 84's second half, which this script was not honouring: a redirected echo
      # reaches the page cache and stops there, so an ungraceful power cycle loses it. Two
      # runs died with an empty log for exactly this reason. sync costs milliseconds.
      sync
    done
  done
done

W1=$(sudo dmesg | grep -c WARNING)
echo "WARNING after $W1 (delta $((W1 - W0)))  taint $(cat /proc/sys/kernel/tainted)"
echo MM_DRAIN_SWEEP_DONE
