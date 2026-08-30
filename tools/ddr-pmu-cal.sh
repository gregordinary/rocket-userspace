#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# ddr-pmu-cal.sh — calibrate the rockchip_ddr PMU against a known number of bytes.
#
# The PMU's sysfs metadata claims `read-bytes`/`write-bytes`/`bytes` in MB via a
# .scale of 2^-20. That is metadata, not a measurement. This drives ddr_pmu_cal at
# two pass counts and DIFFERENCES the counters, so that allocation, page faults, the
# kernel's page zeroing and the idle floor -- identical in both arms -- cancel, and
# what remains is (hi-lo) * size of traffic whose byte count is known exactly.
#
# A counter with no positive control is not a measurement: quote every later number
# through the ratio this prints, not through the sysfs unit.
#
# Usage: ddr-pmu-cal.sh [path-to-ddr_pmu_cal] [MiB] [lo] [hi] [reps]
set -u
BIN=${1:-./ddr_pmu_cal}
MIB=${2:-2048}
LO=${3:-1}
HI=${4:-9}
REPS=${5:-3}
PERF=${PERF:-perf}
EV=rockchip_ddr/read-bytes/,rockchip_ddr/write-bytes/,rockchip_ddr/bytes/

command -v "$PERF" >/dev/null || { echo "no perf"; exit 1; }
[ -d /sys/bus/event_source/devices/rockchip_ddr ] || { echo "no rockchip_ddr PMU"; exit 1; }
[ -x "$BIN" ] || { echo "no $BIN"; exit 1; }

# counted MiB for one arm: prints "read write total"
run_arm() { # $1 wpass  $2 rpass
    sudo "$PERF" stat -a -x, -e "$EV" -- "$BIN" "$MIB" "$1" "$2" 2>&1 >/dev/null \
      | awk -F, '/read-bytes/{r=$1} /write-bytes/{w=$1} /rockchip_ddr\/bytes\//{t=$1}
                 END{printf "%s %s %s\n", r, w, t}'
}

echo "== arm that must SUCCEED: the idle floor, 10 s =="
sudo "$PERF" stat -a -e "$EV" -- sleep 10 2>&1 | grep -E "rockchip_ddr|elapsed"
echo

echo "== the differential: $MIB MiB buffer, $LO vs $HI passes, $REPS reps =="
printf "%-6s %-4s %12s %12s %12s\n" arm pass "read MiB" "write MiB" "total MiB"
declare -A S
for kind in read write; do
  for lvl in lo hi; do
    p=$([ $lvl = lo ] && echo $LO || echo $HI)
    sr=0; sw=0; st=0
    for i in $(seq 1 $REPS); do
      if [ $kind = read ]; then out=$(run_arm 1 $p); else out=$(run_arm $p 1); fi
      set -- $out
      sr=$(echo "$sr + $1" | bc -l); sw=$(echo "$sw + $2" | bc -l); st=$(echo "$st + $3" | bc -l)
      printf "%-6s %-4s %12s %12s %12s\n" "$kind" "$p" "$1" "$2" "$3"
    done
    S[$kind-$lvl-r]=$(echo "$sr/$REPS" | bc -l)
    S[$kind-$lvl-w]=$(echo "$sw/$REPS" | bc -l)
    S[$kind-$lvl-t]=$(echo "$st/$REPS" | bc -l)
  done
done
echo

KNOWN=$(echo "($HI - $LO) * $MIB" | bc -l)   # MiB of known traffic in each differential
echo "== calibration: counted / known, over $KNOWN MiB of known traffic each =="
for kind in read write; do
  dr=$(echo "${S[$kind-hi-r]} - ${S[$kind-lo-r]}" | bc -l)
  dw=$(echo "${S[$kind-hi-w]} - ${S[$kind-lo-w]}" | bc -l)
  dt=$(echo "${S[$kind-hi-t]} - ${S[$kind-lo-t]}" | bc -l)
  echo "$kind differential: d_read=$(printf %.1f $dr) MiB  d_write=$(printf %.1f $dw) MiB  d_total=$(printf %.1f $dt) MiB"
  if [ $kind = read ]; then main=$dr; else main=$dw; fi
  echo "  RATIO counted/known on the $kind column = $(echo "$main / $KNOWN" | bc -l | cut -c1-6)"
  echo "  cross-column (the other direction)      = $(echo "($dt - $main) / $KNOWN" | bc -l | cut -c1-6)"
done
