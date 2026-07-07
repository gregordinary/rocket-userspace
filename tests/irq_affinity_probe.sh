#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# irq_affinity_probe.sh — measure the NPU dispatch floor (us/submit) as a function
# of where the 3 NPU completion IRQs land and where the submitting thread runs.
#
# IRQ affinity is a standard multi-core lever: binding the three NPU completion IRQs to
# a CPU big core (and pinning the app there) can cut submit/wakeup latency. We tested
# *app* taskset before (no win on fp16 prefill) but never the IRQ binding itself. This probes it directly on
# the submit-overhead path (single-fd round-trip; the IRQ-delivery + waiter-wakeup
# latency is inside the timed window).
#
# RK3588: cpu0-3 = A55 little (1.8 GHz), cpu4-7 = A76 big (2.4 GHz).
# NPU IRQs (this kernel): 69 = fdab0000.npu, 70 = fdac0000.npu, 71 = fdad0000.npu.
#
# Usage: sudo ./irq_affinity_probe.sh [iters] [cycles]
set -u
BIN="$(dirname "$0")/../build/submit_overhead_rocket"
ITERS="${1:-3000}"
CYCLES="${2:-5}"
IRQS=(69 70 71)
SHAPE="8 64 16"

if [[ ! -x "$BIN" ]]; then echo "missing $BIN — build first"; exit 1; fi

# save + restore IRQ affinity
declare -A SAVED
for i in "${IRQS[@]}"; do SAVED[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
restore() { for i in "${IRQS[@]}"; do echo "${SAVED[$i]}" > /proc/irq/$i/smp_affinity_list 2>/dev/null; done; }
trap restore EXIT

set_irq() { # $1 = affinity list applied to all 3 NPU irqs
  for i in "${IRQS[@]}"; do echo "$1" > /proc/irq/$i/smp_affinity_list 2>/dev/null || echo "WARN: irq $i set $1 failed"; done
}

# one measurement: $1 = label, $2 = irq affinity list, $3 = taskset spec ("" = none)
run_one() {
  local irq="$2" ts="$3"
  set_irq "$irq"
  local cmd="$BIN $SHAPE $ITERS"
  [[ -n "$ts" ]] && cmd="taskset -c $ts $cmd"
  $cmd 2>/dev/null | grep RESULT | sed 's/RESULT //'
}

# configs: label | irq-affinity | taskset
CFG_LABELS=(A_base_unpinned B_base_appcpu6 C_irq6_app6 D_irq4_app7 E_irq6_unpinned)
CFG_IRQ=(0-7 0-7 6 4 6)
CFG_TS=("" 6 6 7 "")

declare -A ACC
echo "shape=$SHAPE iters=$ITERS cycles=$CYCLES   (median us/submit per run)"
echo "actual NPU IRQ cpu (post-set) reported per run as well"
for ((c=1;c<=CYCLES;c++)); do
  for idx in "${!CFG_LABELS[@]}"; do
    L="${CFG_LABELS[$idx]}"
    out=$(run_one "$L" "${CFG_IRQ[$idx]}" "${CFG_TS[$idx]}")
    med=$(echo "$out" | sed -n 's/.*submit_us_median=\([0-9.]*\).*/\1/p')
    minv=$(echo "$out" | sed -n 's/.*submit_us_min=\([0-9.]*\).*/\1/p')
    # which cpu actually serviced the npu irq (delta in interrupt counts)
    echo "cycle $c  $L  median=$med  min=$minv"
    ACC[$L]="${ACC[$L]:-} $med"
  done
done

echo
echo "=== summary (median of per-cycle medians) ==="
for idx in "${!CFG_LABELS[@]}"; do
  L="${CFG_LABELS[$idx]}"
  vals=$(echo "${ACC[$L]}" | tr ' ' '\n' | grep -v '^$' | sort -n)
  n=$(echo "$vals" | wc -l)
  med=$(echo "$vals" | awk -v n="$n" 'NR==int((n+1)/2){print}')
  mn=$(echo "$vals" | head -1); mx=$(echo "$vals" | tail -1)
  printf "%-20s median=%-8s min=%-8s max=%-8s  irq=%s ts=%s\n" "$L" "$med" "$mn" "$mx" "${CFG_IRQ[$idx]}" "${CFG_TS[$idx]:-none}"
done
