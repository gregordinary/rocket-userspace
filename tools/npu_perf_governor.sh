#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# npu_perf_governor.sh — pin all CPU cores to the `performance` cpufreq governor.
#
# WHY: the rocket NPU workload is host-orchestration-bound (submit ioctl, blocking wait on
# the completion IRQ, host softmax/LayerNorm/GELU, de-tile/readback), and the clock patch
# re-applies the NPU V+clock via SCMI on every runtime-resume — all CPU-side work. Under a
# ramping governor (`schedutil`/`ondemand`) the cores park low between submits, inflating
# AND jittering any submit-bound measurement. On the SigLIP-B/16 encoder, moving the CPU
# governor from `schedutil` to `performance` (NPU clock held at 600 MHz throughout) cut the
# resident warm median ~7.2 s -> ~3.06 s (-57%) and collapsed run-to-run jitter (+-2 s ->
# +-0.06 s). The NPU core clock is untouched — this is purely the CPU-side floor.
#
# Pair with taskset to an A76 (cpu4-7) and npu_set_irq_affinity.sh for the full host-side win.
#
# Usage:
#   sudo ./npu_perf_governor.sh performance   # pin all cores to performance (before a bench)
#   sudo ./npu_perf_governor.sh schedutil     # restore the default ramping governor
#   sudo ./npu_perf_governor.sh show          # print per-core governor + current freq
set -u

CPUS=$(ls -d /sys/devices/system/cpu/cpu[0-9]* 2>/dev/null)

cmd=${1:-show}
case "$cmd" in
  show)
    for c in $CPUS; do
      g=$(cat "$c/cpufreq/scaling_governor" 2>/dev/null) || continue
      f=$(cat "$c/cpufreq/scaling_cur_freq" 2>/dev/null)
      printf "%s: %-12s %s kHz\n" "$(basename "$c")" "$g" "$f"
    done
    ;;
  performance|schedutil|ondemand|powersave|userspace)
    n=0
    for c in $CPUS; do
      [ -w "$c/cpufreq/scaling_governor" ] || continue
      echo "$cmd" > "$c/cpufreq/scaling_governor" && n=$((n+1))
    done
    echo "set $n cores -> $cmd"
    ;;
  *)
    echo "usage: $0 {performance|schedutil|show}" >&2
    exit 1
    ;;
esac
