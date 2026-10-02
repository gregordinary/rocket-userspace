#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
#
# npu_core_diff_gate.sh — one register-setting arm of the lane-0 gating sweep, per pinned core,
# on the vendor RK3588 kernel (rknpu, RKNPU_CORE_MASK pins a job to a core).
#
# usage: npu_core_diff_gate.sh WORKDIR ARM "ENV=VAL ..." [masks]
#   WORKDIR holds: probe (tests/npu_core_diff_probe built against a librocketnpu with the
#   npu_regcmd.c debug knobs ROCKET_CNA_CLK_GATE / ROCKET_CORE_MAC_GATING /
#   ROCKET_CORE_SOFT_GATING), score (tools/npu_core_diff_score.c), set/raw_NNN.{a16,b16} (the
#   operands, NT calls), and ref_mK/raw_NNN.f32 (the control arm's outputs per core).
#   masks default "1 2 4" (cores 0, 1, 2); NT defaults to the number of set/*.a16 files.
#
# Each arm waits for the NPU to power off (runtime_status suspended; the vendor driver powers
# down 3 s after its last user) so no register value can carry over from the previous arm,
# runs every call on each core with the env applied, scores the output against the base model,
# compares each call byte for byte with the control, and counts rknpu timeout/reset lines in the
# kernel log since the arm started. One line per core:
#   ARM mask K: pre <runtime_status> rc <probe rc> klog <n> | calls identical to control x/NT |
#   lane-0 events <n> | off-lane-0 mismatches <n>
# A timed-out job leaves rc 1 and klog > 0; the driver soft-resets the NPU, and the next arm's
# control comparison shows whether anything persisted. The sweep cannot see a write that lands
# and changes nothing on these operands, which reads the same as a write that never lands.
W=$1 ARM=$2 ENVS=$3 MASKS=${4:-"1 2 4"}
cd "$W" || exit 1
NT=${NT:-$(ls set/raw_*.a16 | wc -l)}
D=/sys/devices/platform/fdab0000.npu/power/runtime_status
mkdir -p out
for mk in $MASKS; do
  for i in $(seq 1 60); do [ "$(cat $D)" = suspended ] && break; sleep 0.5; done
  pre=$(cat $D)
  o=out/${ARM}_m$mk; rm -rf "$o"; mkdir -p "$o"
  since=$(date '+%Y-%m-%d %H:%M:%S')
  env $ENVS RKNPU_CORE_MASK=$mk timeout 300 ./probe raw set "$o" "$NT" > "$o.log" 2>&1
  rc=$?
  klog=$(journalctl -k --since "$since" --no-pager 2>/dev/null | grep -i -c "rknpu.*\(timeout\|error\|reset\|fail\)")
  ./score set "$o" "$NT" > "$o.mis" 2> "$o.sum"
  same=0
  for t in $(seq -f %03g 0 $((NT - 1))); do cmp -s "$o/raw_$t.f32" "ref_m$mk/raw_$t.f32" && same=$((same + 1)); done
  l0=$(awk '$3 % 16 == 0' "$o.mis" | wc -l); off=$(awk '$3 % 16 != 0' "$o.mis" | wc -l)
  echo "$ARM mask $mk: pre $pre rc $rc klog $klog | calls identical to control $same/$NT | lane-0 events $l0 | off-lane-0 mismatches $off | env: ${ENVS:-none}"
done
