#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
#
# Run one command (tests/fa_replay_probe) under a PRIVATE tracefs instance recording the
# gpu_scheduler job events, with FA_REPLAY_MARKER pointing at that instance's trace_marker, and
# save the trace. The instance is created here and removed on exit, so the global trace state
# (tracing_on, set_event, buffer size, clock) is never touched. Run as root.
#
# Usage: sudo -E fa_core_trace.sh OUTPREFIX COMMAND [ARGS...]
#   writes OUTPREFIX.log (the command's output plus "rc N") and OUTPREFIX.trace
# Join with tools/fa_trace_join.py.
set -u
OUT=$1; shift
T=/sys/kernel/tracing
I=$T/instances/fa_core
EVS="drm_sched_job_queue drm_sched_job_run drm_sched_job_done"

cleanup() {
    [ -d "$I" ] || return 0
    echo 0 > "$I/tracing_on"
    for e in $EVS; do echo 0 > "$I/events/gpu_scheduler/$e/enable"; done
    echo > "$I/trace"
    rmdir "$I"
}

if [ -d "$I" ]; then echo "tracefs instance $I already exists; refusing" >&2; exit 1; fi
trap cleanup EXIT
mkdir "$I" || exit 1
echo mono > "$I/trace_clock"
echo 4096 > "$I/buffer_size_kb"
for e in $EVS; do echo 1 > "$I/events/gpu_scheduler/$e/enable" || exit 1; done
echo 1 > "$I/tracing_on"
FA_REPLAY_MARKER="$I/trace_marker" "$@" > "$OUT.log" 2>&1
rc=$?
echo 0 > "$I/tracing_on"
cat "$I/trace" > "$OUT.trace"
echo "rc $rc" >> "$OUT.log"
exit $rc
