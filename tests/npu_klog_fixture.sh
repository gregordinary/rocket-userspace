#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
#
# npu_klog_fixture.sh -- the kernel-log bracket around the NPU tests (CTest fixture
# npu_klog in CMakeLists.txt).
#
#   npu_klog_fixture.sh mark  <file>   save a journal cursor to <file>
#   npu_klog_fixture.sh check <file>   read every kernel line after it
#
# The `rocket` driver retires a job that runs 500 ms, resets the core and signals the
# job's fence, and PREP_BO then returns 0 as it does for a job that completed. So a hung
# job fails no syscall, and a gate comparing the untouched output BO can pass on it. The
# kernel log is where it shows, as "NPU job timed out". `check` fails the run on any such
# line, on any rk_iommu line, and on any WARNING from the NPU's drivers, and prints the
# first few.
#
# The mark is a journal cursor, not a dmesg line count: dmesg reads empty under
# dmesg_restrict and stops moving once its ring buffer rolls. The journal is read as the
# caller first (the `adm` or `systemd-journal` group can), then through `sudo -n`, which
# never prompts. When neither works, `mark` still exits 0 so the NPU tests run, and
# `check` exits 2, which CTest reports as a skip: the bracket did not run.
set -u

mode=${1:-}
file=${2:-}
if [ -z "$mode" ] || [ -z "$file" ]; then
    echo "usage: $0 mark|check <cursor-file>" >&2
    exit 1
fi

read_cursor() {   # $1 = "" or "sudo -n"
    $1 journalctl -k -q --no-pager -n 1 --show-cursor -o cat 2>/dev/null \
        | sed -n 's/^-- cursor: //p' | tail -n 1
}

case "$mode" in
mark)
    for via in "" "sudo -n"; do
        cursor=$(read_cursor "$via")
        if [ -n "$cursor" ]; then
            printf '%s\n%s\n' "$via" "$cursor" > "$file"
            if [ -n "$via" ]; then
                echo "kernel journal marked (read through $via)"
            else
                echo "kernel journal marked (read directly)"
            fi
            exit 0
        fi
    done
    printf 'UNREADABLE\n' > "$file"
    echo "the kernel journal is unreadable as $(id -un), directly and through sudo -n:"
    echo "the NPU tests run, and npu_klog_check will report SKIP"
    exit 0
    ;;
check)
    if [ ! -r "$file" ]; then
        echo "no mark at $file: npu_klog_mark did not run"
        exit 1
    fi
    via=$(sed -n 1p "$file")
    cursor=$(sed -n 2p "$file")
    if [ "$via" = "UNREADABLE" ]; then
        echo "SKIP: the kernel journal was unreadable at the mark, so no job timeout this"
        echo "run caused can be seen. Add the user to adm, or allow sudo -n journalctl."
        exit 2
    fi
    # journald reads /dev/kmsg asynchronously: give the last test's lines time to land.
    sleep 1
    # $via is deliberately unquoted: it is empty or the two words "sudo -n".
    if ! log=$($via journalctl -k -q --no-pager -o short-monotonic --after-cursor="$cursor" 2>/dev/null); then
        echo "SKIP: the kernel journal could not be read back after the mark"
        exit 2
    fi
    npu_warn='WARNING: .*(rocket|drivers/accel|drivers/gpu/drm|drivers/iommu)'
    n_to=$(grep -c 'NPU job timed out' <<<"$log" || true)
    n_iommu=$(grep -c 'rk_iommu' <<<"$log" || true)
    n_warn=$(grep -cE "$npu_warn" <<<"$log" || true)
    n_lines=$(grep -c . <<<"$log" || true)
    echo "kernel log since the mark: $n_lines line(s); NPU job timeouts $n_to," \
         "rk_iommu lines $n_iommu, NPU-side WARNINGs $n_warn"
    if [ "$n_to" -gt 0 ] || [ "$n_iommu" -gt 0 ] || [ "$n_warn" -gt 0 ]; then
        echo "FAIL: the first offending lines:"
        grep -E "NPU job timed out|rk_iommu|$npu_warn" <<<"$log" | head -n 12
        echo "A timed-out job's fence signals like a completed one, so any test above that"
        echo "read its output may have scored the BO's previous contents."
        exit 1
    fi
    exit 0
    ;;
*)
    echo "usage: $0 mark|check <cursor-file>" >&2
    exit 1
    ;;
esac
