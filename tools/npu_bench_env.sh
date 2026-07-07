#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# npu_bench_env.sh — run a benchmark command and log the machine state around it.
#
# WHY: the rocket NPU result is host-orchestration-bound, so a number is only
# comparable against another taken under the same conditions. The NPU clock parks
# at 200 MHz idle and rides up only under load; the CPU cores park low under a
# ramping governor; thermals climb across a long sweep; a cold model faults in from
# NVMe. A run that looks like a regression is often just a colder/slower-clocked/
# thermally-throttled environment. This wrapper records the load-time environment
# (sampled WHILE the command runs, not an idle snapshot) so drift is visible instead
# of mistaken for a code change.
#
# Logs, sampled every --interval seconds during the command:
#   - NPU core clock (scmi_clk_npu) — min/median/max under load (needs root to read
#     /sys/kernel/debug/clk/clk_summary; degrades to "n/a" without it)
#   - NPU per-core runtime-PM state (was the NPU actually active during the run?)
#   - big-core (A76, cpu4-7) and little-core (A55, cpu0) cpufreq governor + freq
#   - thermals: npu / bigcore0 / bigcore2 / package (before, max-during, after)
#   - MemAvailable, swap used, and the major-fault delta across the run (cold-NVMe tell)
# And once, after the command exits:
#   - any ROCKET_MM_PROFILE / llama-bench rows the command emitted on stderr
#
# Usage:
#   sudo -E ./npu_bench_env.sh -- <command...>
#   sudo -E ./npu_bench_env.sh --label "qwen9b q4k ub2048" -- llama-bench -m ... -p 2048
#   NPU_BENCH_ENV_LOG=./bench.tsv sudo -E ./npu_bench_env.sh -- <command...>
#
# Options:
#   --label STR      free-text tag recorded with the run (default: empty)
#   --interval SEC   sampler period in seconds (default: 1)
#   --                end of options; everything after is the command + its args
#
# Notes:
#   - Pairs with npu_perf_governor.sh (pin CPUs) and npu_set_irq_affinity.sh.
#   - Run under `sudo -E` so /dev/accel privilege, the ROCKET_* knobs, AND the
#     clk_summary read all survive (sudo strips the env without -E).
#   - The summary goes to stderr (stdout passes the command through untouched); a
#     one-line TSV record is appended to $NPU_BENCH_ENV_LOG if that var is set.
set -u

label=""
interval=1
while [ $# -gt 0 ]; do
  case "$1" in
    --label)    label=${2:-}; shift 2 ;;
    --interval) interval=${2:-1}; shift 2 ;;
    --)         shift; break ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *)          break ;;   # first non-option token starts the command
  esac
done
if [ $# -eq 0 ]; then
  echo "npu_bench_env: no command given (usage: $0 [--label STR] [--interval SEC] -- <command...>)" >&2
  exit 2
fi

# --- sysfs/debugfs locations (probed; absent ones degrade to n/a) ----------------
CLK_SUMMARY=/sys/kernel/debug/clk/clk_summary
BIG_CPU=/sys/devices/system/cpu/cpu4/cpufreq      # A76 cluster representative
LIT_CPU=/sys/devices/system/cpu/cpu0/cpufreq      # A55 cluster representative
NPU_CORES="/sys/devices/platform/fdab0000.npu /sys/devices/platform/fdac0000.npu /sys/devices/platform/fdad0000.npu"

tz() {  # tz <type> -> milli-degrees, or empty
  for z in /sys/class/thermal/thermal_zone*; do
    [ "$(cat "$z/type" 2>/dev/null)" = "$1" ] && { cat "$z/temp" 2>/dev/null; return; }
  done
}
c2m() { awk -v v="${1:-}" 'BEGIN{ if (v=="") print "n/a"; else printf "%.1f", v/1000 }'; }  # milli -> deg
khz2mhz() { awk -v v="${1:-}" 'BEGIN{ if (v=="") print "n/a"; else printf "%.0f", v/1000 }'; }
hz2mhz()  { awk -v v="${1:-}" 'BEGIN{ if (v=="") print "n/a"; else printf "%.0f", v/1000000 }'; }

npu_clk_hz() {  # current scmi_clk_npu rate in Hz (needs root); empty if unreadable
  [ -r "$CLK_SUMMARY" ] || return
  awk '$1=="scmi_clk_npu"{print $5; exit}' "$CLK_SUMMARY" 2>/dev/null
}
npu_pm_state() {  # "active" if any NPU core is active, else the first state seen
  local any="" s
  for d in $NPU_CORES; do
    s=$(cat "$d/power/runtime_status" 2>/dev/null) || continue
    [ "$s" = "active" ] && { echo active; return; }
    any=${any:-$s}
  done
  echo "${any:-n/a}"
}

# median + max of integers on stdin -> "median max" ("n/a n/a" if none).
# Uses `sort -n` + plain awk so it works under mawk (no gawk asort); non-numeric
# tokens (the "-" placeholders below) are skipped.
stats() {
  sort -n | awk '$1 ~ /^[0-9]+$/{a[n++]=$1} END{
    if(!n){print "n/a n/a"; exit}
    m=(n%2)? a[int(n/2)] : int((a[n/2-1]+a[n/2])/2)
    print m, a[n-1]
  }'
}

# --- before snapshot -------------------------------------------------------------
ts_start=$(date '+%Y-%m-%d %H:%M:%S')
maj_before=$(awk '/^pgmajfault/{print $2}' /proc/vmstat 2>/dev/null)
mem_avail_kb=$(awk '/^MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)
swap_used_kb=$(awk '/^SwapTotal/{t=$2}/^SwapFree/{f=$2}END{print t-f}' /proc/meminfo 2>/dev/null)
big_gov=$(cat "$BIG_CPU/scaling_governor" 2>/dev/null || echo n/a)
lit_gov=$(cat "$LIT_CPU/scaling_governor" 2>/dev/null || echo n/a)
t_npu_before=$(tz npu-thermal); t_pkg_before=$(tz package-thermal)
[ -r "$CLK_SUMMARY" ] || echo "npu_bench_env: note — $CLK_SUMMARY unreadable (run under sudo for NPU clock)" >&2

# --- background sampler ----------------------------------------------------------
samp=$(mktemp /tmp/npu_bench_env.XXXXXX)   # cols: npu_clk_hz big_khz lit_khz npu_mC big_mC pkg_mC pm
sampler() {
  local nc bf lf tn tb tp pm
  while :; do
    # Every field defaults to "-" so the row always has 7 tokens — an empty field
    # would shift awk's positional ($1/$2/$4) parsing of the columns below.
    nc=$(npu_clk_hz); bf=$(cat "$BIG_CPU/scaling_cur_freq" 2>/dev/null); lf=$(cat "$LIT_CPU/scaling_cur_freq" 2>/dev/null)
    tn=$(tz npu-thermal); tb=$(tz bigcore0-thermal); tp=$(tz package-thermal); pm=$(npu_pm_state)
    printf '%s %s %s %s %s %s %s\n' "${nc:--}" "${bf:--}" "${lf:--}" "${tn:--}" "${tb:--}" "${tp:--}" "${pm:--}" >> "$samp"
    sleep "$interval"
  done
}
sampler & samp_pid=$!
trap 'kill "$samp_pid" 2>/dev/null; rm -f "$samp" "$errlog" 2>/dev/null' EXIT INT TERM

# --- run the command (stdout passthrough; stderr tee'd for profile extraction) ---
errlog=$(mktemp /tmp/npu_bench_env_err.XXXXXX)
sec_start=$SECONDS
"$@" 2> >(tee "$errlog" >&2)
rc=$?
elapsed=$((SECONDS - sec_start))

kill "$samp_pid" 2>/dev/null; wait "$samp_pid" 2>/dev/null

# --- after snapshot + aggregates -------------------------------------------------
ts_end=$(date '+%Y-%m-%d %H:%M:%S')
maj_after=$(awk '/^pgmajfault/{print $2}' /proc/vmstat 2>/dev/null)
maj_delta=$(( ${maj_after:-0} - ${maj_before:-0} ))
t_npu_after=$(tz npu-thermal)
pm_seen=$(awk '{print $7}' "$samp" 2>/dev/null | sort -u | paste -sd, -)

read -r npu_clk_med npu_clk_max < <(awk '{print $1}' "$samp" | stats)
read -r big_med big_max         < <(awk '{print $2}' "$samp" | stats)
read -r npu_t_med npu_t_max     < <(awk '{print $4}' "$samp" | stats)
nsamp=$(wc -l < "$samp" 2>/dev/null | tr -d ' ')

# --- summary (stderr) ------------------------------------------------------------
{
  echo "──────────────────────────────────────────────────────────────────────"
  echo "npu_bench_env  ${label:+[$label] }rc=$rc  wall=${elapsed}s  samples=$nsamp@${interval}s"
  echo "  when      : $ts_start → $ts_end"
  echo "  NPU clock : median $(hz2mhz "$npu_clk_med") / max $(hz2mhz "$npu_clk_max") MHz under load   pm=${pm_seen:-n/a}"
  echo "  A76 cpufreq: gov=$big_gov  median $(khz2mhz "$big_med") / max $(khz2mhz "$big_max") MHz    A55 gov=$lit_gov"
  echo "  thermals  : npu $(c2m "$t_npu_before")→max $(c2m "$npu_t_max")→$(c2m "$t_npu_after") °C   pkg start $(c2m "$t_pkg_before") °C"
  echo "  memory    : avail $(awk -v k="${mem_avail_kb:-0}" 'BEGIN{printf "%.1f", k/1048576}') GiB  swap_used $(awk -v k="${swap_used_kb:-0}" 'BEGIN{printf "%.2f", k/1048576}') GiB  major_faults +$maj_delta"
  prof=$(grep -iE 'ROCKET_MM_PROFILE|pack[AB]|wDMA|fDMA|readback|wt_rotate|wt_quant|dequant' "$errlog" 2>/dev/null | head -20)
  [ -n "$prof" ] && { echo "  MM_PROFILE:"; echo "$prof" | sed 's/^/    /'; }
  bench=$(grep -iE '\| *pp[0-9]+ +\||\| *tg[0-9]+ +\||t/s' "$errlog" 2>/dev/null | grep -v '+/-.*t/s.*model' | head -12)
  [ -n "$bench" ] && { echo "  bench rows:"; echo "$bench" | sed 's/^/    /'; }
  echo "──────────────────────────────────────────────────────────────────────"
} >&2

# --- one-line TSV record (optional) ----------------------------------------------
if [ -n "${NPU_BENCH_ENV_LOG:-}" ]; then
  [ -s "$NPU_BENCH_ENV_LOG" ] || \
    printf 'ts\tlabel\trc\twall_s\tnpu_clk_med_mhz\tnpu_clk_max_mhz\ta76_med_mhz\ta76_gov\tnpu_t_max_c\tmaj_faults\tmem_avail_gib\n' >> "$NPU_BENCH_ENV_LOG"
  printf '%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%.1f\n' \
    "$ts_start" "$label" "$rc" "$elapsed" \
    "$(hz2mhz "$npu_clk_med")" "$(hz2mhz "$npu_clk_max")" "$(khz2mhz "$big_med")" "$big_gov" \
    "$(c2m "$npu_t_max")" "$maj_delta" "$(awk -v k="${mem_avail_kb:-0}" 'BEGIN{print k/1048576}')" \
    >> "$NPU_BENCH_ENV_LOG"
fi

exit "$rc"
