#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# npu_bench_env.sh — run a benchmark command and log the machine state around it.
#
# WHY: the rocket NPU result is host-orchestration-bound, so a number is only
# comparable against another taken under the same conditions. The NPU clock parks
# at 198 MHz idle and rises only under load; the CPU cores park low under a
# ramping governor; thermals climb across a long sweep; a cold model faults in from
# NVMe. A run that looks like a regression is often just a colder/slower-clocked/
# thermally-throttled environment. This wrapper records the load-time environment
# (sampled WHILE the command runs, not an idle snapshot) so drift is visible instead
# of mistaken for a code change.
#
# The NPU clock is reported as the physical frequency. Above 200 MHz the RK3588 NPU runs
# from a PVTPLL whose frequency follows the rail voltage, so scmi_clk_npu's rate is the
# REQUEST: at a 0.80 V rail the 600 MHz request runs at ~816 MHz. The physical clock comes
# from tools/npu_clock (build it with `cc -O2 -o tools/npu_clock tools/npu_clock.c`).
#
# Logs, sampled every --interval seconds during the command:
#   - NPU clock path: the share of samples on the raised PVTPLL clock (CRU, always safe
#     to read; needs root and tools/npu_clock), and the requested scmi_clk_npu rate
#     (needs root to read /sys/kernel/debug/clk/clk_summary; "n/a" without it)
#   - NPU per-core runtime-PM state (was the NPU actually active during the run?)
#   - big-core (A76, cpu4-7) and little-core (A55, cpu0) cpufreq governor + freq
#   - thermals: npu / bigcore0 / bigcore2 / package (before, max-during, after)
#   - MemAvailable, swap used, and the major-fault delta across the run (cold-NVMe tell)
# And once, after the command exits:
#   - the physical clock at the raised rate and the NPU rail voltage. The PVTPLL counter is
#     inside the NPU power domain, so this holds all three cores at power/control "on" for
#     the read (~0.5 s) and restores each core's previous setting, also on Ctrl-C. Skipped
#     with --no-phys, without root, without tools/npu_clock, or when the run never raised
#     the clock.
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
#   --no-phys        skip the post-run physical-clock read (no power/control writes)
#   --                end of options; everything after is the command + its args
#
# Notes:
#   - Pairs with npu_perf_governor.sh (pin CPUs) and npu_set_irq_affinity.sh.
#   - Run under `sudo -E` so /dev/accel privilege, the ROCKET_* knobs, AND the
#     clk_summary and /dev/mem reads all survive (sudo strips the env without -E).
#   - The summary goes to stderr (stdout passes the command through untouched); a
#     one-line TSV record is appended to $NPU_BENCH_ENV_LOG if that var is set. The
#     TSV's last three columns (npu_phys_mhz, npu_raised_pct, npu_rail_mv) are newer
#     than the rest: start a new log file rather than appending to one with the old header.
#   - The npu_clk_med/max_mhz columns are the REQUESTED rate, kept for continuity.
set -u

label=""
interval=1
phys=1
while [ $# -gt 0 ]; do
  case "$1" in
    --label)    label=${2:-}; shift 2 ;;
    --interval) interval=${2:-1}; shift 2 ;;
    --no-phys)  phys=0; shift ;;
    --)         shift; break ;;
    -h|--help)  sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
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
NPU_CLOCK=${NPU_CLOCK:-$(dirname "$0")/npu_clock}   # tools/npu_clock, built from npu_clock.c
[ -x "$NPU_CLOCK" ] || NPU_CLOCK=$(command -v npu_clock 2>/dev/null || true)

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
npu_clk_path() {  # "P" on the raised PVTPLL clock, else the normal-PLL MHz; empty if unreadable
  [ -n "$NPU_CLOCK" ] || return
  local p; p=$("$NPU_CLOCK" path 2>/dev/null) || return
  [ "$p" = pvtpll ] && echo P || echo "$p"
}
npu_rail_uv() {  # vdd_npu_s0 in microvolts; empty if the regulator is not exposed
  local r
  for r in /sys/class/regulator/regulator.*; do
    [ "$(cat "$r/name" 2>/dev/null)" = vdd_npu_s0 ] && { cat "$r/microvolts" 2>/dev/null; return; }
  done
}

# Hold every core runtime-active, read the PVTPLL counter, then put each core's
# power/control back. saved_ctl doubles as the "a hold is in place" flag for the trap.
saved_ctl=""
release_cores() {
  [ -n "$saved_ctl" ] || return 0
  local d c i=1
  for d in $NPU_CORES; do
    c=$(echo "$saved_ctl" | cut -d' ' -f$i); i=$((i + 1))
    echo "${c:-auto}" > "$d/power/control" 2>/dev/null
  done
  saved_ctl=""
}
read_phys_clock() {  # sets phys_line ("mhz=... ring=... length=... rm=...") and rail_uv
  local d all ok=0
  saved_ctl=$(for d in $NPU_CORES; do cat "$d/power/control" 2>/dev/null || echo auto; done | paste -sd' ' -)
  for d in $NPU_CORES; do echo on > "$d/power/control" 2>/dev/null || { release_cores; return 1; }; done
  for _ in $(seq 40); do  # up to 2 s for every core to resume
    all=1
    for d in $NPU_CORES; do [ "$(cat "$d/power/runtime_status" 2>/dev/null)" = active ] || all=0; done
    [ "$all" = 1 ] && { ok=1; break; }
    sleep 0.05
  done
  if [ "$ok" = 1 ]; then
    phys_line=$("$NPU_CLOCK" count 100 2>/dev/null)
    rail_uv=$(npu_rail_uv)
  fi
  release_cores
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
[ -r "$CLK_SUMMARY" ] || echo "npu_bench_env: note — $CLK_SUMMARY unreadable (run under sudo for the requested NPU rate)" >&2
if [ -z "$NPU_CLOCK" ]; then
  echo "npu_bench_env: note — tools/npu_clock not built; the physical NPU clock is n/a (cc -O2 -o tools/npu_clock tools/npu_clock.c)" >&2
elif ! "$NPU_CLOCK" path >/dev/null 2>&1; then
  echo "npu_bench_env: note — $NPU_CLOCK path failed (not root, or not an RK3588); the physical NPU clock is n/a" >&2
  NPU_CLOCK=""
fi

# --- background sampler ----------------------------------------------------------
samp=$(mktemp /tmp/npu_bench_env.XXXXXX)   # cols: npu_req_hz big_khz lit_khz npu_mC big_mC pkg_mC pm npu_path
sampler() {
  local nc bf lf tn tb tp pm np
  while :; do
    # Every field defaults to "-" so the row always has 8 tokens — an empty field
    # would shift awk's positional ($1/$2/$4) parsing of the columns below.
    nc=$(npu_clk_hz); bf=$(cat "$BIG_CPU/scaling_cur_freq" 2>/dev/null); lf=$(cat "$LIT_CPU/scaling_cur_freq" 2>/dev/null)
    tn=$(tz npu-thermal); tb=$(tz bigcore0-thermal); tp=$(tz package-thermal); pm=$(npu_pm_state)
    np=$(npu_clk_path)
    printf '%s %s %s %s %s %s %s %s\n' "${nc:--}" "${bf:--}" "${lf:--}" "${tn:--}" "${tb:--}" "${tp:--}" "${pm:--}" "${np:--}" >> "$samp"
    sleep "$interval"
  done
}
sampler & samp_pid=$!
trap 'kill "$samp_pid" 2>/dev/null; release_cores; rm -f "$samp" "$errlog" 2>/dev/null' EXIT INT TERM

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

# Share of samples on the raised PVTPLL clock, and the normal-PLL rate seen otherwise.
read -r raised_pct pll_mhz < <(awk '$8=="P"{p++} $8~/^[0-9]+$/{m=$8} $8!="-"{n++}
  END{ if(n) printf "%.0f %s\n", 100*p/n, (m==""?"n/a":m); else print "n/a n/a" }' "$samp")

# Physical clock at the raised rate: one held read after the command, never during it.
phys_line=""; rail_uv=""
if [ "$phys" = 1 ] && [ -n "$NPU_CLOCK" ] && [ "$raised_pct" != n/a ] && [ "$raised_pct" != 0 ]; then
  read_phys_clock
fi
phys_mhz=$(echo "$phys_line" | sed -n 's/^mhz=\([0-9.]*\).*/\1/p')
phys_len=$(echo "$phys_line" | sed -n 's/.* length=\([0-9]*\).*/\1/p')
rail_note=""   # without the held read, the rail is the parked (idle) voltage
[ -n "$rail_uv" ] || { rail_uv=$(npu_rail_uv); rail_note=" (idle)"; }

# --- summary (stderr) ------------------------------------------------------------
{
  echo "──────────────────────────────────────────────────────────────────────"
  echo "npu_bench_env  ${label:+[$label] }rc=$rc  wall=${elapsed}s  samples=$nsamp@${interval}s"
  echo "  when      : $ts_start → $ts_end"
  echo "  NPU clock : ${phys_mhz:-n/a} MHz physical when raised${phys_len:+ (PVTPLL length $phys_len)}, rail $(awk -v u="${rail_uv:-}" 'BEGIN{ if (u=="") print "n/a"; else printf "%.3f V", u/1e6 }')${rail_note}   raised in ${raised_pct}% of samples, ${pll_mhz} MHz otherwise   pm=${pm_seen:-n/a}"
  echo "  NPU request: median $(hz2mhz "$npu_clk_med") / max $(hz2mhz "$npu_clk_max") MHz (scmi_clk_npu, the requested rate)"
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
    printf 'ts\tlabel\trc\twall_s\tnpu_clk_med_mhz\tnpu_clk_max_mhz\ta76_med_mhz\ta76_gov\tnpu_t_max_c\tmaj_faults\tmem_avail_gib\tnpu_phys_mhz\tnpu_raised_pct\tnpu_rail_mv\n' >> "$NPU_BENCH_ENV_LOG"
  printf '%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%.1f\t%s\t%s\t%s\n' \
    "$ts_start" "$label" "$rc" "$elapsed" \
    "$(hz2mhz "$npu_clk_med")" "$(hz2mhz "$npu_clk_max")" "$(khz2mhz "$big_med")" "$big_gov" \
    "$(c2m "$npu_t_max")" "$maj_delta" "$(awk -v k="${mem_avail_kb:-0}" 'BEGIN{print k/1048576}')" \
    "${phys_mhz:-n/a}" "$raised_pct" "$(awk -v u="${rail_uv:-}" 'BEGIN{ if (u=="") print "n/a"; else printf "%.0f", u/1000 }')" \
    >> "$NPU_BENCH_ENV_LOG"
fi

exit "$rc"
