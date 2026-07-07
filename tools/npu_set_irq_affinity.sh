#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# npu_set_irq_affinity.sh — bind the 3 RK3588 NPU completion IRQs to A76 big core(s).
#
# WHY: the default IRQ affinity mask is all-CPUs (`0-7`); GICv3 then services the NPU
# completion IRQ on the LOWEST cpu in the mask = cpu0, an A55 *little* core @1.8 GHz.
# The completion handler + waiter wakeup runs there, so the per-submit dispatch floor is
# ~51 us. Binding the IRQs to an A76 big core (cpu4-7 @2.4 GHz) cuts it to ~27 us (-47%),
# measured on-device (RK3588 @600 MHz).
#
# This is a runtime knob (no driver change). Re-apply after reboot (or wire into a unit).
#
# Usage:
#   sudo ./npu_set_irq_affinity.sh latency [BIGCORE]   # all 3 IRQs -> one big core (default 7)
#                                                        # then run: taskset -c BIGCORE <app>
#   sudo ./npu_set_irq_affinity.sh throughput           # IRQ 69->5 70->6 71->7 (multi-fd pool)
#   sudo ./npu_set_irq_affinity.sh reset                # restore default 0-7
#   sudo ./npu_set_irq_affinity.sh show                 # print current binding + serviced cpu
set -u

# discover the NPU IRQ numbers (robust to kernel-specific GIC numbering)
mapfile -t IRQS < <(grep -E 'fdab0000|fdac0000|fdad0000' /proc/interrupts | sed 's/^ *//;s/:.*//')
if [[ ${#IRQS[@]} -lt 3 ]]; then echo "could not find 3 NPU IRQs in /proc/interrupts"; exit 1; fi

set_one() { echo "$2" > /proc/irq/"$1"/smp_affinity_list || echo "WARN: irq $1 <- $2 failed"; }

case "${1:-show}" in
  latency)
    CORE="${2:-7}"
    for q in "${IRQS[@]}"; do set_one "$q" "$CORE"; done
    echo "NPU IRQs ${IRQS[*]} -> cpu$CORE (latency). Run the app with: taskset -c $CORE <app>"
    ;;
  throughput)
    CORES=(5 6 7)
    for i in "${!IRQS[@]}"; do set_one "${IRQS[$i]}" "${CORES[$i]}"; done
    echo "NPU IRQs ${IRQS[*]} -> cpu${CORES[*]} (throughput). Pin each worker to its core."
    ;;
  reset)
    for q in "${IRQS[@]}"; do set_one "$q" 0-7; done
    echo "NPU IRQs reset to 0-7 (default; services on cpu0 = A55 little)."
    ;;
  show|*)
    for q in "${IRQS[@]}"; do
      printf "irq %s  affinity=%s\n" "$q" "$(cat /proc/irq/"$q"/smp_affinity_list)"
    done
    ;;
esac
