#!/bin/bash
# Per-rep readout of the drain deadline's two failure forms, at the heights the shipped
# planner still honours on `k7ic64-g2`. One process per pass, so the position of a rep
# within a process is readable; the aggregate `rowmap` line is printed alongside.
#
#   ROCKET_LG_MAP_SEQ=1 prints one line per rep: the dropped-to-zero and wrong-valued
#   counts and the span. The question is whether the severe form (a surface wrong across
#   every row and channel) is the row tail's own mechanism cut earlier, or a write landing
#   in a following job's window.
#
# Env: PASSES (default 6), REPS (default 10), LO/HI (default 33/36).
set -u
cd "$(dirname "$0")/.." || exit 1
passes=${PASSES:-6}
reps=${REPS:-10}
lo=${LO:-33}
hi=${HI:-36}
p=1
while [ "$p" -le "$passes" ]; do
    echo "=== process $p ==="
    sudo -E env ROCKET_LG_FILTER=k7ic64-g2 ROCKET_LG_MAP_ONLY=1 \
        ROCKET_LG_MAP_LO="$lo" ROCKET_LG_MAP_HI="$hi" ROCKET_LG_MAP_REPS="$reps" \
        ROCKET_LG_MAP_SEQ=1 ./build/rk3576_conv_lib_gate rowmap 2>/dev/null |
        grep -E '^ +(seq|ih)'
    p=$((p + 1))
done
echo "=== DONE ==="
