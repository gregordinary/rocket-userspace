#!/bin/sh
# rk3576-lag-pressure.sh — the divisor lag's RATE against a DIALLED memory load.
#
# The lag's two recorded high-rate contexts (inside a kick; reading a host-scattered
# source) and its two quiet ones (its own submit; reading a producer's cube) are not
# separated by geometry, data, program or the gap between submits — all four are refuted.
# What the loud contexts share is concurrent traffic on the memory system while the
# pooling program runs. `rk3576-memhog` applies that traffic from the CPU, touching no NPU
# buffer and issuing no ioctl, so a rate that moves with it is about pressure and not
# about the submit or the source surface.
#
# The arms are INTERLEAVED, because this hazard's rate at a fixed configuration has been
# seen to read 0 and 10 of 10 across identical runs: a block of one arm followed by a block
# of the other prices the stretch, not the arm.
#
#   PASSES=5 N=60 THREADS="0 1 4" bash tools/rk3576-lag-pressure.sh
set -u
BIN=${BIN:-$HOME/npu/rocket-userspace/build/rk3576_pool_probe}
HOG=${HOG:-/tmp/memhog}
PASSES=${PASSES:-5}
N=${N:-60}
THREADS=${THREADS:-"0 1 4"}
HOGSECS=${HOGSECS:-8}

p=1
while [ "$p" -le "$PASSES" ]; do
    for t in $THREADS; do
        # A trailing `s` is the SPIN arm: the same core count, no memory traffic. Without
        # it a memory arm cannot say whether the effect is DRAM or just competition.
        spin=0
        nt=$t
        case "$t" in
            *s) spin=1; nt=${t%s} ;;
        esac
        echo "=== pass $p  hog ${nt} thread(s) spin=${spin} ==="
        hpid=""
        if [ "$nt" != "0" ]; then
            MEMHOG_SPIN=$spin "$HOG" "$nt" "$HOGSECS" > "/tmp/hog_p${p}_t${t}.log" 2>&1 &
            hpid=$!
            sleep 1
        fi
        sudo -E env ROCKET_POOLR_GAPS=0 ROCKET_POOLA_N="$N" \
            "$BIN" rate 2>&1 | grep -E "^  (geometry|c[0-9])"
        if [ -n "$hpid" ]; then
            wait "$hpid"
            cat "/tmp/hog_p${p}_t${t}.log"
        fi
        sleep 1
    done
    p=$((p + 1))
done
