#!/bin/bash
# Make the drain deadline's cut land EARLIER and read what the failure's extent does.
#
# The tail form is confined to one 16-channel group (`c 32-47` in 56 of 56 wrong reps at
# grace 500), whose whole extent is 25% of the surface; the severe form is 74-84%. If the
# two are one mechanism the extent must be a CONTINUUM as the cut moves earlier, and the
# channel span must widen past one group. If they are two, the gap between them stays
# empty at every grace value.
#
# `dpu_grace_us` is an INSTRUMENT here, never a fix. The script restores 500 on every exit
# path, including a signal.
set -u
cd "$(dirname "$0")/.." || exit 1
reps=${REPS:-10}
lo=${LO:-33}
hi=${HI:-36}

restore() {
    echo 500 | sudo tee /sys/module/rocket/parameters/dpu_grace_us > /dev/null
    echo "restored dpu_grace_us=$(cat /sys/module/rocket/parameters/dpu_grace_us)"
}
trap restore EXIT INT TERM

for g in ${GRACES:-500 250 100 0}; do
    echo "$g" | sudo tee /sys/module/rocket/parameters/dpu_grace_us > /dev/null
    echo "=== dpu_grace_us=$(cat /sys/module/rocket/parameters/dpu_grace_us) ==="
    sudo -E env ROCKET_LG_FILTER=k7ic64-g2 ROCKET_LG_MAP_ONLY=1 \
        ROCKET_LG_MAP_LO="$lo" ROCKET_LG_MAP_HI="$hi" ROCKET_LG_MAP_REPS="$reps" \
        ROCKET_LG_MAP_SEQ=1 ./build/rk3576_conv_lib_gate rowmap 2>/dev/null |
        grep -E '^ +(seq|ih)'
done
echo "=== DONE ==="
