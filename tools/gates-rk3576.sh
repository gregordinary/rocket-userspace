#!/usr/bin/env bash
# The RK3576 gate list: every asserting gate this part has, one row each.
#
# Run it from the source root (the network gates read their blobs from
# tests/data/rk3576-net/), with a build tree at build/:
#
#     bash tools/gates-rk3576.sh            # the whole list
#     bash tools/gates-rk3576.sh conv       # only rows whose name matches
#
# Each row's full output is /tmp/g36_<name>.log; the table prints the rc only, so a
# failing gate's detail is in its log and re-running it standalone is the other way in.
# rc=2 is a gate skipping itself (wrong chip, no device, no blob) and is not a failure.
#
# Read the taint and WARNING lines at the end, not just the rcs: a gate can pass while
# the kernel takes a splat. The backstop column is cumulative dmesg.
#
# The two poison rows deliberately manufacture the wide-output hazard, so they are the
# rows expected to record backstop hits.
set -u

ROOT=${ROCKET_SRC_DIR:-$PWD}
B=${ROCKET_BUILD_DIR:-$ROOT/build}
FILTER=${1:-}
export ROCKET_SRC_DIR=$ROOT

if [ ! -x "$B/rk3576_conv_lib_gate" ]; then
    echo "no build at $B — cmake -S . -B build && cmake --build build -j8"
    exit 1
fi

pass=0; fail=0; skip=0; failed_names=""
t0=$(date +%s)

# g <name> <env assignments...> -- <argv...>
g() {
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then return; fi
    local log=/tmp/g36_$name.log
    local s=$(date +%s%3N)
    sudo -E env "${envs[@]}" "$B/$1" "${@:2}" > "$log" 2>&1
    local rc=$?
    local e=$(date +%s%3N)
    printf '%-28s rc=%-3d %6d ms\n' "$name" "$rc" "$((e - s))"
    case $rc in
        0) pass=$((pass + 1)) ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)); failed_names="$failed_names $name" ;;
    esac
}

NETENV=(ROCKET_RK3576_NET_RESIDENT=1 ROCKET_RK3576_NET_CUBE=1)

# --- host-only: no device, never skips ------------------------------------------
# `chain_layout_rocket` is NOT here: it is host-only but not chip-neutral — it drives
# gen_matmul_fp16/int8/int4, which refuse on this part by construction, so it fails
# rather than skipping. It is an RK3588 gate.
g regcmd_rk3576              -- regcmd_rk3576_gate
# The claim-time plan against the run's own verdict, over the whole envelope table. Pure —
# no submit, no device — so it belongs with the host-only gates even though its subject is
# the convolution path. It is what says the two have not drifted apart.
g conv_claimplan             -- rk3576_conv_lib_gate claimplan

# --- the encoder and its envelope -----------------------------------------------
g first_light                -- rk3576_first_light
g conv_gate                  -- rk3576_conv_gate all
g conv_lib_gate              -- rk3576_conv_lib_gate
g conv_sym                   -- rk3576_conv_sym all
g refusal_gate               -- rk3576_refusal_gate
g matmul_gate                -- rk3576_matmul_gate
g perchannel_gate            -- rk3576_perchannel_gate
g coeff_c                    -- rk3576_coeff_c all

# --- cube geometry: bases, strides, pitches, tails ------------------------------
g offset_cube                -- rk3576_offset_cube gate
g pad_channels               -- rk3576_pad_channels
g surf_stride                -- rk3576_surf_stride
g conv_pitch                 -- rk3576_conv_pitch
g row_pitch                  -- rk3576_row_pitch

# --- the packed-image first conv ------------------------------------------------
g argb_ic1                   -- rk3576_argb_ic1
g argb_pad                   -- rk3576_argb_pad
g argb_extent                -- rk3576_argb_extent
g argb_extend                -- rk3576_argb_extend

# --- chained streams ------------------------------------------------------------
g chain_raw                  -- rk3576_chain_raw
g chain_pool                 -- rk3576_chain_pool
g chain_argb                 -- rk3576_chain_argb
g chain_len_uniform          -- rk3576_chain_len uniform 40
g chain_len_mixed            -- rk3576_chain_len mixed 60

# --- the PPU --------------------------------------------------------------------
g pool_gate                  -- rk3576_pool_probe gate
g pool_lib                   -- rk3576_pool_probe lib
g pool_lib_packed            ROCKET_RK3576_POOL_PACK_SRC=1 -- rk3576_pool_probe lib
g pool_split                 -- rk3576_pool_probe split
g pool_place                 -- rk3576_pool_probe place
g pool_bound                 -- rk3576_pool_probe bound
g pool_avg                   -- rk3576_pool_probe avg

# --- the DPU: elementwise, the residual add, the LUT ----------------------------
g add_probe                  -- rk3576_add_probe gate
g residual_add               -- rk3576_residual_add gate
g lut_probe                  -- rk3576_lut_probe gate
g act_gate                   -- rk3576_act_gate gate

# --- the uAPI: the paths an unprivileged caller reaches -------------------------
g uapi_selftest              -- uapi_selftest_rocket
g uapi_submit_errpath        -- uapi_submit_errpath_rocket
g uapi_bo_ranges             -- uapi_bo_ranges_rocket
g uapi_bo_lifetime           -- uapi_bo_lifetime_rocket
g uapi_regcmd_fault          -- uapi_regcmd_fault_rocket

# --- whole networks: every pass, then the numbers -------------------------------
for n in v1 v2 r18 iv1 iv3; do
    g "net_${n}_all"         "${NETENV[@]}" ROCKET_NET=$n -- rk3576_net_gate all
done
for n in v1 v2 r18 iv1 iv3; do
    g "net_${n}_bench100"    "${NETENV[@]}" ROCKET_NET=$n -- rk3576_net_gate bench 100
done
for n in v1 v2 r18 iv1 iv3; do
    g "net_${n}_perkick"     "${NETENV[@]}" ROCKET_NET=$n \
        ROCKET_RK3576_GUARD_PER_KICK=1 -- rk3576_net_gate bench 100
done
g net_iv3_cube               "${NETENV[@]}" ROCKET_NET=iv3 -- rk3576_net_gate cube

# The write guard's coverage: two graphs run with a wide-output job injected before each
# inference and two alternating inputs, each scored against its own clean answer. These
# are the rows that should record backstop hits.
g net_guard_coverage_v1      "${NETENV[@]}" ROCKET_NET=v1 \
    ROCKET_RK3576_NET_VARY=1 ROCKET_RK3576_NET_POISON=1 -- rk3576_net_gate bench 50
g net_guard_coverage_r18     "${NETENV[@]}" ROCKET_NET=r18 \
    ROCKET_RK3576_NET_VARY=1 ROCKET_RK3576_NET_POISON=1 -- rk3576_net_gate bench 50

echo
echo "pass=$pass skip=$skip fail=$fail   ($(($(date +%s) - t0)) s)"
[ -n "$failed_names" ] && echo "FAILED:$failed_names"
echo "taint: $(cat /proc/sys/kernel/tainted)"
echo "dmesg WARNING: $(sudo dmesg | grep -c WARNING)"
# The backstop is a job that raised no PC_DONE at all, which is what a poisoned submit
# looks like from the driver. Expected non-zero, and only from the two poison rows.
echo "backstop hits (cumulative dmesg): $(sudo dmesg | grep -c 'raised no PC_DONE')"
exit $([ "$fail" -gt 0 ] && echo 1 || echo 0)
