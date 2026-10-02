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
#
# THE LIST FAILS WHAT IT DOES NOT RUN. It passes only when every selected row ran and
# passed, and the kernel stayed quiet while it did:
#
#   - Every ROCKET_* variable inherited from the caller is stripped before the first
#     row, and each row gets only the variables written on it. One inherited knob can
#     blind a whole class: ROCKET_RK3576_I32_SENTINEL=0 turns off the output stamps
#     that every same-input gate relies on to see a dropped write.
#   - rc=2 is a gate skipping itself (wrong chip, no device, no blob). A skip is a
#     failure unless the row is marked skippable (-s), because a skipped row checked
#     nothing and used to count toward a green list.
#   - The pass count must reach the number of selected rows that cannot skip.
#   - Each row gets its own kernel-journal delta, read from a cursor taken just before
#     it. A row fails on any "NPU job timed out", rk_iommu line or NPU-side WARNING in
#     its delta unless its -k pattern says that row causes them, and on any change of
#     the taint word across it. The cumulative dmesg count this list used to print
#     could not say which row a line came from, and dmesg rolls.
#
# A row runs directly when this user can open /dev/accel/accel0 (group render), and
# through `sudo -E` otherwise. The journal is read directly (group adm) or through
# `sudo -n`. With neither, every row's kernel side goes unchecked, and the list fails
# unless ALLOW_NO_KLOG=1 says the caller accepts that.
set -u

ROOT=${ROCKET_SRC_DIR:-$PWD}
B=${ROCKET_BUILD_DIR:-$ROOT/build}
FILTER=${1:-}

# Strip every inherited ROCKET_* before anything runs, then set only what the list needs.
stripped=()
for v in $(compgen -e); do
    case $v in ROCKET_*) stripped+=("$v"); unset "$v" ;; esac
done
export ROCKET_SRC_DIR=$ROOT
if [ ${#stripped[@]} -gt 0 ]; then
    echo "stripped from the caller's environment: ${stripped[*]}"
fi

if [ ! -x "$B/rk3576_conv_lib_gate" ]; then
    echo "no build at $B — cmake -S . -B build && cmake --build build -j8"
    exit 1
fi

# sudo only when there is a device this user cannot open. With no device node at all the
# rows run as they are and skip themselves, and a skip then fails the list below.
if [ -e /dev/accel/accel0 ] && ! { [ -r /dev/accel/accel0 ] && [ -w /dev/accel/accel0 ]; }; then
    RUN=(sudo -E env)
else
    RUN=(env)
fi

KLOG_VIA=UNREADABLE
for via in "" "sudo -n"; do
    if $via journalctl -k -q --no-pager -n 1 --show-cursor -o cat 2>/dev/null | grep -q '^-- cursor: '; then
        KLOG_VIA=$via
        break
    fi
done
if [ "$KLOG_VIA" = "UNREADABLE" ]; then
    echo "WARNING: the kernel journal is unreadable as $(id -un); no row's kernel side is checked"
fi

klog_cursor() {
    [ "$KLOG_VIA" = "UNREADABLE" ] && return 0
    $KLOG_VIA journalctl -k -q --no-pager -n 1 --show-cursor -o cat 2>/dev/null \
        | sed -n 's/^-- cursor: //p' | tail -n 1
}

# Lines that fail a row unless its -k pattern names them.
KLOG_BAD='NPU job timed out|rk_iommu|WARNING: .*(rocket|drivers/accel|drivers/gpu/drm|drivers/iommu)'

pass=0; fail=0; skip=0; need=0; failed_names=""; klog_notes=""; backstop=0
t0=$(date +%s)

# g [-s] [-k <regex>] <name> <env assignments...> -- <argv...>
#   -s          the row may skip (rc=2) without failing the list
#   -k <regex>  kernel lines this row causes by design: counted and reported, not failed
g() {
    local skippable=0 allow=""
    while :; do
        case $1 in
            -s) skippable=1; shift ;;
            -k) allow=$2; shift 2 ;;
            *) break ;;
        esac
    done
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then return; fi
    [ "$skippable" -eq 0 ] && need=$((need + 1))

    local log=/tmp/g36_$name.log
    local taint0 cursor
    taint0=$(cat /proc/sys/kernel/tainted)
    cursor=$(klog_cursor)
    local s=$(date +%s%3N)
    "${RUN[@]}" "${envs[@]}" "$B/$1" "${@:2}" > "$log" 2>&1
    local rc=$?
    local e=$(date +%s%3N)
    local taint1 note=""
    taint1=$(cat /proc/sys/kernel/tainted)

    # The row's own kernel delta.
    if [ "$KLOG_VIA" != "UNREADABLE" ] && [ -n "$cursor" ]; then
        local klog nbad nok nback
        # $KLOG_VIA is deliberately unquoted: empty, or the two words "sudo -n".
        klog=$($KLOG_VIA journalctl -k -q --no-pager -o short-monotonic \
               --after-cursor="$cursor" 2>/dev/null)
        nback=$(grep -c 'retiring it' <<<"$klog" || true)
        backstop=$((backstop + nback))
        if [ -n "$allow" ]; then
            nok=$(grep -cE "$allow" <<<"$klog" || true)
            nbad=$(grep -E "$KLOG_BAD" <<<"$klog" | grep -cvE "$allow" || true)
        else
            nok=0
            nbad=$(grep -cE "$KLOG_BAD" <<<"$klog" || true)
        fi
        [ "$nok" -gt 0 ] && note="$note expected-klog=$nok"
        [ "$nback" -gt 0 ] && note="$note backstop=$nback"
        if [ "$nbad" -gt 0 ]; then
            note="$note KLOG=$nbad"
            klog_notes="$klog_notes\n  [$name] $(grep -E "$KLOG_BAD" <<<"$klog" | head -n 1)"
            { echo "---- kernel lines this row caused ----"; grep -E "$KLOG_BAD" <<<"$klog"; } >> "$log"
            [ "$rc" -eq 0 ] && rc=1
        fi
    fi
    if [ "$taint0" != "$taint1" ]; then
        note="$note TAINT $taint0->$taint1"
        [ "$rc" -eq 0 ] && rc=1
    fi
    if [ "$rc" -eq 2 ] && [ "$skippable" -eq 0 ]; then
        note="$note skipped-but-not-skippable"
        rc=1
    fi

    printf '%-28s rc=%-3d %6d ms%s\n' "$name" "$rc" "$((e - s))" "$note"
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
# The resident A/B is off in the row above, which keeps its timings comparable: two
# alternating inputs through one held handle, each against its own transient answer.
g conv_lib_resident          ROCKET_LG_RESIDENT=1 -- rk3576_conv_lib_gate
g conv_sym                   -- rk3576_conv_sym all
# The fp16 convolution returns drained: back-to-back calls at the shapes whose programs run
# longest, two alternating inputs, every element against an exact host sum. A job fenced
# before its DPU finished writing leaves its last rows unwritten, and the write guard, which
# asks whether a task wrote anything, cannot see that.
g fp16_drain                 -- rk3576_drain_probe gate
g refusal_gate               -- rk3576_refusal_gate
g matmul_gate                -- rk3576_matmul_gate
# The matmul-form fp16 program, one job for the whole of K: its host half holds the
# transcription to charsiu's own output, and its device half is exact against the CPU with a
# wrong-layout control that must come back wrong.
g mm_fp16_gate               -- rk3576_mm_fp16_gate
# The matmul entry's REQUANT against a host model of it, per-tensor and per-column. The
# only gate anywhere that scores what the DPU's output convertor computes on the matmul
# path; every int8 accuracy claim about that entry rests on the two agreeing.
g mm_requant                 -- rk3576_mm_requant
# The tie rule, which no gate's own scales reach: bit 30 clear rounds half to even.
g requant_round             -- requant_round_probe
# The derivation's carry edge against the FLOAT scale, which the models cannot see: they
# share the derivation. A regression puts a scale in 32768 at half its value.
g requant_edge              -- requant_edge_probe
# The W8A8 route COMPOSED: the frontend's two-pass calibration, its frozen scale and the
# de-quantize, driven end to end on the part. Every other number on that route is host
# arithmetic over a simulator, and this is the only row where the part supplies the
# accumulator the calibration reads.
g w8a8_route                 -- rk3576_w8a8_route
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
g chain_raw                  -- rk3576_chain_raw gate
g chain_pool                 -- rk3576_chain_pool gate
g chain_argb                 -- rk3576_chain_argb gate
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
# Faults the NPU on purpose: an unmapped regcmd address, which this part absorbs as IOMMU
# stall timeouts (two per faulting job) and a job the kernel retires. Those lines are this
# row's by design. A WARNING is not, and still fails it.
g -k 'rk_iommu|NPU job timed out' \
  uapi_regcmd_fault          -- uapi_regcmd_fault_rocket

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

# Resident handles fed a CHANGED input: two alternating images, each scored against its own
# first answer. A bench that repeats one image cannot see a stale surface, and v1 and r18
# already vary below, under the poison.
for n in v2 iv1 iv3; do
    g "net_${n}_vary"        "${NETENV[@]}" ROCKET_NET=$n ROCKET_RK3576_NET_VARY=1 \
        -- rk3576_net_gate bench 20
done

# The write guard's coverage: two graphs run with a wide-output job injected before each
# inference and two alternating inputs, each scored against its own clean answer. Each row
# fails if the injection never trips the guard, since a row that never redoes a kick passes
# whether or not the guard works.
g net_guard_coverage_v1      "${NETENV[@]}" ROCKET_NET=v1 \
    ROCKET_RK3576_NET_VARY=1 ROCKET_RK3576_NET_POISON=1 -- rk3576_net_gate bench 50
g net_guard_coverage_r18     "${NETENV[@]}" ROCKET_NET=r18 \
    ROCKET_RK3576_NET_VARY=1 ROCKET_RK3576_NET_POISON=1 -- rk3576_net_gate bench 50
# The per-kick guard under the same injection. The perkick rows above assert only the rc
# and the last top-1 of one repeated image.
g net_guard_perkick_v1       "${NETENV[@]}" ROCKET_NET=v1 ROCKET_RK3576_GUARD_PER_KICK=1 \
    ROCKET_RK3576_NET_VARY=1 ROCKET_RK3576_NET_POISON=1 -- rk3576_net_gate bench 50

echo
# Every full-list row is unskippable today, so the full list must pass all 69. A filter
# lowers the minimum to the rows it selected.
FULL_ROWS=69
min=$need
[ -z "$FILTER" ] && [ "$min" -lt "$FULL_ROWS" ] && min=$FULL_ROWS
echo "pass=$pass skip=$skip fail=$fail   minimum pass=$min   ($(($(date +%s) - t0)) s)"
[ -n "$failed_names" ] && echo "FAILED:$failed_names"
[ -n "$klog_notes" ] && echo -e "kernel lines that failed a row:$klog_notes"
echo "taint at the end: $(cat /proc/sys/kernel/tainted)"
# The backstop is a job that never reported itself finished, which is what a poisoned
# submit looks like from the driver. On 7.2.3-1 and rocket 1.6.0 a clean run records about
# 163: refusal_gate 2-3, uapi_regcmd_fault 1, and one per kick each guard row redoes, about
# 52 a row. The per-row journal delta above is what attributes them.
# Matched on "retiring it" rather than on the reason: the driver's wording for it has
# changed once already, and a grep that misses the message reports zero hits rather than
# no column, which reads as a clean run.
if [ "$KLOG_VIA" = "UNREADABLE" ]; then
    echo "kernel journal: UNREADABLE, so no row's kernel side was checked"
else
    echo "backstop hits this run: $backstop"
fi

rc=0
[ "$fail" -gt 0 ] && rc=1
[ "$pass" -lt "$min" ] && { echo "FAIL: $pass passes, below the minimum of $min"; rc=1; }
if [ "$KLOG_VIA" = "UNREADABLE" ] && [ "${ALLOW_NO_KLOG:-0}" != "1" ]; then
    echo "FAIL: the kernel side went unchecked (set ALLOW_NO_KLOG=1 to accept that)"
    rc=1
fi
exit $rc
