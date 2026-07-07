#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
#
# correctness_matrix.sh — packed-layout correctness matrix driver.
#
# Drives matmul_correctness_matrix_rocket (cosine-similarity gate, realistic
# random fp16 inputs) across an M/K/N grid, the four fp16 entry points, and the
# operating-mode env knobs (default / KACC / DATA_REUSE). Catches silent layout /
# scatter / readback corruption that the integer-input gates cannot. Run on the
# RK3588 device as root (the NPU needs it):
#
#     cmake --build build -j --target matmul_correctness_matrix_rocket
#     sudo -E tests/correctness_matrix.sh          # full matrix
#     sudo -E COS_THRESH=0.999 tests/correctness_matrix.sh
#
# Exit 0 iff every run PASSed or SKIPped; nonzero (and a FAIL list) otherwise.
set -u
BIN="${BIN:-build/matmul_correctness_matrix_rocket}"
[ -x "$BIN" ] || { echo "missing $BIN (build it first)"; exit 2; }

pass=0; fail=0; skip=0; fails=""

tally() {  # tally <out> <rc> <label>
    echo "$1"
    # The test binary's EXIT STATUS is the authority: report() returns 0 on PASS, 1 on
    # FAIL, and the harness returns 0 on SKIP (2/3 on usage/alloc/no-NPU). So a nonzero
    # rc is a failure regardless of text (covers crashes too); the printed text only
    # disambiguates PASS vs SKIP when rc==0.
    if [ "${2:-1}" -ne 0 ]; then
        fail=$((fail+1)); fails="$fails\n  [$3] (rc=$2) $1"
        return
    fi
    case "$1" in
        *SKIP*) skip=$((skip+1));;
        *PASS*) pass=$((pass+1));;
        *FAIL*) fail=$((fail+1)); fails="$fails\n  [$3] (rc=0 but FAIL text!) $1";;
        *)      fail=$((fail+1)); fails="$fails\n  [$3] (rc=0, no PASS/SKIP) $1";;
    esac
}
run() {  # run <mode> <M> <K> <N> [env...]   (fp16)
    local mode=$1 M=$2 K=$3 N=$4; shift 4
    local out; out="$(env "$@" "$BIN" "$M" "$K" "$N" "$mode" 2>/dev/null)"
    tally "$out" "$?" "fp16 $mode $M $K $N $*"
}
run_dt() {  # run_dt <mode> <M> <K> <N> <dtype> [env...]
    local mode=$1 M=$2 K=$3 N=$4 dt=$5; shift 5
    local out; out="$(env "$@" "$BIN" "$M" "$K" "$N" "$mode" "$dt" 2>/dev/null)"
    tally "$out" "$?" "$dt $mode $M $K $N $*"
}

# Gemma-4-12B projection shapes (label K N) + synthetic edge shapes.
#   ffn_down has K>8192 (forces K-tiling + host fp32 accum)
#   N=4112 / K=8224 are %16/%32 but NOT %256 (tiling remainders)
SHAPES=(
  "attn_q   3840 4096"
  "attn_kv  3840 2048"
  "attn_o   4096 3840"
  "ffn_up   3840 15360"
  "ffn_down 15360 3840"   # K>8192
  "small    1024 1024"
  "Kbig     8224 2048"    # K>8192, K%256!=0
  "Nodd     3840 4112"    # N%256!=0
)

echo "############ PART 1 — shape grid, one-shot (tiled), CPU-accum oracle ############"
# K-accum is the default operating mode now, so PART 1 forces ROCKET_KACC=0 to keep the
# byte-exact host fp64-accum (CPU-accum) path gated across the full shape grid; PART 2/3
# below exercise the now-default KACC path. M grid spans: 1 (padded GEMV), 4 (min tile),
# 130 (caller-padded M%4!=0 -> 132), 256 (one full tile), 300 (remainder M-tile),
# 512 (multi-tile).
for s in "${SHAPES[@]}"; do
  set -- $s; lbl=$1 K=$2 N=$3
  for M in 1 4 130 256 300 512; do
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=0
  done
done

echo "############ PART 2 — all four entry points, representative shapes ############"
for s in "attn_q 3840 4096" "ffn_down 15360 3840"; do
  set -- $s; K=$2 N=$3
  for mode in tiled mt stream prepacked; do
    for M in 1 4 512; do
      run "$mode" "$M" "$K" "$N" SAMPLE_ROWS=48
    done
  done
done

echo "############ PART 3 — operating-mode env knobs (KACC / DATA_REUSE / WEIGHT_REUSE) ############"
# KACC arms NPU-side eltwise K-accum (falls back to CPU oracle for M<12); REUSE=2
# is DATA_REUSE, REUSE=1 is WEIGHT_REUSE. Validate each produces correct results vs
# the fp32 reference. KACC+REUSE=1 gates the WEIGHT_REUSE branch of the KACC path
# (mode 1) that AUTO can now select for nMt>nNt shapes (see PART 3b).
for s in "attn_q 3840 4096" "ffn_down 15360 3840"; do
  set -- $s; K=$2 N=$3
  for M in 4 16 512; do
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=1
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_REUSE=2
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=1 ROCKET_REUSE=2
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=1 ROCKET_REUSE=1
  done
done

echo "############ PART 3b — tall shapes (nMt>nNt): KACC AUTO depth-pick selects WEIGHT_REUSE ############"
# Tall shapes have MORE M-tiles than N-tiles, so WEIGHT_REUSE has the deeper
# consecutive same-operand run and KACC's AUTO (default, ROCKET_REUSE unset) picks it.
# N=256 -> nNt=1, so DATA_REUSE gets zero reuse here while WEIGHT_REUSE gets depth nMt
# -> the exact case the depth-pick exists for. Gate AUTO, forced WEIGHT, and forced
# DATA (oracle cross-check) all bit-exact vs the fp32 reference.
for s in "tall_a 2048 256" "tall_b 8192 512"; do
  set -- $s; K=$2 N=$3
  for M in 512 1024; do
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=1                  # AUTO -> WEIGHT
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=1 ROCKET_REUSE=1   # forced WEIGHT
    run tiled "$M" "$K" "$N" SAMPLE_ROWS=48 ROCKET_KACC=1 ROCKET_REUSE=2   # forced DATA
  done
done

echo "############ PART 4 — dtype matrix (int8/int4/int16-exact/bf16/tf32), one-shot ############"
# Shapes chosen to satisfy every dtype's alignment (N%64 for int4, K%32). M grid:
# 1 (padded GEMV, the bug this addresses), 4 (min tile), 512 (multi-tile).
for dt in int8 int4 int16 bf16 tf32; do
  for s in "3840 4096" "3840 2048" "3840 15360" "15360 3840"; do
    set -- $s; K=$1 N=$2
    for M in 1 4 512; do run_dt tiled "$M" "$K" "$N" "$dt" SAMPLE_ROWS=32; done
  done
done

echo "############ PART 5 — resident (prepacked) int8/int4: M==1 must SKIP, M>=4 PASS ############"
for dt in int8 int4; do
  for M in 1 4 512; do run_dt prepacked "$M" 3840 4096 "$dt" SAMPLE_ROWS=32; done
done

echo
echo "================ SUMMARY: PASS=$pass  SKIP=$skip  FAIL=$fail ================"
if [ "$fail" -ne 0 ]; then echo -e "FAILURES:$fails"; exit 1; fi
echo "ALL GREEN (every run PASS or SKIP)"
