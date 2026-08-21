/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 The rocket-userspace authors
 *
 * rocket_graph_rk3576.h — the RK3576 graph planner: which of a network's tensors stay in
 * cube layout, which producers write slices of a shared buffer, and which runs of layers
 * go out as ONE hardware kick.
 *
 * WHY THIS IS A COMPONENT AND NOT PART OF THE LIBRARY. `librocketnpu` holds no
 * graph-execution object: a frontend owns scheduling, buffer lifetimes and its own tensor
 * model, and only the RUN FINDER (rocket_chain_plan_rk3576(), pure) belongs to the
 * library because it is the one thing a frontend cannot own — which streams the part will
 * run. What is here is between those two: the placement and linking rules, which are
 * generic to the part but are not the part, and which every frontend with a graph would
 * otherwise write again. There are twelve refusal classes in them and every one closed so
 * far was a host-buffer rule rather than the hardware's, so a second copy of them would
 * fork silently.
 *
 * WHAT IT IS WORTH, on MobileNetV1-224: per-op entries with transient weights ~115 ms,
 * resident weights ~21, tensors kept in cube layout 10.4-10.5, the run as one kick 5.0.
 * A caller that only invokes the op entries gets about a twentieth of the part's
 * demonstrated throughput on that graph. [HW sweep, H96 MAX M9]
 *
 * WHAT THE CALLER OWNS AND THIS DOES NOT:
 *   - packing the weights (the resident handles it hands over, already packed)
 *   - the tensors, the ping-pong, the skip lifetimes, the de-scatter points
 *   - which layer runs when: the plan says what MAY be linked, not what to submit
 *
 * HOW TO USE IT. Fill one rocket_graph_layer per layer of the network in execution order,
 * with the handles already packed (one warm-up inference is the usual way), call
 * rocket_graph_plan_new(), then run the graph reading the plan's arrays: a layer with
 * `kick[i]` starts a chained run ending at `kick_end[i]`, a layer with `cube_in[i]` needs
 * no host scatter, a layer with `cube_out[i]` needs no de-scatter, and a layer that
 * `placement[i]` marks has no program at all.
 *
 * THE ENVIRONMENT KNOBS ARE THE A/B ARMS of each pass and keep their names, because every
 * measurement on record is quoted against them: ROCKET_RK3576_NET_CAT, _CATN,
 * _CATN_OVERLAP, _TAIL, _MULTI, _KICK, _KICK_AT, _KICK_MAX, _NOJOIN, _PADJOIN.
 */
#ifndef ROCKET_GRAPH_RK3576_H
#define ROCKET_GRAPH_RK3576_H

#include <stddef.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A layer's KIND, which is what decides the shape of every rule here. ROCKET_GRAPH_HOST is
 * a layer the caller runs on the host and the part never sees (a softmax): it produces no
 * tensor a join could carry, so it is stepped over rather than counted as a break. */
typedef enum {
    ROCKET_GRAPH_CONV = 0,
    ROCKET_GRAPH_DWCONV,
    ROCKET_GRAPH_AVGPOOL,
    ROCKET_GRAPH_MAXPOOL,
    ROCKET_GRAPH_ADD,
    ROCKET_GRAPH_CONCAT,
    ROCKET_GRAPH_HOST
} rocket_graph_kind;

/* Both pooling kinds are the same PPU program with a different reduction, so every rule
 * that asks "is this the pool" means both. */
#define ROCKET_GRAPH_IS_POOL(k) \
    ((k) == ROCKET_GRAPH_AVGPOOL || (k) == ROCKET_GRAPH_MAXPOOL)

#define ROCKET_GRAPH_MAX_SRC 4
#define ROCKET_GRAPH_NO_SRC  0xFFFFFFFFu

/* ONE LAYER, as the passes need it.
 *
 * The operands are named BY PRODUCER and never as "the layer before". A feed-forward chain
 * does not need that and a residual one does: MobileNetV2's adds read a skip three to five
 * layers back on operand B and the main line from the layer before, and a ResNet block
 * that changes width reads them the OTHER WAY ROUND. A rule keyed on adjacency wires one
 * family and silently feeds the other the wrong tensor. */
typedef struct {
    rocket_graph_kind kind;

    /* Geometry. `ic` and `w_zp` decide the one join class whose correctness rests on the
     * producer (a consumer walking round32(ic) past the producer's live channels pays a B
     * term for them); `oc oh ow` are the tensor a join carries, which is what prices it. */
    unsigned ic, oc, oh, ow;
    int      w_zp, out_zp;

    unsigned src[ROCKET_GRAPH_MAX_SRC];   /* producer indices, ROCKET_GRAPH_NO_SRC for none */

    /* THE CALLER'S LOWERING, which no rule here can see. Set it where the caller prepares
     * this layer's input on the host — a materialised asymmetric border, a widened or
     * shifted stem, anything that means the tensor the entry receives is not a producer's
     * output. Such a layer is never a cube consumer and never a chain member: a chain
     * scatters what it is handed and has no way to know a prepared copy was meant. */
    unsigned host_input;

    /* Set where this layer's output must stay a ROW-MAJOR tensor because something
     * outside the description reads it — in a frontend, a partition output. Every other
     * kind says that by leaving its handle NULL below, which costs one join and never a
     * wrong answer; a CONCATENATION has no handle to leave, and placing one leaves its
     * consumers reading a cube and nothing writing the tensor the outside reader wants.
     * Without this the caller's only recourse is to refuse the whole graph. */
    unsigned row_major_out;

    /* The resident handles, already packed. Exactly one of them is set, by kind; a NULL
     * here is how a layer says it cannot be linked or chained at all. */
    rocket_conv2d_int8_weights_rk3576 *conv;
    rocket_pool_int8_rk3576_handle    *pool;
} rocket_graph_layer;

/* WHY A JOIN WAS REFUSED. On a feed-forward chain almost every adjacent pair links and the
 * breakdown is not interesting; on a branchy one it is the whole answer to how much of the
 * graph can go out as one kick. Each bucket is a different thing to fix. */
enum { ROCKET_GRAPH_NJ_ADD_CONSUMER = 0, ROCKET_GRAPH_NJ_SKIP_SOURCE,
       ROCKET_GRAPH_NJ_IC_ALIGN, ROCKET_GRAPH_NJ_SURFACE, ROCKET_GRAPH_NJ_OTHER,
       ROCKET_GRAPH_NJ_FORCED, ROCKET_GRAPH_NJ_N };

/* WHY A LAYER IS OUTSIDE EVERY RUN — the term a submit count does not carry. A join is a
 * property of a PAIR and a run is broken by ONE layer, so the join table cannot say why a
 * graph takes n submits rather than one. */
enum { ROCKET_GRAPH_KB_HOST_PREP = 0, ROCKET_GRAPH_KB_ADD_HOST_CAT,
       ROCKET_GRAPH_KB_SKIP_MATERIALISE, ROCKET_GRAPH_KB_POOL_UNRESIDENT,
       ROCKET_GRAPH_KB_POOL_LAGGY, ROCKET_GRAPH_KB_NOT_RESIDENT,
       ROCKET_GRAPH_KB_NO_CUBE_OUT, ROCKET_GRAPH_KB_NO_CUBE_IN, ROCKET_GRAPH_KB_ALONE,
       ROCKET_GRAPH_KB_N };

/* THE PLAN. The arrays are indexed by layer and are what a caller reads as it runs the
 * graph; the counters are what a session quotes. */
/* The refused pairs, NAMED. A bucket's count prices a submit and its bytes price the
 * transposes, but neither prices a PAIR: the two classes already closed measure 8x apart
 * per join, so the pairs a session proposes to close are listed one at a time with the
 * tensor each of them carries. */
#define ROCKET_GRAPH_NJ_LIST_MAX 64

typedef struct rocket_graph_plan {
    const rocket_graph_layer *L;
    unsigned n;
    int fd;
    int verbose;

    unsigned char *cube_in;    /* this layer reads a producer's surface: no host scatter  */
    unsigned char *cube_out;   /* this layer leaves a cube: no host de-scatter            */
    unsigned char *placed;     /* it writes a SLICE of a shared buffer (an add, a concat) */
    unsigned char *shared;     /* it leaves a cube that several readers take              */
    unsigned char *skip;       /* some layer other than the next one reads it             */

    rocket_rk3576_cube *buf;   /* the buffer layer i names, where it owns one             */

    /* A run starting at layer i, and the layer after it. Everything between is covered by
     * that one submit. */
    rocket_conv2d_int8_chain_rk3576 **kick;
    unsigned *kick_end;

    int joins;                 /* adjacent-pair joins                                     */
    int far_links;             /* links whose reader is not the next layer                */
    /* AND WHAT EACH BUCKET IS WORTH, which the COUNT does not say. What a join removes is
     * the two transposes at it, and those are BYTES — a 56x56 pair moves 64x more than a
     * 7x7 one, so twelve refusals at the deep end and twelve at the shallow end are
     * different numbers. A count prices a submit; only the bytes price the transposes. */
    size_t join_bytes, far_bytes;
    int nojoin[ROCKET_GRAPH_NJ_N];
    size_t nojoin_bytes[ROCKET_GRAPH_NJ_N];
    int cat_adds, catn_wired, tails_given, multi_srcs;
    /* Concatenations the caller declared row-major, so the placement above was not
     * taken. Counted rather than silent: it is the difference between a graph that
     * pays a host copy and one whose shape the planner could not have wired. */
    int catn_escaped;
    int kick_runs, kick_layers;
    int kb_count[ROCKET_GRAPH_KB_N];
    unsigned char *kb_of;      /* per layer: the bucket + 1, or 0 for "in a run"          */

    struct { unsigned i, j; size_t bytes; int why; } nj_list[ROCKET_GRAPH_NJ_LIST_MAX];
    int nj_list_n;
} rocket_graph_plan;

/* Build the plan. `verbose` prints the same report the passes have always printed — the
 * join count and its denominator, the refusal buckets with their bytes, the runs and where
 * they break. Returns NULL only on allocation failure; a graph nothing can be linked in is
 * a valid plan with zero joins. */
rocket_graph_plan *rocket_graph_plan_new(int fd, const rocket_graph_layer *L, unsigned n,
                                         int verbose);
void rocket_graph_plan_free(rocket_graph_plan *p);

/* Whether the cross-layer kick is on at all (ROCKET_RK3576_NET_KICK). A caller that
 * asserts on the runs needs it: with the knob off, "no runs" is the control and not a
 * refusal to explain. */
int rocket_graph_kick_on(void);

/* Find the runs and build one chained stream each. A second step because the links are
 * what make a run eligible: a caller that runs the graph between the two calls measures
 * the joins and the kicks apart. */
void rocket_graph_plan_kicks(rocket_graph_plan *p);

/* A layer that is PLACEMENT: a concatenation whose operands are slices of one buffer. It
 * has no program and no host work, so it is neither a break in a run nor a submit. */
int rocket_graph_is_placement(const rocket_graph_plan *p, unsigned i);

/* The graph queries the passes are written over, exposed because a caller needs the same
 * answers for its own buffers: a layer read by anything other than the layer immediately
 * after it needs a buffer that outlives the ping-pong. */
unsigned rocket_graph_srcs(const rocket_graph_layer *L, unsigned *out);
int      rocket_graph_is_skip_source(const rocket_graph_layer *L, unsigned n, unsigned i);

/* The channel offset operand B of an add starts at, given operand A's channel count: a
 * slice starts every 16 channels, which is what one 16-byte atom interleaves. */
unsigned rocket_graph_add_boff(unsigned oc);

#ifdef __cplusplus
}
#endif

#endif /* ROCKET_GRAPH_RK3576_H */
