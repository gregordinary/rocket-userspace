/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 The rocket-userspace authors
 *
 * rocket_graph_rk3576.c — the RK3576 graph planner.
 *
 * The placement and linking passes, over a frontend-neutral layer description. They were
 * written and validated inside tests/rk3576_net_gate.c against five whole classifiers and
 * are moved here unchanged in behavior: every rule, every refusal bucket and every A/B
 * knob is the one those measurements were taken with.
 *
 * THE ORDER THE PASSES RUN IN IS LOAD-BEARING. Placement first (an add's buffer, then a
 * concatenation's, then a declared tail), because a placed producer's slice is cheaper
 * than a surface of its own and a pair placement wires is a join the adjacent-pair loop
 * must not undo; then the shared surfaces, which take what is left of the skip sources;
 * then the adjacent pairs; then the runs, because the links are what make a run eligible.
 *
 * WHAT IS NOT HERE. Packing the weights, the tensors, the ping-pong and the de-scatter
 * points are the caller's — see rocket_graph_rk3576.h.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_graph_rk3576.h"

/* The report is the caller's to want: a delegate running an inference does not print, and
 * a gate bisecting a graph reads nothing else. */
#define RG_LOG(g, ...) do { if ((g)->verbose) printf(__VA_ARGS__); } while (0)

/* A refusal's reason names one END of the pair, so the report prints what the two layers
 * ARE: the reasons that are the caller's lowering and the reasons that are the part's read
 * the same until you can see that. */
static const char *KIND_NAME[] = { "conv", "dwconv", "avgpool", "maxpool", "add",
                                   "concat", "host" };

/* ------------------------------------------------------------------ the A/B arms ---
 *
 * Each pass has one, default-on, and they keep the names every measurement on record is
 * quoted against. A pass turned off here is not a smaller plan: it is the CONTROL the
 * pass's own value was measured as the difference from.
 */
static int env_on(const char *name)
{
    const char *e = getenv(name);
    return !e || !*e || *e != '0';
}

static int cat_on(void)   { static int c = -1; if (c < 0) c = env_on("ROCKET_RK3576_NET_CAT");   return c; }
static int catn_on(void)  { static int c = -1; if (c < 0) c = env_on("ROCKET_RK3576_NET_CATN");  return c; }
static int tail_on(void)  { static int c = -1; if (c < 0) c = env_on("ROCKET_RK3576_NET_TAIL");  return c; }
static int multi_on(void) { static int c = -1; if (c < 0) c = env_on("ROCKET_RK3576_NET_MULTI"); return c; }

/* ROCKET_RK3576_NET_KICK=0 is the A/B for the whole cross-layer kick: the same graph one
 * submit per layer must give byte-identical results. */
int rocket_graph_kick_on(void) { static int c = -1; if (c < 0) c = env_on("ROCKET_RK3576_NET_KICK"); return c; }
#define kick_on() rocket_graph_kick_on()

/* Build ONLY the run starting at this layer, leaving every other layer one submit each.
 * The localizer: a byte comparison can only see MATERIALISED layers, so a wrong byte
 * anywhere in a chain first shows up at the next one — with a single run in play, a diff
 * there is a diff caused by THAT run. -1 (the default) builds them all.
 * ROCKET_RK3576_NET_KICK_AT=<i>. */
static int kick_at(void)
{
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("ROCKET_RK3576_NET_KICK_AT");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : -1;
    }
    return cached;
}

/* The longest run a kick may cover. Not a hardware bound — it is the instrument for
 * bisecting one: a chain that computes at 2 layers and not at 24 is a length effect, and a
 * chain that fails at every length is not. It is also what makes a kick's own bucket
 * readable, since the finite difference between m and m+1 layers is what one program costs
 * inside it. ROCKET_RK3576_NET_KICK_MAX=<n>. */
static unsigned kick_max(void)
{
    static unsigned cached = 0;
    if (!cached) {
        const char *e = getenv("ROCKET_RK3576_NET_KICK_MAX");
        long v = (e && *e) ? strtol(e, NULL, 0) : 0;
        cached = (v >= 2 && v <= 512) ? (unsigned)v : 512u;
    }
    return cached;
}

/* A CONCATENATION LAYER whose operands are placed slices of one buffer: its consumers read
 * the buffer as a cube and the layer itself is a name for an address. cube_out is the
 * marker because that is what it means everywhere else — this layer leaves a cube and
 * materialises no row-major tensor. */
static int cat_wired(rocket_graph_plan *g, unsigned i)
{
    return g->L[i].kind == ROCKET_GRAPH_CONCAT && g->cube_out[i];
}

#define add_boff(L) rocket_graph_add_boff((L)->oc)


/* A layer's operands, in order, as LAYER indices — one for a convolution or a pool, two
 * for an add, up to four for a concatenation. ROCKET_GRAPH_NO_SRC (the network input, or
 * an unused slot) is dropped, so the count is the number of LAYER operands. */
unsigned rocket_graph_srcs(const rocket_graph_layer *L, unsigned *out)
{
    unsigned k, n = 0;
    for (k = 0; k < ROCKET_GRAPH_MAX_SRC; k++)
        if (L->src[k] != ROCKET_GRAPH_NO_SRC) out[n++] = L->src[k];
    return n;
}

/* The gap between operand A's channels and the group boundary operand B starts on carries
 * zero weights, so it contributes nothing at any content. */
unsigned rocket_graph_add_boff(unsigned oc)
{
    return (oc + 15u) & ~15u;
}

/* Whether layer j can read its input from a producer's surface. Every clause here is a
 * property of the CALLER's lowering rather than of the hardware, so it is asked here and
 * the hardware's own bounds are left to the library's refusal:
 *
 *   host_input — the caller writes a prepared tensor (a materialised asymmetric border, a
 *   widened or shifted stem), so what the entry sees is not the producer's output at all;
 *   a layer with no resident handle (a host softmax takes row-major and runs nowhere near
 *   the part). */
static int cube_consumer_ok(rocket_graph_plan *g, unsigned j)
{
    const rocket_graph_layer *L = &g->L[j];
    /* The POOLING layer is a consumer too, and it is the same join: the PPU reads the
     * same NC1HWC2 cube the convolution path packs. It is not a chain member — a pool is
     * a different program in a BO of its own — so this only removes its host scatter. */
    if (ROCKET_GRAPH_IS_POOL(L->kind)) return g->L[j].pool != NULL;
    /* An ADD lowers onto a convolution over its two operands CONCATENATED along channels
     * — the DPU's elementwise stage takes one operand — so there is no single producer
     * surface for it to read. It is a cube producer like any other convolution, and it is
     * a consumer only where a placement pass gives it one buffer holding both operands. */
    if (L->kind == ROCKET_GRAPH_ADD) return 0;
    if (L->kind != ROCKET_GRAPH_CONV && L->kind != ROCKET_GRAPH_DWCONV) return 0;
    if (L->host_input) return 0;
    return g->L[j].conv != NULL;
}
/* WHY A JOIN WAS REFUSED, counted. On a feed-forward chain almost every adjacent pair
 * links and the breakdown is not interesting; on a residual one it is the whole answer to
 * how much of the graph can go out as one kick, so the reasons are separated rather than
 * left as "32 of 63". Each bucket is a DIFFERENT thing to fix, and two of them are
 * properties of this network rather than of the part. */
static const char *NOJOIN_WHY[ROCKET_GRAPH_NJ_N] = {
    "the consumer is an ADD, whose input is a host-built concatenation",
    "the producer feeds a SKIP and so must leave a row-major tensor",
    "the consumer's input channel count is not a multiple of 32",
    "the producer's output surface is not a feature cube (tiled, or a materialising stem)",
    "the consumer is not a resident convolution",
    "forced off by ROCKET_RK3576_NET_NOJOIN (the per-pair A/B)",
};
static const char *NOJOIN_TAG[ROCKET_GRAPH_NJ_N] = {
    "add-consumer", "skip-source", "ic-align", "surface", "not-resident", "forced",
};

/* The tensor that crosses one adjacent pair: the producer's whole output surface, which
 * is de-scattered at the producer and scattered again at the consumer. */
static size_t pair_bytes(rocket_graph_plan *g, unsigned i)
{
    return (size_t)g->L[i].oc * g->L[i].oh * g->L[i].ow;
}

/* THE PRODUCER SIDE, for either kind of handle. A POOL is a cube producer like any
 * convolution — the PPU writes the same 16-byte-atom surface, at round4(ow*oh) rather than
 * the plane, which is the consumer's DDR channel-group jump and a register. Keeping the
 * two behind one set of calls is what lets a pool sit on either side of a join instead of
 * only on the consumer side, and ResNet-18's max pool is the case: it feeds the next
 * convolution AND a residual add three layers on. */
static int prod_ok(rocket_graph_plan *g, unsigned i)
{
    if (ROCKET_GRAPH_IS_POOL(g->L[i].kind)) return g->L[i].pool != NULL;
    return g->L[i].conv != NULL;
}

/* WHETHER LAYER j READS LAYER i AT ALL — the denominator of every join ratio, and the
 * link pass's only dataflow guard.
 *
 * "n join(s) of m adjacent pair(s)" is a ratio only if every pair in m is a relation a
 * join could remove. On a feed-forward chain index-adjacency and dataflow are the same
 * relation, so the distinction never mattered; on a branchy graph most index-adjacent
 * pairs are not edges at all — an Inception module's four branches sit consecutively in
 * the layer table and none of them reads the one before it. Counting those understates
 * the ratio, and putting them through the refusal buckets lets a bucket name a producer
 * whose index-successor it never feeds.
 *
 * It is also the only thing here that asks about DATAFLOW. Every clause below is a
 * geometry or an ownership question, so a non-edge pair was rejected only because the
 * producer's output shape and the consumer's input shape happen to differ — which is not
 * a guarantee. A non-edge whose shapes agreed would be joined, and layer j would read a
 * tensor nothing in the graph says it reads. */
static int reads(rocket_graph_plan *g, unsigned j, unsigned i)
{
    unsigned s[ROCKET_GRAPH_MAX_SRC], n, k;

    n = rocket_graph_srcs(&g->L[j], s);
    for (k = 0; k < n; k++)
        if (s[k] == i) return 1;
    return 0;
}

static int prod_cube_of(rocket_graph_plan *g, unsigned i, rocket_rk3576_cube *c)
{
    return ROCKET_GRAPH_IS_POOL(g->L[i].kind)
             ? rocket_pool_int8_cube_of_rk3576(g->L[i].pool, c)
             : rocket_conv2d_int8_cube_of_rk3576(g->L[i].conv, c);
}

static int prod_cube_out(rocket_graph_plan *g, unsigned i, int on)
{
    return ROCKET_GRAPH_IS_POOL(g->L[i].kind)
             ? rocket_pool_int8_cube_out_rk3576(g->L[i].pool, on)
             : rocket_conv2d_int8_cube_out_rk3576(g->L[i].conv, on);
}

static int prod_cube_out_at(rocket_graph_plan *g, unsigned i,
                            const rocket_rk3576_cube *dst)
{
    return ROCKET_GRAPH_IS_POOL(g->L[i].kind)
             ? rocket_pool_int8_cube_out_at_rk3576(g->L[i].pool, dst)
             : rocket_conv2d_int8_cube_out_at_rk3576(g->L[i].conv, dst);
}

/* A pool is never a chain member — a pooling program lives in a BO of its own — so nothing
 * re-stamps its surface behind its back and it has no declaration to make. */
static void prod_cube_shared(rocket_graph_plan *g, unsigned i, int on)
{
    if (!ROCKET_GRAPH_IS_POOL(g->L[i].kind))
        rocket_conv2d_int8_cube_shared_rk3576(g->L[i].conv, on);
}

static int cons_cube_in(rocket_graph_plan *g, unsigned j, const rocket_rk3576_cube *c)
{
    return ROCKET_GRAPH_IS_POOL(g->L[j].kind)
             ? rocket_pool_int8_cube_in_rk3576(g->L[j].pool, c)
             : rocket_conv2d_int8_cube_in_rk3576(g->L[j].conv, c);
}

/* The layer that follows i in execution order — a SOFTMAX is not run as a layer here, so
 * it is stepped over. g->n when there is none. */
static unsigned next_layer(rocket_graph_plan *g, unsigned i)
{
    unsigned j = i + 1u;
    while (j < g->n && g->L[j].kind == ROCKET_GRAPH_HOST) j++;
    return j;
}

/* ONE WIRED PRODUCER -> READER RELATION, counted. The headline is "joins of ADJACENT
 * pairs", because that is the unit the refusal buckets are counted in and the unit every
 * earlier measurement is quoted in — so a relation whose reader is not the producer's
 * immediate successor is real work removed and is counted apart rather than folded in.
 * ResNet-18's downsample adds are all of that kind: their distant operand is the 3x3 leg,
 * one layer further back than the 1x1 the add sits behind. */

static void cube_joined(rocket_graph_plan *g, unsigned p, unsigned r)
{
    if (next_layer(g, p) == r) { g->joins++; g->join_bytes += pair_bytes(g, p); }
    else { g->far_links++; g->far_bytes += pair_bytes(g, p); }
}

static void cube_unjoined(rocket_graph_plan *g, unsigned p, unsigned r)
{
    if (next_layer(g, p) == r) { g->joins--; g->join_bytes -= pair_bytes(g, p); }
    else { g->far_links--; g->far_bytes -= pair_bytes(g, p); }
}

/* The refused pairs, NAMED. A bucket's count prices a submit and its bytes price the
 * transposes, but neither prices a PAIR: the two classes already closed measure 8x apart
 * per join, so the pairs a session proposes to close have to be listed one at a time with
 * the tensor each of them carries. Printed for the refusals only — a joined pair has
 * nothing left to decide, and its price is the A/B below. */

static void nojoin_note(rocket_graph_plan *g, unsigned i, unsigned j, int why)
{
    size_t bytes = pair_bytes(g, i);
    g->nojoin[why]++;
    g->nojoin_bytes[why] += bytes;
    if (g->nj_list_n < ROCKET_GRAPH_NJ_LIST_MAX) {
        g->nj_list[g->nj_list_n].i = i; g->nj_list[g->nj_list_n].j = j;
        g->nj_list[g->nj_list_n].bytes = bytes; g->nj_list[g->nj_list_n].why = why;
        g->nj_list_n++;
    }
}

/* THE PER-PAIR A/B. What a join is WORTH is measured, not derived from its bytes: the
 * measurement is the finite difference between the graph as it stands and the same graph
 * with exactly one join refused. ROCKET_RK3576_NET_NOJOIN=<i> refuses the join whose
 * PRODUCER is layer i and leaves every other join intact. A concatenation-wired add is
 * wired as a PAIR, so naming either of its producers takes both of that add's joins off —
 * which is what its price is, since neither half stands alone. */
static int nojoin_at(void)
{
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("ROCKET_RK3576_NET_NOJOIN");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : -2;
    }
    return cached;
}

static int nojoin_forced(unsigned i) { return nojoin_at() == (int)i; }

/* THE LOCALIZER for a join whose consumer's channel count is not a multiple of 32 at a
 * non-zero weight zero point. That join is sound only because a direct producer's partial
 * output group carries its output zero point, so it is the one class of join whose
 * correctness rests on a property of the PRODUCER — and a graph that disagrees needs to be
 * able to enable them one at a time. ROCKET_RK3576_NET_PADJOIN=-1 refuses them all (the
 * A/B), =<i> allows only the pair whose producer is layer i, unset allows every one. */
static int padjoin_at(void)
{
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("ROCKET_RK3576_NET_PADJOIN");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : -2;
    }
    return cached;
}

static int padjoin_ok(rocket_graph_plan *g, unsigned i, unsigned j)
{
    if (g->L[j].ic % 32u == 0 || !g->L[j].w_zp) return 1;   /* not that class */
    if (padjoin_at() == -2) return 1;
    return padjoin_at() == (int)i;
}

/* Every layer that reads layer i's output. A SOFTMAX is not run as a layer here, so its
 * read is not one. Returns the count and fills `out` up to `max`, or `max + 1` when there
 * are more than that — a caller may not silently wire a subset of a producer's readers. */
static unsigned consumers_of(rocket_graph_plan *g, unsigned i, unsigned *out,
                             unsigned max)
{
    unsigned j, n = 0;
    for (j = i + 1; j < g->n; j++) {
        unsigned s[ROCKET_GRAPH_MAX_SRC], ns, k, reads = 0;
        if (g->L[j].kind == ROCKET_GRAPH_HOST) continue;
        ns = rocket_graph_srcs(&g->L[j], s);
        for (k = 0; k < ns; k++) if (s[k] == i) reads = 1;
        if (!reads) continue;
        if (n < max) out[n] = j;
        n++;
        if (n > max) return max + 1u;
    }
    return n;
}
/*
 * THE CONCATENATION BUFFERS — what the two big refusal buckets actually wanted.
 *
 * Twenty of MobileNetV2's thirty-one refused joins are one shape of problem: an add reads
 * a CONCATENATION of two tensors from different places, and the layer that produces its
 * second operand therefore has to leave a row-major tensor for it to be copied out of,
 * three to five layers later. Neither is a property of the part. Both are the host
 * building a tensor that the hardware could have been asked to write in the first place.
 *
 * A cube's base is a plain address on both sides of a convolution, so it can be: allocate
 * ONE buffer per add, give operand A the low slice and operand B the high one, and point
 * the two producers at them with rocket_conv2d_int8_cube_out_at_rk3576(). The add then
 * reads the whole buffer as its feature cube and no concatenation is built at all.
 *
 * WHAT MAKES IT A PAIR, and why it is refused as one. Operand B's producer is also the
 * layer BEFORE the block's expand convolution, so making it write a slice means that
 * layer has to read the same slice as its cube — a producer that leaves no row-major
 * tensor for a consumer that cannot take a cube would leave the graph with nothing at
 * all. So an add is wired only when BOTH producers can be placed, and every OTHER reader
 * of either producer can read the slice back.
 *
 * NEITHER OPERAND IS "THE LAYER BEFORE", AND NEITHER IS THE PLACEMENT RULE. MobileNetV2's
 * adds take operand A from the layer before and operand B from a skip three to five layers
 * back; a ResNet block that changes width takes them the OTHER WAY ROUND, operand B from
 * the 1x1 downsample immediately before the add and operand A from the 3x3 leg past it. A
 * pass keyed on "operand A's producer is the previous layer" wires the first family and
 * refuses the second — which is what left ResNet-18's three downsample adds reading a host
 * concatenation. The buffer is a dedicated allocation per add and each slice has exactly
 * one writer, so the ORDER the two producers run in does not matter at all; what has to
 * hold is that both precede the add, which they do by construction.
 *
 * The gap between operand A's channels and the group boundary operand B starts on carries
 * zero weights (add_boff), so it contributes nothing at any content.
 */
static void cat_link(rocket_graph_plan *g)
{
    unsigned j;
    if (!cat_on()) return;
    for (j = 0; j < g->n; j++) {
        const rocket_graph_layer *L = &g->L[j];
        rocket_rk3576_cube lo, hi, all;
        rocket_rk3576_cube pc[2];
        unsigned a, b, boff, chans, p, ok = 1;
        unsigned rd[2][4], nrd[2];
        if (L->kind != ROCKET_GRAPH_ADD || L->src[1] == ROCKET_GRAPH_NO_SRC || L->src[0] == ROCKET_GRAPH_NO_SRC) continue;
        a = L->src[0]; b = L->src[1];
        /* ONE of the two producers is the layer immediately before the add — either one.
         * Without that the add has no adjacent producer at all and the layer between them
         * would be reading and writing the buffer this pass is placing. */
        if (a + 1u != j && b + 1u != j) continue;
        /* The per-pair A/B: naming either producer refuses the whole wiring, and the two
         * joins it would have made fall to the buckets below like any other refusal. */
        if (nojoin_forced(a) || nojoin_forced(b)) continue;
        if (!prod_ok(g, a) || !prod_ok(g, b) || !g->L[j].conv) continue;
        /* EVERY OTHER READER OF EITHER PRODUCER has to be able to read the slice: a placed
         * producer leaves no row-major tensor, so a reader that cannot take a cube would be
         * left with nothing. MobileNetV2's operand B has one such reader (the next block's
         * expand convolution); ResNet-18's downsample operands have none. */
        for (p = 0; p < 2 && ok; p++) {
            unsigned q, src = p ? b : a, n;
            n = consumers_of(g, src, rd[p], 4u);
            if (n > 4u) { ok = 0; break; }
            nrd[p] = n;
            for (q = 0; q < n; q++)
                if (rd[p][q] != j && !cube_consumer_ok(g, rd[p][q])) ok = 0;
        }
        if (!ok) continue;
        /* Neither producer may be a layer the CALLER prepares on the host, and the add
         * itself has to be a plain resident convolution. */
        if (g->L[a].host_input || g->L[b].host_input) continue;
        boff = add_boff(L);
        /* The buffer is as deep as the add's feature DMA WALKS, which is the register
         * count rounded to the 32-channel MAC group — not the live channels. */
        chans = (boff + L->oc + 31u) & ~31u;
        if (rocket_rk3576_cube_alloc(g->fd, chans, L->oh, L->ow, &g->buf[j]) != ROCKET_OK)
            continue;
        /* The add reads `boff + oc` channels, which is what its descriptor says and not
         * the whole allocation — the groups past it are the 32-channel rounding the
         * feature DMA walks and the cube's own size check covers. */
        if (rocket_rk3576_cube_slice(&g->buf[j], 0u, boff + L->oc, &all) != ROCKET_OK ||
            rocket_rk3576_cube_slice(&g->buf[j], 0u, L->oc, &lo) != ROCKET_OK ||
            rocket_rk3576_cube_slice(&g->buf[j], boff, L->oc, &hi) != ROCKET_OK ||
            prod_cube_out_at(g, a, &lo) != ROCKET_OK ||
            prod_cube_out_at(g, b, &hi) != ROCKET_OK ||
            rocket_conv2d_int8_cube_in_rk3576(g->L[j].conv, &all) != ROCKET_OK) {
            prod_cube_out_at(g, a, NULL);
            prod_cube_out_at(g, b, NULL);
            rocket_conv2d_int8_cube_in_rk3576(g->L[j].conv, NULL);
            rocket_rk3576_cube_free(g->fd, &g->buf[j]);
            continue;
        }
        /* THE OTHER READERS ARE ASKED FOR THE PRODUCER'S CUBE, NOT FOR THE SLICE THIS PASS
         * BUILT. The two name the same bytes, but only the producer can say what its tail
         * holds — a partial output group lands on its output zero point — and a consumer
         * whose own channel count is not a multiple of 32 needs exactly that to be stated. */
        if (prod_cube_of(g, a, &pc[0]) != ROCKET_OK ||
            prod_cube_of(g, b, &pc[1]) != ROCKET_OK) ok = 0;
        for (p = 0; p < 2 && ok; p++) {
            unsigned q;
            for (q = 0; q < nrd[p]; q++) {
                unsigned r = rd[p][q];
                if (r == j) continue;
                if (cons_cube_in(g, r, &pc[p]) != ROCKET_OK) { ok = 0; break; }
                g->cube_in[r] = 1;
                cube_joined(g, p ? b : a, r);
            }
        }
        if (!ok) {
            for (p = 0; p < 2; p++) {
                unsigned q;
                for (q = 0; q < nrd[p]; q++) {
                    unsigned r = rd[p][q];
                    if (r == j || !g->cube_in[r]) continue;
                    cons_cube_in(g, r, NULL);
                    g->cube_in[r] = 0;
                    cube_unjoined(g, p ? b : a, r);
                }
            }
            prod_cube_out_at(g, a, NULL);
            prod_cube_out_at(g, b, NULL);
            rocket_conv2d_int8_cube_in_rk3576(g->L[j].conv, NULL);
            rocket_rk3576_cube_free(g->fd, &g->buf[j]);
            continue;
        }
        g->cube_out[a] = 1; g->cube_out[b] = 1;
        g->cube_in[j] = 1;
        g->placed[a] = 1; g->placed[b] = 1;
        /* Counted HERE or not at all: an adjacent pair the loop below then skips would
         * otherwise leave the joined-bytes total understating the denominator every
         * refusal ratio is quoted against. */
        cube_joined(g, a, j); cube_joined(g, b, j);
        g->cat_adds++;
    }
    if (g->cat_adds)
        RG_LOG(g, "   concatenation buffers: %d of the adds read a buffer their two producers "
               "wrote directly — no host concatenation, and the skip source no longer has "
               "to materialise\n", g->cat_adds);
}
/*
 * THE CONCATENATION LAYERS, wired the way the adds are — and this is what a graph built
 * out of concatenated branches loses almost everything to.
 *
 * An Inception module is four branches whose outputs are joined along the channel axis,
 * and every one of Inception V1's 46 refused joins is that topology: 27 branch outputs
 * that must materialise because a concatenation several layers later copies them, and 18
 * pairs whose producer or consumer is the concatenation itself, which is not a resident
 * convolution and so can be neither side of a join. None of it is a property of the part.
 *
 * A CONCATENATION IS PLACEMENT when every operand already carries the output's
 * quantization, which the blob builder asserts. So allocate ONE buffer per concatenation,
 * give operand k the slice at its own channel offset, point the four producers at those
 * slices with rocket_conv2d_int8_cube_out_at_rk3576(), and let every consumer read the
 * buffer as its feature cube. The layer then costs nothing at all: no copy, no transpose,
 * no program.
 *
 * FOUR THINGS IT HAS TO GET RIGHT, and three of them are not in the add's version.
 *
 * (1) THE PRODUCERS OVERLAP, AND THE ORDER IS WHAT MAKES THAT SAFE. A handle writes its
 * REGISTER channel count — round-32 — so operand k's program writes past its own live
 * channels into the head of operand k+1's slice. That is harmless exactly when the later
 * operand's program runs after it and overwrites those bytes, so the pass asserts that
 * the producers are in increasing LAYER order and that their write ends do not decrease.
 * A chained stream emits its programs in array order, so "later layer" is "later program".
 *
 * (1a) SO THIS GRAPH ASKS WHETHER A CHAINED STREAM HONOURS WRITE-AFTER-WRITE, which
 * nothing here has needed before. Read-after-write between two programs is measured
 * (40/40, tests/rk3576_chain_raw.c); two programs writing the same bytes in one stream is
 * a different guarantee, and it is reached only by an operand whose channel count is not a
 * multiple of 32 — two of Inception V1's nine concatenations, at 16 channels each. If it
 * did NOT hold, those 16 channels would carry the earlier producer's padding instead of
 * the later producer's output, and every consumer of that concatenation would be wrong;
 * the byte comparison and the logits are what would say so.
 * ROCKET_RK3576_NET_CATN_OVERLAP=0 refuses exactly that case, which is the A/B that
 * would tell a disagreement here from any other.
 *
 * (1b) AND THE OVERLAP WEAKENS THE WRITE GUARD FOR THOSE PRODUCERS. "Did this task write"
 * asks whether any byte of its extent differs from the sentinel, and its extent covers the
 * padding groups the NEXT producer overwrites — so a program that died could read as one
 * that wrote. It is the mid-stream death the guard's per-kick form already trades away and
 * that nothing on this part has been observed to do; the whole-submit poisoning is
 * unaffected, since it leaves every other layer of the kick unwritten too.
 *
 * (2) THE LAST PRODUCER OVERHANGS INTO THE TAIL, and that is what makes the tail sound.
 * Its padding channels land on its own output zero point, which is the concatenation's,
 * which is every consumer's input zero point — so a consumer whose channel count is not a
 * multiple of 32 can be told what its tail holds instead of having the join refused. The
 * buffer is filled with that constant once at link time for the groups no producer
 * reaches.
 *
 * (3) A SLICE IS SIZED FOR A WRITER AND A VIEW FOR A READER. The producers take slices of
 * their own live channels; the consumers take the whole buffer narrowed to the
 * concatenation's channel count, because a feature DMA walks round32(ic) and the buffer
 * carries those groups.
 *
 * (4) IT IS ALL ONE DECISION. A producer that placed its output for a consumer that then
 * refused the buffer would have left no row-major tensor at all, so the whole
 * concatenation is wired or none of it is — the same rule the add's pair has, over up to
 * eight relations instead of three.
 *
 * ROCKET_RK3576_NET_CATN=0 refuses them all, which is the A/B.
 */
/* Whether an operand whose round-32 padding runs into the NEXT operand's slice may be
 * wired. See (1a): it is the only case that asks the part for write-after-write between
 * two programs of one stream, and this is the arm that isolates it. */
static int catn_overlap_ok(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_NET_CATN_OVERLAP");
        cached = !e || !*e || *e != '0';
    }
    return cached;
}

static void concat_link(rocket_graph_plan *g)
{
    unsigned j;
    if (!catn_on()) return;
    for (j = 0; j < g->n; j++) {
        const rocket_graph_layer *L = &g->L[j];
        unsigned s[ROCKET_GRAPH_MAX_SRC], off[ROCKET_GRAPH_MAX_SRC], ns, k, p, q;
        unsigned rd[ROCKET_GRAPH_MAX_SRC][4], nrd[ROCKET_GRAPH_MAX_SRC];
        unsigned cons[4], ncons;
        rocket_rk3576_cube view, sl[ROCKET_GRAPH_MAX_SRC], pc[ROCKET_GRAPH_MAX_SRC];
        unsigned chans, o = 0, end = 0;
        int ok = 1;

        if (L->kind != ROCKET_GRAPH_CONCAT || nojoin_forced(j)) continue;
        ns = rocket_graph_srcs(L, s);
        if (!ns) continue;
        /* THE OFFSETS ARE THE PRODUCERS' CHANNEL COUNTS IN OPERAND ORDER — the same
         * arithmetic concat_run() does on the host, which is what makes the two lowerings
         * the same tensor. */
        for (k = 0; k < ns && ok; k++) {
            unsigned pc_end;
            p = s[k];
            off[k] = o;
            o += g->L[p].oc;
            pc_end = off[k] + ((g->L[p].oc + 31u) & ~31u);
            if (off[k] % 16u) ok = 0;                       /* a slice starts on an atom */
            if (k && s[k] <= s[k - 1]) ok = 0;              /* (1): program order */
            if (pc_end < end) ok = 0;                       /* (1): write ends rise */
            if (k + 1u < ns && g->L[p].oc % 32u && !catn_overlap_ok()) ok = 0;  /* (1a) */
            end = pc_end;
            if (g->L[p].out_zp != L->out_zp) ok = 0;      /* (2): one constant */
            if (!prod_ok(g, p) || nojoin_forced(p)) ok = 0;
            if (g->cube_out[p] || g->placed[p] || g->shared[p]) ok = 0;
            if (g->L[p].host_input) ok = 0;
        }
        if (!ok || o != L->oc) continue;
        /* Every OTHER reader of a producer has to be able to read that producer's own
         * slice back, for the reason cat_link() gives: a placed producer leaves no
         * row-major tensor. */
        for (k = 0; k < ns && ok; k++) {
            unsigned n = consumers_of(g, s[k], rd[k], 4u);
            if (n > 4u) { ok = 0; break; }
            nrd[k] = n;
            /* A reader another pass has already linked is not this one's to clear on the
             * unwind path, so it refuses the whole concatenation instead. */
            for (q = 0; q < n; q++)
                if (rd[k][q] != j &&
                    (!cube_consumer_ok(g, rd[k][q]) || g->cube_in[rd[k][q]])) ok = 0;
        }
        if (!ok) continue;
        /* And every reader of the CONCATENATION has to take the buffer. */
        ncons = consumers_of(g, j, cons, 4u);
        if (!ncons || ncons > 4u) continue;
        for (q = 0; q < ncons; q++)
            if (!cube_consumer_ok(g, cons[q]) || g->cube_in[cons[q]] ||
                !padjoin_ok(g, j, cons[q])) ok = 0;
        if (!ok) continue;

        /* The buffer is as deep as the widest thing that walks it: a consumer's feature
         * DMA reads round32 of the concatenation's channels, and the last producer's
         * program writes round32 of its own. */
        chans = (L->oc + 31u) & ~31u;
        if (end > chans) chans = end;
        if (rocket_rk3576_cube_alloc(g->fd, chans, L->oh, L->ow, &g->buf[j]) != ROCKET_OK)
            continue;
        /* (2) THE TAIL, ONCE. cube_alloc leaves zeros and the consumers' border constant
         * is the concatenation's zero point; the last producer rewrites the part of it
         * that its own program reaches, with the same value. Bracketed, never bare — a
         * dirty line races the DPU's DMA. */
        if (chans > L->oc) {
            size_t g0 = (size_t)(L->oc / 16u);
            size_t bytes = ((size_t)chans / 16u - g0) * (size_t)L->oh * L->ow * 16u;
            rocket_bo_prep(g->fd, &g->buf[j].bo, 1, 0);
            memset((char *)g->buf[j].bo.ptr + g0 * (size_t)L->oh * L->ow * 16u,
                   (unsigned char)(int8_t)L->out_zp, bytes);
            rocket_bo_fini(g->fd, &g->buf[j].bo);
        }
        /* (3) the producers' slices and the consumers' view. */
        if (rocket_rk3576_cube_slice(&g->buf[j], 0u, chans, &view) != ROCKET_OK ||
            rocket_rk3576_cube_declare_tail(&view, L->oc, L->out_zp) != ROCKET_OK)
            ok = 0;
        /* A PRODUCER'S SLICE IS SIZED FOR WHAT IT WRITES, not for what it lives: the DPU
         * writes every group it is programmed with, which is the round-32 register count,
         * so a slice of the live channels alone is one group short and the placement is
         * refused. That is the same overlap (1a) names — the slice really does reach into
         * the next operand's, and the next operand's program is what puts it right. */
        for (k = 0; k < ns && ok; k++)
            if (rocket_rk3576_cube_slice(&g->buf[j], off[k],
                                         (g->L[s[k]].oc + 31u) & ~31u, &sl[k])
                    != ROCKET_OK ||
                prod_cube_out_at(g, s[k], &sl[k]) != ROCKET_OK ||
                prod_cube_of(g, s[k], &pc[k]) != ROCKET_OK) ok = 0;
        for (q = 0; q < ncons && ok; q++)
            if (cons_cube_in(g, cons[q], &view) != ROCKET_OK) ok = 0;
        /* A producer's OTHER readers are asked for the PRODUCER's cube, not for the slice
         * this pass built: the two name the same bytes and only the producer can say what
         * its own tail holds. */
        for (k = 0; k < ns && ok; k++)
            for (q = 0; q < nrd[k]; q++)
                if (rd[k][q] != j && cons_cube_in(g, rd[k][q], &pc[k]) != ROCKET_OK) ok = 0;
        if (!ok) {
            /* (4) ALL OF IT OR NONE. Unwound in the order it was built, so a producer that
             * placed its output owns a surface again and every reader is back on a
             * row-major tensor. */
            for (q = 0; q < ncons; q++) cons_cube_in(g, cons[q], NULL);
            for (k = 0; k < ns; k++) {
                for (q = 0; q < nrd[k]; q++)
                    if (rd[k][q] != j) cons_cube_in(g, rd[k][q], NULL);
                prod_cube_out_at(g, s[k], NULL);
            }
            rocket_rk3576_cube_free(g->fd, &g->buf[j]);
            continue;
        }
        for (k = 0; k < ns; k++) {
            g->cube_out[s[k]] = 1;
            g->placed[s[k]] = 1;
            cube_joined(g, s[k], j);
            for (q = 0; q < nrd[k]; q++)
                if (rd[k][q] != j) { g->cube_in[rd[k][q]] = 1; cube_joined(g, s[k], rd[k][q]); }
        }
        for (q = 0; q < ncons; q++) { g->cube_in[cons[q]] = 1; cube_joined(g, j, cons[q]); }
        g->cube_out[j] = 1;                                    /* the wired marker */
        g->catn_wired++;
    }
    if (g->catn_wired)
        RG_LOG(g, "   concatenation layers: %d of them are placement — their operands are "
               "slices of one buffer their consumers read as a cube, so no copy, no "
               "transpose and no program\n", g->catn_wired);
}
/*
 * A PRODUCER THAT CANNOT DECLARE ITS OWN TAIL CAN BE GIVEN ONE.
 *
 * The last refused join of this graph is a 528-channel pool feeding a convolution whose
 * weight zero point is not zero: the coefficient group's B term sums the channels past
 * 528 that the consumer's feature DMA walks, and a POOL cannot say what they hold — it
 * reduces WITHIN a channel and writes only the channels it has, so its surface is a group
 * short and its content past them is whatever the input cube's padding was. A DIRECT
 * convolution has no such problem: it is programmed with the round-32 count, writes every
 * group it is told, and its padding lands on its own output zero point.
 *
 * That is a statement about a BUFFER, not about the part, and a buffer is something a
 * caller owns. Give the producer a destination that is round-32 deep, fill the groups past
 * its channels with its output zero point ONCE — nothing writes there afterwards, the
 * program does not address them and the sentinel covers only the groups the producer owns
 * — and hand the consumer a view that says so. It is the concatenation buffer with one
 * operand.
 *
 * IT IS WORTH FAR MORE THAN ONE JOIN, and that is not obvious from the refusal. A run
 * cannot span a layer that leaves a row-major tensor, so this single pair split the graph
 * at layer 59 — and every later module's concatenation takes its first operand from a
 * layer before the split, which `r76_run_fed` needs inside the same run. One refusal at
 * the 528-channel module therefore cost every kick after it.
 *
 * ROCKET_RK3576_NET_TAIL=0 refuses them all, which is the A/B.
 */
static void tail_link(rocket_graph_plan *g)
{
    unsigned i;
    if (!tail_on()) return;
    for (i = 0; i < g->n; i++) {
        const rocket_graph_layer *L = &g->L[i];
        rocket_rk3576_cube pc, dst, view;
        unsigned rd[4], n, q, chans, ok = 1;
        size_t g0, bytes;

        if (g->cube_out[i] || g->placed[i] || nojoin_forced(i) || !prod_ok(g, i)) continue;
        /* The live channels have to end on an atom, or the LAST one is half this
         * producer's and half a tail nobody can name. */
        if (L->oc % 16u || L->oc % 32u == 0) continue;
        n = consumers_of(g, i, rd, 4u);
        if (!n || n > 4u) continue;
        /* Only the class this closes: a reader that walks past the producer's channels and
         * has a B term to pay for them. Anything else already joins or is refused for a
         * different reason, and giving it a buffer would be a second way to do the same
         * thing. */
        for (q = 0; q < n; q++) {
            const rocket_graph_layer *C = &g->L[rd[q]];
            if (!cube_consumer_ok(g, rd[q]) || g->cube_in[rd[q]] || !padjoin_ok(g, i, rd[q])) ok = 0;
            else if (ROCKET_GRAPH_IS_POOL(C->kind) || C->ic % 32u == 0 || !C->w_zp) ok = 0;
        }
        if (!ok) continue;
        if (prod_cube_of(g, i, &pc) != ROCKET_OK) continue;
        if (pc.pad_from) continue;                       /* it can already say */
        /* cube_alloc lays a buffer out at the PLANE, so a producer whose channel-group
         * stride is padded (a depthwise convolution's round4, a pool's) needs a buffer this
         * pass cannot build. Refused rather than mis-strided. */
        if (pc.surf_elems != (size_t)L->oh * L->ow) continue;

        chans = (L->oc + 31u) & ~31u;
        if (rocket_rk3576_cube_alloc(g->fd, chans, L->oh, L->ow, &g->buf[i]) != ROCKET_OK)
            continue;
        g0 = (size_t)(L->oc / 16u);
        bytes = ((size_t)chans / 16u - g0) * (size_t)L->oh * L->ow * 16u;
        rocket_bo_prep(g->fd, &g->buf[i].bo, 1, 0);
        memset((char *)g->buf[i].bo.ptr + g0 * (size_t)L->oh * L->ow * 16u,
               (unsigned char)(int8_t)L->out_zp, bytes);
        rocket_bo_fini(g->fd, &g->buf[i].bo);

        if (rocket_rk3576_cube_slice(&g->buf[i], 0u, chans, &dst) != ROCKET_OK) ok = 0;
        view = dst;
        if (ok && rocket_rk3576_cube_declare_tail(&view, L->oc, L->out_zp) != ROCKET_OK)
            ok = 0;
        if (ok && prod_cube_out_at(g, i, &dst) != ROCKET_OK) ok = 0;
        for (q = 0; q < n && ok; q++)
            if (cons_cube_in(g, rd[q], &view) != ROCKET_OK) ok = 0;
        if (!ok) {
            for (q = 0; q < n; q++) cons_cube_in(g, rd[q], NULL);
            prod_cube_out_at(g, i, NULL);
            rocket_rk3576_cube_free(g->fd, &g->buf[i]);
            continue;
        }
        g->cube_out[i] = 1; g->placed[i] = 1;
        for (q = 0; q < n; q++) { g->cube_in[rd[q]] = 1; cube_joined(g, i, rd[q]); }
        g->tails_given++;
    }
    if (g->tails_given)
        RG_LOG(g, "   declared tails: %d producer(s) write into a round-32 buffer whose tail "
               "holds their output zero point, which is what their readers' B term "
               "assumes\n", g->tails_given);
}
/*
 * A SKIP SOURCE MAY WRITE A CUBE AFTER ALL — when every layer that reads it can read one.
 *
 * "A producer some later layer reads must materialise" is a statement about the HOST
 * buffers, not about the part: it holds because the reader three layers on wants a
 * row-major tensor. A resident handle owns its output surface and nothing else writes
 * there until that handle runs again, so the surface outlives the whole inference — which
 * is exactly the lifetime a distant reader needs. Two readers of one surface is not a
 * second copy of anything; it is the same cube described twice.
 *
 * That closes the class this graph loses the most to. ResNet-18's identity blocks end in
 * an add whose output is read TWICE — by the next block's 3x3 convolution and, three
 * layers later, by that block's 1x1 downsample — and both are ordinary convolutions.
 *
 * WHAT IT COSTS THE CHAIN, and why the library has to be told. A cross-layer kick
 * re-stamps every interior layer's surface in its verify bracket, sound only because the
 * next kick rewrites it before anything reads it. A shared surface is read first, so the
 * stamp would replace the layer's output with 0xA5 and the outside reader would compute a
 * full and plausible surface from it — invisible until the next materialised layer.
 * rocket_conv2d_int8_cube_shared_rk3576() moves that stamp to the start of the producer's
 * next call, one PREP/FINI pair and nothing else.
 *
 * ROCKET_RK3576_NET_MULTI=0 refuses them all, which is the A/B.
 */
static void multi_link(rocket_graph_plan *g)
{
    unsigned i;
    if (!multi_on()) return;
    for (i = 0; i < g->n; i++) {
        rocket_rk3576_cube c;
        unsigned rd[4], n, q, ok = 1;
        if (!g->skip[i] || g->placed[i] || g->cube_out[i]) continue;
        if (!prod_ok(g, i) || nojoin_forced(i)) continue;
        n = consumers_of(g, i, rd, 4u);
        if (!n || n > 4u) continue;
        for (q = 0; q < n; q++)
            if (!cube_consumer_ok(g, rd[q]) || g->cube_in[rd[q]] || !padjoin_ok(g, i, rd[q])) ok = 0;
        if (!ok) continue;
        if (prod_cube_of(g, i, &c) != ROCKET_OK) continue;
        for (q = 0; q < n && ok; q++)
            if (cons_cube_in(g, rd[q], &c) != ROCKET_OK) ok = 0;
        if (ok && prod_cube_out(g, i, 1) != ROCKET_OK) ok = 0;
        if (!ok) {
            for (q = 0; q < n; q++) cons_cube_in(g, rd[q], NULL);
            continue;
        }
        prod_cube_shared(g, i, 1);
        g->cube_out[i] = 1; g->shared[i] = 1;
        for (q = 0; q < n; q++) { g->cube_in[rd[q]] = 1; cube_joined(g, i, rd[q]); }
        g->multi_srcs++;
    }
    if (g->multi_srcs)
        RG_LOG(g, "   shared surfaces: %d skip source(s) leave a cube that every one of their "
               "readers takes, so they no longer materialise\n", g->multi_srcs);
}
/* Link every adjacent pair the library accepts. Called once, after a warm-up inference has
 * packed the handles. */
static void cube_link(rocket_graph_plan *g)
{
    unsigned i, pairs = 0, nonedge = 0;
    /* The concatenation buffers first: they decide two of the refusal buckets below, and
     * a pair they wire is a join this pass must not undo. The shared surfaces then take
     * what is left of the skip sources — placement first, because a placed producer's
     * slice is cheaper than its own surface (one buffer for the add instead of two). */
    cat_link(g);
    concat_link(g);
    tail_link(g);
    multi_link(g);
    for (i = 0; i + 1 < g->n; i++) {
        rocket_rk3576_cube c;
        unsigned j = i + 1;
        while (j < g->n && g->L[j].kind == ROCKET_GRAPH_HOST) j++;
        if (j >= g->n) break;
        /* NOT AN EDGE: layer j does not read layer i, so there is no tensor crossing this
         * pair, nothing for a join to remove, and nothing to refuse. Out of the
         * denominator and out of the refusal buckets both. */
        if (!reads(g, j, i)) { nonedge++; continue; }
        pairs++;
        /* Already wired as a concatenation: layer i writes a slice and layer j reads it.
         * Counted there, and nothing left to do here. */
        if (g->cube_out[i] && g->cube_in[j]) continue;
        /* A WIRED CONCATENATION LAYER is on one side of this pair, and both of its
         * relations — its producers' slices and its consumers' view — were counted in
         * concat_link(). There is no pair left to decide and none to refuse. */
        if (cat_wired(g, i) || cat_wired(g, j)) continue;
        if (nojoin_forced(i)) { nojoin_note(g, i, j, ROCKET_GRAPH_NJ_FORCED); continue; }
        if (!prod_ok(g, i)) { nojoin_note(g, i, j, ROCKET_GRAPH_NJ_OTHER); continue; }
        if (g->L[j].kind == ROCKET_GRAPH_ADD) { nojoin_note(g, i, j, ROCKET_GRAPH_NJ_ADD_CONSUMER); continue; }
        if (!cube_consumer_ok(g, j)) { nojoin_note(g, i, j, ROCKET_GRAPH_NJ_OTHER); continue; }
        /* A SKIP SOURCE MUST MATERIALISE unless it was PLACED. A cube-out layer writes no
         * row-major tensor, and the add that names it as an operand runs three to five
         * layers later with host work in between — so unless that add reads the slice this
         * layer wrote, there would be nothing for it to read. */
        if (g->skip[i] && !g->placed[i] && !g->shared[i]) {
            nojoin_note(g, i, j, ROCKET_GRAPH_NJ_SKIP_SOURCE); continue; }
        if (g->cube_out[i] || g->cube_in[j]) { nojoin_note(g, i, j, ROCKET_GRAPH_NJ_OTHER); continue; }
        if (!padjoin_ok(g, i, j)) { nojoin_note(g, i, j, ROCKET_GRAPH_NJ_IC_ALIGN); continue; }
        if (prod_cube_of(g, i, &c) != ROCKET_OK) {
            nojoin_note(g, i, j, ROCKET_GRAPH_NJ_SURFACE); continue;
        }
        if (cons_cube_in(g, j, &c) != ROCKET_OK) {
            /* The library refuses a cube whose channel count is not a multiple of 32 —
             * the consumer's own cube relies on the zero it memset into the padding
             * channels, which a producer does not control. Every other reason it can
             * refuse (plane, stride, g->fd) is impossible for an adjacent pair here. */
            nojoin_note(g, i, j, g->L[j].ic % 32u ? ROCKET_GRAPH_NJ_IC_ALIGN : ROCKET_GRAPH_NJ_OTHER);
            continue;
        }
        if (prod_cube_out(g, i, 1) != ROCKET_OK) {
            /* The consumer took the cube and the producer will not leave one, so the pair
             * has to come apart again — the consumer's row-major input is the only thing
             * that still exists. */
            cons_cube_in(g, j, NULL);
            nojoin_note(g, i, j, ROCKET_GRAPH_NJ_SURFACE);
            continue;
        }
        g->cube_out[i] = 1; g->cube_in[j] = 1;
        cube_joined(g, i, j);
    }
    RG_LOG(g, "   cube chain: %d join(s) of %u adjacent pair(s) — layer n's output surface is "
           "layer n+1's feature cube, so neither transpose runs at those joins\n",
           g->joins, pairs);
    /* The denominator is DATAFLOW-adjacent pairs. On a branchy graph the index-adjacent
     * count is much larger and is not a denominator: those pairs carry no tensor. */
    if (nonedge)
        RG_LOG(g, "      (%u further index-adjacent pair(s) are not edges at all — layer n+1 "
               "does not read layer n — so they are neither joins nor refusals)\n",
               nonedge);
    if (g->far_links)
        RG_LOG(g, "      and %d NON-adjacent link(s), %.0f KiB: a reader further on than the "
               "next layer takes the same cube, which is real work removed but not an "
               "adjacent pair\n", g->far_links, g->far_bytes / 1024.0);
    {
        size_t refused = 0;
        int k;
        for (k = 0; k < ROCKET_GRAPH_NJ_N; k++) refused += g->nojoin_bytes[k];
        for (k = 0; k < ROCKET_GRAPH_NJ_N; k++)
            if (g->nojoin[k])
                RG_LOG(g, "      %2d refused, %5.0f KiB: %s\n", g->nojoin[k],
                       g->nojoin_bytes[k] / 1024.0, NOJOIN_WHY[k]);
        /* THE REFUSED PAIRS, ONE LINE EACH. The bytes ratio above is an ordering, NOT a
         * cap: scaling the existing joins' measured value by it puts more milliseconds on
         * the refusals than the whole wall holds. What prices a pair is the finite
         * difference with that one join forced off — ROCKET_RK3576_NET_NOJOIN=<producer>,
         * over the joins that EXIST — read against these tensor sizes. */
        if (refused) {
            RG_LOG(g, "      the joins that exist carry %.0f KiB and the refused ones %.0f "
                   "KiB (%.2fx) — an ORDERING, not a cap: price a pair with "
                   "ROCKET_RK3576_NET_NOJOIN\n",
                   g->join_bytes / 1024.0, refused / 1024.0,
                   g->join_bytes ? (double)refused / (double)g->join_bytes : 0.0);
            for (k = 0; k < g->nj_list_n; k++)
                /* WITH BOTH KINDS. A refusal's reason names one END of the pair, so the
                 * bucket alone does not say which layer to go and look at — and the
                 * reasons that are the GATE's lowering and the reasons that are the
                 * part's read the same until you can see what the two layers are. */
                RG_LOG(g, "      pair %2u %-8s -> %2u %-8s  %6.1f KiB  %s\n",
                       g->nj_list[k].i, KIND_NAME[g->L[g->nj_list[k].i].kind],
                       g->nj_list[k].j, KIND_NAME[g->L[g->nj_list[k].j].kind],
                       g->nj_list[k].bytes / 1024.0,
                       NOJOIN_TAG[g->nj_list[k].why]);
        }
        /* The producers the A/B has to sweep, so the list does not have to be reconstructed
         * from the graph by hand. */
        if (g->joins) {
            unsigned m;
            RG_LOG(g, "      joined at producer(s):");
            for (m = 0; m < g->n; m++) if (g->cube_out[m]) RG_LOG(g, " %u", m);
            RG_LOG(g, "\n");
        }
    }
}
/* Build one chain per run of two or more cube-linked layers.
 *
 * FINDING the runs is the LIBRARY's: it needs no hardware knowledge beyond the program
 * bound, so rocket_chain_plan_rk3576() does it and this is what gates it — including its
 * program-count split, which is why a build-and-shorten loop is not here.
 *
 * What stays here is what the CALLER's lowering decides rather than the handles: a NULL
 * entry is how the finder is told a layer cannot be in a run, and the array below carries
 * one wherever the caller prepares that layer's input on the host. A chain scatters the
 * tensor it is handed and has no way to know a widened or bordered copy was meant. */
/* WHY A LAYER IS NOT IN A RUN — the term a submit count does not carry.
 *
 * A join is a property of a PAIR and a run is broken by ONE layer, so the join table above
 * cannot say why a graph takes n submits rather than one: Inception V3 joins 82 of its 129
 * adjacent pairs and still fragments into sixteen runs. Every bucket here is a different
 * thing to fix, and which of them are the PART's is the whole question — on this corpus so
 * far none of them has been.
 *
 * Counted over layers rather than pairs, and only for layers that would otherwise carry a
 * program (a host concatenation and the softmax cost no submit and are not fragmentation). */
static const char *KB_WHY[ROCKET_GRAPH_KB_N] = {
    "its input is prepared on the host (a materialised border or a widened stem)",
    "it is an ADD whose input is a host-built concatenation",
    "it feeds a SKIP that reads a row-major tensor, so it must materialise one",
    "it is a pooling layer with no resident handle (a column split has one surface per slice)",
    "it is a padded average pool whose divisor can lag, so its redo is its own submit "
    "rather than a re-run of every program of a kick",
    "it is not a resident convolution",
    "it leaves no cube, so the layer after it would need a host de-scatter",
    "it reads no cube any earlier member of the run wrote",
    "it is linked at both ends but is the only program between two breaks",
};

/* ROCKET_RK3576_NET_POOL_OUT IS GONE, and it was removed rather than left inert. The
 * library refuses a pool whose divisor can lag as a run node, so the knob reached no
 * placement ROCKET_RK3576_CHAIN_POOL_LAG does not — and with the check off it did not even
 * reach that, because its condition went through _lag_observable(). A knob that silently
 * does nothing reads as a MEASURED ZERO, and this one had 9.5-25.5 ms on record against it,
 * so the next A/B of it would have recorded the opposite of the measurement. */

static void kb_mark(rocket_graph_plan *g, unsigned i, int why)
{
    if (g->kb_of && !g->kb_of[i]) g->kb_of[i] = (unsigned char)(why + 1);
}

static void kick_build(rocket_graph_plan *g)
{
    rocket_chain_node_rk3576 *cand;
    rocket_conv2d_int8_run_rk3576 runs[32];
    unsigned i, nruns, r;

    if (!kick_on()) {
        RG_LOG(g, "   cross-layer kick: OFF (ROCKET_RK3576_NET_KICK=0) — one submit per "
               "layer, which is the A/B control\n");
        return;
    }
    cand = calloc(g->n, sizeof *cand);
    g->kb_of = calloc(g->n, 1);
    if (!cand) { free(g->kb_of); g->kb_of = NULL; return; }
    for (i = 0; i < g->n; i++) {
        if (g->L[i].host_input) { kb_mark(g, i, ROCKET_GRAPH_KB_HOST_PREP); continue; }
        /* A WIRED CONCATENATION LAYER HAS NO PROGRAM AT ALL and no host work either, so
         * it must not break the run it sits in — which on this graph is every run there
         * is. It is declared as a PLACEMENT node and the finder spans it. Asked before the
         * skip-source test below, which it would otherwise fail: a concatenation feeds
         * four readers and none of them is the layer after it. */
        if (cat_wired(g, i)) { cand[i].placement = 1; continue; }
        /* An ADD reading a host-built concatenation, and a SKIP SOURCE whose output is
         * read on the host later, are both host work between two programs — so neither
         * can sit inside a stream, and a NULL here is how the finder is told so. A WIRED
         * one is neither: its operands and its consumer are slices of one buffer, and
         * there is nothing left between the two programs at all. */
        if (g->L[i].kind == ROCKET_GRAPH_ADD && !g->cube_in[i]) {
            kb_mark(g, i, ROCKET_GRAPH_KB_ADD_HOST_CAT); continue;
        }
        if (g->skip[i] && !g->placed[i] && !g->shared[i]) {
            kb_mark(g, i, ROCKET_GRAPH_KB_SKIP_MATERIALISE); continue;
        }
        /* THE LOCALIZER. The finder returns MAXIMAL runs, so naming a layer that sits
         * inside a longer one would name no run at all — cut the array before it instead,
         * and the run the finder then finds is the one that starts there. */
        if (kick_at() >= 0 && i < (unsigned)kick_at()) continue;
        /* A POOLING layer is a node of the run rather than a break in it: a pool program
         * runs inside a convolution stream on this part [HW sweep, rk3576_chain_pool].
         * The finder places it — it may only be interior — so this only has to offer it. */
        if (ROCKET_GRAPH_IS_POOL(g->L[i].kind)) {
            cand[i].pool = g->L[i].pool;
            /* A PADDED AVERAGE POOL INSIDE A KICK MAKES THE KICK'S REDO ITS REDO: a kick is
             * one submit, so a redo re-runs every program of the run and every observable
             * pool in it has to come out clean on the same attempt. The LIBRARY refuses one
             * as a run node for that reason, so the candidate is offered unchanged and there
             * is one rule rather than two — but the bucket has to be marked here or the
             * table reports the layers stranded AROUND the pool and never the pool itself,
             * which reads as fragmentation the topology caused. */
            if (cand[i].pool && rocket_pool_int8_rk3576_lag_can_show(cand[i].pool))
                kb_mark(g, i, ROCKET_GRAPH_KB_POOL_LAGGY);
            if (!cand[i].pool) kb_mark(g, i, ROCKET_GRAPH_KB_POOL_UNRESIDENT);
            continue;
        }
        cand[i].conv = g->L[i].conv;
        if (!cand[i].conv) kb_mark(g, i, ROCKET_GRAPH_KB_NOT_RESIDENT);
    }
    nruns = rocket_chain_plan_rk3576(cand, g->n, runs,
                                     (unsigned)(sizeof runs / sizeof *runs));
    if (nruns > sizeof runs / sizeof *runs) nruns = sizeof runs / sizeof *runs;
    for (r = 0; r < nruns; r++) {
        unsigned first = runs[r].first, n = runs[r].count;
        if (kick_at() >= 0 && (unsigned)kick_at() != first) continue;
        if (n > kick_max()) n = kick_max();          /* the bisection instrument */
        /* A CUT RUN MUST STILL END ON A LAYER THAT MAY END ONE. The finder trims a
         * trailing pool and a trailing placement layer; shortening its answer here can put
         * one back, and the constructor would refuse the whole run rather than the layer. */
        while (n >= 2u && (ROCKET_GRAPH_IS_POOL(g->L[first + n - 1u].kind) ||
                           cat_wired(g, first + n - 1u))) n--;
        if (n < 2u) continue;
        g->kick[first] = rocket_chain_new_rk3576(g->fd, cand + first, n);
        if (!g->kick[first]) {
            RG_LOG(g, "   cross-layer kick: the run at layer %u (%u layers, %u programs) was "
                   "refused; those layers keep one submit each\n",
                   first, n, runs[r].programs);
            continue;
        }
        g->kick_end[first] = first + n;
        g->kick_runs++;
        g->kick_layers += (int)n;
    }
    /* EVERY LAYER A RUN DID NOT REACH, CLASSIFIED. The buckets above are the ones this
     * gate's own lowering set; what is left is a layer that WAS offered and still fell
     * outside, which the finder's two rules separate: it leaves no cube (nothing after it
     * can be in its run) or it reads none an earlier member wrote (nothing before it can).
     * A layer failing neither is simply alone between two breaks — that one is not a thing
     * to fix on its own, it is what the neighbours cost. */
    for (i = 0; i < g->n; i++) {
        unsigned m;
        int covered = 0;
        if (g->L[i].kind == ROCKET_GRAPH_HOST) continue;
        if (g->L[i].kind == ROCKET_GRAPH_CONCAT) continue;   /* no program either way */
        for (m = 0; m < g->n; m++)
            if (g->kick[m] && i >= m && i < g->kick_end[m]) { covered = 1; break; }
        if (covered) { if (g->kb_of) g->kb_of[i] = 0; continue; }
        if (g->kb_of && g->kb_of[i]) { g->kb_count[g->kb_of[i] - 1]++; continue; }
        if (!g->cube_out[i])      kb_mark(g, i, ROCKET_GRAPH_KB_NO_CUBE_OUT);
        else if (!g->cube_in[i])   kb_mark(g, i, ROCKET_GRAPH_KB_NO_CUBE_IN);
        else                                 kb_mark(g, i, ROCKET_GRAPH_KB_ALONE);
        if (g->kb_of) g->kb_count[g->kb_of[i] - 1]++;
    }
    free(cand);
    if (g->kick_runs) {
        unsigned m;
        RG_LOG(g, "   cross-layer kick: %d run(s) covering %d layer(s) — one hardware kick "
               "each where the per-layer path takes one per layer\n",
               g->kick_runs, g->kick_layers);
        /* WHERE THE RUNS BREAK, which the count does not say and which is the whole
         * diagnosis when a graph does not reach one kick. */
        RG_LOG(g, "      run(s) at layer(s):");
        for (m = 0; m < g->n; m++)
            if (g->kick[m]) RG_LOG(g, " %u-%u", m, g->kick_end[m] - 1u);
        RG_LOG(g, "\n");
    }
    else
        RG_LOG(g, "   cross-layer kick: no run of two or more linked layers, so nothing to "
               "chain\n");
    {
        int k, out = 0;
        for (k = 0; k < ROCKET_GRAPH_KB_N; k++) out += g->kb_count[k];
        if (out) {
            RG_LOG(g, "      %d layer(s) outside any run, and each one is a submit:\n", out);
            for (k = 0; k < ROCKET_GRAPH_KB_N; k++)
                if (g->kb_count[k]) RG_LOG(g, "      %3d  %s\n", g->kb_count[k], KB_WHY[k]);
            RG_LOG(g, "      layer(s):");
            for (k = 0; k < (int)g->n; k++)
                if (g->kb_of && g->kb_of[k]) RG_LOG(g, " %d", k);
            RG_LOG(g, "\n");
        }
    }
}

/* Run one layer. `how` is set to the path it took. `in2` is the second operand of an add
 * and NULL everywhere else. */
/* ---------------------------------------------------------------- the plan -------- */

/* A layer read by anything other than the layer immediately after it. A caller's ping-pong
 * pair only ever holds the previous layer's output, so every other reader needs this one
 * to have written somewhere of its own — and the passes need it because such a producer
 * cannot silently stop materialising. A HOST layer is stepped over: it is not run on the
 * part, so its read is not one. */
int rocket_graph_is_skip_source(const rocket_graph_layer *L, unsigned n, unsigned i)
{
    unsigned j;
    for (j = i + 1; j < n; j++) {
        unsigned s[ROCKET_GRAPH_MAX_SRC], ns, k;
        if (L[j].kind == ROCKET_GRAPH_HOST) continue;
        ns = rocket_graph_srcs(&L[j], s);
        for (k = 0; k < ns; k++)
            if (s[k] == i && j != i + 1) return 1;
    }
    return 0;
}

/* THE RUNS, as a second step. The links are what make a run eligible, and a caller that
 * wants to price them apart runs the graph between the two calls — which is the A/B that
 * separates "the transposes a join removes" from "the submits a kick removes". */
void rocket_graph_plan_kicks(rocket_graph_plan *g)
{
    if (g) kick_build(g);
}

int rocket_graph_is_placement(const rocket_graph_plan *g, unsigned i)
{
    return i < g->n && g->L[i].kind == ROCKET_GRAPH_CONCAT && g->cube_out &&
           g->cube_out[i];
}

void rocket_graph_plan_free(rocket_graph_plan *g)
{
    unsigned i;
    if (!g) return;
    /* The chains BORROW the handles, so they go first. */
    if (g->kick)
        for (i = 0; i < g->n; i++)
            if (g->kick[i]) rocket_conv2d_int8_chain_free_rk3576(g->fd, g->kick[i]);
    if (g->buf)
        for (i = 0; i < g->n; i++) rocket_rk3576_cube_free(g->fd, &g->buf[i]);
    free(g->kick); free(g->kick_end);
    free(g->cube_in); free(g->cube_out); free(g->placed); free(g->shared);
    free(g->skip); free(g->buf); free(g->kb_of);
    free(g);
}

rocket_graph_plan *rocket_graph_plan_new(int fd, const rocket_graph_layer *L, unsigned n,
                                         int verbose)
{
    rocket_graph_plan *g = calloc(1, sizeof *g);
    unsigned i;

    if (!g || !L || !n) { free(g); return NULL; }
    g->L = L; g->n = n; g->fd = fd; g->verbose = verbose;
    g->cube_in  = calloc(n, 1);
    g->cube_out = calloc(n, 1);
    g->placed   = calloc(n, 1);
    g->shared   = calloc(n, 1);
    g->skip     = calloc(n, 1);
    g->buf      = calloc(n, sizeof *g->buf);
    g->kick     = calloc(n, sizeof *g->kick);
    g->kick_end = calloc(n, sizeof *g->kick_end);
    if (!g->cube_in || !g->cube_out || !g->placed || !g->shared || !g->skip ||
        !g->buf || !g->kick || !g->kick_end) {
        rocket_graph_plan_free(g);
        return NULL;
    }
    for (i = 0; i < n; i++)
        g->skip[i] = (unsigned char)rocket_graph_is_skip_source(L, n, i);

    cube_link(g);
    return g;
}
