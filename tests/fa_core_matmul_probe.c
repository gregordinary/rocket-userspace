// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * fa_core_matmul_probe.c — run flash-attention AV matmuls on every NPU core and report, per
 * call, how the fp16 output compares with a reference, so a per-core arithmetic difference
 * reads as a count.
 *
 * A probe, not a gate. The FA
 * replay's odd output follows the core that runs the owning worker's AV job. This runs the FA
 * handler's matmul path (rocket_mm_batch_run: tiles of the AV plan, fp32 accumulate within a
 * tile, DPU fp32->fp16 on the way out, host fp32 accumulation across K passes) from worker
 * threads with an fd each, released together by a barrier every round so the kernel's
 * scheduler spreads the jobs over the cores.
 *
 * Synthetic mode (no OPDIR): one AV tile, M 256, K 384, N 256, two items a call.
 *   item 0, "grid": a = i/64, b = j/64 with small integers i, j. Every product and partial sum
 *     is an exact fp32 value in any MAC order, so the only rounding is fp32 -> fp16. The host
 *     has the exact value, so each output is classed as round-to-nearest-even, round-half-away
 *     at a tie, or neither. Ties are common by construction.
 *   item 1, "rand": full-mantissa fp16 in [-1, 1); the fp32 accumulation rounds, so a
 *     difference here and not in "grid" is an accumulation (order or width) difference.
 * Operands mode (OPDIR holding fa_av_{a,b,c}.f16 from tests/fa_av_dump_wrap.c): one head's real
 *   AV operands, M 512, K 1024, N 256, four calls a round, each ONE job:
 *     X  K pass 0 through rocket_matmul_fp16_f32out: the tile's fp32 accumulator, unrounded,
 *        against the kind's first result and the exact sum rounded to fp32, in fp32 ulps;
 *     F  the full AV as the handler runs it (two identical items), against the dumped result;
 *     S  K passes 0 and 1 (k 0-383, 384-767) as two single-pass items;
 *     T  K pass 2 (k 768-1023), twice.
 *   Each call is compared with the first result of its kind (ndiff), and each pass's outputs
 *   with the exact (double) sum rounded to nearest even (mis). It prints the element the FA
 *   replay flips (row 256, column 240) for every call. The exact sum is a double sum of exact
 *   products, exact for these operands' range but not checked to be.
 * With FA_REPLAY_MARKER naming a trace_marker, each worker writes one line after each call from
 * its own thread, so the call's job is the queue event just before it on the same tid.
 *
 * What it does NOT show: which core ran a call (that needs the trace), the QK and softmax
 * around the AV, or which rounding is right for the model.
 *
 * Usage: fa_core_matmul_probe rounds [nworkers [OPDIR]]      (nworkers default 3, max 8)
 * Exit: 0, or 1 on a load or call failure, 2 no NPU.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/syscall.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int g_marker = -1;
static pthread_barrier_t g_bar;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_rounds;

static uint64_t fnv1a(const void *data, size_t n)
{
    const unsigned char *p = data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

static void mark(const char *buf)
{
    if (g_marker >= 0) { ssize_t w = write(g_marker, buf, strlen(buf)); (void)w; }
}

static uint16_t bits16(_Float16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }

/* An exact double to fp16 with ties away from zero; *is_tie says whether x was a tie. */
static _Float16 to_f16_rha(double x, int *is_tie)
{
    _Float16 r = (_Float16)x, up, dn;
    uint16_t bits = bits16(r), bu = bits + 1, bd = bits - 1;
    memcpy(&up, &bu, 2); memcpy(&dn, &bd, 2);
    const double rd = (double)r;
    const double other = (fabs((double)up - x) < fabs((double)dn - x)) ? (double)up : (double)dn;
    *is_tie = 0;
    if (fabs(other - x) == fabs(rd - x) && other != rd) {
        *is_tie = 1;
        if (fabs(other) > fabs(rd)) return (_Float16)other;
    }
    return r;
}

/* ---- synthetic mode -------------------------------------------------------------------- */
enum { SM = 256, SK = 384, SN = 256 };
static _Float16 *gA[2], *gB[2];
static int32_t *gnum;
static _Float16 *gref_rne;
static int *gtie;

static void *worker_syn(void *p)
{
    const int w = *(int *)p;
    const int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "worker %d: no NPU (%d)\n", w, fd); exit(2); }
    rocket_mm_batch *b = rocket_mm_batch_create(fd);
    if (!b) { fprintf(stderr, "worker %d: rocket_mm_batch_create failed\n", w); exit(1); }
    _Float16 *C[2] = { malloc((size_t)SM * SN * 2), malloc((size_t)SM * SN * 2) };
    const long tid = syscall(SYS_gettid);
    for (int r = 0; r < g_rounds; r++) {
        memset(C[0], 0xAA, (size_t)SM * SN * 2);
        memset(C[1], 0xAA, (size_t)SM * SN * 2);
        pthread_barrier_wait(&g_bar);
        const _Float16 *const A[2] = { gA[0], gA[1] }, *const B[2] = { gB[0], gB[1] };
        int rc = rocket_mm_batch_run(b, SM, SK, SN, 2, A, B, C);
        if (rc) { fprintf(stderr, "worker %d round %d: rc %d\n", w, r, rc); exit(1); }
        long rne = 0, rha = 0, oth = 0, tie = 0;
        for (size_t i = 0; i < (size_t)SM * SN; i++) {
            tie += gtie[i];
            if (bits16(C[0][i]) == bits16(gref_rne[i])) { rne++; continue; }
            int t; _Float16 ha = to_f16_rha(gnum[i] / 4096.0, &t);
            if (t && bits16(C[0][i]) == bits16(ha)) { rha++; continue; }
            oth++;
        }
        const uint64_t hg = fnv1a(C[0], (size_t)SM * SN * 2), hr = fnv1a(C[1], (size_t)SM * SN * 2);
        printf("round %3d worker %d tid %ld: grid rne %ld tie_away %ld other %ld of %d, ties %ld; "
               "hash grid %016llx rand %016llx\n", r, w, tid, rne, rha, oth, SM * SN, tie,
               (unsigned long long)hg, (unsigned long long)hr);
        char buf[200];
        snprintf(buf, sizeof buf, "FCM r=%d w=%d rne=%ld rha=%ld oth=%ld hg=%016llx hr=%016llx",
                 r, w, rne, rha, oth, (unsigned long long)hg, (unsigned long long)hr);
        mark(buf);
        fflush(stdout);
    }
    rocket_mm_batch_free(b);
    rocket_close(fd);
    free(C[0]); free(C[1]);
    return NULL;
}

static void setup_syn(void)
{
    srand(12345);
    for (int it = 0; it < 2; it++) { gA[it] = malloc((size_t)SM * SK * 2); gB[it] = malloc((size_t)SN * SK * 2); }
    static int8_t ia[SM * SK], ib[SN * SK];
    for (int i = 0; i < SM * SK; i++) { ia[i] = (int8_t)(rand() % 65 - 32); gA[0][i] = (_Float16)(ia[i] / 64.0); }
    for (int i = 0; i < SN * SK; i++) { ib[i] = (int8_t)(rand() % 65 - 32); gB[0][i] = (_Float16)(ib[i] / 64.0); }
    for (int i = 0; i < SM * SK; i++) gA[1][i] = (_Float16)((rand() / (double)RAND_MAX) * 2.0 - 1.0);
    for (int i = 0; i < SN * SK; i++) gB[1][i] = (_Float16)((rand() / (double)RAND_MAX) * 2.0 - 1.0);
    gnum = malloc((size_t)SM * SN * sizeof *gnum);
    gref_rne = malloc((size_t)SM * SN * 2);
    gtie = malloc((size_t)SM * SN * sizeof *gtie);
    long nties = 0;
    for (int m = 0; m < SM; m++)
        for (int n = 0; n < SN; n++) {
            int32_t s = 0;
            for (int k = 0; k < SK; k++) s += (int32_t)ia[m * SK + k] * ib[n * SK + k];
            const size_t i = (size_t)m * SN + n;
            gnum[i] = s;
            gref_rne[i] = (_Float16)(s / 4096.0);
            to_f16_rha(s / 4096.0, &gtie[i]);
            nties += gtie[i];
        }
    printf("fa_core_matmul_probe synthetic: M %d K %d N %d; grid item has %ld fp16 ties of %d\n",
           SM, SK, SN, nties, SM * SN);
}

/* ---- operands mode --------------------------------------------------------------------- */
enum { OM = 512, OK = 1024, ON = 256, OKT = 384, TR = 256, TC = 240 };
static _Float16 *oA, *oB, *oC;              /* P [OM][OK], V [ON][OK], the dumped result      */
static _Float16 *sA[3], *sB[3];             /* the three K passes, contiguous                  */
static int sK[3] = { OKT, OKT, OK - 2 * OKT };
static _Float16 *xrne[3];                   /* exact pass sums rounded to nearest even         */
static float *xf0;                          /* exact pass-0 sums rounded to fp32               */
static _Float16 *ref[3];                    /* first result of each call kind (F, S, T)        */
static float *refx;                         /* first result of the fp32-out pass-0 call (X)    */
static uint64_t seen[4][8];
static int nseen[4];

static _Float16 *load(const char *dir, const char *tag, size_t n)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/fa_av_%s.f16", dir, tag);
    _Float16 *p = malloc(n * 2);
    FILE *f = fopen(path, "rb");
    size_t got = (p && f) ? fread(p, 2, n, f) : 0;
    if (f) fclose(f);
    if (got != n) { fprintf(stderr, "%s: read %zu of %zu\n", path, got, n); exit(1); }
    return p;
}

static void setup_ops(const char *dir)
{
    oA = load(dir, "a", (size_t)OM * OK);
    oB = load(dir, "b", (size_t)ON * OK);
    oC = load(dir, "c", (size_t)OM * ON);
    long sub = 0, zero = 0;
    for (size_t i = 0; i < (size_t)OM * OK; i++) {
        const uint16_t b = bits16(oA[i]) & 0x7fff;
        if (b == 0) zero++; else if (b < 0x0400) sub++;
    }
    int k0 = 0;
    for (int s = 0; s < 3; s++) {
        sA[s] = malloc((size_t)OM * sK[s] * 2);
        sB[s] = malloc((size_t)ON * sK[s] * 2);
        for (int m = 0; m < OM; m++) memcpy(sA[s] + (size_t)m * sK[s], oA + (size_t)m * OK + k0, (size_t)sK[s] * 2);
        for (int n = 0; n < ON; n++) memcpy(sB[s] + (size_t)n * sK[s], oB + (size_t)n * OK + k0, (size_t)sK[s] * 2);
        xrne[s] = malloc((size_t)OM * ON * 2);
        for (int m = 0; m < OM; m++)
            for (int n = 0; n < ON; n++) {
                double acc = 0.0;   /* each product is exact in double; the sum is (near) exact */
                for (int k = 0; k < sK[s]; k++)
                    acc += (double)sA[s][(size_t)m * sK[s] + k] * (double)sB[s][(size_t)n * sK[s] + k];
                xrne[s][(size_t)m * ON + n] = (_Float16)acc;
                if (s == 0) {
                    if (!xf0) xf0 = malloc((size_t)OM * ON * sizeof(float));
                    xf0[(size_t)m * ON + n] = (float)acc;
                }
            }
        k0 += sK[s];
    }
    printf("fa_core_matmul_probe operands %s: M %d K %d N %d; P has %ld zero and %ld subnormal of %d; "
           "dumped result at (%d,%d) = 0x%04x; exact pass sums there 0x%04x 0x%04x 0x%04x\n",
           dir, OM, OK, ON, zero, sub, OM * OK, TR, TC, bits16(oC[(size_t)TR * ON + TC]),
           bits16(xrne[0][TR * ON + TC]), bits16(xrne[1][TR * ON + TC]), bits16(xrne[2][TR * ON + TC]));
}

static long ndiff_first(const _Float16 *a, const _Float16 *b, size_t n, long *first)
{
    long d = 0; *first = -1;
    for (size_t i = 0; i < n; i++) if (bits16(a[i]) != bits16(b[i])) { if (*first < 0) *first = (long)i; d++; }
    return d;
}

/* Compare a call's output against its kind's reference (set by the first call of the kind),
 * print the first differing positions the first time a hash is seen. */
static long vs_ref(int kind, const _Float16 *out, int r, int w, uint64_t h)
{
    const size_t n = (size_t)OM * ON;
    long f, d;
    pthread_mutex_lock(&g_mu);
    if (!ref[kind]) { ref[kind] = malloc(n * 2); memcpy(ref[kind], out, n * 2); }
    d = ndiff_first(out, ref[kind], n, &f);
    int k = 0;
    while (k < nseen[kind] && seen[kind][k] != h) k++;
    if (k == nseen[kind] && k < 8) {
        seen[kind][nseen[kind]++] = h;
        if (d) {
            printf("  new hash %016llx (kind %c, round %d worker %d): %ld differ from the kind's first; at",
                   (unsigned long long)h, "FST"[kind], r, w, d);
            int shown = 0;
            for (size_t i = 0; i < n && shown < 8; i++)
                if (bits16(out[i]) != bits16(ref[kind][i])) {
                    printf(" (%zu,%zu) 0x%04x/0x%04x", i / ON, i % ON, bits16(out[i]), bits16(ref[kind][i]));
                    shown++;
                }
            printf("\n");
        }
    }
    pthread_mutex_unlock(&g_mu);
    return d;
}

static long mis_exact(int s, const _Float16 *out)
{
    long m = 0;
    for (size_t i = 0; i < (size_t)OM * ON; i++) m += bits16(out[i]) != bits16(xrne[s][i]);
    return m;
}

static uint32_t bits32(float v) { uint32_t b; memcpy(&b, &v, 4); return b; }

/* Pass 0 through the fp32-output entry: the tile's fp32 accumulator, unrounded. Counts elements
 * that differ from the kind's first result and from the exact sum rounded to fp32, with the
 * largest distance in fp32 ulps, and prints where the first time a hash is seen. */
static void call_x(int fd, float *Cf, int r, int w, long tid)
{
    const size_t n = (size_t)OM * ON;
    for (size_t i = 0; i < n; i++) Cf[i] = NAN;
    pthread_barrier_wait(&g_bar);
    int rc = rocket_matmul_fp16_f32out(fd, OM, OKT, ON, sA[0], sB[0], Cf);
    if (rc) { fprintf(stderr, "worker %d round %d kind X: rc %d\n", w, r, rc); exit(1); }
    const uint64_t h = fnv1a(Cf, n * 4);
    long d = 0, xm = 0, maxu = 0, xmaxu = 0;
    pthread_mutex_lock(&g_mu);
    if (!refx) { refx = malloc(n * 4); memcpy(refx, Cf, n * 4); }
    int k = 0;
    while (k < nseen[3] && seen[3][k] != h) k++;
    const int fresh = (k == nseen[3] && k < 8);
    if (fresh) seen[3][nseen[3]++] = h;
    int shown = 0;
    for (size_t i = 0; i < n; i++) {
        const long u = labs((long)(int32_t)bits32(Cf[i]) - (long)(int32_t)bits32(refx[i]));
        const long ux = labs((long)(int32_t)bits32(Cf[i]) - (long)(int32_t)bits32(xf0[i]));
        if (u) { d++; if (u > maxu) maxu = u;
                 if (fresh && shown < 8) { if (!shown) printf("  new hash %016llx (kind X, round %d worker %d) vs first at", (unsigned long long)h, r, w);
                                           printf(" (%zu,%zu) %.9g/%.9g", i / ON, i % ON, Cf[i], refx[i]); shown++; } }
        if (ux) { xm++; if (ux > xmaxu) xmaxu = ux; }
    }
    if (shown) printf("\n");
    pthread_mutex_unlock(&g_mu);
    const float v = Cf[(size_t)TR * ON + TC];
    printf("round %3d worker %d tid %ld call X: vs first %ld (max %ld ulp), vs exact-fp32 %ld (max %ld ulp); "
           "(%d,%d) %.9g 0x%08x exact %.9g; hash %016llx\n", r, w, tid, d, maxu, xm, xmaxu, TR, TC, v, bits32(v),
           xf0[(size_t)TR * ON + TC], (unsigned long long)h);
    char buf[240];
    snprintf(buf, sizeof buf, "FCO r=%d w=%d call=X d=%ld dump=%ld x0=%ld x1=%ld v0=%08x v1=%08x h0=%016llx",
             r, w, d, maxu, xm, xmaxu, bits32(v), bits32(xf0[(size_t)TR * ON + TC]), (unsigned long long)h);
    mark(buf);
    fflush(stdout);
}

static void *worker_ops(void *p)
{
    const int w = *(int *)p;
    const int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "worker %d: no NPU (%d)\n", w, fd); exit(2); }
    rocket_mm_batch *bf = rocket_mm_batch_create(fd), *bs = rocket_mm_batch_create(fd),
                    *bt = rocket_mm_batch_create(fd);
    if (!bf || !bs || !bt) { fprintf(stderr, "worker %d: rocket_mm_batch_create failed\n", w); exit(1); }
    const size_t n = (size_t)OM * ON;
    _Float16 *C[2] = { malloc(n * 2), malloc(n * 2) };
    float *Cf = malloc(n * sizeof(float));
    const long tid = syscall(SYS_gettid);
    for (int r = 0; r < g_rounds; r++) {
        char buf[240];
        call_x(fd, Cf, r, w, tid);
        for (int kind = 0; kind < 3; kind++) {
            memset(C[0], 0xAA, n * 2); memset(C[1], 0xAA, n * 2);
            pthread_barrier_wait(&g_bar);
            int rc;
            if (kind == 0) {
                const _Float16 *const A[2] = { oA, oA }, *const B[2] = { oB, oB };
                rc = rocket_mm_batch_run(bf, OM, OK, ON, 2, A, B, C);
            } else if (kind == 1) {
                const _Float16 *const A[2] = { sA[0], sA[1] }, *const B[2] = { sB[0], sB[1] };
                rc = rocket_mm_batch_run(bs, OM, OKT, ON, 2, A, B, C);
            } else {
                const _Float16 *const A[2] = { sA[2], sA[2] }, *const B[2] = { sB[2], sB[2] };
                rc = rocket_mm_batch_run(bt, OM, sK[2], ON, 2, A, B, C);
            }
            if (rc) { fprintf(stderr, "worker %d round %d kind %d: rc %d\n", w, r, kind, rc); exit(1); }
            const uint64_t h0 = fnv1a(C[0], n * 2), h1 = fnv1a(C[1], n * 2);
            const long d0 = vs_ref(kind, C[0], r, w, h0);
            long x0 = -1, x1 = -1, dump = -1, f;
            if (kind == 0) dump = ndiff_first(C[0], oC, n, &f);
            if (kind == 1) { x0 = mis_exact(0, C[0]); x1 = mis_exact(1, C[1]); }
            if (kind == 2) { x0 = mis_exact(2, C[0]); }
            const uint16_t v0 = bits16(C[0][(size_t)TR * ON + TC]), v1 = bits16(C[1][(size_t)TR * ON + TC]);
            printf("round %3d worker %d tid %ld call %c: vs first %ld, vs dump %ld, vs exact %ld %ld; "
                   "(%d,%d) 0x%04x 0x%04x; hash %016llx %016llx\n", r, w, tid, "FST"[kind], d0, dump, x0, x1,
                   TR, TC, v0, v1, (unsigned long long)h0, (unsigned long long)h1);
            snprintf(buf, sizeof buf, "FCO r=%d w=%d call=%c d=%ld dump=%ld x0=%ld x1=%ld v0=%04x v1=%04x h0=%016llx",
                     r, w, "FST"[kind], d0, dump, x0, x1, v0, v1, (unsigned long long)h0);
            mark(buf);
            fflush(stdout);
        }
    }
    rocket_mm_batch_free(bf); rocket_mm_batch_free(bs); rocket_mm_batch_free(bt);
    rocket_close(fd);
    free(C[0]); free(C[1]); free(Cf);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s rounds [nworkers [OPDIR]]\n", argv[0]); return 1; }
    g_rounds = atoi(argv[1]);
    int nw = argc > 2 ? atoi(argv[2]) : 3;
    if (nw < 1) nw = 1;
    if (nw > 8) nw = 8;
    const char *opdir = argc > 3 ? argv[3] : NULL;
    if (opdir) setup_ops(opdir); else setup_syn();
    printf("%d rounds x %d workers\n", g_rounds, nw);

    const char *mpath = getenv("FA_REPLAY_MARKER");
    if (mpath && *mpath) {
        g_marker = open(mpath, O_WRONLY | O_CLOEXEC);
        if (g_marker < 0) { perror(mpath); return 1; }
    }
    pthread_barrier_init(&g_bar, NULL, (unsigned)nw);
    pthread_t th[8];
    int idx[8];
    for (int w = 0; w < nw; w++) {
        idx[w] = w;
        if (pthread_create(&th[w], NULL, opdir ? worker_ops : worker_syn, &idx[w]) != 0) {
            fprintf(stderr, "pthread_create\n"); return 1;
        }
    }
    for (int w = 0; w < nw; w++) pthread_join(th[w], NULL);
    pthread_barrier_destroy(&g_bar);
    if (g_marker >= 0) close(g_marker);
    return 0;
}
