// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * fa_replay_probe.c — replay one flash-attention op from its dumped tiles, many times, and
 * count how often its output differs from itself.
 *
 * A probe, not a gate. On
 * gemma4-12b F16 at pp2048 the handler's output takes one of two values about one run in
 * five, and the disagreement starts at one op whose inputs are byte-identical in both states
 * (ggml-rocket ROCKET_FA_CHECKSUM=3). ROCKET_FA_DUMP_OP writes that op's dense tiles and its
 * fp16 output. This feeds the tiles back through rocket_flash_attn_fp16_ctx, the path the
 * handler takes, so a disagreement costs seconds rather than a model run, and the handler's
 * own inputs are held fixed by construction.
 *
 * Each trial runs the op once and compares its output element by element against the
 * reference dump (a modal run's), and against a second dump when one is given (an odd run's).
 * It prints, per trial, how many elements differ from each and where the first one is, and per
 * mode how many distinct outputs the trials produced, by hash, so a knob that changes the
 * arithmetic (and so matches neither dump) can still be read for stability.
 * Trials run first in one context, then each in a fresh context, since a context owns worker
 * fds and resident scratch that persist across calls. Each mode also prints its trials' states
 * in order (R the reference, A the alternate, O neither).
 *
 * Core placement: with FA_REPLAY_MARKER naming a tracefs trace_marker (an instance's, so the
 * global trace state is untouched), each trial is bracketed by "FAR B m=<mode> t=<trial>" and
 * "FAR E m=<mode> t=<trial> s=<R|A|O>" lines, and each mode opens with "FAR M". With the
 * gpu_scheduler events enabled in that instance, every job the trial submits lands between its
 * two markers with its ring (the core) and its fd's DRM client id; tools/fa_trace_join.py joins
 * them. It prints the QK and AV tile plans, from which the join derives which job of which
 * worker holds a given element. The markers cost a write(2) per trial on the main thread, not
 * inside the workers, but tracing is not free, so a rate read under it is its own reading.
 *
 * What it does NOT show: the model around the op (host threads, the other ops' device work,
 * memory pressure), or which of two values is right. The trace shows where a job ran, not
 * what the core held when it started.
 *
 * Usage: fa_replay_probe DIR_REF [DIR_ALT] n_tokens n_head n_kv n_kv_heads dk dv scale softcap
 *                        nthreads trials
 *   DIR_*  a ROCKET_FA_DUMP_DIR holding fa_op*_{q,k,v,m,o}.f16 for one op
 *   FA_REPLAY_MARKER  optional: a trace_marker path to bracket each trial in (needs root)
 * Exit: 0, or 1 on a load or call failure, 2 no NPU.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <glob.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdarg.h>

#include "rocket_npu.h"
#include "rocket_attn.h"
#include "rocket_matmul.h"

/* One line to the trace marker, if FA_REPLAY_MARKER opened one. A failed write is ignored:
 * the join refuses a trial whose markers are missing rather than guessing. */
static int g_marker = -1;
static void mark(const char *fmt, ...)
{
    if (g_marker < 0) return;
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) { ssize_t w = write(g_marker, buf, (size_t)n < sizeof buf ? (size_t)n : sizeof buf - 1); (void)w; }
}

static _Float16 *load(const char *dir, const char *tag, size_t n)
{
    char pat[1024];
    glob_t g;
    snprintf(pat, sizeof pat, "%s/fa_op*_pid*_%s.f16", dir, tag);
    if (glob(pat, 0, NULL, &g) != 0 || g.gl_pathc != 1) {
        fprintf(stderr, "%s: expected one file matching %s\n", dir, pat);
        return NULL;
    }
    _Float16 *p = malloc(n * sizeof(_Float16));
    FILE *f = fopen(g.gl_pathv[0], "rb");
    size_t got = (p && f) ? fread(p, sizeof(_Float16), n, f) : 0;
    if (f) fclose(f);
    globfree(&g);
    if (got != n) { fprintf(stderr, "%s/%s: read %zu of %zu elements\n", dir, tag, got, n); free(p); return NULL; }
    return p;
}

static uint64_t fnv1a(const void *data, size_t n)
{
    const unsigned char *p = data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

static long compare(const _Float16 *a, const _Float16 *b, size_t n, long *first)
{
    long d = 0;
    *first = -1;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], sizeof(_Float16)) != 0) { if (*first < 0) *first = (long)i; d++; }
    return d;
}

int main(int argc, char **argv)
{
    if (argc != 12 && argc != 13) {
        fprintf(stderr, "usage: %s DIR_REF [DIR_ALT] n_tokens n_head n_kv n_kv_heads dk dv scale softcap "
                "nthreads trials\n", argv[0]);
        return 1;
    }
    const int alt = (argc == 13);
    const char *dref = argv[1], *dalt = alt ? argv[2] : NULL;
    int ai = alt ? 3 : 2;
    const int nt = atoi(argv[ai++]), nh = atoi(argv[ai++]), nkv = atoi(argv[ai++]);
    const int nkvh = atoi(argv[ai++]), dk = atoi(argv[ai++]), dv = atoi(argv[ai++]);
    const float scale = (float)atof(argv[ai++]), softcap = (float)atof(argv[ai++]);
    const int nthreads = atoi(argv[ai++]), trials = atoi(argv[ai++]);

    const size_t nq = (size_t)nh * nt * dk, nk = (size_t)nkvh * nkv * dk, nv = (size_t)nkvh * dv * nkv;
    const size_t nm = (size_t)nt * nkv, no = (size_t)nh * nt * dv;
    _Float16 *Q = load(dref, "q", nq), *K = load(dref, "k", nk), *V = load(dref, "v", nv);
    _Float16 *M = load(dref, "m", nm), *O_ref = load(dref, "o", no);
    _Float16 *O_alt = alt ? load(dalt, "o", no) : NULL, *O = malloc(no * sizeof(_Float16));
    if (!Q || !K || !V || !M || !O_ref || (alt && !O_alt) || !O) return 1;
    if (alt) {
        _Float16 *Qa = load(dalt, "q", nq), *Ka = load(dalt, "k", nk), *Va = load(dalt, "v", nv), *Ma = load(dalt, "m", nm);
        long f;
        if (!Qa || !Ka || !Va || !Ma) return 1;
        printf("inputs of the two dumps: q %ld, k %ld, v %ld, m %ld elements differ; outputs %ld",
               compare(Q, Qa, nq, &f), compare(K, Ka, nk, &f), compare(V, Va, nv, &f), compare(M, Ma, nm, &f),
               compare(O_ref, O_alt, no, &f));
        if (f >= 0) printf(" (first at head %ld, token %ld, dv %ld)", f / ((long)nt * dv), (f / dv) % nt, f % dv);
        printf("\n");
        free(Qa); free(Ka); free(Va); free(Ma);
    }
    printf("replay: n_tokens %d n_head %d n_kv %d n_kv_heads %d dk %d dv %d scale %g softcap %g, "
           "%d worker threads, %d trials a mode\n", nt, nh, nkv, nkvh, dk, dv, scale, softcap, nthreads, trials);
    {
        /* The chained path's two shapes: QK (Tp,dk,Kn) and AV (Tp,Kn,dv). */
        const int Tp = (nt + 3) & ~3, Kn = (nkv + 31) & ~31;
        int Mt, Kt, Nt;
        int tq = rocket_matmul_plan(Tp, dk, Kn, &Mt, &Kt, &Nt);
        printf("plan: QK M %d K %d N %d -> %d tiles a head, Mt %d Kt %d Nt %d\n", Tp, dk, Kn, tq, Mt, Kt, Nt);
        int ta = rocket_matmul_plan(Tp, Kn, dv, &Mt, &Kt, &Nt);
        printf("plan: AV M %d K %d N %d -> %d tiles a head, Mt %d Kt %d Nt %d\n", Tp, Kn, dv, ta, Mt, Kt, Nt);
    }
    const char *mpath = getenv("FA_REPLAY_MARKER");
    if (mpath && *mpath) {
        g_marker = open(mpath, O_WRONLY | O_CLOEXEC);
        if (g_marker < 0) { perror(mpath); return 1; }
        printf("marker: %s\n", mpath);
    }

    const char *modes[2] = { "one ctx", "fresh ctx" };
    char *states = malloc((size_t)trials + 1);
    if (!states) return 1;
    for (int mode = 0; mode < 2; mode++) {
        rocket_fa_ctx *ctx = NULL;
        long n_ref = 0, n_alt = 0, n_other = 0;
        uint64_t uh[16];
        long uc[16];
        int nu = 0;
        mark("FAR M m=%d nthreads=%d trials=%d", mode, nthreads, trials);
        for (int t = 0; t < trials; t++) {
            if (!ctx) ctx = rocket_fa_ctx_create(nthreads);
            if (!ctx) { fprintf(stderr, "rocket_fa_ctx_create failed\n"); return 2; }
            memset(O, 0xAA, no * sizeof(_Float16));
            mark("FAR B m=%d t=%d", mode, t);
            int rc = rocket_flash_attn_fp16_ctx(ctx, nt, nkv, dk, dv, nh, nkvh, scale, softcap, Q, K, V, M, O);
            if (rc != 0) { fprintf(stderr, "trial %d: rc %d\n", t, rc); return 1; }
            long fr, fa = -1;
            const long dr = compare(O, O_ref, no, &fr);
            const long da = alt ? compare(O, O_alt, no, &fa) : -1;
            if (dr == 0) n_ref++;
            else if (da == 0) n_alt++;
            else n_other++;
            states[t] = dr == 0 ? 'R' : (da == 0 ? 'A' : 'O');
            mark("FAR E m=%d t=%d s=%c", mode, t, states[t]);
            {
                const uint64_t h = fnv1a(O, no * sizeof(_Float16));
                int k = 0;
                while (k < nu && uh[k] != h) k++;
                if (k == nu && nu < 16) { uh[nu] = h; uc[nu] = 0; nu++; }
                if (k < nu) uc[k]++;
            }
            if (dr != 0 && (!alt || da != 0) && t < 3) {
                printf("  %-9s trial %3d: %ld elements differ from the reference", modes[mode], t, dr);
                if (alt) printf(", %ld from the alternate", da);
                printf(", first at head %ld, token %ld, dv %ld\n", fr / ((long)nt * dv), (fr / dv) % nt, fr % dv);
            }
            if (mode == 1) { rocket_fa_ctx_free(ctx); ctx = NULL; }
        }
        if (ctx) rocket_fa_ctx_free(ctx);
        printf("  %-9s: %ld of %d trials equal the reference, %ld the alternate, %ld neither; "
               "%d distinct output(s):", modes[mode], n_ref, trials, n_alt, n_other, nu);
        for (int k = 0; k < nu; k++) printf(" %016llx x%ld", (unsigned long long)uh[k], uc[k]);
        printf("\n");
        states[trials] = '\0';
        printf("  %-9s states: %s\n", modes[mode], states);
    }
    if (g_marker >= 0) close(g_marker);
    free(states);
    free(Q); free(K); free(V); free(M); free(O_ref); free(O_alt); free(O);
    return 0;
}
