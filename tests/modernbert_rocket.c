// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * modernbert_rocket.c — gate for rocket_modernbert: the resident ModernBERT encoder (and its
 * post stack) against ONNX Runtime's fp32 taps, layer by layer.
 *
 * The artifacts come from tools/modernbert_extract.py: a checkpoint's weights blob and one
 * .case file per reference input. Every tap (the embedding norm, each encoder layer, the
 * final norm, each post layer) is scored over the live rows by cosine and max abs error. The
 * gate fails a tap under ROCKET_MB_MIN_COS (default 0.9999); a vacuous run (no case read)
 * fails too.
 *
 *   ./modernbert_rocket DIR [CASE ...]        # NPU, 3 workers; every .case in DIR by default
 *   ./modernbert_rocket --host DIR [CASE ...] # the host mode: same glue, no device
 *   ROCKET_MB_BENCH=11 ./modernbert_rocket DIR   # also the warm median over 11 encodes
 *
 * No DIR (or an empty one) -> SKIP (2), which is what ctest sees without artifacts.
 *
 * mb_weights.f16: int32[12] {magic 'MBRT', version 1, d, n_layers, n_head, d_ff, window,
 *   n_post, post_n_head, post_d_ff, post_act, n_rope}, float {eps, post_eps},
 *   int32[n_layers][3] {sliding, has_attn_norm, rope index}, float[n_rope][head_dim/2],
 *   then fp16: emb_norm, final_norm, per layer [attn_norm] Wqkv Wo mlp_norm Wi Wo_mlp,
 *   per post layer ln1_g ln1_b Wqkv bqkv Wo bo ln2_g ln2_b W1 b1 W2 b2.
 * CASE.case: int32 {B, T, n_taps, d}, int32 len[B], float emb[B][T][d],
 *   float post_bias[B][d], float taps[n_taps][B][T][d].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "rocket_modernbert.h"
#include "rocket_npu.h"

#define MB_MAGIC 0x5452424D

typedef struct { void *map; size_t size; } mapping;

static void *map_file(const char *path, mapping *mp)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    mp->map = p; mp->size = (size_t)st.st_size;
    return p;
}

static float rope_tab[8][128];

static int load_model(const char *path, rocket_modernbert_model *m, mapping *mp)
{
    const uint8_t *p = map_file(path, mp);
    if (!p) { fprintf(stderr, "cannot map %s\n", path); return -1; }
    const int32_t *h = (const int32_t *)p;
    if (h[0] != MB_MAGIC || h[1] != 1) { fprintf(stderr, "%s: bad magic/version\n", path); return -1; }
    memset(m, 0, sizeof *m);
    m->d = h[2]; m->n_layers = h[3]; m->n_head = h[4]; m->d_ff = h[5]; m->window = h[6];
    m->n_post = h[7]; m->post_n_head = h[8]; m->post_d_ff = h[9]; m->post_act = h[10];
    const int n_rope = h[11], d = m->d, dh = d / m->n_head, dff = m->d_ff, pff = m->post_d_ff;
    if (m->n_layers > ROCKET_MB_MAX_LAYERS || m->n_post > ROCKET_MB_MAX_POST || n_rope > 8 || dh / 2 > 128)
        return -1;
    const float *fh = (const float *)(h + 12);
    m->eps = fh[0]; m->post_eps = fh[1];
    const int32_t *lf = (const int32_t *)(fh + 2);
    const float *rf = (const float *)(lf + 3 * m->n_layers);
    for (int r = 0; r < n_rope; r++) memcpy(rope_tab[r], rf + (size_t)r * (dh / 2), (size_t)(dh / 2) * sizeof(float));
    const _Float16 *w = (const _Float16 *)(rf + (size_t)n_rope * (dh / 2));
#define TAKE(dst, n) do { (dst) = w; w += (n); } while (0)
    TAKE(m->emb_norm_g, d);
    TAKE(m->final_norm_g, d);
    for (int l = 0; l < m->n_layers; l++) {
        rocket_mb_layer *L = &m->layers[l];
        L->sliding = lf[3 * l];
        L->inv_freq = rope_tab[lf[3 * l + 2]];
        if (lf[3 * l + 1]) TAKE(L->attn_norm_g, d);
        TAKE(L->Wqkv, (size_t)3 * d * d); TAKE(L->Wo, (size_t)d * d); TAKE(L->mlp_norm_g, d);
        TAKE(L->Wi, (size_t)2 * dff * d); TAKE(L->Wo_mlp, (size_t)d * dff);
    }
    for (int j = 0; j < m->n_post; j++) {
        rocket_mb_post_layer *P = &m->post[j];
        TAKE(P->ln1_g, d); TAKE(P->ln1_b, d); TAKE(P->Wqkv, (size_t)3 * d * d); TAKE(P->bqkv, 3 * d);
        TAKE(P->Wo, (size_t)d * d); TAKE(P->bo, d); TAKE(P->ln2_g, d); TAKE(P->ln2_b, d);
        TAKE(P->W1, (size_t)pff * d); TAKE(P->b1, pff); TAKE(P->W2, (size_t)d * pff); TAKE(P->b2, d);
    }
#undef TAKE
    if ((const uint8_t *)w != p + mp->size) {
        fprintf(stderr, "%s: size mismatch (%zu parsed, %zu on disk)\n", path,
                (size_t)((const uint8_t *)w - p), mp->size);
        return -1;
    }
    return 0;
}

static double now_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Score one tap over the live rows. */
static void score(const float *got, const float *ref, int B, int T, int d, const int *len,
                  double *cosv, double *maxerr, double *maxref)
{
    double dot = 0, ng = 0, nr = 0, me = 0, mr = 0;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < len[b]; t++)
            for (int j = 0; j < d; j++) {
                size_t i = ((size_t)b * T + t) * d + j;
                double g = got[i], r = ref[i];
                dot += g * r; ng += g * g; nr += r * r;
                if (fabs(g - r) > me || !isfinite(g)) me = isfinite(g) ? fabs(g - r) : INFINITY;
                if (fabs(r) > mr) mr = fabs(r);
            }
    *cosv = dot / (sqrt(ng) * sqrt(nr) + 1e-300);
    *maxerr = me; *maxref = mr;
}

static int run_case(rocket_mb_ctx *ctx, const rocket_modernbert_model *m, const char *path,
                    double min_cos, int bench)
{
    mapping mp;
    const int32_t *h = map_file(path, &mp);
    if (!h) { fprintf(stderr, "cannot map %s\n", path); return 1; }
    const int B = h[0], T = h[1], nt = h[2], d = h[3];
    if (d != m->d || nt != m->n_layers + 2 + m->n_post) { fprintf(stderr, "%s: shape mismatch\n", path); return 1; }
    const int *len = h + 4;
    const float *emb = (const float *)(len + B);
    const float *pb = emb + (size_t)B * T * d;
    const float *taps = pb + (size_t)B * d;
    const size_t slab = (size_t)B * T * d;
    float *out = malloc(slab * sizeof(float));
    float *hid = malloc((size_t)nt * slab * sizeof(float));
    if (!out || !hid) { fprintf(stderr, "oom\n"); return 1; }

    int rc = rocket_modernbert_encode(ctx, B, T, len, emb, pb, out, hid);
    int fail = 0;
    printf("  %s: B=%d T=%d len=", path, B, T);
    for (int b = 0; b < B; b++) printf("%s%d", b ? "," : "", len[b]);
    printf(" rc=%d\n", rc);
    if (rc) fail = 1;
    double worst = 1.0;
    for (int k = 0; !rc && k < nt; k++) {
        double cv, me, mr;
        score(hid + (size_t)k * slab, taps + (size_t)k * slab, B, T, d, len, &cv, &me, &mr);
        char lab[32];
        if (k == 0) snprintf(lab, sizeof lab, "emb_norm");
        else if (k <= m->n_layers) snprintf(lab, sizeof lab, "enc%d", k - 1);
        else if (k == m->n_layers + 1) snprintf(lab, sizeof lab, "final_norm");
        else snprintf(lab, sizeof lab, "post%d", k - m->n_layers - 2);
        int bad = !(cv >= min_cos);
        if (cv < worst) worst = cv;
        fail |= bad;
        if (bad || k == 0 || k % 7 == 0 || k >= m->n_layers)
            printf("    %-10s cos %.9f  maxerr %.3e  (max|ref| %.1f)%s\n", lab, cv, me, mr, bad ? "  <- FAIL" : "");
    }
    if (!rc) {   /* the returned output is the last tap */
        double cv, me, mr;
        score(out, hid + (size_t)(nt - 1) * slab, B, T, d, len, &cv, &me, &mr);
        if (me != 0.0) { printf("    out differs from the last hidden tap (maxerr %.3e) <- FAIL\n", me); fail = 1; }
        printf("    worst tap cos %.9f -> %s\n", worst, fail ? "FAIL" : "PASS");
    }
    if (!rc && bench > 0) {
        double *t = malloc((size_t)bench * sizeof(double));
        rocket_modernbert_encode(ctx, B, T, len, emb, pb, out, NULL);   /* warm */
        for (int i = 0; i < bench; i++) {
            double t0 = now_ms();
            rocket_modernbert_encode(ctx, B, T, len, emb, pb, out, NULL);
            t[i] = now_ms() - t0;
        }
        qsort(t, (size_t)bench, sizeof(double), cmp_dbl);
        printf("    bench: median %.1f ms, min %.1f, max %.1f over %d encodes\n", t[bench / 2], t[0], t[bench - 1], bench);
        free(t);
    }
    free(out); free(hid);
    munmap(mp.map, mp.size);
    return fail;
}

int main(int argc, char **argv)
{
    int host = 0, a = 1;
    if (a < argc && !strcmp(argv[a], "--host")) { host = 1; a++; }
    if (a >= argc) { fprintf(stderr, "no artifact directory -> SKIP\n"); return 2; }
    const char *dir = argv[a++];
    char path[4096];
    snprintf(path, sizeof path, "%s/mb_weights.f16", dir);
    if (access(path, R_OK)) { fprintf(stderr, "%s missing -> SKIP\n", path); return 2; }
    if (!host) {
        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "no NPU (%d) -> SKIP\n", fd); return 2; }
        rocket_close(fd);
    }
    const char *mc = getenv("ROCKET_MB_MIN_COS");
    const double min_cos = mc ? atof(mc) : 0.9999;
    const int bench = getenv("ROCKET_MB_BENCH") ? atoi(getenv("ROCKET_MB_BENCH")) : 0;

    rocket_modernbert_model m;
    mapping wm;
    if (load_model(path, &m, &wm)) return 1;
    printf("modernbert_rocket: %s  d=%d layers=%d heads=%d d_ff=%d window=%d post=%d  %s\n", dir, m.d,
           m.n_layers, m.n_head, m.d_ff, m.window, m.n_post, host ? "HOST" : "NPU x3");
    double t0 = now_ms();
    rocket_mb_ctx *ctx = rocket_modernbert_ctx_create(&m, host ? 0 : 3);
    if (!ctx) { fprintf(stderr, "ctx_create failed\n"); return 1; }
    printf("  ctx create (weight pack) %.0f ms\n", now_ms() - t0);

    int fail = 0, ran = 0;
    if (a < argc) {
        for (; a < argc; a++) {
            snprintf(path, sizeof path, "%s/%s.case", dir, argv[a]);
            fail |= run_case(ctx, &m, path, min_cos, bench); ran++;
        }
    } else {
        DIR *dp = opendir(dir);
        struct dirent *e;
        while (dp && (e = readdir(dp))) {
            size_t n = strlen(e->d_name);
            if (n < 6 || strcmp(e->d_name + n - 5, ".case")) continue;
            snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
            fail |= run_case(ctx, &m, path, min_cos, bench); ran++;
        }
        if (dp) closedir(dp);
    }
    rocket_modernbert_ctx_free(ctx);
    munmap(wm.map, wm.size);
    if (!ran) { printf("no case files in %s -> FAIL\n", dir); return 1; }
    printf("==== %s (%d case%s) ====\n", fail ? "FAIL" : "PASS", ran, ran == 1 ? "" : "s");
    return fail ? 1 : 0;
}
