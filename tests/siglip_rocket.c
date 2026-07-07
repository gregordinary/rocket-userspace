// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * siglip_rocket.c — HW gate + bench for the full SigLIP-B/16 vision encoder on the NPU
 * (rocket_siglip_encode): patch-embed -> +pos -> 12 x pre-norm block -> post-LN.
 *
 * Validates per-layer cosine vs the fp32 HF oracle dumped by tools/siglip_reference.py
 * (the SHARD fidelity metric: cosine of each layer output, averaged over the 12 layers),
 * plus the final post_layernorm output. PASS = mean-layer cosine >= 0.99 AND post-LN >= 0.99.
 *
 * Needs the artifacts on disk (weight blob + oracle). Missing artifacts or no NPU -> SKIP (2).
 *
 * Usage:   siglip_rocket [artifact_dir]      (default ./siglip-artifacts)
 *   env    ROCKET_SIGLIP_BENCH=N             time N warm encodes (discard cold), print median ms
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_siglip.h"

static float *read_f32(const char *path, size_t want, const char **err)
{
    FILE *f = fopen(path, "rb");
    if (!f) { *err = "open"; return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz != (long)(want * sizeof(float))) { *err = "size"; fclose(f); return NULL; }
    float *b = malloc(want * sizeof(float));
    if (!b) { *err = "oom"; fclose(f); return NULL; }
    size_t got = fread(b, sizeof(float), want, f);
    fclose(f);
    if (got != want) { *err = "read"; free(b); return NULL; }
    return b;
}

/* cosine of an fp16 vector (got) against an fp32 vector (ref) */
static double cosine_f16_f32(const _Float16 *got, const float *ref, size_t n)
{
    double dot = 0, ng = 0, nr = 0;
    for (size_t i = 0; i < n; i++) {
        double g = (double)got[i], r = (double)ref[i];
        dot += g * r; ng += g * g; nr += r * r;
    }
    return dot / (sqrt(ng) * sqrt(nr) + 1e-30);
}

static double now_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static int cmp_d(const void *a, const void *b)
{ double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "./siglip-artifacts";
    char wpath[1024], ppath[1024], hpath[1024], lpath[1024];
    snprintf(wpath, sizeof wpath, "%s/siglip_weights.f16", dir);
    snprintf(ppath, sizeof ppath, "%s/pixels.f32", dir);
    snprintf(hpath, sizeof hpath, "%s/hidden.f32", dir);
    snprintf(lpath, sizeof lpath, "%s/postln.f32", dir);

    rocket_siglip_model m;
    int lrc = rocket_siglip_load(wpath, &m);
    if (lrc != 0) { printf("note: no weight blob %s (%d) — SKIP\n", wpath, lrc); return 2; }
    printf("siglip: d=%d layers=%d heads=%d d_ff=%d L=%d patch_dim=%d eps=%g\n",
           m.d, m.n_layers, m.n_head, m.d_ff, m.L, m.patch_dim, m.eps);

    const int d = m.d, L = m.L, nL = m.n_layers;
    const size_t Ld = (size_t)L * d;
    const size_t pix_n = (size_t)m.ic * m.image_size * m.image_size;

    const char *err = NULL;
    float *pix = read_f32(ppath, pix_n, &err);
    float *href = read_f32(hpath, (size_t)(nL + 1) * Ld, &err);
    float *lref = read_f32(lpath, Ld, &err);
    if (!pix || !href || !lref) {
        printf("note: missing/!size oracle in %s (%s) — SKIP\n", dir, err ? err : "?");
        free(pix); free(href); free(lref); rocket_siglip_free(&m); return 2;
    }

    int fd = rocket_open();
    if (fd < 0) {
        printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd);
        free(pix); free(href); free(lref); rocket_siglip_free(&m); return 2;
    }

    _Float16 *pix16 = malloc(pix_n * sizeof(_Float16));
    _Float16 *hid = malloc((size_t)(nL + 1) * Ld * sizeof(_Float16));
    _Float16 *out = malloc(Ld * sizeof(_Float16));
    if (!pix16 || !hid || !out) { printf("oom\n"); return 1; }
    for (size_t i = 0; i < pix_n; i++) pix16[i] = (_Float16)pix[i];

    int rc = rocket_siglip_encode(fd, &m, pix16, out, hid);
    if (rc != 0) { printf("encode rc=%d -> FAIL\n", rc); return 1; }

    /* per-layer cosine vs the oracle */
    double cos_emb = cosine_f16_f32(hid, href, Ld);
    printf("  layer  0 (embeddings)  cos=%.6f\n", cos_emb);
    double sum = 0; double worst = 1.0;
    for (int k = 1; k <= nL; k++) {
        double c = cosine_f16_f32(hid + (size_t)k * Ld, href + (size_t)k * Ld, Ld);
        printf("  layer %2d              cos=%.6f\n", k, c);
        sum += c; if (c < worst) worst = c;
    }
    double mean_layers = sum / nL;
    double cos_post = cosine_f16_f32(out, lref, Ld);
    printf("  post_layernorm        cos=%.6f\n", cos_post);
    printf("  MEAN over %d layers    cos=%.6f   (worst layer %.6f)\n", nL, mean_layers, worst);

    int ok = (mean_layers >= 0.99) && (cos_post >= 0.99);
    printf("  fidelity %s (target mean & post-LN >= 0.99)\n", ok ? "PASS" : "FAIL");

    /* optional: resident (prepacked, multicore) path — cross-check vs oracle + latency bench */
    const char *be = getenv("ROCKET_SIGLIP_BENCH");
    int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        const char *nt = getenv("ROCKET_SIGLIP_NTHREADS");
        int nthreads = nt ? atoi(nt) : 3;
        double tc = now_ms();
        rocket_siglip_ctx *c = rocket_siglip_ctx_create(&m, nthreads);
        tc = now_ms() - tc;
        if (!c) {
            printf("  resident ctx create FAILED (IOVA / fd?) — bench skipped\n");
        } else {
            printf("  resident ctx created in %.0f ms (packed %d static weights, nthreads=%d)\n",
                   tc, 1 + 6 * nL, nthreads);
            _Float16 *hid2 = malloc((size_t)(nL + 1) * Ld * sizeof(_Float16));
            int rc2 = rocket_siglip_encode_ctx(c, pix16, out, hid2);
            if (rc2 == 0 && hid2) {
                double s2 = 0;
                for (int kk = 1; kk <= nL; kk++)
                    s2 += cosine_f16_f32(hid2 + (size_t)kk * Ld, href + (size_t)kk * Ld, Ld);
                double cpost = cosine_f16_f32(out, lref, Ld);
                printf("  resident path: mean-layer cos=%.6f  post-LN cos=%.6f  %s\n",
                       s2 / nL, cpost, (s2 / nL >= 0.99 && cpost >= 0.99) ? "OK" : "DRIFT");
            } else {
                printf("  resident encode rc=%d\n", rc2);
            }
            free(hid2);

            double cold = now_ms();
            rocket_siglip_encode_ctx(c, pix16, out, NULL);
            cold = now_ms() - cold;
            double *t = malloc((size_t)bn * sizeof(double));
            for (int i = 0; i < bn; i++) {
                double s = now_ms();
                rocket_siglip_encode_ctx(c, pix16, out, NULL);
                t[i] = now_ms() - s;
            }
            qsort(t, bn, sizeof(double), cmp_d);
            printf("  bench (resident): cold=%.1f ms | warm median=%.1f ms (min %.1f, max %.1f, n=%d) = %.3f img/s\n",
                   cold, t[bn / 2], t[0], t[bn - 1], bn, 1000.0 / t[bn / 2]);
            free(t);
            rocket_siglip_ctx_free(c);
        }
    }

    free(pix16); free(hid); free(out); free(pix); free(href); free(lref);
    rocket_close(fd); rocket_siglip_free(&m);
    printf("==== %s ====\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
