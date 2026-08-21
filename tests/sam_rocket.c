// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * sam_rocket.c — gate + bench for the SAM ViT-Det image encoder (facebook/sam-vit-base) on
 * the NPU (rocket_sam_encode): patch-embed -> +pos -> 12x block (windowed + periodic global
 * attention with a decomposed relative-position bias) -> conv neck.
 *
 * Validates per-layer cosine vs the HF fp32 reference dumped by tools/sam_extract.py (the
 * 13 hidden states + the neck output). It ALWAYS runs the exact host reference (fd<0), which
 * needs no NPU -- the off-device datapath self-check runs on x86; then, if /dev/accel/accel0
 * is present, it runs the NPU one-shot and (with ROCKET_SAM_BENCH=N) the resident ctx path.
 * PASS = mean-layer cos >= 0.99 AND neck cos >= 0.99. Missing artifacts -> SKIP (2).
 *
 *   ./sam_rocket [artifacts-dir]            # default ./sam-artifacts
 *   env ROCKET_SAM_BENCH=N                  # time N warm resident encodes, print median ms
 *   env ROCKET_SAM_NTHREADS=3               # resident worker fds (NPU cores)
 *   env ROCKET_SAM_HOSTONLY=1               # skip the NPU path even if present (x86 gate)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_sam.h"
#include "rocket_npu.h"

static float *read_f32(const char *path, size_t n, const char **err)
{
    FILE *f = fopen(path, "rb");
    if (!f) { *err = "open"; return NULL; }
    float *b = malloc(n * sizeof(float));
    if (!b) { *err = "oom"; fclose(f); return NULL; }
    size_t got = fread(b, sizeof(float), n, f);
    fclose(f);
    if (got != n) { *err = "size"; free(b); return NULL; }
    return b;
}

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
{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6; }

static int cmp_d(const void *a, const void *b)
{ double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* score a (nL+1)-hidden + neck run against the refs; returns mean-layer cos, sets *neck_cos. */
static double score(const rocket_sam_model *m, const _Float16 *hid, const _Float16 *out,
                    const float *href, const float *nref, const char *tag, double *neck_cos)
{
    const size_t Gd = (size_t)m->grid * m->grid * m->d;
    const size_t Nn = (size_t)m->neck_out * m->grid * m->grid;
    printf("  [%s] layer  0 (embeddings)  cos=%.6f\n", tag, cosine_f16_f32(hid, href, Gd));
    double sum = 0, worst = 1.0;
    for (int k = 1; k <= m->n_layers; k++) {
        double c = cosine_f16_f32(hid + (size_t)k * Gd, href + (size_t)k * Gd, Gd);
        printf("  [%s] layer %2d %-7s     cos=%.6f\n", tag, k,
               m->windowed[k - 1] ? "(win)" : "(GLOBAL)", c);
        sum += c; if (c < worst) worst = c;
    }
    *neck_cos = cosine_f16_f32(out, nref, Nn);
    double mean = sum / m->n_layers;
    printf("  [%s] neck output          cos=%.6f\n", tag, *neck_cos);
    printf("  [%s] MEAN over %d layers   cos=%.6f  (worst %.6f)\n", tag, m->n_layers, mean, worst);
    return mean;
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "./sam-artifacts";
    char wpath[1024], ppath[1024], hpath[1024], npath[1024];
    snprintf(wpath, sizeof wpath, "%s/sam_weights.f16", dir);
    snprintf(ppath, sizeof ppath, "%s/pixels.f32", dir);
    snprintf(hpath, sizeof hpath, "%s/hidden.f32", dir);
    snprintf(npath, sizeof npath, "%s/neck.f32", dir);

    rocket_sam_model m;
    int lrc = rocket_sam_load(wpath, &m);
    if (lrc != 0) { printf("note: no weight blob %s (%d) — SKIP\n", wpath, lrc); return 2; }
    printf("sam: d=%d layers=%d heads=%d dhead=%d d_ff=%d grid=%d win=%d neck_out=%d eps=%g\n",
           m.d, m.n_layers, m.n_head, m.dhead, m.d_ff, m.grid, m.win, m.neck_out, m.eps);

    const size_t Gd = (size_t)m.grid * m.grid * m.d;
    const size_t Nn = (size_t)m.neck_out * m.grid * m.grid;
    const size_t pix_n = (size_t)m.ic * m.image_size * m.image_size;

    const char *err = NULL;
    float *pix  = read_f32(ppath, pix_n, &err);
    float *href = read_f32(hpath, (size_t)(m.n_layers + 1) * Gd, &err);
    float *nref = read_f32(npath, Nn, &err);
    if (!pix || !href || !nref) {
        printf("note: missing/!size refs in %s (%s) — SKIP\n", dir, err ? err : "?");
        free(pix); free(href); free(nref); rocket_sam_free(&m); return 2;
    }

    _Float16 *pix16 = malloc(pix_n * sizeof(_Float16));
    _Float16 *hid   = malloc((size_t)(m.n_layers + 1) * Gd * sizeof(_Float16));
    _Float16 *out   = malloc(Nn * sizeof(_Float16));
    if (!pix16 || !hid || !out) { printf("oom\n"); return 1; }
    for (size_t i = 0; i < pix_n; i++) pix16[i] = (_Float16)pix[i];

    int ok = 1;
    double neck_cos;

    /* Off-device (or ROCKET_SAM_HOSTONLY): run the exact host reference (fd<0) as the datapath
     * self-check. On the NPU, run the device path directly -- it validates the same datapath
     * against the same refs, and the naive-scalar host_matmul reference is far too slow at
     * 1024px to run as a device-side redundancy. */
    const char *ho = getenv("ROCKET_SAM_HOSTONLY");
    int fd = (ho && atoi(ho)) ? -1 : rocket_open();
    if (fd < 0) {
        printf("=== host reference (fd<0) ===\n");
        int rc = rocket_sam_encode(-1, &m, pix16, out, hid);
        if (rc != 0) { printf("host encode rc=%d -> FAIL\n", rc); return 1; }
        double hmean = score(&m, hid, out, href, nref, "host", &neck_cos);
        ok &= (hmean >= 0.99) && (neck_cos >= 0.99);
        printf("note: no NPU (host-only run) — device gate skipped\n");
    } else {
        printf("=== NPU one-shot (fd>=0) ===\n");
        double t0 = now_ms();
        int rc = rocket_sam_encode(fd, &m, pix16, out, hid);
        double t1 = now_ms();
        if (rc != 0) { printf("npu encode rc=%d -> FAIL\n", rc); rocket_close(fd); return 1; }
        double nmean = score(&m, hid, out, href, nref, "npu", &neck_cos);
        printf("  one-shot encode %.0f ms\n", t1 - t0);
        ok &= (nmean >= 0.99) && (neck_cos >= 0.99);

        const char *be = getenv("ROCKET_SAM_BENCH");
        int bn = be ? atoi(be) : 0;
        const char *nt = getenv("ROCKET_SAM_NTHREADS");
        int nthreads = nt ? atoi(nt) : 3;
        printf("=== resident ctx (nthreads=%d) ===\n", nthreads);
        double tc = now_ms();
        rocket_sam_ctx *c = rocket_sam_ctx_create(&m, nthreads);
        tc = now_ms() - tc;
        if (!c) {
            printf("  resident ctx create FAILED — skipping\n");
        } else {
            printf("  resident ctx created in %.0f ms\n", tc);
            rc = rocket_sam_encode_ctx(c, pix16, out, hid);
            if (rc == 0) {
                double rmean = score(&m, hid, out, href, nref, "res", &neck_cos);
                ok &= (rmean >= 0.99) && (neck_cos >= 0.99);
            } else printf("  resident encode rc=%d -> FAIL\n", rc);
            if (bn > 0 && rc == 0) {
                double *ts = malloc(bn * sizeof(double));
                for (int i = 0; i < bn; i++) {
                    double s = now_ms();
                    rocket_sam_encode_ctx(c, pix16, out, NULL);
                    ts[i] = now_ms() - s;
                }
                qsort(ts, bn, sizeof(double), cmp_d);
                printf("  warm resident encode median %.1f ms over %d runs (discard run 0 cold)\n",
                       ts[bn / 2], bn);
                free(ts);
            }
            rocket_sam_ctx_free(c);
        }
        rocket_close(fd);
    }

    printf("fidelity %s (target mean-layer & neck cos >= 0.99)\n", ok ? "PASS" : "FAIL");
    free(pix); free(href); free(nref); free(pix16); free(hid); free(out);
    rocket_sam_free(&m);
    return ok ? 0 : 1;
}
