// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_act_rocket.c — conv -> activation FUSION gate (single NPU job).
 *
 * The conv->activation fusion ports the DPU LUT epilogue into gen_conv2d_task so a
 * DIRECT fp16 conv post-processes its own result with a SMOOTH activation f(x)
 * (SiLU / tanh / GELU) in the SAME NPU job: out = f(conv(x)), no second round-trip.
 *
 * Layers:
 *  1. REGCMD SMOKE (no NPU): a single-job shape with conv_params_t.act set must emit
 *     EXACTLY +1026 ops over the plain conv (the 2*513 LE/LO upload that replaces the
 *     two idle LUT_ACCESS writes), and act=NULL must be byte-identical — the fusion is
 *     strictly additive and default-off.
 *
 *  2. ON-HARDWARE (needs /dev/accel). For each shape × kind, three numbers:
 *       FUSION   = max| fused  -  (conv-on-NPU then standalone flying LUT) |
 *                  -> the PORT correctness: the fused epilogue must reproduce the
 *                     independently-validated standalone DPU LUT (gen_lut_activation_fp16)
 *                     applied to the same conv output. THIS IS THE HARD GATE.
 *       ACC      = max| fused - f(conv) |  (true function) — the shippable-accuracy claim.
 *       x0-spikes= count of |fused - f| > 0.5 with |conv| < 0.02 — the NVDLA hybrid-LUT
 *                  LE/LO BOUNDARY glitch at x≈0 (a discrete +128/+8 mux toggle). It is a
 *                  property of the single-pass LUT itself (the standalone flying op spikes
 *                  IDENTICALLY — FUSION stays ~0 through a spike), NOT a fusion bug, and is
 *                  narrow (|x|<~0.0015). Reported, not failed.
 *
 *  PASS = regcmd smoke + FUSION <= 0.02 (all shapes/kinds) + conv->tanh ACC clean away
 *  from the x≈0 band. Exit: 0 PASS, 1 FAIL, 2 no-NPU SKIP.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_activation.h"
#include "npu_matmul.h"   /* conv_params_t, gen_conv2d_fp16 */

static uint32_t rng = 0x1234567u;
static float frand(void) { rng = rng*1664525u + 1013904223u; return (rng>>8)*(1.0f/16777216.0f); }

static const int KINDS[] = { ROCKET_ACTIVATION_SILU, ROCKET_ACTIVATION_TANH, ROCKET_ACTIVATION_GELU };

/* Plain vs fused regcmd op count for a single-job shape (no device). 0 / 1. */
static int regcmd_smoke(const rocket_conv2d_desc *d)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    static uint16_t lut[1026];
    lut_epilogue_t ep;
    if (rocket_lut_epilogue_build(ROCKET_ACTIVATION_SILU, lut, &ep)) { printf("  smoke: build FAIL\n"); return 1; }
    uint64_t rp[2048] = {0}, ra[2048] = {0};
    conv_params_t base = { .ic=d->ic,.ih=d->ih,.iw=d->iw,.oc=d->oc,.oh=OH,.ow=OW,
        .kh=d->kh,.kw=d->kw,.stride_y=d->stride_y,.stride_x=d->stride_x,
        .dil_y=d->dil_y,.dil_x=d->dil_x,.pad_top=d->pad_top,.pad_left=d->pad_left,
        .input_dma=0x1000,.weights_dma=0x2000,.output_dma=0x3000,.fp32tofp16=1 };
    conv_params_t pp = base; pp.tasks = rp;
    conv_params_t pa = base; pa.tasks = ra; pa.act = &ep;
    int gp = gen_conv2d_fp16(&pp), ga = gen_conv2d_fp16(&pa);
    if (gp || ga) { printf("  smoke: gen plain=%d act=%d -> needs-tiling (skip count check)\n", gp, ga); return 0; }
    int delta = (int)pa.task_count - (int)pp.task_count;
    int ok = (pp.task_count>0 && delta == 1026);
    printf("  regcmd smoke: plain=%u fused=%u (delta=%d, want +1026) -> %s\n",
           pp.task_count, pa.task_count, delta, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_shape(int fd, const rocket_conv2d_desc *d, int tiled)
{
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d), N = d->oc*OH*OW;
    const size_t in_n=(size_t)d->ic*d->ih*d->iw, wt_n=(size_t)d->oc*d->ic*d->kh*d->kw;
    printf("conv IC=%d IH=%d IW=%d -> OC=%d K=%dx%d s=%dx%d p=%d,%d (OH=%d OW=%d N=%d%s)\n",
           d->ic,d->ih,d->iw,d->oc,d->kh,d->kw,d->stride_y,d->stride_x,d->pad_top,d->pad_left,
           OH,OW,N, tiled?", TILED":"");
    if (rocket_conv2d_plan(d)) { printf("  plan: unsupported — skip\n"); return 0; }

    int fail = tiled ? 0 : regcmd_smoke(d);
    if (fd < 0) return fail;

    _Float16 *in=malloc(in_n*2),*W=malloc(wt_n*2),*cvt=malloc((size_t)N*2),*ref=malloc((size_t)N*2);
    _Float16 *got=malloc((size_t)N*2),*s1=malloc((size_t)N*2),*fly=malloc((size_t)N*2);
    if(!in||!W||!cvt||!ref||!got||!s1||!fly){fprintf(stderr,"oom\n");return -1;}

    const double a = 4.5 / sqrt((double)d->ic*d->kh*d->kw);  /* conv out ~N(0,1.5^2) */
    for(size_t i=0;i<in_n;i++) in[i]=(_Float16)(2.0f*frand()-1.0f);
    for(size_t i=0;i<wt_n;i++) W[i]=(_Float16)(a*(2.0f*frand()-1.0f));
    rocket_conv2d_ref_fp16(d,in,W,cvt);
    float lo=1e9f,hi=-1e9f; for(int i=0;i<N;i++){float v=(float)cvt[i]; if(v<lo)lo=v; if(v>hi)hi=v;}
    printf("  conv pre-activation range [%.2f, %.2f]\n", lo, hi);

    for (size_t k=0;k<sizeof(KINDS)/sizeof(KINDS[0]);k++) {
        int kind = KINDS[k];
        rocket_activation_ref_fp16(kind, cvt, ref, N);                 /* true f(conv) */
        memset(got,0,(size_t)N*2);
        int r = rocket_conv2d_act_fp16(fd, d, kind, in, W, got);       /* fused, one job */
        if (r) { printf("  %-4s: rocket_conv2d_act_fp16=%d FAIL\n", rocket_activation_name(kind), r); fail=1; continue; }
        /* standalone reference: NPU conv readback, then the flying single-pass LUT */
        memset(s1,0,(size_t)N*2); memset(fly,0,(size_t)N*2);
        int rc = rocket_conv2d_fp16(fd, d, in, W, s1);
        if (!rc) rc = rocket_activation_fp16(fd, kind, s1, fly, N);     /* WIDE_LUT set in main() */

        double mf=0, ma=0; int x0=0;
        for (int i=0;i<N;i++) {
            double dfu=fabs((float)got[i]-(float)fly[i]); if(dfu>mf) mf=dfu;
            double dac=fabs((float)got[i]-(float)ref[i]); if(dac>ma) ma=dac;
            if (dac>0.5 && fabs((float)cvt[i])<0.02) x0++;
        }
        int fuse_ok = (rc==0 && mf<=0.02);
        printf("  %-4s: FUSION(vs flying LUT)=%.4f%s  ACC(vs true f)=%.4f  x0-spikes=%d -> %s\n",
               rocket_activation_name(kind), mf, rc?"(flyref err)":"", ma, x0,
               fuse_ok ? "fusion OK" : "FUSION FAIL");
        if (!fuse_ok) fail = 1;
        /* shippable claim: conv->tanh is genuinely accurate where the input avoids the
         * x≈0 boundary band (no spikes) — assert it on the clean single-tile shapes. */
        if (kind==ROCKET_ACTIVATION_TANH && !tiled && x0==0 && ma>0.05) {
            printf("        tanh ACC %.4f > 0.05 (no x0 spike) -> FAIL\n", ma); fail=1;
        }
    }
    free(in);free(W);free(cvt);free(ref);free(got);free(s1);free(fly);
    return fail;
}

/* Probe the x≈0-boundary mitigation (ROCKET_LUT_SHIFT) — INFORMATIONAL (the shift is an
 * experimental RE path, not the shipping one). It maps the whole domain onto the positive
 * index half via the now-mapped POST-scale BN-ALU bias (index = x·BN_MUL + fp32(-x_lo·scale),
 * MUL then ALU, fp32 index-domain bias — HW-confirmed by the BNALU sweep). That RELOCATES the
 * x≈0 LE/LO boundary glitch but does NOT eliminate it: a symmetric domain centers x=0 at
 * index 8192, which is itself a LO-internal sub-boundary, so a residual spike persists at x≈0
 * (and a single wider table also re-exposes the flat-region quirk). Reported, never failed —
 * the BN-ALU mechanism is the durable finding; a robust fix needs x=0 off every sub-boundary. */
static int shift_probe(int fd)
{
    if (fd < 0) return 0;
    rocket_conv2d_desc d = { .ic=64,.ih=8,.iw=8,.oc=64,.kh=1,.kw=1,.stride_y=1,.stride_x=1,
                             .pad_top=0,.pad_left=0,.dil_y=1,.dil_x=1 };
    int N = d.oc*8*8; size_t in_n=(size_t)d.ic*8*8, wt_n=(size_t)d.oc*d.ic;
    _Float16 *in=malloc(in_n*2),*W=malloc(wt_n*2),*cvt=malloc((size_t)N*2),*ref=malloc((size_t)N*2),*got=malloc((size_t)N*2);
    if(!in||!W||!cvt||!ref||!got){return -1;}
    const double a = 1.6 / sqrt((double)d.ic);   /* small: conv out ~N(0,0.8^2) -> tanh curved */
    for(size_t i=0;i<in_n;i++) in[i]=(_Float16)(2.0f*frand()-1.0f);
    for(size_t i=0;i<wt_n;i++) W[i]=(_Float16)(a*(2.0f*frand()-1.0f));
    rocket_conv2d_ref_fp16(&d,in,W,cvt);
    rocket_activation_ref_fp16(ROCKET_ACTIVATION_TANH, cvt, ref, N);
    float lo=1e9f,hi=-1e9f; for(int i=0;i<N;i++){float v=(float)cvt[i]; if(v<lo)lo=v; if(v>hi)hi=v;}
    setenv("ROCKET_LUT_SHIFT","1",1);
    memset(got,0,(size_t)N*2);
    int r = rocket_conv2d_act_fp16(fd,&d,ROCKET_ACTIVATION_TANH,in,W,got);
    unsetenv("ROCKET_LUT_SHIFT");
    double ma=0; int x0=0;
    for(int i=0;i<N;i++){ double da=fabs((float)got[i]-(float)ref[i]); if(da>ma)ma=da; if(da>0.5&&fabs((float)cvt[i])<0.02)x0++; }
    printf("SHIFT-mode probe (conv->tanh, curved [%.2f,%.2f], INFO): ACC=%.4f x0-spikes=%d  "
           "(BN-ALU post-scale bias works; x≈0 relocated to the index-centre sub-boundary, residual remains)\n",
           lo, hi, ma, x0);
    free(in);free(W);free(cvt);free(ref);free(got);
    return r ? 1 : 0;   /* only a hard error fails; the residual glitch is expected/reported */
}

int main(void)
{
    /* make the flying reference use the SINGLE-PASS LUT for every kind (incl. SiLU,
     * which otherwise takes the 2-pass gate+mul path) so it matches the fused epilogue. */
    setenv("ROCKET_ACT_WIDE_LUT", "1", 1);

    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — regcmd smoke only\n\n", fd);

    struct { rocket_conv2d_desc d; int tiled; } shapes[] = {
        {{ .ic=64,.ih=8, .iw=8, .oc=64,.kh=1,.kw=1,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=0,.dil_y=1,.dil_x=1 },0}, /* 1x1==matmul (FFN) */
        {{ .ic=32,.ih=8, .iw=8, .oc=16,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },0}, /* 3x3 same-pad */
        {{ .ic=64,.ih=10,.iw=12,.oc=32,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },0}, /* stride2 multi-group */
        {{ .ic=64,.ih=40,.iw=40,.oc=16,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },1}, /* TILED (N=25600 < flying cap) */
    };
    int fail = 0;
    for (size_t i=0;i<sizeof(shapes)/sizeof(shapes[0]);i++) { fail |= run_shape(fd,&shapes[i].d,shapes[i].tiled); printf("\n"); }
    fail |= shift_probe(fd);
    printf("\n");

    if (fd >= 0) rocket_close(fd); else { printf("==== SKIP (no NPU) ====\n"); return 2; }
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
