/* What the RK3576 int8 matmul entry PLANS and EMITS for a shape, with NO DEVICE.
 *
 * Every quantity this prints is pure: the route the entry will take, the K slice, the
 * plane, the output-channel tile, the submit count, the resident weight slice against the
 * cap it is compared to, and the regcmd words themselves. That makes it the instrument to
 * reach for FIRST on any question about a shape this entry mishandles — a shape in the K
 * split class can take the device down across processes, so a map that costs nothing is
 * worth having before a cell that costs a reboot.
 *
 * It includes the entry's translation unit so the static planners are visible. Build it
 * with every OTHER src/*.c beside it:
 *
 *   gcc -O1 -Iinclude -Isrc -o planprobe tools/rk3576-planprobe.c \
 *       $(ls src/ *.c | grep -v rocket_matmul_rk3576.c) -lm -lpthread
 *
 *   ./planprobe             the shape table, plan only
 *   ./planprobe -r          the same, plus the emitted regcmd for the first row task
 *   ./planprobe map         which (M, K, N) reach the K split, and the tile against K
 *   ./planprobe ladder      the K-split cells ordered by submit count, for a device arm
 *   ./planprobe cell M K N  one arbitrary shape, same detail as the table's rows
 *
 * It forces ROCKET_CHIP=rk3576 itself, so it runs anywhere. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_matmul.h"

#include "rocket_matmul_rk3576.c"

struct shape { int M, K, N; const char *note; };

static void dump_regs(const uint64_t *ops, unsigned n, const char *tag)
{
    unsigned i;
    printf("    -- %s: %u ops --\n", tag, n);
    for (i = 0; i < n; i++) {
        uint32_t reg = (uint32_t)(ops[i] >> 32);
        uint32_t val = (uint32_t)(ops[i] & 0xffffffffu);
        printf("      [%3u] 0x%04x = 0x%08x (%u)\n", i, reg & 0xffff, val, val);
    }
}

static void one(const struct shape *s, int dump)
{
    unsigned iw = 0, ih = 0, nt = 0, ks, mult;
    int ntile = 0, single;

    printf("== M=%d K=%d N=%d  (%s)\n", s->M, s->K, s->N, s->note);
    single = rocket_matmul_plan_int8_rk3576(s->M, s->K, s->N, NULL, NULL, &ntile);
    if (single >= 0) {
        printf("   ROUTE single-task int8, N tile %d, %d tiles\n", ntile, single);
        return;
    }
    printf("   ROUTE i32 K-split\n");

    mult = r76_i32_oc_mult((unsigned)s->M, (unsigned)s->K, (unsigned)s->N);
    ks = r76_i32_plan((unsigned)s->M, (unsigned)s->K, (unsigned)s->N, mult, &iw, &ih, &nt);
    if (!ks) { printf("   REFUSED: no K slice fits\n"); return; }
    {
        unsigned nslices = ((unsigned)s->K + ks - 1u) / ks;
        unsigned lastk = (unsigned)s->K % ks ? (unsigned)s->K % ks : ks;
        unsigned surf_elems = rocket_rk3576_out_surf_elems(iw, ih, 0);
        unsigned n0;
        size_t in_max = (size_t)((ks + C2 - 1) / C2) * ih * iw * C2;
        size_t in_slot = (in_max + 63u) & ~(size_t)63u;
        printf("   plane %ux%u  ks=%u  nslices=%u (last %u)  nt=%u  mult=%u\n",
               iw, ih, ks, nslices, lastk, nt, mult);
        printf("   feature BO = %zu (%zu KiB), slot %zu\n",
               in_slot * nslices, in_slot * nslices / 1024, in_slot);
        for (n0 = 0; n0 < (unsigned)s->N; n0 += nt) {
            unsigned tile_n = (unsigned)s->N - n0 < nt ? (unsigned)s->N - n0 : nt;
            unsigned oc_prog = rocket_rk3576_pad_oc(mult * tile_n);
            size_t w_bytes = (size_t)((oc_prog + 31) / 32) * ((ks + 31) / 32) * 32 * 32;
            size_t atoms_per_px = (mult == 4u ? 1u : 2u) * ((oc_prog + C2 - 1) / C2);
            size_t surf_bytes = atoms_per_px * surf_elems * C2;
            unsigned wslice = 32u * ks;                 /* r76_weight_resident_bytes k=1 */
            unsigned cube = ((oc_prog + 31u) / 32u) * wslice;
            unsigned rows = rocket_rk3576_max_task_rows(iw, ks, oc_prog, 1, 1, 0);
            printf("   tile n0=%u tile_n=%u oc_prog=%u  w=%zu (%zu KiB)  surf=%zu (%zu KiB)"
                   "  wslice=%u (%u KiB)  cube=%u (%u KiB)  rows=%u\n",
                   n0, tile_n, oc_prog, w_bytes, w_bytes / 1024,
                   surf_bytes, surf_bytes / 1024,
                   wslice, wslice / 1024, cube, cube / 1024, rows);
            if (n0 == 0 && dump) {
                uint64_t ops[RK3576_CONV_TASK_OPS];
                conv_params_t q = {0};
                rocket_rk3576_row_task rp[4096];
                unsigned ntask = 1;
                q.ic = (uint16_t)ks; q.ih = (uint16_t)ih; q.iw = (uint16_t)iw;
                q.oc = (uint16_t)oc_prog; q.oh = (uint16_t)ih; q.ow = (uint16_t)iw;
                q.kh = 1; q.kw = 1; q.stride_y = 1; q.stride_x = 1;
                q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
                if (rocket_rk3576_plan_rows(&q, 0, rp, 4096, &ntask) < 0)
                    printf("      NO ROW PLAN\n");
                else
                    printf("      row plan: %u tasks, task0 ih=%u oh=%u foff=%u ooff=%u\n",
                           ntask, rp[0].ih, rp[0].oh,
                           (unsigned)rp[0].feature_off, (unsigned)rp[0].output_off);
                q.ih = rp[0].ih; q.oh = rp[0].oh; q.pad_top = rp[0].pad_top;
                q.int8_out = 1;
                q.in_scale = 1.0f; q.w_scale = 1.0f; q.out_scale = 1.0f;
                q.input_zero_point = 0x80; q.output_zero_point = 0x80;
                q.weight_zero_point = 0x80;
                q.tasks = ops;
                q.input_dma = 0x10000000u; q.weights_dma = 0x20000000u;
                q.bias_dma = 0x30000000u; q.output_dma = 0x40000000u;
                if ((mult == 4u ? gen_conv2d_int8_rk3576_i32out(&q)
                                : gen_conv2d_int8_rk3576_i32out_wide(&q)) != 0)
                    printf("      EMITTER REFUSED\n");
                else
                    dump_regs(ops, q.task_count, "task 0");
            }
            if (n0 >= nt * 2) { printf("   ... (tiles repeat)\n"); break; }
        }
    }
}

/* THE ROUTE MAP. For each (M, N) the smallest K at which the entry stops finding a
 * single-task plan and falls onto the int32 K-split — which is the class the wedge was
 * observed in. K walks the 32-multiples the entry accepts. */
static void routemap(void)
{
    static const int Ms[] = { 64, 128, 256, 512, 1024, 2048 };
    static const int Ns[] = { 512, 1024, 2048, 4096, 8192 };
    unsigned mi, ni;
    printf("\n== route map: smallest K taking the int32 K-split (32-multiples, K<=32768)\n");
    printf("%6s", "M\\N");
    for (ni = 0; ni < sizeof Ns / sizeof Ns[0]; ni++) printf("%9d", Ns[ni]);
    printf("\n");
    for (mi = 0; mi < sizeof Ms / sizeof Ms[0]; mi++) {
        printf("%6d", Ms[mi]);
        for (ni = 0; ni < sizeof Ns / sizeof Ns[0]; ni++) {
            int K, hit = 0;
            for (K = 32; K <= 32768; K += 32) {
                int ntile = 0;
                if (rocket_matmul_plan_int8_rk3576(Ms[mi], K, Ns[ni], NULL, NULL,
                                                   &ntile) < 0) { hit = K; break; }
            }
            if (hit) printf("%9d", hit); else printf("%9s", "none");
        }
        printf("\n");
    }
}

/* What the single-task route costs as K grows: the N tile it has to shrink to, and the
 * submit count that follows. A tile that collapses is a shape that is legal and slow. */
static void tilewalk(int M, int N)
{
    int K;
    printf("\n== single-task N tile against K at M=%d N=%d\n", M, N);
    printf("%8s %8s %10s\n", "K", "Ntile", "submits");
    for (K = 1024; K <= 16384; K += 512) {
        int ntile = 0, subs = rocket_matmul_plan_int8_rk3576(M, K, N, NULL, NULL, &ntile);
        if (subs < 0) printf("%8d %8s %10s\n", K, "-", "K-SPLIT");
        else printf("%8d %8d %10d\n", K, ntile, subs);
    }
}

/* The K-split ladder: for each candidate device cell, what the i32 route will actually
 * do, and how many submits it is. Ordered so an arm can walk it smallest first. */
static void ladder(void)
{
    static const struct shape L[] = {
        { 512, 6176,   32, "smallest K in class, 1 N tile" },
        { 512, 6176,  256, "1 N tile at the wedge's tile width" },
        { 512, 8192,   32, "the wedge's K, 1 N tile" },
        { 512, 8192,  256, "the wedge's K, 1 tile at its width" },
        { 512, 6176, 2048, "smallest K, the wedge's N" },
        { 512, 8192, 2048, "THE WEDGE" },
        {  64, 8192,   32, "small M, the wedge's K" },
        { 128, 8192,  256, "small M" },
    };
    unsigned i;
    printf("\n== the i32 K-split ladder\n");
    printf("%6s %6s %6s %6s %8s %6s %5s %9s  %s\n",
           "M", "K", "N", "ks", "plane", "nt", "mult", "submits", "note");
    for (i = 0; i < sizeof L / sizeof L[0]; i++) {
        unsigned iw = 0, ih = 0, nt = 0, ks, mult, subs;
        char pl[16];
        mult = r76_i32_oc_mult((unsigned)L[i].M, (unsigned)L[i].K, (unsigned)L[i].N);
        ks = r76_i32_plan((unsigned)L[i].M, (unsigned)L[i].K, (unsigned)L[i].N,
                          mult, &iw, &ih, &nt);
        subs = r76_i32_submit_count((unsigned)L[i].M, (unsigned)L[i].K,
                                    (unsigned)L[i].N, mult);
        snprintf(pl, sizeof pl, "%ux%u", iw, ih);
        printf("%6d %6d %6d %6u %8s %6u %5u %9u  %s\n",
               L[i].M, L[i].K, L[i].N, ks, pl, nt, mult, subs, L[i].note);
    }
}

int main(int argc, char **argv)
{
    static const struct shape sh[] = {
        { 512, 2048,  512, "sgemm A/B clean" },
        { 512, 2048, 2048, "sgemm A/B clean" },
        {2048, 2048, 2048, "sgemm A/B clean" },
        { 512, 2048, 8192, "sgemm A/B clean" },
        { 512, 4096, 4096, "sgemm A/B clean" },
        { 256, 2048, 2048, "drain-deadline shape" },
        { 512, 4608, 2048, "K exactly at KS_MAX" },
        { 512, 4640, 2048, "K one group past KS_MAX -> 2 slices" },
        { 512, 8192, 2048, "THE WEDGE" },
    };
    unsigned i;
    int dump = argc > 1 && !strcmp(argv[1], "-r");
    setenv("ROCKET_CHIP", "rk3576", 1);
    setenv("ROCKET_LOG_LEVEL", "error", 1);
    printf("hw profile = %s\n", rocket_hw_current()->name);
    if (argc > 1 && !strcmp(argv[1], "map")) { routemap(); tilewalk(512, 2048);
                                               tilewalk(512, 4096); return 0; }
    if (argc > 1 && !strcmp(argv[1], "ladder")) { ladder(); return 0; }
    if (argc > 4 && !strcmp(argv[1], "cell")) {
        struct shape s = { atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), "cell" };
        one(&s, 0);
        if (rocket_matmul_plan_int8_rk3576(s.M, s.K, s.N, NULL, NULL, NULL) < 0) {
            unsigned mult = r76_i32_oc_mult((unsigned)s.M, (unsigned)s.K, (unsigned)s.N);
            printf("   submits=%u\n",
                   r76_i32_submit_count((unsigned)s.M, (unsigned)s.K,
                                        (unsigned)s.N, mult));
        }
        return 0;
    }
    for (i = 0; i < sizeof sh / sizeof sh[0]; i++) one(&sh[i], dump);
    return 0;
}
