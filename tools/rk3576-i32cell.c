/* ONE cell of the int32 K-split ladder, one per PROCESS (rule 84).
 *
 * Calls rocket_matmul_int8_rk3576_i32() directly — the route the int8 entry falls onto
 * once no single-task plan exists — scores the surface against an exact CPU int32 GEMM,
 * and prints the entry's own repair columns. It exists because a cell in this class can
 * take the device down, so nothing may share a process with it and nothing may be
 * inferred from a cell that ran after one.
 *
 *   i32cell M K N [seed]
 *
 * Exit: 0 exact, 1 wrong surface, 2 the entry refused, 3 setup. Scratch, not a gate. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    int M, K, N, fd, rc, i, j;
    unsigned seed = 1;
    int8_t *A, *B;
    int32_t *C, *ref;
    rocket_rk3576_i32_stats st;
    size_t wrong = 0, nz = 0;
    double t0, t1;

    if (argc < 4) { fprintf(stderr, "usage: i32cell M K N [seed]\n"); return 3; }
    M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]);
    if (argc > 4) seed = (unsigned)atoi(argv[4]);

    A = malloc((size_t)M * K);
    B = malloc((size_t)N * K);
    C = calloc((size_t)M * N, sizeof *C);
    ref = calloc((size_t)M * N, sizeof *ref);
    if (!A || !B || !C || !ref) { fprintf(stderr, "oom\n"); return 3; }

    srand(seed);
    for (i = 0; i < M * K; i++) A[i] = (int8_t)((rand() % 15) - 7);
    for (i = 0; i < N * K; i++) B[i] = (int8_t)((rand() % 15) - 7);

    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no /dev/accel\n"); return 3; }

    printf("CELL M=%d K=%d N=%d seed=%u\n", M, K, N, seed);
    fflush(stdout);

    t0 = now_ms();
    rc = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, NULL, C);
    t1 = now_ms();
    rocket_rk3576_i32_last_stats(&st);

    printf("RC=%d  wall=%.1f ms  submits=%u tasks=%u redo_empty=%u redo_zeroed=%u "
           "atoms_empty=%u atoms_zeroed=%u accepted_zero=%u refused=%u\n",
           rc, t1 - t0, st.submits, st.tasks, st.redo_empty, st.redo_zeroed,
           st.atoms_empty, st.atoms_zeroed, st.accepted_zero, st.refused);
    fflush(stdout);
    if (rc != 0) { rocket_close(fd); return 2; }

    /* The reference is a naive O(M*N*K) GEMM and this part is one A72 thread, so past
     * about 1.5e9 MACs it scores a random SUBSET instead. A subset bounds a DROPPED-write
     * class well and says nothing about a defect confined to a few elements — which is
     * exactly the caveat that has to travel with the number. */
    {
        long long macs = (long long)M * N * K;
        int subset = macs > 1500000000LL;
        int cells = subset ? 20000 : M * N;
        size_t shown = 0;
        for (i = 0; i < M * N; i++) if (C[i] != 0) nz++;
        srand(seed ^ 0x5eedu);
        for (i = 0; i < cells; i++) {
            int r = subset ? rand() % M : i / N;
            int c = subset ? rand() % N : i % N;
            const int8_t *a = A + (size_t)r * K;
            const int8_t *b = B + (size_t)c * K;
            int32_t acc = 0;
            int k;
            for (k = 0; k < K; k++) acc += (int32_t)a[k] * (int32_t)b[k];
            if (C[(size_t)r * N + c] != acc) {
                wrong++;
                if (shown < 6) {
                    printf("   [%d,%d] got %d want %d\n", r, c,
                           C[(size_t)r * N + c], acc);
                    shown++;
                }
            }
        }
        printf("WRONG=%zu of %d scored (%s)   nonzero=%zu of %d\n",
               wrong, cells, subset ? "RANDOM SUBSET" : "whole surface", nz, M * N);
    }
    (void)ref;
    fflush(stdout);
    rocket_close(fd);
    return wrong ? 1 : 0;
}
