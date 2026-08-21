/* rk3576-memhog — a DRAM-bandwidth load, so memory pressure can be DIALLED rather than
 * inferred from a context.
 *
 * It exists for one question: several RK3576 hazards read as properties of "the graph" or
 * of "a kick" when the only thing those contexts share is concurrent memory traffic. This
 * touches no NPU buffer and issues no ioctl, so it separates pressure on the memory system
 * from anything about the submit, the source surface's provenance or its cache state.
 *
 * One A72 thread already saturates this part's DRAM (half the RK3588's bandwidth at the
 * same 2112 MHz), so the useful dial is 1 to 4 threads and the top of the range is flat.
 *
 *   cc -O2 -pthread -o rk3576-memhog rk3576-memhog.c
 *   ./rk3576-memhog <threads> <seconds> [mib_per_thread]
 *
 * Prints the aggregate GB/s it achieved, which is what says the load was actually applied
 * — a hog whose own throughput is unknown cannot bound a negative.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile int stop;
static volatile char sink;
static size_t buf_bytes = 64u << 20;

struct worker { pthread_t t; unsigned long long bytes; };

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* MEMHOG_SPIN=1 keeps the cores just as busy and moves no memory — the control that says
 * whether an effect is about DRAM traffic or about the host having competition. Its
 * reported GB/s is ~0 by construction, which is the check that it applied nothing. */
static void *spin(void *arg)
{
    struct worker *w = arg;
    unsigned long long x = 1;

    while (!stop) {
        int i;
        for (i = 0; i < 4096; i++) x = x * 6364136223846793005ull + 1442695040888963407ull;
        sink = (char)x;
    }
    (void)w;
    return NULL;
}

static void *run(void *arg)
{
    struct worker *w = arg;
    char *a = malloc(buf_bytes), *b = malloc(buf_bytes);

    if (!a || !b) { free(a); free(b); return NULL; }
    memset(a, 0x5A, buf_bytes);
    memset(b, 0xA5, buf_bytes);
    /* The sink is not decoration: without a read of the destination the compiler deletes
     * the copy outright and the hog becomes a counter loop that reports terabytes a
     * second. Its own throughput is how you know it did not. */
    while (!stop) {
        memcpy(b, a, buf_bytes);
        sink = b[w->bytes % buf_bytes];
        w->bytes += buf_bytes * 2u; /* one read plus one write */
    }
    free(a); free(b);
    return NULL;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 1;
    double secs = argc > 2 ? atof(argv[2]) : 10.0;
    struct worker *w;
    unsigned long long total = 0;
    double t0, el;
    int i, spinning;

    if (argc > 3) buf_bytes = (size_t)atoi(argv[3]) << 20;
    if (n <= 0) { printf("memhog: 0 threads, nothing applied\n"); return 0; }
    w = calloc((size_t)n, sizeof *w);
    if (!w) return 1;

    spinning = getenv("MEMHOG_SPIN") && *getenv("MEMHOG_SPIN") != '0';
    t0 = now_s();
    for (i = 0; i < n; i++)
        pthread_create(&w[i].t, NULL, spinning ? spin : run, &w[i]);
    while (now_s() - t0 < secs) {
        struct timespec ts = { 0, 20 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    stop = 1;
    for (i = 0; i < n; i++) { pthread_join(w[i].t, NULL); total += w[i].bytes; }
    el = now_s() - t0;
    printf("memhog: %d %s thread(s), %.1f s, %.2f GB/s aggregate\n",
           n, spinning ? "SPIN" : "copy", el, (double)total / el / 1e9);
    return 0;
}
