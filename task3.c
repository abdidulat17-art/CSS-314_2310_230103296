// Task 3: false sharing demonstration.
// Build: gcc -O2 -pthread -o task3 task3.c
// Run:   ./task3 [threads]   (default = physical core count)
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ITERS 50000000UL
#define REPEATS 5
#define MAX_THREADS 256
#define CACHE_LINE 64

typedef struct { volatile unsigned long v; } plain_t;
typedef struct { volatile unsigned long v; char pad[CACHE_LINE - sizeof(unsigned long)]; }
    __attribute__((aligned(CACHE_LINE))) padded_t;

static plain_t  unpadded[MAX_THREADS] __attribute__((aligned(CACHE_LINE)));
static padded_t padded[MAX_THREADS];
static volatile unsigned long local_result[MAX_THREADS]; // written once at the end
static int mode;                                          // 0 unpadded, 1 padded, 2 local
static pthread_barrier_t barrier;

static int physical_cores(void) {
    // Count unique (package, core) pairs from sysfs; fall back to logical CPUs.
    int logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
    static char seen[4096][2];
    int n = 0;
    char path[256];
    for (int c = 0; c < logical && c < 4096; c++) {
        int pkg = 0, core = c;
        FILE *f;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        if ((f = fopen(path, "r"))) { if (fscanf(f, "%d", &pkg) != 1) pkg = 0; fclose(f); }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        if ((f = fopen(path, "r"))) { if (fscanf(f, "%d", &core) != 1) core = c; fclose(f); }
        int dup = 0;
        for (int i = 0; i < n; i++)
            if (seen[i][0] == (char)pkg && seen[i][1] == (char)core) { dup = 1; break; }
        if (!dup && n < 4096) { seen[n][0] = (char)pkg; seen[n][1] = (char)core; n++; }
    }
    return n > 0 ? n : logical;
}

static void cpu_model(char *out, size_t len) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[512];
    snprintf(out, len, "unknown");
    if (!f) return;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "model name", 10)) {
            char *p = strchr(line, ':');
            if (p) { snprintf(out, len, "%s", p + 2); out[strcspn(out, "\n")] = 0; }
            break;
        }
    fclose(f);
}

static void *worker(void *arg) {
    long i = (long)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((int)(i % sysconf(_SC_NPROCESSORS_ONLN)), &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);   // one thread per core
    pthread_barrier_wait(&barrier);                             // start together

    if (mode == 0) {
        for (unsigned long k = 0; k < ITERS; k++) unpadded[i].v++;   // adjacent 8-byte slots: same cache line
    } else if (mode == 1) {
        for (unsigned long k = 0; k < ITERS; k++) padded[i].v++;     // 64-byte separated slots
    } else {
        unsigned long local = 0;                                     // private accumulator
        for (unsigned long k = 0; k < ITERS; k++) { local++; __asm__ volatile("" : "+r"(local)); }
        local_result[i] = local;                                     // single shared write at the end
    }
    return NULL;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double run(int m, int threads) {
    pthread_t th[MAX_THREADS];
    mode = m;
    memset((void *)unpadded, 0, sizeof unpadded);
    memset((void *)padded, 0, sizeof padded);
    memset((void *)local_result, 0, sizeof local_result);
    pthread_barrier_init(&barrier, NULL, threads + 1);
    for (long i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    pthread_barrier_wait(&barrier);
    double t0 = now_ms();
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double t = now_ms() - t0;
    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < threads; i++) {
        unsigned long got = m == 0 ? unpadded[i].v : m == 1 ? padded[i].v : local_result[i];
        if (got != ITERS) { fprintf(stderr, "count mismatch in slot %d: %lu\n", i, got); exit(1); }
    }
    return t;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_run(int m, int threads) {
    double t[REPEATS];
    for (int r = 0; r < REPEATS; r++) t[r] = run(m, threads);
    qsort(t, REPEATS, sizeof(double), cmp);
    return t[REPEATS / 2];
}

int main(int argc, char **argv) {
    const char *sid = "230103296";
    int logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int threads = physical_cores();
    if (argc > 1) threads = atoi(argv[1]);
    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    char cpu[256];
    cpu_model(cpu, sizeof cpu);

    printf("Student ID: %s\n", sid);
    printf("CPU: %s\n", cpu);
    printf("Logical cores: %d | Physical cores: %d | Threads used: %d\n", logical, physical_cores(), threads);
    printf("Increments per thread: %lu | Cache line: %d bytes | Median of %d runs\n\n", ITERS, CACHE_LINE, REPEATS);

    run(1, threads);  // warm-up

    double t_unpadded = median_run(0, threads);
    double t_padded   = median_run(1, threads);
    double t_local    = median_run(2, threads);

    printf("%-34s | %12s | %s\n", "Variant", "Wall Time ms", "Ratio vs Unpadded");
    printf("-----------------------------------+--------------+------------------\n");
    printf("%-34s | %12.2f | %17s\n", "Unpadded shared (false sharing)", t_unpadded, "1.00x");
    printf("%-34s | %12.2f | %16.2fx\n", "Padded shared (64-byte slots)", t_padded, t_unpadded / t_padded);
    printf("%-34s | %12.2f | %16.2fx\n", "Local variable, one final write", t_local, t_unpadded / t_local);
    printf("\nExecution ratio Unpadded / Padded: %.2fx slower with false sharing\n", t_unpadded / t_padded);
    return 0;
}
