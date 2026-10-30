import sys
import time
from multiprocessing import Pool


def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers, total_items=1_000_000, chunk_size=1):
    numbers = range(1, total_items + 1)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        # chunk_size=1 -> one IPC task envelope per number (micro-tasks)
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    return time.perf_counter() - t0, sum(results)


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    cores = [1, 2, 4, 8]
    for chunk in (1, 10_000):
        label = "fine (k=1)" if chunk == 1 else "coarse (k=10000)"
        times, checks = {}, set()
        for p in cores:
            t, s = run_granularity(p, total, chunk)
            times[p] = t
            checks.add(s)
        assert len(checks) == 1, "checksum mismatch between runs"
        t1 = times[1]
        print(f"\n--- Exp A: {label}, N={total:,} ---")
        print(f"T1={t1:.3f}s  checksum={checks.pop()}")
        for p in cores[1:]:
            tp = times[p]
            print(f"P={p:2d} | Time={tp:7.3f}s | Speedup={t1/tp:5.2f}x | "
                  f"Eff={t1/tp/p*100:5.1f}% | Karp-Flatt e={karp_flatt(t1, tp, p):6.3f}")
