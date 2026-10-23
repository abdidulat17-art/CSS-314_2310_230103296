import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Same Student ID / seed / array as Task 1
STUDENT_ID = "230103296"
N = 20_000_000
P = 0.95          # parallelizable fraction for Amdahl's Law
REPEATS = 5       # each configuration is run several times, median is reported


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def chunk_sum(chunk):
    # NumPy releases the GIL inside np.sum, so threads run truly in parallel.
    return float(np.sum(chunk))


def parallel_sum(data, workers):
    bounds = np.linspace(0, len(data), workers + 1, dtype=np.int64)
    chunks = [data[bounds[i]:bounds[i + 1]] for i in range(workers)]
    if workers == 1:
        return chunk_sum(chunks[0])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(chunk_sum, chunks))


def amdahl(p_workers, parallel_fraction=P):
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p_workers)


def bench(data, workers):
    times = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = parallel_sum(data, workers)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], result


def main():
    seed = int("".join(ch for ch in STUDENT_ID if ch.isdigit()))
    max_cores = os.cpu_count()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores (os.cpu_count): {max_cores}")
    print(f"Array: {N} float64, seed {seed}; Amdahl P = {P}; median of {REPEATS} runs")
    print("Parallelization: ThreadPoolExecutor, one contiguous chunk per worker\n")

    data = np.random.default_rng(seed).random(N, dtype=np.float64)
    parallel_sum(data, 1)  # warm-up

    # [1, 2, 4, max_cores, max_cores * 2], duplicates removed (keeps order)
    counts = list(dict.fromkeys([1, 2, 4, max_cores, max_cores * 2]))

    results = []
    for w in counts:
        t_ms, s = bench(data, w)
        results.append((w, t_ms, s))

    t1 = results[0][1]
    ref = results[0][2]

    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for w, t_ms, s in results:
        print(f"{w:>6} | {t_ms:>14.2f} | {t1 / t_ms:>8.2f} | {amdahl(w):>18.2f}")

    print(f"\nAmdahl limit for p -> infinity: {1 / (1 - P):.1f}x")
    ok = all(abs(s - ref) < 1e-3 * abs(ref) for _, _, s in results)
    print(f"Sums consistent across worker counts: {ok} (sum = {ref:.6f})")


if __name__ == "__main__":
    main()
