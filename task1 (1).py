"""Task 1: row-major vs. column-major (strided) parallel cache traversal.

8000 x 8000 float64 matrix (C order: each row is contiguous in memory), summed by one worker thread
per physical core in two ways:
  Mode A: workers take a band of rows and walk each row linearly (consecutive addresses)
  Mode B: workers take a band of columns and walk each column top to bottom (8000 * 8 bytes = 64,000-byte jumps)
Both modes make the same number of NumPy calls (8000) over the same number of elements, so the
difference is the memory access pattern.
"""
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103296"
N = 8000
REPEATS = 5


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def physical_cores():
    logical = os.cpu_count() or 1
    seen = set()
    for c in range(logical):
        base = f"/sys/devices/system/cpu/cpu{c}/topology/"
        try:
            with open(base + "physical_package_id") as f:
                pkg = int(f.read())
            with open(base + "core_id") as f:
                core = int(f.read())
        except (OSError, ValueError):
            return logical
        seen.add((pkg, core))
    return len(seen) or logical


def sum_rows(m, lo, hi):
    """Mode A: for each row in [lo, hi), sum it linearly along the columns."""
    total = 0.0
    for i in range(lo, hi):
        total += float(m[i, :].sum())
    return total


def sum_cols(m, lo, hi):
    """Mode B: for each column in [lo, hi), sum it across the rows (stride = N * 8 bytes)."""
    total = 0.0
    for j in range(lo, hi):
        total += float(m[:, j].sum())
    return total


def parallel(fn, m, workers, pool):
    edges = [N * w // workers for w in range(workers + 1)]
    futs = [pool.submit(fn, m, edges[w], edges[w + 1]) for w in range(workers)]
    return sum(f.result() for f in futs)


def bench(fn, m, workers, pool):
    result = parallel(fn, m, workers, pool)          # warm-up
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = parallel(fn, m, workers, pool)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], result


def main():
    logical = os.cpu_count()
    workers = physical_cores()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical core count: {logical} | Physical cores used as workers: {workers}")
    print(f"Matrix: {N} x {N} float64 = {N * N * 8 / 2**20:.0f} MiB, row-major (C order), seed {int(STUDENT_ID)}")
    print(f"Row stride: {N * 8:,} bytes | Median of {REPEATS} runs | ThreadPoolExecutor, {workers} workers\n")

    m = np.random.default_rng(int(STUDENT_ID)).random((N, N), dtype=np.float64)
    reference = float(m.sum())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        t_a, s_a = bench(sum_rows, m, workers, pool)
        t_b, s_b = bench(sum_cols, m, workers, pool)

    assert abs(s_a - reference) < 1e-9 * reference, "Mode A sum mismatch"
    assert abs(s_b - reference) < 1e-9 * reference, "Mode B sum mismatch"

    print(f"Mode A (row-major, linear rows):       {t_a:>10.2f} ms   sum = {s_a:.4f}")
    print(f"Mode B (column-major, strided cols):   {t_b:>10.2f} ms   sum = {s_b:.4f}")
    print(f"Cache penalty factor (Mode B / Mode A): {t_b / t_a:.2f}x")


if __name__ == "__main__":
    main()
