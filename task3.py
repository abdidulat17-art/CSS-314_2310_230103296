"""Task 3: prefix scan, step efficiency vs. work efficiency.

Serial inclusive scan:   output[i] = output[i-1] + input[i]
Parallel chunked scan (3 phases around the 2 passes):
  pass 1 (parallel): each worker computes the sum of its chunk
  offsets (serial, tiny): exclusive prefix sum of the chunk sums
  pass 2 (parallel): each worker scans its chunk and adds its offset
The comparison uses the same primitive (np.cumsum, one thread) for the serial baseline, so the speedup measures
the parallel structure, not NumPy vs. a Python loop. A literal Python-loop timing is printed for reference.
"""
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103296"
N = 2_000_000
REPEATS = 15


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def serial_scan(x):
    return np.cumsum(x, dtype=np.int64)


def serial_scan_python_loop(values):
    out = [0] * len(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = out[i - 1] + values[i]
    return out


def chunk_sum(x, lo, hi):
    return int(x[lo:hi].sum(dtype=np.int64))


def chunk_scan(x, out, lo, hi, offset):
    np.cumsum(x[lo:hi], dtype=np.int64, out=out[lo:hi])
    if offset:
        out[lo:hi] += offset


def parallel_scan(x, workers, pool):
    n = len(x)
    edges = [n * w // workers for w in range(workers + 1)]
    out = np.empty(n, dtype=np.int64)

    sums = [f.result() for f in [pool.submit(chunk_sum, x, edges[w], edges[w + 1]) for w in range(workers)]]   # pass 1

    offsets = [0] * workers                      # offset reduction: exclusive prefix over chunk sums
    for w in range(1, workers):
        offsets[w] = offsets[w - 1] + sums[w - 1]

    for f in [pool.submit(chunk_scan, x, out, edges[w], edges[w + 1], offsets[w]) for w in range(workers)]:   # pass 2
        f.result()
    return out


def median_ms(fn):
    fn()  # warm-up
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    workers = os.cpu_count() or 1
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()} | Parallel workers (threads): {workers}")
    print(f"Array: {N:,} int64 values in [1, 10], seed {int(STUDENT_ID)} | Median of {REPEATS} runs\n")

    x = np.random.default_rng(int(STUDENT_ID)).integers(1, 11, size=N, dtype=np.int64)

    serial_out = serial_scan(x)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_out = parallel_scan(x, workers, pool)
        t_serial = median_ms(lambda: serial_scan(x))
        t_parallel = median_ms(lambda: parallel_scan(x, workers, pool))

    # reference: the literal recurrence in a Python loop (single run, also cross-checks the NumPy scan)
    values = x.tolist()
    t0 = time.perf_counter()
    loop_out = serial_scan_python_loop(values)
    t_loop = (time.perf_counter() - t0) * 1000

    last_ok = int(parallel_out[-1]) == int(serial_out[-1])
    full_ok = bool(np.array_equal(parallel_out, serial_out)) and loop_out[-1] == int(serial_out[-1])
    status = "PASS" if (last_ok and full_ok) else "FAIL"

    print(f"{'Serial Time (ms)':>17} | {'Parallel Time (ms)':>19} | {'Speedup':>8} | Verification Status")
    print("-" * 70)
    print(f"{t_serial:>17.3f} | {t_parallel:>19.3f} | {t_serial / t_parallel:>7.2f}x | {status}")
    print(f"\nparallel_output[-1] = {int(parallel_out[-1]):,} | serial_output[-1] = {int(serial_out[-1]):,} "
          f"| full array equal: {full_ok}")
    print(f"Reference: literal Python-loop serial scan = {t_loop:.1f} ms "
          f"(NumPy parallel is {t_loop / t_parallel:.0f}x faster than it, not a like-for-like comparison)")
    print(f"\nWork: serial does N = {N:,} additions; the chunked scan does ~3N (chunk-sum pass, local scan, offset add over every element),")
    print("so it is not work-efficient: it spends about 3x the operations to shorten the sequential chain per worker.")

    if status != "PASS":
        print("ERROR: verification failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
