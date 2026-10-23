"""Task 5: thread oversubscription threshold.

10,000 small CPU-bound tasks: each repeatedly hashes a 1 KB buffer (SHA-256, 500 chained rounds).
The batch is run on pools of [1, Cores/2, Cores, Cores*2, Cores*4, Cores*16, 256] workers (duplicates removed).

Table 1 (the requested one): thread pool.
  hashlib only releases the GIL for inputs larger than 2 KB, so 1 KB hashes keep the GIL: Python threads cannot
  run these tasks in parallel, and extra threads mostly add GIL hand-off and OS scheduling overhead.
Table 2 (for contrast): process pool, same batch, where each worker is a separate OS thread of execution with its own GIL.
"""
import hashlib
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103296"
TASKS = 10_000
ROUNDS = 500
PAYLOAD = 1024
REPEATS = 3
WORSEN_TOLERANCE = 0.05     # a pool size "worsens" if it is more than 5% slower than the previous size
PROC_CHUNK = 50             # tasks per process-pool message (threads take tasks one at a time)

BASE = np.random.default_rng(int(STUDENT_ID)).bytes(PAYLOAD)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def task(i):
    data = i.to_bytes(4, "little") + BASE[4:]
    for _ in range(ROUNDS):
        data = hashlib.sha256(data).digest() + data[32:]      # stays exactly 1 KB
    return data[:8]


def run_threads(workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        t0 = time.perf_counter()
        out = list(pool.map(task, range(TASKS)))
        return (time.perf_counter() - t0) * 1000, out


def run_processes(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        t0 = time.perf_counter()
        out = list(pool.map(task, range(TASKS), chunksize=PROC_CHUNK))
        return (time.perf_counter() - t0) * 1000, out


def measure(runner, workers, reference):
    times = []
    for _ in range(REPEATS):
        t, out = runner(workers)
        assert out == reference, "results differ between pool sizes"
        times.append(t)
    times.sort()
    return times[len(times) // 2]


def classify(sizes, times):
    """Return per-row status strings and the first size that starts worsening (or None)."""
    best_i = min(range(len(times)), key=lambda i: times[i])
    worsen_i = None
    for i in range(1, len(times)):
        if times[i] > times[i - 1] * (1 + WORSEN_TOLERANCE):
            worsen_i = i
            break
    status = []
    for i in range(len(sizes)):
        if i == 0:
            s = "baseline"
        elif i == worsen_i:
            s = "<-- STARTS WORSENING (context-switch / GIL overhead)"
        elif worsen_i is not None and i > worsen_i:
            s = "degraded" if times[i] > times[best_i] * (1 + WORSEN_TOLERANCE) else "near best"
        elif i == best_i:
            s = "best so far"
        else:
            s = "scaling" if times[i] < times[i - 1] else "flat"
        status.append(s)
    return status, best_i, worsen_i


def table(title, runner, sizes, reference):
    times = [measure(runner, s, reference) for s in sizes]
    status, best_i, worsen_i = classify(sizes, times)
    print(title)
    header = f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status"
    print(header)
    print("-" * (len(header) + 30))
    for s, t, st in zip(sizes, times, status):
        print(f"{s:>9} | {t:>19.1f} | {times[0] / t:>18.2f}x | {st}")
    print(f"Fastest pool size: {sizes[best_i]} ({times[best_i]:.1f} ms).")
    if worsen_i is None:
        print(f"No pool size was more than {WORSEN_TOLERANCE:.0%} slower than the previous one: "
              "no oversubscription degradation observed up to the largest pool.")
    else:
        print(f"Execution time starts worsening at pool size {sizes[worsen_i]} "
              f"({times[worsen_i - 1]:.1f} ms at {sizes[worsen_i - 1]} -> {times[worsen_i]:.1f} ms at {sizes[worsen_i]}).")
    print()
    return times


def main():
    cores = os.cpu_count() or 1
    sizes = list(dict.fromkeys(max(1, s) for s in [1, cores // 2, cores, cores * 2, cores * 4, cores * 16, 256]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores: {cores} | Pool sizes (duplicates removed): {sizes}")
    print(f"Batch: {TASKS:,} tasks x {ROUNDS} chained SHA-256 rounds over a {PAYLOAD}-byte buffer | Median of {REPEATS} runs")
    print(f"Status rule: a size 'starts worsening' when it is more than {WORSEN_TOLERANCE:.0%} slower than the previous size\n")

    _, reference = run_threads(max(sizes[:3]))     # also a warm-up; every run must reproduce these results

    table("Table 1: ThreadPoolExecutor (the requested thread pool; GIL held by 1 KB hashes)", run_threads, sizes, reference)
    table(f"Table 2: ProcessPoolExecutor for contrast (chunksize {PROC_CHUNK}; true parallelism across cores)",
          run_processes, sizes, reference)


if __name__ == "__main__":
    main()
