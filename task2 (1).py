"""Task 2: pseudo-random generator state bottleneck (Monte Carlo pi, 40,000,000 samples, 4 threads).

Shared:      one NumPy Generator, every draw is done while holding a lock (the generator state is mutated).
Independent: each thread owns Generator(seed = Student_ID * 10 + thread_index), no lock at all.
Both modes draw points in batches of BATCH (x, y) pairs, so the only difference is the locking.
"""
import math
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103296
SAMPLES = 40_000_000
THREADS = 4
BATCH = 1_000          # points per generator call (the same in both modes)
REPEATS = 3
SENSITIVITY_BATCHES = (100, 1_000, 10_000)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def count_inside(pts):
    return int(np.count_nonzero(pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1] <= 1.0))


def worker_shared(rng, lock, n, batch):
    inside = 0
    left = n
    while left > 0:
        b = min(batch, left)
        with lock:                         # generator state is shared: draws must be serialized
            pts = rng.random((b, 2))
        inside += count_inside(pts)
        left -= b
    return inside


def worker_independent(seed, n, batch):
    rng = np.random.default_rng(seed)      # private state: nothing to lock
    inside = 0
    left = n
    while left > 0:
        b = min(batch, left)
        inside += count_inside(rng.random((b, 2)))
        left -= b
    return inside


def split(total, parts):
    return [total * (i + 1) // parts - total * i // parts for i in range(parts)]


def run_shared(batch):
    rng = np.random.default_rng(STUDENT_ID * 10)
    lock = threading.Lock()
    counts = split(SAMPLES, THREADS)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        t0 = time.perf_counter()
        futs = [pool.submit(worker_shared, rng, lock, c, batch) for c in counts]
        inside = sum(f.result() for f in futs)
        t = (time.perf_counter() - t0) * 1000
    return 4.0 * inside / SAMPLES, t


def run_independent(batch):
    counts = split(SAMPLES, THREADS)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        t0 = time.perf_counter()
        futs = [pool.submit(worker_independent, STUDENT_ID * 10 + i, counts[i], batch) for i in range(THREADS)]
        inside = sum(f.result() for f in futs)
        t = (time.perf_counter() - t0) * 1000
    return 4.0 * inside / SAMPLES, t


def median_of(fn, batch):
    fn(batch)  # warm-up
    runs = sorted((fn(batch) for _ in range(REPEATS)), key=lambda r: r[1])
    return runs[len(runs) // 2]


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()} | Threads: {THREADS} | Samples: {SAMPLES:,} | Batch: {BATCH:,} points per draw")
    print(f"Independent seeds: Seed = ({STUDENT_ID} * 10) + thread_index = "
          f"{', '.join(str(STUDENT_ID * 10 + i) for i in range(THREADS))} | Median of {REPEATS} runs\n")

    pi_s, t_s = median_of(run_shared, BATCH)
    pi_i, t_i = median_of(run_independent, BATCH)

    print(f"Shared PRNG (lock):      pi = {pi_s:.8f} | abs error = {abs(pi_s - math.pi):.3e} | wall time = {t_s:.2f} ms")
    print(f"Independent PRNG/thread: pi = {pi_i:.8f} | abs error = {abs(pi_i - math.pi):.3e} | wall time = {t_i:.2f} ms")
    print(f"Speedup from eliminating generator state locking: {t_s / t_i:.2f}x")

    print("\nSensitivity to batch size (smaller batches = more lock acquisitions):")
    print(f"{'Batch':>8} | {'Lock acquisitions':>17} | {'Shared ms':>10} | {'Independent ms':>14} | Speedup")
    for b in SENSITIVITY_BATCHES:
        _, ts = median_of(run_shared, b)
        _, ti = median_of(run_independent, b)
        print(f"{b:>8,} | {SAMPLES // b:>17,} | {ts:>10.2f} | {ti:>14.2f} | {ts / ti:>6.2f}x")

    print("\nNote: the shared run is also statistically a single stream, while the independent run uses four streams;")
    print("both estimate pi with error of order 1/sqrt(N), so their errors differ only by random noise.")


if __name__ == "__main__":
    main()
