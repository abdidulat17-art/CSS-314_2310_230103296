import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103296"
A, B = 2.5, 1.25
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
WORKERS = os.cpu_count()
TRIALS = 9              # median of this many timing trials per measurement
TARGET_TRIAL_S = 0.003  # each trial runs enough calls to last about this long
ROUNDS = 7              # interleaved sweep rounds used for the crossover search


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def serial(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def _chunk(args):
    x, y, lo, hi = args
    np.multiply(x[lo:hi], A, out=y[lo:hi])
    np.add(y[lo:hi], B, out=y[lo:hi])


def make_parallel(pool, workers):
    def parallel(x, y):
        n = len(x)
        bounds = [n * i // workers for i in range(workers + 1)]
        list(pool.map(_chunk, [(x, y, bounds[i], bounds[i + 1]) for i in range(workers)]))
    return parallel


def time_call(fn, x, y):
    """Median per-call time in seconds."""
    fn(x, y)  # warm-up
    t0 = time.perf_counter()
    fn(x, y)
    one = max(time.perf_counter() - t0, 1e-7)
    reps = max(1, int(TARGET_TRIAL_S / one))
    samples = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x, y)
        samples.append((time.perf_counter() - t0) / reps)
    samples.sort()
    return samples[len(samples) // 2]


def measure(n, par):
    rng = np.random.default_rng(int(STUDENT_ID))
    x = rng.random(n)
    y = np.empty(n)
    ts = time_call(serial, x, y)
    tp = time_call(par, x, y)
    ref = A * x + B
    assert np.allclose(y, ref), "parallel result mismatch"
    serial(x, y)
    assert np.allclose(y, ref), "serial result mismatch"
    return ts, tp


def fmt_time(s):
    return f"{s * 1e6:,.2f} us" if s < 1e-3 else f"{s * 1e3:,.3f} ms"


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Workers (all logical cores): {WORKERS}")
    print(f"Operation: y[i] = {A} * x[i] + {B}  (float64, NumPy, persistent ThreadPoolExecutor)")
    print(f"Timing: median of {TRIALS} trials per measurement\n")

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    par = make_parallel(pool, WORKERS)

    # ---- Required sizes -------------------------------------------------
    header = f"{'N':>12} | {'Serial':>14} | {'Parallel':>14} | {'Par/Serial':>10} | Winner"
    print(header)
    print("-" * len(header))
    for n in SIZES:
        ts, tp = measure(n, par)
        winner = "PARALLEL" if tp < ts else "serial"
        print(f"{n:>12,} | {fmt_time(ts):>14} | {fmt_time(tp):>14} | {tp / ts:>10.2f} | {winner}")

    # ---- Fine log-spaced sweep, repeated in interleaved rounds -------------
    # A single pass is too noisy near the crossover, so every size is measured
    # ROUNDS times and the median Parallel/Serial ratio decides the winner.
    grid = sorted(set(int(v) for v in np.geomspace(100, 10_000_000, 70)))
    ratios = {n: [] for n in grid}
    for _ in range(ROUNDS):
        for n in grid:
            ts, tp = measure(n, par)
            ratios[n].append(tp / ts)
    med = {n: float(np.median(ratios[n])) for n in grid}
    wins = {n: sum(r < 1.0 for r in ratios[n]) for n in grid}

    print(f"\n=== Sweep: median Parallel/Serial ratio over {ROUNDS} rounds ===")
    print(f"{'N':>12} | {'median ratio':>12} | rounds parallel won")
    for n in grid:
        print(f"{n:>12,} | {med[n]:>12.2f} | {wins[n]}/{ROUNDS}")

    # stable crossover: smallest N from which the median ratio is < 1 for every larger N
    stable_idx = None
    for i in range(len(grid)):
        if all(med[g] < 1.0 for g in grid[i:]):
            stable_idx = i
            break

    print("\n=== Crossover analysis ===")
    if stable_idx is None:
        print(f"Parallel ({WORKERS} workers) never became consistently faster than serial "
              f"up to N = {grid[-1]:,} on this hardware.")
    elif stable_idx == 0:
        print(f"Parallel was already faster than serial at N = {grid[0]:,} (smallest size tested).")
    else:
        lo, hi = grid[stable_idx - 1], grid[stable_idx]
        print(f"Multi-threading is SLOWER than serial for N <= {lo:,}")
        print(f"Multi-threading is FASTER than serial for N >= {hi:,}")
        print(f"Crossover range: N in [{lo:,}, {hi:,}] (~{int((lo * hi) ** 0.5):,} elements, geometric midpoint)")
        print(f"= about {int((lo * hi) ** 0.5) * 8 / 1024:,.0f} KiB per float64 array")
        first_win = next((n for n in grid if med[n] < 1.0), None)
        if first_win is not None and first_win < hi:
            print(f"Gray zone: from N = {first_win:,} up to N = {hi:,} the result flips between "
                  f"serial and parallel from run to run (median ratio ~1.0, noise-dominated);")
            print(f"           the win becomes sustained only from N >= {hi:,}.")
        else:
            print("No gray zone: the winner changes cleanly at the crossover.")

    pool.shutdown()


if __name__ == "__main__":
    main()
