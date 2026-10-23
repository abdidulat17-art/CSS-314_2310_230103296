import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103296"
N = 30_000_000
REPEATS = 5
BLOCK = 1 << 17          # elements per block in Kernel B (keeps temporaries in cache)
STALL_THRESHOLD = 0.90   # efficiency at max cores below this counts as "stalled"
MIN_GAP = 0.10           # and it must trail the other kernel by at least this much


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def kernel_a(a, b, lo, hi):
    """Memory-bound: B[i] = A[i]"""
    np.copyto(b[lo:hi], a[lo:hi])


def kernel_b(a, b, lo, hi):
    """Compute-bound: B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])"""
    t1 = np.empty(BLOCK)
    t2 = np.empty(BLOCK)
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        n = e - s
        x = a[s:e]
        u, v = t1[:n], t2[:n]
        np.sin(x, out=u)
        np.cos(x, out=v)
        np.multiply(u, v, out=u)
        np.tan(x, out=v)
        np.add(u, v, out=b[s:e])


def run(kernel, a, b, workers, pool):
    bounds = [N * i // workers for i in range(workers + 1)]
    if workers == 1:
        kernel(a, b, 0, N)
        return
    futs = [pool.submit(kernel, a, b, bounds[i], bounds[i + 1]) for i in range(workers)]
    for f in futs:
        f.result()


def bench(kernel, a, b, workers):
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        run(kernel, a, b, workers, pool)  # warm-up (also first-touches B)
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            run(kernel, a, b, workers, pool)
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    max_cores = os.cpu_count()
    counts = list(dict.fromkeys([1, 2, max_cores]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores (os.cpu_count): {max_cores}")
    print(f"Array: {N:,} float64 = {N * 8 / 1e6:.0f} MB per array; threads tested: {counts}")
    print(f"Median of {REPEATS} runs; NumPy ufuncs/copies release the GIL, one contiguous chunk per thread\n")

    rng = np.random.default_rng(int(STUDENT_ID))
    a = rng.random(N, dtype=np.float64) * 1.2 + 0.1     # keep tan() away from its poles
    b = np.empty(N, dtype=np.float64)

    summary = {}
    for name, kernel in (("Kernel A (copy, memory-bound)", kernel_a),
                         ("Kernel B (sin*cos+tan, compute-bound)", kernel_b)):
        rows = []
        for w in counts:
            rows.append((w, bench(kernel, a, b, w)))
        t1 = rows[0][1]

        # correctness check on a sample
        idx = slice(0, N, 997)
        if kernel is kernel_a:
            assert np.array_equal(b[idx], a[idx])
        else:
            x = a[idx]
            assert np.allclose(b[idx], np.sin(x) * np.cos(x) + np.tan(x))

        print(name)
        header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Efficiency':>10}"
        if kernel is kernel_a:
            header += f" | {'Bandwidth (GB/s)':>16}"
        print(header)
        print("-" * len(header))
        for w, t in rows:
            sp = t1 / t
            line = f"{w:>6} | {t * 1000:>14.2f} | {sp:>8.2f} | {sp / w:>10.2f}"
            if kernel is kernel_a:
                line += f" | {2 * 8 * N / t / 1e9:>16.2f}"   # read A + write B
            print(line)
        print()
        summary[name] = (rows[-1][0], (t1 / rows[-1][1]) / rows[-1][0])

    (nm_a, (cores, eff_a)), (nm_b, (_, eff_b)) = summary.items()
    print("=== Verdict ===")
    print(f"Efficiency at {cores} cores: Kernel A (copy) = {eff_a:.2f} | Kernel B (sin*cos+tan) = {eff_b:.2f}")
    if cores == 1:
        print("Only 1 core available: scaling cannot be assessed.")
    elif eff_a < STALL_THRESHOLD and eff_b - eff_a >= MIN_GAP:
        print(f"Kernel A (vector copy) STALLED in scaling due to memory channel / RAM bandwidth saturation: "
              f"efficiency {eff_a:.2f} < {STALL_THRESHOLD:.2f}, while compute-bound Kernel B kept efficiency {eff_b:.2f}.")
        print("The copy does almost no arithmetic per byte moved, so extra cores just wait on the shared memory bus;")
        print("the transcendental math in Kernel B keeps each core busy on its own, so it scales almost linearly.")
    elif eff_b < STALL_THRESHOLD and eff_a - eff_b >= MIN_GAP:
        print(f"Kernel B (sin*cos+tan) scaled worse than Kernel A (efficiency {eff_b:.2f} vs {eff_a:.2f}); "
              "memory saturation is NOT the explanation here (other limits such as thread count vs. physical cores).")
    else:
        print(f"No kernel stalled: Kernel A efficiency {eff_a:.2f}, Kernel B {eff_b:.2f} with {cores} cores "
              f"(stall = below {STALL_THRESHOLD:.2f} and trailing the other kernel by {MIN_GAP:.2f}+). "
              "No clear memory channel saturation observed on this hardware.")


if __name__ == "__main__":
    main()
