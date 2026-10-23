"""Task 8 (CPU path): 1024x1024 matrix multiplication, tiled (16x16, 64x64) vs non-tiled, across threads.

No dedicated NVIDIA GPU is used; the CPU-parallel variant of the task is implemented.
BLAS is pinned to one thread so the parallelism measured here is the thread pool's.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import platform
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103296"
N = 1024
REPEATS = 3
TILES = (16, 64)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cache_sizes():
    out = {}
    for lvl, name in ((0, "L1d"), (2, "L2"), (3, "L3")):
        try:
            with open(f"/sys/devices/system/cpu/cpu0/cache/index{lvl}/size") as f:
                out[name] = f.read().strip()
        except OSError:
            pass
    return out


# ---- kernels: each computes rows [lo, hi) of C = A @ B ------------------------

def rowwise(A, B, C, lo, hi):
    """Non-tiled: one row of A at a time, so all of B is streamed through the cache for every row."""
    for i in range(lo, hi):
        np.dot(A[i], B, out=C[i])


def tiled(A, B, C, lo, hi, bs):
    """Tiled: C[ii:,jj:] += A[ii:,kk:] @ B[kk:,jj:] on bs x bs tiles, so each tile is reused while hot."""
    tmp = np.empty((bs, bs))
    for ii in range(lo, hi, bs):
        for jj in range(0, N, bs):
            acc = C[ii:ii + bs, jj:jj + bs]
            acc[:] = 0.0
            for kk in range(0, N, bs):
                np.matmul(A[ii:ii + bs, kk:kk + bs], B[kk:kk + bs, jj:jj + bs], out=tmp)
                np.add(acc, tmp, out=acc)


def blas_band(A, B, C, lo, hi):
    """Reference: single-thread BLAS gemm on a band of rows (BLAS blocks internally)."""
    np.matmul(A[lo:hi], B, out=C[lo:hi])


def bands(workers, align=64):
    nb = N // align
    workers = min(workers, nb)
    edges = [round(nb * i / workers) * align for i in range(workers + 1)]
    return [(edges[i], edges[i + 1]) for i in range(workers)]


def run(kernel, A, B, C, workers, pool):
    bs = bands(workers)
    if len(bs) == 1:
        kernel(A, B, C, 0, N)
        return
    futs = [pool.submit(kernel, A, B, C, lo, hi) for lo, hi in bs]
    for f in futs:
        f.result()


def bench(kernel, A, B, C, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        run(kernel, A, B, C, workers, pool)       # warm-up
        ts = []
        for _ in range(REPEATS):
            C[:] = -1.0
            t0 = time.perf_counter()
            run(kernel, A, B, C, workers, pool)
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    max_cores = os.cpu_count()
    counts = list(dict.fromkeys([1, max_cores]))
    caches = cache_sizes()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores: {max_cores} | Caches (cpu0): " + ", ".join(f"{k} {v}" for k, v in caches.items()))
    gpu = shutil.which("nvidia-smi")
    print(f"NVIDIA GPU: {'detected' if gpu else 'none (nvidia-smi not found)'} -> running the CPU tiled matmul variant")
    print(f"Matrix: {N}x{N} float64 ({N * N * 8 / 2**20:.0f} MiB each), threads tested: {counts}, median of {REPEATS} runs")
    print("BLAS threads fixed to 1; parallelism = Python thread pool over 64-row bands of C\n")

    rng = np.random.default_rng(int(STUDENT_ID))
    A = rng.random((N, N))
    B = rng.random((N, N))
    C = np.empty((N, N))
    ref = A @ B

    variants = [("Non-tiled (row-wise A[i] @ B)", rowwise, None)]
    for bs in TILES:
        variants.append((f"Tiled {bs}x{bs}", lambda A_, B_, C_, lo, hi, bs=bs: tiled(A_, B_, C_, lo, hi, bs), bs))
    variants.append(("Reference: BLAS gemm per band", blas_band, None))

    results = {}
    for name, kern, bs in variants:
        for w in counts:
            t = bench(kern, A, B, C, w)
            assert np.allclose(C, ref), f"wrong result for {name}"
            results[(name, w)] = t

    header = f"{'Variant':<32} | " + " | ".join(f"{w:>2} thread(s) ms" for w in counts) + " | Speedup"
    print(header)
    print("-" * len(header))
    for name, _, _ in variants:
        ts = [results[(name, w)] for w in counts]
        print(f"{name:<32} | " + " | ".join(f"{t:>14.1f}" for t in ts) + f" | {ts[0] / ts[-1]:>7.2f}x")

    wmax = counts[-1]
    base = results[(variants[0][0], wmax)]
    print(f"\nRuntime relative to the non-tiled kernel at {wmax} thread(s) (higher = faster than non-tiled):")
    for name, _, _ in variants[1:]:
        print(f"  {name:<32} {base / results[(name, wmax)]:>6.2f}x")

    print("\nCache behaviour model (estimates from tile geometry; hardware miss counters are not available here):")
    l1 = caches.get("L1d", "?")
    print(f"  {'Variant':<30} | {'tile working set A+B+tmp':>24} | {'tile products':>13} | {'est. data moved (GB)':>20}")
    row_traffic = (N * N * N * 8 + N * N * 8) / 1e9      # all of B streamed per row of A
    print(f"  {'Non-tiled row-wise':<30} | {'B = ' + format(N * N * 8 // 1024, ',') + ' KiB (whole)':>24} | {N:>13,} | {row_traffic:>20.2f}")
    for bs in TILES:
        ws = 3 * bs * bs * 8 / 1024
        products = (N // bs) ** 3
        traffic = 2 * N ** 3 * 8 / bs / 1e9              # each A and B tile reloaded N/bs times
        print(f"  {f'Tiled {bs}x{bs}':<30} | {ws:>20.1f} KiB | {products:>13,} | {traffic:>20.2f}")
    print(f"  (L1d per core = {l1}; a tile triple fits in L1 when its working set is below that.)")

    t16, t64 = (results[(f"Tiled {bs}x{bs}", wmax)] for bs in TILES)
    print("\n=== Verdict ===")
    print(f"Non-tiled {base:.1f} ms | 16x16 {t16:.1f} ms | 64x64 {t64:.1f} ms at {wmax} thread(s).")
    best = min((("non-tiled", base), ("16x16 tiles", t16), ("64x64 tiles", t64)), key=lambda x: x[1])
    print(f"Fastest of the three: {best[0]}. Small tiles (16x16) keep data in L1 but pay for {(N // 16) ** 3:,} tile calls;")
    print(f"64x64 tiles amortize call overhead ({(N // 64) ** 3:,} tile calls) while their working set still fits in L2.")
    sp = {n: results[(n, counts[0])] / results[(n, wmax)] for n, _, _ in variants}
    print(f"Thread scaling ({counts[0]} -> {wmax}): non-tiled {sp[variants[0][0]]:.2f}x, 16x16 {sp['Tiled 16x16']:.2f}x, "
          f"64x64 {sp['Tiled 64x64']:.2f}x, BLAS reference {sp[variants[-1][0]]:.2f}x.")
    print("The tiled kernels run a Python loop around many small NumPy calls that hold the GIL, so extra threads add")
    print("little; long GIL-free calls (row-wise dot, BLAS) scale with the cores. Tiling helps cache reuse, not Python overhead.")


if __name__ == "__main__":
    main()
