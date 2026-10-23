"""Task 4: work stealing & scheduling under skew.

1,000 items, item i costs sleep(0.0001 * (i % 20)); 4 worker threads.
A: static equal blocks | B: shared dynamic queue (chunk 1 and 5) | C: work stealing (per-worker deques, chunk 5)
"""
import os
import platform
import queue
import threading
import time
from collections import deque

STUDENT_ID = "230103296"
ITEMS = 1000
WORKERS = 4
REPEATS = 7


def cost(i):
    return 0.0001 * (i % 20)


def do_item(i):
    c = cost(i)
    if c > 0:
        time.sleep(c)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_workers(target):
    """Start WORKERS threads together; returns (start, finish_times[]) via perf_counter."""
    barrier = threading.Barrier(WORKERS + 1)
    finish = [0.0] * WORKERS
    done_items = [0] * WORKERS

    def wrap(w):
        barrier.wait()
        done_items[w] = target(w)
        finish[w] = time.perf_counter()

    ts = [threading.Thread(target=wrap, args=(w,)) for w in range(WORKERS)]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    assert sum(done_items) == ITEMS, f"processed {sum(done_items)} items"
    return t0, finish, done_items


# ---- Strategy A: static equal blocks ---------------------------------------
def static_blocks():
    edges = [ITEMS * w // WORKERS for w in range(WORKERS + 1)]

    def target(w):
        for i in range(edges[w], edges[w + 1]):
            do_item(i)
        return edges[w + 1] - edges[w]
    return run_workers(target)


# ---- Strategy B: shared dynamic queue --------------------------------------
def dynamic_queue(chunk):
    q = queue.Queue()
    for s in range(0, ITEMS, chunk):
        q.put(range(s, min(s + chunk, ITEMS)))

    def target(w):
        n = 0
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return n
            for i in r:
                do_item(i)
            n += len(r)
    return run_workers(target)


# ---- Strategy C: work stealing ---------------------------------------------
def work_stealing(chunk):
    edges = [ITEMS * w // WORKERS for w in range(WORKERS + 1)]
    deques = [deque(range(s, min(s + chunk, edges[w + 1]))
                    for s in range(edges[w], edges[w + 1], chunk)) for w in range(WORKERS)]
    locks = [threading.Lock() for _ in range(WORKERS)]
    steals = [0] * WORKERS

    def target(w):
        n = 0
        while True:
            with locks[w]:
                r = deques[w].popleft() if deques[w] else None   # own work from the front
            if r is None:
                victim = max(range(WORKERS), key=lambda v: len(deques[v]))   # busiest worker
                with locks[victim]:
                    r = deques[victim].pop() if deques[victim] else None     # steal from the back
                if r is None:
                    if all(len(d) == 0 for d in deques):
                        return n
                    continue
                steals[w] += 1
            for i in r:
                do_item(i)
            n += len(r)
    out = run_workers(target)
    return out + (sum(steals),)


def median_run(fn):
    runs = []
    for _ in range(REPEATS):
        res = fn()
        t0, finish, items = res[0], res[1], res[2]
        total = (max(finish) - t0) * 1000
        delta = (max(finish) - min(finish)) * 1000
        runs.append((total, delta, [(f - t0) * 1000 for f in finish], items, res[3] if len(res) > 3 else None))
    runs.sort(key=lambda r: r[0])
    return runs[len(runs) // 2], runs


def main():
    total_work = sum(cost(i) for i in range(ITEMS))
    ideal = total_work / WORKERS * 1000

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()} | Workers (threads): {WORKERS} | Items: {ITEMS} | Median of {REPEATS} runs")
    print(f"Item cost: sleep(0.0001 * (i % 20)); total requested sleep = {total_work * 1000:.1f} ms")
    print(f"Ideal perfectly balanced time = {ideal:.2f} ms (before sleep/scheduling overhead)\n")

    strategies = [
        ("A: Static equal blocks (250 each)", static_blocks),
        ("B1: Dynamic shared queue, chunk=1", lambda: dynamic_queue(1)),
        ("B2: Dynamic shared queue, chunk=5", lambda: dynamic_queue(5)),
        ("C: Work stealing, chunk=5", lambda: work_stealing(5)),
    ]
    results = []
    for name, fn in strategies:
        fn()  # warm-up
        results.append((name,) + (median_run(fn),))

    header = f"{'Strategy':<36} | {'Total ms':>9} | {'First done':>10} | {'Last done':>9} | {'Delta ms':>8} | {'vs ideal':>8}"
    print(header)
    print("-" * len(header))
    for name, ((total, delta, fin, items, steals), runs) in results:
        print(f"{name:<36} | {total:>9.2f} | {min(fin):>10.2f} | {max(fin):>9.2f} | {delta:>8.2f} | {total / ideal:>7.2f}x")

    print("\nPer-thread finish time (ms after start) and items processed, median run:")
    for name, ((total, delta, fin, items, steals), runs) in results:
        fins = ", ".join(f"{f:.2f}" for f in fin)
        extra = f" | steals: {steals}" if steals is not None else ""
        print(f"  {name:<36} finish [{fins}] | items {items}{extra}")

    print("\nSpread over runs (min..max Delta ms): " + "; ".join(
        f"{n.split(':')[0]} {min(r[1] for r in runs):.2f}..{max(r[1] for r in runs):.2f}" for n, (_, runs) in results))

    best = min(results, key=lambda r: r[1][0][0])
    best_delta = min(results, key=lambda r: r[1][0][1])
    print("\n=== Verdict ===")
    print(f"Fastest total time: {best[0]} ({best[1][0][0]:.2f} ms). "
          f"Smallest first-to-last finish delta: {best_delta[0]} ({best_delta[1][0][1]:.2f} ms).")
    totals = [r[1][0][0] for r in results]
    spread = max(totals) - min(totals)
    print(f"Total times differ by only {spread:.2f} ms ({100 * spread / min(totals):.1f}%) across strategies: the cost pattern")
    print("(i % 20) repeats every 20 items, so equal contiguous blocks are already nearly balanced and the run is dominated")
    print("by sleep granularity. What changes is balance: static blocks leave a "
          f"{results[0][1][0][1]:.2f} ms first-to-last gap, dynamic chunk=1 shrinks it to {results[1][1][0][1]:.2f} ms.")

if __name__ == "__main__":
    main()
