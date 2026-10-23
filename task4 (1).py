"""Task 4 (CPU-only variant): inter-process transfer overhead, serialization vs. zero-copy shared memory.

A parent process hands an array of N float64 values to a persistent child process; the child checks a sampled
checksum and acknowledges. Transfer overhead = time from just before the send until the acknowledgement arrives.
Process start-up is excluded (the child is started once per path and reused).

  Path A1: multiprocessing.Pipe     (pickle -> pipe -> unpickle)
  Path A2: multiprocessing.Queue    (pickle in a feeder thread -> pipe -> unpickle)
  Path B1: shared memory, data already in the shared block: only the name is sent, the child attaches (zero-copy)
  Path B2: shared memory including creating the block and copying the array into it
"""
import multiprocessing as mp
import os
import platform
import time
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = "230103296"
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 7


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def checksum(arr):
    step = max(1, len(arr) // 1000)
    return float(arr[::step].sum())


def attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python 3.13+: child must not own the block
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def child(mode, req, ack):
    while True:
        msg = req.get() if mode == "queue" else req.recv()
        if msg is None:
            break
        if mode == "shm":
            name, n = msg
            shm = attach(name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            cs = checksum(arr)
            del arr
            shm.close()
        else:
            cs = checksum(msg)
        ack.send(cs)


class Channel:
    def __init__(self, mode):
        self.mode = mode
        ctx = mp.get_context("fork")
        if mode == "queue":
            self.req_parent = ctx.Queue()
            req_child = self.req_parent
        else:
            self.req_parent, req_child = ctx.Pipe()
        self.ack_parent, ack_child = ctx.Pipe()
        self.proc = ctx.Process(target=child, args=(mode, req_child, ack_child), daemon=True)
        self.proc.start()

    def send(self, msg):
        if self.mode == "queue":
            self.req_parent.put(msg)
        else:
            self.req_parent.send(msg)

    def close(self):
        self.send(None)
        self.proc.join(timeout=5)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def time_serialized(ch, arr, expected):
    ts = []
    for r in range(REPEATS + 1):
        t0 = time.perf_counter()
        ch.send(arr)
        cs = ch.ack_parent.recv()
        t = (time.perf_counter() - t0) * 1000
        assert cs == expected, "checksum mismatch"
        if r:                                  # first run is a warm-up
            ts.append(t)
    return median(ts)


def time_shm_attach(ch, arr, expected):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    try:
        np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)   # fill once, outside the timing
        ts = []
        for r in range(REPEATS + 1):
            t0 = time.perf_counter()
            ch.send((shm.name, len(arr)))
            cs = ch.ack_parent.recv()
            t = (time.perf_counter() - t0) * 1000
            assert cs == expected, "checksum mismatch"
            if r:
                ts.append(t)
    finally:
        shm.close()
        shm.unlink()
    return median(ts)


def time_shm_create_copy(ch, arr, expected):
    ts = []
    for r in range(REPEATS + 1):
        t0 = time.perf_counter()
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)
        ch.send((shm.name, len(arr)))
        cs = ch.ack_parent.recv()
        t = (time.perf_counter() - t0) * 1000
        shm.close()
        shm.unlink()
        assert cs == expected, "checksum mismatch"
        if r:
            ts.append(t)
    return median(ts)


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()} | No GPU used -> CPU inter-process variant | Median of {REPEATS} runs")
    print("Overhead = send -> child receives & checksums a sample -> acknowledgement (child already running)\n")

    pipe, queue_, shm = Channel("pipe"), Channel("queue"), Channel("shm")
    rng = np.random.default_rng(int(STUDENT_ID))

    rows = []
    for n in SIZES:
        arr = rng.random(n)
        expected = checksum(arr)
        a1 = time_serialized(pipe, arr, expected)
        a2 = time_serialized(queue_, arr, expected)
        b1 = time_shm_attach(shm, arr, expected)
        b2 = time_shm_create_copy(shm, arr, expected)
        rows.append((n, arr.nbytes / 1e6, a1, a2, b1, b2))

    for ch in (pipe, queue_, shm):
        ch.close()

    header = (f"{'N':>12} | {'Size MB':>8} | {'A1 Pipe ms':>10} | {'A2 Queue ms':>11} | "
              f"{'B1 SHM attach ms':>16} | {'B2 SHM create+copy ms':>21} | {'A1 / B1':>9}")
    print(header)
    print("-" * len(header))
    for n, mb, a1, a2, b1, b2 in rows:
        print(f"{n:>12,} | {mb:>8.1f} | {a1:>10.3f} | {a2:>11.3f} | {b1:>16.3f} | {b2:>21.3f} | {a1 / b1:>8.0f}x")

    print("\nEffective throughput of Path A1 (Pipe): " + ", ".join(
        f"N={n:,}: {mb / 1000 / (a1 / 1000):.2f} GB/s" for n, mb, a1, a2, b1, b2 in rows))
    first, last = rows[0], rows[-1]
    print(f"Growth from N={first[0]:,} to N={last[0]:,} ({last[0] // first[0]}x more data): "
          f"A1 Pipe {last[2] / first[2]:.0f}x, A2 Queue {last[3] / first[3]:.0f}x, B1 SHM attach {last[4] / first[4]:.0f}x, "
          f"B2 SHM create+copy {last[5] / first[5]:.0f}x.")
    print("B1 sends only the block name, so it scales far more slowly (what remains is the OS mapping and faulting in pages);")
    print("B2 is the honest cost when the data starts in ordinary memory: one memcpy into the shared block, no pickling.")

if __name__ == "__main__":
    main()
