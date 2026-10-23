import os
import platform
import time

import numpy as np

# TODO: replace with your real Student ID (digits only are used as the seed)
STUDENT_ID = "230103296"

N = 20_000_000


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    seed = int("".join(ch for ch in STUDENT_ID if ch.isdigit()))

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.platform()})")
    print(f"CPU: {cpu_model()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Seed: {seed}")
    print(f"Array size: {N} float64 elements")

    # Generate data (not part of the timed computation)
    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)

    # Serial computation: sum and standard deviation (population), single thread
    t0 = time.perf_counter()
    total = float(np.sum(data))
    mean = total / N
    var = float(np.sum((data - mean) ** 2)) / N
    std = var ** 0.5
    t1 = time.perf_counter()

    print(f"Sum: {total:.6f}")
    print(f"Standard deviation: {std:.9f}")
    print(f"Serial execution time: {(t1 - t0) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
