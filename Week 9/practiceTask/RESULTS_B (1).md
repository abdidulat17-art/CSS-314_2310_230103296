# RESULTS_B — Variant B: Granularity & IPC Contention

Formulas: S_p = T1/Tp, e(p) = (p·Tp/T1 − 1)/(p − 1).

| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(8) | Verdict |
|---|---|---|---|---|---|---|---|---|
| A: Micro (k=1) * | ~20 | ~11 | ~8 | ~7 | ~2.9 | ~0.10 | ~0.40 | Master dispatch serialization, e(p) grows |
| A: Coarse (k=10000) * | 4.80 | 2.50 | 1.35 | 0.75 | 6.4 | 0.042 | 0.038 | Flat e, healthy strong scaling |
| B: Queue Choke * | 1.20 | 1.95 | 3.80 | 7.60 | 0.16 | 2.25 | 7.10 | IPC lock contention explosion, negative speedup (T8 > T1) |
| C: Buffered Batch (measured) | 0.910 | 0.523 | 0.349 | 0.335 | 2.72 | 0.148 | 0.278 | Amortized synchronization |

\* Not measured on my machine: values are the reference values from the lab handout (A coarse, B) or a rough estimate (A micro). Only row C was measured (N = 10,000,000, Windows, Python 3.14).

## Notes

- **A:** with chunksize=1 every number is a separate IPC task, so the master process spends its time dispatching and efficiency drops as P grows. With chunksize=10000 e(p) stays flat (~0.04), i.e. healthy scaling.
- **B:** every `put` crosses the pipe and takes locks, so adding processes increases runtime (T8 > T1) and e(p) rises monotonically. The bottleneck is communication, not computation. (Handout lists e(2)=1.250, e(8)=6.280, which do not match its own times; values here are computed from the times.)
- **C:** local buffering sends one message per 10,000 items. Throughput is ~11M items/s versus ~0.33M items/s in B (about 33x), meeting the ">10x" criterion. S8 = 2.72 and e(8) = 0.278 do not reach S8 > 5 and e(8) ≤ 0.05: the run lasts under a second, so process start-up on Windows and the physical core count limit the speedup.
