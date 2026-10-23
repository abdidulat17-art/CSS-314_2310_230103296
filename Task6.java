import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 6: race condition forensics.
 * Build/run: javac Task6.java && java Task6
 */
public class Task6 {
    static final String STUDENT_ID = "230103296";
    static final int THREADS = 4;
    static final int INCREMENTS = 2_000_000;
    static final long EXPECTED = (long) THREADS * INCREMENTS;
    static final int REPEATS = 7;

    // No lock, no atomic. 'volatile' only forces a real memory read and write on every
    // iteration (otherwise the JIT collapses the loop into one addition and no race happens);
    // counter++ is still a separate read, add, write, so concurrent updates are still lost.
    static volatile int unsafeCounter;
    static final AtomicInteger atomicCounter = new AtomicInteger();
    static int lockedCounter;
    static final ReentrantLock lock = new ReentrantLock();

    interface Body { void run(); }

    /** Runs THREADS threads released together; returns {wall time in ms}. */
    static double timed(Body body) throws InterruptedException {
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch go = new CountDownLatch(1);
        Thread[] ts = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            ts[i] = new Thread(() -> {
                ready.countDown();
                try { go.await(); } catch (InterruptedException e) { return; }
                body.run();
            });
            ts[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : ts) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    static void unsafeBody() { for (int k = 0; k < INCREMENTS; k++) unsafeCounter++; }
    static void atomicBody() { for (int k = 0; k < INCREMENTS; k++) atomicCounter.incrementAndGet(); }
    static void lockBody() {
        for (int k = 0; k < INCREMENTS; k++) {
            lock.lock();
            try { lockedCounter++; } finally { lock.unlock(); }
        }
    }

    static double median(double[] a) {
        double[] c = a.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Logical cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads: " + THREADS + " | Increments per thread: " + INCREMENTS
                + " | Runs per variant: " + REPEATS + "\n");

        // warm-up so the JIT has compiled all three loops
        timed(Task6::unsafeBody); timed(Task6::atomicBody); timed(Task6::lockBody);

        double[] tUnsafe = new double[REPEATS], tAtomic = new double[REPEATS], tLock = new double[REPEATS];
        long[] finalUnsafe = new long[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            unsafeCounter = 0;
            tUnsafe[r] = timed(Task6::unsafeBody);
            finalUnsafe[r] = unsafeCounter;

            atomicCounter.set(0);
            tAtomic[r] = timed(Task6::atomicBody);
            if (atomicCounter.get() != EXPECTED) throw new AssertionError("atomic counter wrong");

            lockedCounter = 0;
            tLock[r] = timed(Task6::lockBody);
            if (lockedCounter != EXPECTED) throw new AssertionError("locked counter wrong");
        }

        System.out.println("=== Unsynchronized counter (volatile int, counter++, no lock/atomic) ===");
        for (int r = 0; r < REPEATS; r++) {
            double lost = 100.0 * (EXPECTED - finalUnsafe[r]) / EXPECTED;
            System.out.printf("  run %d: final = %,d | lost = %,d (%.2f%%) | %.2f ms%n",
                    r + 1, finalUnsafe[r], EXPECTED - finalUnsafe[r], lost, tUnsafe[r]);
        }
        long[] sorted = finalUnsafe.clone();
        Arrays.sort(sorted);
        long medFinal = sorted[REPEATS / 2];
        System.out.println();
        System.out.printf("Expected final value:  %,d%n", EXPECTED);
        System.out.printf("Actual final value:    %,d  (median run)%n", medFinal);
        System.out.printf("Updates lost:          %,d = %.2f%%  (range over runs: %.2f%% to %.2f%%)%n",
                EXPECTED - medFinal, 100.0 * (EXPECTED - medFinal) / EXPECTED,
                100.0 * (EXPECTED - sorted[REPEATS - 1]) / EXPECTED,
                100.0 * (EXPECTED - sorted[0]) / EXPECTED);

        double mU = median(tUnsafe), mA = median(tAtomic), mL = median(tLock);
        System.out.println("\n=== Correct versions (always exactly " + String.format("%,d", EXPECTED) + ") ===");
        System.out.printf("%-34s | %12s | %10s | %s%n", "Variant", "Median ms", "Ratio", "Change vs unsafe");
        System.out.println("-----------------------------------+--------------+------------+------------------");
        System.out.printf("%-34s | %12.2f | %10s | %s%n", "Unsafe (wrong result)", mU, "1.00x", "-");
        System.out.printf("%-34s | %12.2f | %9.2fx | %+.2f ms%n", "AtomicInteger.incrementAndGet", mA, mA / mU, mA - mU);
        System.out.printf("%-34s | %12.2f | %9.2fx | %+.2f ms%n", "ReentrantLock", mL, mL / mU, mL - mU);
        System.out.printf("%nLock penalty (ReentrantLock vs unsynchronized): %+.2f ms (%.2fx the unsynchronized time, %+.1f%%)%n",
                mL - mU, mL / mU, 100.0 * (mL - mU) / mU);
        System.out.printf("Atomic (AtomicInteger) vs unsynchronized:         %+.2f ms (%.2fx, %+.1f%%)%n",
                mA - mU, mA / mU, 100.0 * (mA - mU) / mU);
        System.out.printf("Lock vs atomic:                                   %+.2f ms (%.2fx)%n", mL - mA, mL / mA);
        System.out.println("\nNote: the unsafe counter is 'volatile' (visibility only, NOT atomic), so every counter++");
        System.out.println("is a real memory read, add and write that other threads can overwrite.");
    }
}
