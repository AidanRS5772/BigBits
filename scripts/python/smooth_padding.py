from math import inf
from statistics import mean, stdev, median
import math
import random


def best_smooth(target, primes):
    best = [inf]

    def search(val, idx):
        if val >= best[0]:
            return

        if idx == len(primes):
            if val >= target:
                best[0] = val
            return

        p, max_exp = primes[idx]

        curr_val = val
        for exp in range(max_exp + 1):
            search(curr_val, idx + 1)
            curr_val *= p
            if curr_val >= best[0]:
                break

    search(1, 0)
    return best[0] if best[0] != inf else None


def max_primes(primes):
    res = 1
    for p, pow in primes:
        res *= p**pow
    return res


def get_data(primes, min = 1 << 20, max = 1 << 45, sample_size=100_000):
    log_lo = math.log(min)
    log_hi = math.log(max) - math.log(2)  # stay one octave below ceiling

    data = []
    for _ in range(sample_size):
        s = int(math.exp(random.uniform(log_lo, log_hi)))
        result = best_smooth(s, primes)
        data.append((s, result))

    return data


def analyze_padding(primes, sample_size=1_000_000):
    data = get_data(primes, sample_size)

    # Compute relative gaps
    relative_gaps = [(r - t) / t for t, r in data]
    absolute_gaps = [r - t for t, r in data]

    print(f"\nPrime configuration: {primes}")
    print(f"Max smooth number: {max_primes(primes):,}")
    print(f"Sample size: {len(data):,}")

    # Relative gap statistics (as fraction of target)
    print("\n--- Relative Gap (padding / target) ---")
    print(f"  Mean:   {mean(relative_gaps):.4f} ({mean(relative_gaps) * 100:.2f}%)")
    print(f"  Median: {median(relative_gaps):.4f} ({median(relative_gaps) * 100:.2f}%)")
    print(f"  Std:    {stdev(relative_gaps):.4f}")
    print(f"  Max:    {max(relative_gaps):.4f} ({max(relative_gaps) * 100:.2f}%)")
    print(f"  Min:    {min(relative_gaps):.4f}")

    # Absolute gap statistics
    print("\n--- Absolute Gap (result - target) ---")
    print(f"  Mean:   {mean(absolute_gaps):.2f}")
    print(f"  Median: {median(absolute_gaps):.2f}")
    print(f"  Max:    {max(absolute_gaps)}")

    # Percentiles
    sorted_rel = sorted(relative_gaps)
    p95 = sorted_rel[int(0.95 * len(sorted_rel))]
    p99 = sorted_rel[int(0.99 * len(sorted_rel))]
    print("\n--- Percentiles (Relative Gap) ---")
    print(f"  95th: {p95:.4f} ({p95 * 100:.2f}%)")
    print(f"  99th: {p99:.4f} ({p99 * 100:.2f}%)")

    return {
        "mean_rel": mean(relative_gaps),
        "median_rel": median(relative_gaps),
        "std_rel": stdev(relative_gaps),
        "max_rel": max(relative_gaps),
    }


def main():
    configs = [
        [(2, 46), (3, 3), (5, 5)],
        [(2, 46), (3, 3), (5, 2)],
        [(2, 50), (3, 4), (5, 2)],
    ]
    
    for primes in configs:
        analyze_padding(primes)


if __name__ == "__main__":
    main()
