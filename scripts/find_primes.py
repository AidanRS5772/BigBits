import math
from sympy import isprime
from dataclasses import dataclass


@dataclass
class Prime:
    p: int
    k: int
    exps: list[int]


def min_smooth(radix_primes):
    val = 1
    for p, mn in radix_primes:
        val *= p**mn
    return val


def smoothes_in_range(lo, hi, k, radix_primes):
    results = []
    primes = [rp[0] for rp in radix_primes]
    mins = [rp[1] for rp in radix_primes]

    init_val = k
    for p, mn in radix_primes:
        init_val *= p**mn

    def recurse(val, idx, pows):
        if val >= hi:
            return

        if idx == len(primes):
            if val >= lo:
                results.append((val, pows.copy()))
            return

        p = primes[idx]
        exp = mins[idx]
        while val < hi:
            new_pows = pows.copy()
            new_pows[idx] = exp
            recurse(val, idx + 1, new_pows)
            val *= p
            exp += 1

    recurse(init_val, 0, list(mins))
    results.sort(reverse=True)
    return results


def find_primes(lo, hi, radix_primes, max_k_cap=None):
    prime_prod = math.prod(p for p, _ in radix_primes)

    def next_k(prev_k):
        k = prev_k + 1
        while math.gcd(k, prime_prod) != 1:
            k += 1
        return k

    ms = min_smooth(radix_primes)
    max_k_intrinsic = hi // ms
    max_k = max_k_intrinsic if max_k_cap is None else min(max_k_intrinsic, max_k_cap)

    found = []
    k = 1
    while k <= max_k:
        for val, pows in smoothes_in_range(lo - 1, hi - 1, k, radix_primes):
            if isprime(val + 1):
                found.append(Prime(p=val + 1, k=k, exps=pows))
        k = next_k(k)

    return found


def main():
    prime_config = [(2, 46), (3, 3), (5, 2)]
    primes = find_primes(
        1 << 62,
        1 << 63,
        prime_config,
    )
    print(f"Prime Config: {prime_config}")
    print(f"Found {len(primes)} primes.")
    for P in primes:
        print(f"  p = {P.p}, k = {P.k}, exps = {P.exps}")


if __name__ == "__main__":
    main()
