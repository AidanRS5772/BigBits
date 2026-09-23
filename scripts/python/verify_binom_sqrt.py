#!/usr/bin/env python3
"""Check the binom_sqrt_verify adapter against Python's exact math.isqrt."""

import argparse
import math
import random
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    args = parser.parse_args()
    rng = random.Random(0x53515254)
    mask = (1 << 64) - 1
    cases = []
    for n in range(1, 64):
        for width in range(n + 1, 2 * n + 1):
            for pattern in range(4):
                limbs = [rng.getrandbits(64) for _ in range(width)]
                if pattern == 0:
                    limbs[-1] |= 1 << 63
                elif pattern == 1:
                    limbs[-1] = 1
                elif pattern == 2:
                    limbs[-1] = 1 << ((n + width) % 64)
                else:
                    limbs = [mask] * width
                cases.append((n, limbs))
    for n in (1, 2, 3, 4, 8, 16, 32, 63, 64, 128, 256):
        for _ in range(16):
            root = rng.getrandbits(64 * n) | (1 << (64 * n - 1))
            for value in (root * root - 1, root * root, root * root + 1,
                          root * root + 2 * root):
                cases.append((n, [(value >> (64 * i)) & mask for i in range(2 * n)]))
    for h in (1 << 63, (1 << 63) + 1, mask - 1, mask):
        for rem in (0, 1, h - 1, h, h + 1, 2 * h - 1, 2 * h):
            hi = h * h + rem
            for a in (0, 1, 1 << 63, mask - 1, mask):
                for b in (0, 1, mask):
                    x = [b, a, hi & mask, hi >> 64]
                    cases.append((2, x))
                    if b == 0:
                        cases.append((2, x[1:]))
    data = "".join(f"{n} " + " ".join(f"{limb:x}" for limb in x) + "\n" for n, x in cases)
    result = subprocess.run([args.binary], input=data, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    for (n, x), line in zip(cases, lines):
        root_text, remainder_text = line.split("|")
        actual_root = sum(int(v, 16) << (64 * i) for i, v in enumerate(root_text.split()))
        actual_rem = sum(int(v, 16) << (64 * i) for i, v in enumerate(remainder_text.split()))
        value = sum(v << (64 * i) for i, v in enumerate(x)) << (64 * (2 * n - len(x)))
        expected = math.isqrt(value)
        assert actual_root == expected, (n, x, actual_root, expected)
        assert actual_rem == value - expected * expected, (n, x, actual_rem)
    print(f"PASS: {len(cases)} exact Python math.isqrt root/remainder comparisons")


if __name__ == "__main__":
    main()
