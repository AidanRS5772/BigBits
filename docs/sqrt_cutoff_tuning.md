# Square-root cutoff tuning

Machine: Intel i7-8750H, `powersave` governor, pinned to one core (`taskset -c 3`).
Probe: `benches/probes/sqrt_cutoffs.rs`, built by `scripts/python/tune_sqrt_cutoffs.py`
from a constant-cutoff copy of `src/utils/sqrt.rs` under `target/`. Every timed
candidate is first validated against `binom_sqrt`; variants alternate within each
round and the median of 7-15 rounds of thread CPU time is used.

The former single `ZIMMERMAN_SQRT_CUTOFF = 17` is replaced by seven constants:

| Constant | Old | New | 0.2% plateau |
|---|---:|---:|---|
| `ZIMMERMAN_SQRT_LEAF_CUTOFF` (recursive) | 17 | 15 | 12-20 end-to-end |
| `DYN_SQRT_REM_ZIMMERMAN_CUTOFF` | 17 | 20 | 18-23 |
| `DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 17 | 15 | 12-17 |
| `DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 17 | 15 | 11-17 |
| `STATIC_SQRT_REM_ZIMMERMAN_CUTOFF` | 17 | 20 | 18-23 |
| `STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 17 | 15 | 12-17 |
| `STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 17 | 14 | 11-16 |

## Recursive leaf

`core` mode times one Zimmermann descent whose half-width child is solved by
binomial sqrt against `binom_sqrt_core` at the same root width, which is exactly
the decision made at each level of `zimmermann_sqrt_core`. Ratio descent/binomial:

| Root limbs | 10 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| dyn | 1.066 | 1.053 | 1.038 | **0.988** | 0.969 | 0.966 | 0.942 | 0.924 | 0.882 | 0.834 |
| static, N=next_pow2(2n) | 1.070 | 1.048 | 1.013 | 1.007 | **0.966** | 0.971 | 0.925 | 0.925 | 0.907 | 0.846 |
| static, N=1024 | 1.060 | 1.060 | 1.034 | 1.012 | **0.994** | 0.958 | 0.949 | 0.963 | 0.928 | 0.851 |
| static, N=4096 | 1.193 | 1.169 | 1.112 | 1.094 | 1.015 | 1.018 | **0.977** | 0.967 | 0.930 | 0.861 |

Complete root-remainder calls at root widths 30-512 (dyn, static with tight N,
static with N=4096) sweeping the leaf over 12-20 are flat: every leaf's geometric
mean is within 1.0-2.8% of the per-size best, which is inside run-to-run noise.
The one-step crossover is therefore the deciding signal, and one shared leaf of 15
is used; the recursion is shared by both allocation models.

## Top-level cutoffs

Each output mode and allocation model compares `binom_sqrt_output` against the
Zimmermann entry (leaf 15) for root widths 4-48 across four input shapes:
`full` (`x.len() = 2n`), `near` (`2n-1`), `mid` (`3n/2+1`) and `padded` (`n+1`),
with alternating normalized and shifted top limbs. The cutoff minimizes the
geometric-mean slowdown against the per-case best over all sizes and shapes. Two
independent runs (seeds differ; 7x10 ms and 15x15 ms) agree within one limb.

Geometric-mean Zimmermann/binomial over shapes, combined runs:

| Root limbs | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dyn rem | 1.14 | 1.13 | 1.11 | 1.11 | 1.07 | 1.04 | 1.01 | 1.01 | 0.98 | 0.97 | 0.96 |
| dyn only | 1.01 | 1.02 | 1.03 | 0.98 | 0.98 | 0.93 | 0.94 | 0.90 | 0.91 | 0.87 | 0.90 |
| dyn approx | 1.02 | 1.01 | 1.01 | 0.99 | 0.97 | 0.93 | 0.92 | 0.92 | 0.88 | 0.88 | 0.87 |
| static rem | 1.14 | 1.10 | 1.11 | 1.08 | 1.07 | 1.05 | 1.09 | 1.02 | 1.08 | 0.98 | 0.97 |
| static only | 1.03 | 1.01 | 1.02 | 0.98 | 0.98 | 0.95 | 0.94 | 0.91 | 0.91 | 0.88 | 0.89 |
| static approx | 1.00 | 1.00 | 0.99 | 0.99 | 0.95 | 0.94 | 0.95 | 0.92 | 0.89 | 0.87 | 0.88 |

Root-remainder calls switch later because the Zimmermann entry always pays for
the final low square and denormalization, while root-only calls usually return
before squaring.

The crossover also depends on the input shape, which dispatch does not see:
`full` and `padded` inputs favor Zimmermann near 10-12 (only/approx) or 17
(rem), while `near` and `mid` favor it only from 16-24. The constants are the
best single threshold on root width.

## Static capacity

The static constants were measured with `N = next_pow2(2n)`, the natural case in
which `x` nearly fills the capacity. Static division and squaring zero-initialize
`[u64; N]` stack scratch on every call, so their cost grows with `N` rather than
with the operands. With oversized capacity, the best cutoffs rise:

| Capacity | rem | only | approx |
|---|---:|---:|---:|
| next_pow2(2n) | 20 | 15 | 14 |
| 1024 | 23 | 15 | 20 |
| 4096 | >= 49 | 17 | 37 |

At N=4096, Zimmermann is 2.0-2.4x slower than binomial at 8 root limbs in the
rem and approx modes. The approx mode suffers most because quotient-only
`div_static` copies operands into additional capacity-sized buffers. A
capacity-aware threshold, or scratch that avoids zeroing all `N` limbs, would
remove this sensitivity.

## Reproduction

```bash
cargo bench-cutoffs --no-run
python3 scripts/python/tune_sqrt_cutoffs.py --rlib target/release/deps/libbig_bits-HASH.rlib \
  --output target/sqrt_cutoff_tuning/v4
cd target/sqrt_cutoff_tuning/v4
# Recursive leaf: one descent (4096) vs binomial (0).
taskset -c 3 ./sqrt_cutoffs --sizes 4:48 --leaves 0,4096 --families dyn,static \
  --modes core --pattern random --rounds 9 --ms 25
# Top level: binomial (0) vs Zimmermann with leaf 15.
taskset -c 3 ./sqrt_cutoffs --sizes 4:48 --leaves 0,15 --modes rem,only,approx \
  --shapes full,near,mid,padded --pattern mixed --rounds 7 --ms 10 --families dyn,static
```
