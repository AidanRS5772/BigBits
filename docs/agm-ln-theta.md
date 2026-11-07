# Theta-AGM logarithm: relation between n, N, and s

## Notation

| symbol | meaning |
|---|---|
| `p` | target precision in bits (relative error ≤ 2⁻ᵖ) |
| `s` | scaled argument, `s = x·2^m`, `s > 1` |
| `q` | `1/s` |
| `Q` | theta nome, `Q = q⁴ = s⁻⁴` |
| `Q'` | complementary nome |
| `N` | theta truncation index: θ3 keeps `k = 1..N`, θ2 keeps `k = 0..N` |
| `n` | number of AGM iterations |

## Identity

```
θ2(q⁴) = 2q · Σ_{k≥0} q^{4k(k+1)}
θ3(q⁴) = 1 + 2 · Σ_{k≥1} q^{4k²}

ln s = π / (4 · AGM(θ2², θ3²))      exact for all 0 < q < 1
```

## Modular structure

The Jacobi imaginary transformation connects the nome `Q` with its complement `Q'`:

```
θ3(Q) = √(π/ln(1/Q)) · θ3(Q')
θ2(Q) = √(π/ln(1/Q)) · θ4(Q')
ln(1/Q) · ln(1/Q') = π²
```

In terms of `s`:

```
ln(1/Q)  = 4·ln s
ln(1/Q') = π² / (4·ln s)
```

One AGM step maps the pair at nome `r` to the pair at nome `r²`:

```
((θ3² + θ4²)/2, θ3·θ4)(r) = (θ3², θ4²)(r²)
```

As a result, after `n` steps the pair sits at nome `Q'^{2ⁿ}`. The theta series is evaluated at `Q`, and the AGM squares its way down from `Q'`. Both describe the same object, seen from opposite ends of the transformation.

## Error conditions

| stage | leading error | p-bit condition |
|---|---|---|
| theta series, truncated at `N` | `2·Q^{(N+1)²}` | `(N+1)² · ln(1/Q) ≈ p·ln2`   **(B)** |
| AGM, `n` steps | `8·Q'^{2ⁿ}` | `2ⁿ · ln(1/Q') ≈ p·ln2`   **(A)** |

The θ2 tail, `Q^{(N+1)(N+2)}`, is always smaller than the θ3 tail, so (B) is set by θ3.

## Core relation

Multiply (A) by (B). Since `ln(1/Q)·ln(1/Q') = π²`, the scale `s` cancels:

```
(N+1)² · 2ⁿ ≈ (p·ln2 / π)²

n ≈ 2·log2( p·ln2 / (π(N+1)) )
```

## Dependence on s

Substituting `ln(1/Q) = 4 ln s` into (B) and `ln(1/Q') = π²/(4 ln s)` into (A):

```
N+1 ≈ √( p / (4·log2 s) )                 from (B)
n   ≈ log2( 4·p·(ln2)²·log2 s / π² )       from (A)
```

The scale `s` selects a point on the core relation:

- **Increasing `s`:** `Q` gets smaller, so fewer theta terms are needed. `Q'` moves toward 1, so more AGM steps are needed.
- **Doubling `log2 s`:** adds 1 to `n` and divides `N+1` by √2.
- **Doubling `N+1`:** removes exactly 2 from `n`, because `n` depends only on `p/(N+1)`.

The two extremes:

- **`N = 0`, `log2 s ≈ p/4`:** classical Brent–Salamin, `n ≈ 2·log2(0.22·p)`.
- **`n = 0`:** requires `N+1 ≈ 0.22·p` terms, so the series alone does O(p) work.

## Admissible window for s

Once `(N, n)` is fixed, any `s` in this range meets both conditions:

```
p / (4(N+1)²)  ≤  log2 s  ≤  2ⁿ · π² / (4·(ln2)²·p)
     (B)                          (A)
```

Example: `p = 65536`, `N = 3`, `n = 24` gives `1024 ≤ log2 s ≤ ~1314`.

## Causal tuning chain

```
p  →  (n, N)  →  window for log2 s  →  m = log2 s − log2 x
```

1. **Choose `n` as an integer, then take the minimal `N`.** Because `n` is discrete, any `N` between two thresholds buys nothing.
   ```
   N+1 = ⌈ p·ln2 / (π·2^{n/2}) ⌉
   ```
2. **Pick the `(n, N)` pair with the lowest cost.** Let `c_A` be the cost of one AGM step and `c_θ` the cost of one theta term, both in units of a p-bit multiply:
   ```
   cost ≈ c_A·n + c_θ·N
   ```
   The continuous optimum is
   ```
   N+1 ≈ 2·c_A / (c_θ·ln2)
   ```
   This does not depend on `p`, assuming multiplication cost is linear in precision.
3. **Set `s` from `N`.** Use the lower edge of the window, `log2 s ≈ p / (4(N+1)²)`. The upper edge is the slack you have before `n` increases by 1.
4. **Get `m` from `s`.** The input enters only through `m = log2 s − log2 x`. Nothing in the `(n, N)` choice depends on `x`.

## Check (p = 65536, iterate until |a−b|/a < 2⁻ᵖ)

| N | log2 s | n measured | n predicted |
|---|---|---|---|
| 0 | 16386 | 28 | 27.64 |
| 1 | 4098 | 26 | 25.64 |
| 2 | 1822 | 25 | 24.47 |
| 3 | 1026 | 24 | 23.64 |
| 4 | 657 | 24 | 23.00 |
| 5 | 457 | 23 | 22.47 |
| 7 | 258 | 22 | 21.64 |

`N = 4` gives the same `n` as `N = 3`, which is the discreteness described in step 1.

## Computing the powers (`src/utils/theta_plan.rs`)

The series needs q^(n²) for n = 1..2N+1: odd n go to θ2 and even n to θ3. A power's window is about a fraction `1 − n²/(4(N+1)²)` of the frame, which depends only on N. So for each N the cheapest way to build the powers can be fixed at compile time.

Squares cost about ⅔ of a product. The old recurrence, `q^((n+1)²) = q^(n²)·q^(2n+1)`, spends two products per term. The plans mix squares and products instead: for N = 3, `2 4 8 9 16 25 18 36 13 49` is 6 squares and 4 products, about 0.59× the old work once each step is weighted by its width.

There are two plans per N:

| plan | built by | objective |
|---|---|---|
| `SERIAL` | greedy, targets in order (one op if possible, else one helper) | least width-weighted work |
| `PARALLEL` | earliest-finish DP over every exponent ≤ (2N+1)² (squares 2, products 3) | shortest dependency path |

`SERIAL` takes whichever of the two does less work. Small frames run `SERIAL` in order. Large frames run `PARALLEL`, threaded as a dataflow with one join: each op starts once its operands finish and a slot is free. The plan depends only on the frame size, so threading never changes the result.

Because the work per term pair stays roughly constant (about 1.46 product-widths asymptotically, against a floor near 1.1), theta's cost still grows linearly in N while the AGM saves only 2 steps per doubling of N+1. Cheaper powers move the best N from about 3 to about 5–6, not further.

Measured end to end (`cargo bench-bin-split lnagm`, M1, 2026-11-07, with the AGM's closed-form tail), on generic X:

| | best N | N = 3 | N = 5 | N = 6 | N = 8 |
|---|---|---|---|---|---|
| dynamic, 1 thread, 1024–32768 limbs | 5 | 1.000–1.007 | 1.000 | 1.000–1.017 | 1.012–1.045 |
| dynamic, 8 threads, 1024–32768 limbs | 10–12 (8 at 32768) | 1.044–1.087 | 1.024–1.055 | 1.014–1.040 | 1.000–1.022 |
| static, 1024 and 4096 limbs (1 and 8 threads) | 6 | 1.017–1.027 | 1.007–1.016 | 1.000 | 1.049–1.065 |

The defaults are N = 5 (`DYN_LN_AGM_TERMS`) and 6 (`STATIC_LN_AGM_TERMS`). On several threads the parallel plan's terms are nearly free next to the serial AGM, which moves the best N past 10, but there the series logarithm is about twice as fast as the AGM at every size to 32768 limbs. Static's theta threads through `available_parallelism`, not the rayon pool, so its rows hardly move with the pool. Past N = 6 its parallel plan no longer fits `THETA_STATIC_SLOTS` (8 and 9 run the in-order plan), and N = 8 and 9 measured 5–7% slower.
