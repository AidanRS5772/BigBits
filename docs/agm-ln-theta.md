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
