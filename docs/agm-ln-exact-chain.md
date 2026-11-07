# Theta-AGM logarithm: exact parameter chain N → s → n

The chain fixes the theta truncation `N` once, by tuning. The scale `s` then follows from `N` and the precision, and the AGM iteration count `n` follows from the actual `s` and the precision. Every `n` produced this way is the exact minimal integer, not an estimate.

## Variables

| symbol | definition |
|---|---|
| `p` | target relative precision of `ln s` in bits (the working precision) |
| `ε` | total relative error budget for `ln s`, `ε = 2⁻ᵖ` |
| `ε_θ`, `ε_A` | split of `ε` between theta truncation and AGM, `ε_θ + ε_A ≤ ε`; default `ε_θ = ε_A = 2^{−(p+1)}` |
| `x` | input |
| `m` | integer scale exponent |
| `s` | scaled argument, `s = x·2^m`, `s > 1` |
| `L` | `log2 s` |
| `q` | `1/s` |
| `Q` | theta nome, `Q = q⁴ = s⁻⁴`, so `ln(1/Q) = 4 ln s` |
| `Q'` | complementary nome, `Q' = exp(−π² / (4 ln s))` |
| `N` | theta truncation index: θ3 keeps `k = 1..N`, θ2 keeps `k = 0..N` |
| `n` | number of AGM iterations (sqrt steps) |
| `r_n` | AGM nome after `n` steps, `r_n = Q'^{2ⁿ}` |
| `λ` | AGM limit, `λ = AGM(θ2(Q)², θ3(Q)²)` |
| `C` | stopping-criterion constant (see Stage 3) |
| `R` | cost of one theta term ÷ cost of one AGM step |

## Identity

```
θ2(Q) = 2q · Σ_{k≥0} Q^{k(k+1)}
θ3(Q) = 1 + 2 · Σ_{k≥1} Q^{k²}

ln s = π / (4λ)
```

This holds exactly for all `0 < q < 1`. Then:

```
ln x = ln s − m·ln2
```

## Stage 1: choose N (input-independent)

Minimize `R·(N+1) + n`. The continuous optimum is

```
N + 1 = 2 / (R·ln2)
```

Round to an integer and treat it as a constant. It does not depend on `p` or `x`.

Validity: Stage 2 needs `L_min ≥ 1` (so that `Q ≤ 1/16`), which means `p + 3 ≥ 4(N+1)²`. Use a smaller `N` below that precision.

## Stage 2: N, p → s

**Theta truncation error.** Let `T = 2·Σ_{k>N} Q^{k²}` be the dropped θ3 tail. For `Q ≤ 1/16`:

```
2·Q^{(N+1)²}  ≤  T  ≤  2·Q^{(N+1)²}·(1 + 2Q^{2N+3})
```

The relative error this puts on θ3² is `2T + T² ≈ 4·Q^{(N+1)²}·(1 + O(Q))`. The θ2 tail is smaller by a factor `Q^{N+1}`.

**Propagation through the AGM.** AGM is homogeneous of degree 1 and monotone in each argument. So relative errors `δ_a, δ_b` in its inputs give a relative error in `λ` of at most `max(δ_a, δ_b)`. More precisely:

```
∂ ln λ / ∂ ln b = 1 − 1/ln(4b/a)
∂ ln λ / ∂ ln a =     1/ln(4b/a)
```

The relative error of `λ` passes unchanged into `ln s`.

**Condition.** Require `4·Q^{(N+1)²} ≤ ε_θ`:

```
4(N+1)² · ln s  ≥  ln(4/ε_θ)

L_min = log2(4/ε_θ) / (4(N+1)²)
```

With the default split this is `L_min = (p + 3) / (4(N+1)²)`.

**Choice of s.** Any `L ≥ L_min` is valid. The scale exponent is

```
m = ⌈L_min⌉ − ⌊log2 x⌋        (or rounded up further, e.g. to a multiple of 64)
```

so in general `L ∈ [L_min, L_min + Δ)`, where `Δ` is the overshoot from rounding `m` and from the mantissa of `x`.

## Stage 3: s, p → n (exact)

**Closed form of the AGM sequence.** Under the imaginary transformation `ln(1/Q)·ln(1/Q') = π²`, the AGM squares the complementary nome at every step:

```
a_n = λ·θ3(r_n)²
b_n = λ·θ4(r_n)²
r_n = Q'^{2ⁿ} = exp( −2ⁿ · π² / (4 ln s) )
```

**Error at step n.** The bounds below hold exactly for `0 < r ≤ 1/8`:

| quantity | exact form | bounds |
|---|---|---|
| gap `(a_n − b_n)/a_n` | `2θ2(r²)² / θ3(r)²` | `8r(1 − 4r) ≤ · ≤ 8r` |
| error of `a_n` | `θ3(r)² − 1` | `4r ≤ · ≤ 4r(1 + 2r)` |
| error of `(a_n + b_n)/2` | `θ3(r²)² − 1` | `4r² ≤ · ≤ 4r²(1 + 2r²)` |

At the stopping point `r ≈ ε_A ≪ 1`, the correction factors are `1 + O(2⁻ᵖ)`.

**Condition.** The stopping condition is `C·r_n ≤ ε_A`, i.e. `2ⁿ · π² / (4 ln s) ≥ ln(C/ε_A)`. Solving for the minimal integer:

```
n = max(0, ⌈ log2( 4 · ln s · ln(C/ε_A) / π² ) ⌉)
```

| stopping rule | C | n with default split (`ε_A = 2^{−(p+1)}`) |
|---|---|---|
| return `a_n` | 4 | `⌈ log2( 4·ln s·(p+3)·ln2 / π² ) ⌉` |
| return `(a_n + b_n)/2` | `4r²` form | `⌈ log2( 2·ln s·(p+3)·ln2 / π² ) ⌉` (one fewer sqrt) |
| stop when gap `< ε'` | 8 | `⌈ log2( 4·ln s·ln(8/ε')/π² ) ⌉`; with `ε' = 2⁻ᵖ`: `⌈ log2( 4·ln s·(p+3)·ln2 / π² ) ⌉` |

The input enters only through `ln s`, where `s` is the actual scaled value, rounding included.

**Evaluating the ceiling.** `ln s` only needs to be known to about 53 bits, from the exponent and the top bits of the mantissa. To guard against floating-point rounding when the argument is within roughly 1e-15 of a power of 2, add a relative margin of about 1e-12 before taking the ceiling. A wrong guess then costs at most one extra iteration, never one too few.

## Relation to N (approximate)

At `L = L_min` (no overshoot), substitute `ln s = ln(4/ε_θ) / (4(N+1)²)`:

```
n = ⌈ log2( ln(4/ε_θ) · ln(C/ε_A) / (π²(N+1)²) ) ⌉
  = ⌈ 2·log2( (p+3)·ln2 / (π(N+1)) ) ⌉              (default split, C = 4)
```

Equivalently, `(N+1)² · 2ⁿ ≳ ((p+3)·ln2/π)²`.

This is exact only when `s = s_min`. An overshoot `Δ` raises the argument by `log2(1 + Δ/L_min)`, so

```
n(s) ∈ [ n(s_min),  n(s_min) + ⌈log2(1 + Δ/L_min)⌉ ]
```

Always compute `n` from the actual `s` (Stage 3).

## Admissible window

For fixed `(N, n)`, every `L` in this range satisfies both conditions:

```
log2(4/ε_θ) / (4(N+1)²)  ≤  L  ≤  2ⁿ · π² / (4 · ln2 · ln(C/ε_A))
```

The lower edge minimizes `m`. The upper edge is the overshoot that can be absorbed without increasing `n`.

## Summary

```
N        = round( 2/(R·ln2) − 1 )                              tuned once
L_min    = log2(4/ε_θ) / (4(N+1)²)                             from N, p
m        = ⌈L_min⌉ − ⌊log2 x⌋   (rounded up as needed)         from x
s        = x·2^m
n        = ⌈ log2( 4·ln s·ln(C/ε_A) / π² ) ⌉                   from s, p — exact
ln x     = π / (4·AGM_n(θ2², θ3²)) − m·ln2
```

## Verification

The Stage 3 formula (gap test, `ε' = 2⁻ᵖ`) was compared with iteration counts from a direct AGM run. The test covered:

- `p ∈ {256, 1024, 4096, 16384}`
- `N = 0..9`
- `L` from `L_min + 1` up to `L_min + 41`
- random mantissa

It matched in 160 of 160 cases. That includes boundary cases such as `p = 65536`, `N = 4`, where the argument is `2^{23.0006}` and the formula gives `n = 24` as observed.

The bounds in the Stage 2 and Stage 3 tables were checked numerically over `0 < r ≤ 1/8` and `Q ∈ {1/16, 10⁻³}`, `N = 0..7`.

## Implementation (`ln_agm_dyn_at`, `ln_agm_static_at` in src/utils/trans.rs)

**q without a reciprocal.** The code passes theta q = X·2^-shift, so s := 1/q = 2^shift/X and

```
ln X = c·ln2 − ln(1/q) = c·ln2 − π/(4λ),     c = shift + 64b   for X = x·2^(64b)
```

The chain above applies unchanged with this s. With shift = bits(x) + l, q ∈ [2^-(l+1), 2^-l) and c = ⌊log2 X⌋ + 1 + l.

**Frames**, for o output limbs and δ = |u − 1|'s lead λ (u = X·2^-⌊log2 X⌋, or X itself on [1/2, 1)):

| quantity | value | why |
|---|---|---|
| `f` | `o + 1`, plus `⌈(λ + 2)/64⌉` when X ∈ [1/2, 2) | fraction limbs of T and of the result: c·ln2 and T cancel to \|ln X\| > 2^-(λ+2) |
| `p` | `64(f + 1)` | the chain's working precision: relative error 2^-p in λ is absolute error T·2^-p, T < 2^64 |
| `l` | `max(2, ⌈(p + 3)/(4(N+1)²)⌉)` | Stage 2's L_min; theta needs q < 1/4 |
| `g` | `f + 2 + ⌈l/64⌉` | the AGM's fraction limbs; see below |
| `n` | none: the AGM stops on its own gap | Stage 3's "stop when gap < ε′" row, read from bit lengths each step; with no tail it takes the (a_n + b_n)/2 row's n |

The AGM frame: its smaller input after the first step is b1 = θ2θ3 ≈ 2q, and ∂ln λ/∂b1 ≈ 1/(b1·ln(4a1/b1)) ≈ 2^l/(l·ln2). A unit at 2^-64g therefore moves T ≈ l·ln2 by about 2^(l − 64g), and the 1/l cancels. So 64g ≥ 64f + l + O(1), and the two extra limbs absorb the rounding of every step.

**Free and replaced iterations.**
- The first step's pair is formed directly: a1 = (θ2² + θ3²)/2 = (θ2 + θ3)²/2 − θ2θ3 and b1 = θ2θ3, one square and one product instead of two squares, a product and a root.
- The AGM ends in the closed-form series of its limit, AGM(m + D, m − D) = m·(1 − u/4 − 5u²/64 − 11u³/256 − …), u = (D/m)², whose zeroth term is the old final (a_n + b_n)/2 without its root. Each step before it doubles the gap and halves the terms, so ending j steps early takes about 2^(j−1)…2^j − 1 terms. `agm_dyn` ends 3 roots early (6–7 terms) and `agm_static` 2 (3 terms), set by `DYN_AGM_TAIL_TERMS`/`STATIC_AGM_TAIL_TERMS`; see `docs/agm_profile.md`.
- Starting from (a1, b1), the means-only AGM takes n − 1 roots (n from the table above), against the n + 1 roots of returning a_n from (θ2², θ3²); the tail leaves n − 1 − j.
- The AGM targets its own frame's last unit, about l + 64 bits finer than p, which the series reaches with at most a term more.

**Geometric means from half the product.** √(a·b) to ⌈t/2⌉ limbs needs only the product's top ⌈t/2⌉ + 2 limbs. The root reads them as padded by zeros, which moves it by under 2^-33 units. Each step's buffers then fit out.len() + 2 limbs, which lets the static AGM run at N ≈ g + 2.

**Dispatch** (`agm_wins`). Per M(P), with P = 64o bits:
- **Series:** about L², with L = log2(P / max(λ, j0)). Bit-burst chunk [s, 2s) sums in log2(P/s) binary-splitting levels, chunks run up to P/3, and j0 = A·log2 P is how far the series' square roots move δ.
- **AGM:** about κ·(1 + w/P)·log2(P + w), with w = λ when it must widen.

The constant κ makes the two tie at the cutoff for a generic X on one thread. Since L²/log2 P grows with P, the AGM takes a generic X exactly from the cutoff. It never takes any X below the cutoff: the series only gets cheaper as λ grows and the AGM only dearer. At fixed P it takes leads up to about P^(1 − √(log P*/log P)), so X near 1, or near any power of 2, stays on the series longer.

**Threads.** The two methods use extra threads very differently, so the series' cost is divided by its speedup S(k) on k threads (`ln_series_speedup`, `ln_takes_agm`):
- **Series:** each bit-burst chunk's series and its binary-splitting tree run on their own threads. Only `ln_chain`'s residual updates, the tail's ratio and the final combine are serial.
- **AGM:** a chain of means, each step needing the one before. Only theta's powers run in parallel, plus products past the 16-bit FFT's limit (about 65533 limbs), which go to the NTT and its thread pool.
- **The model:** Amdahl's law, S(k) = 1/((1 − p) + p/k), with p = 0.85 (`DYN_LN_SERIES_PAR_FRACTION`). k is the rayon pool's size for `ln_dyn`, and `available_parallelism` for `ln_static`, whose series and theta use scoped threads.

Measured on the M1 (4 performance and 4 efficiency cores) on generic X at T = 5, as the AGM's time over the series' (below 1, the AGM wins):

| Limbs | 1 thread | 2 threads | 4 threads | 8 threads |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.06 | 1.52 | 2.29 | 2.47 |
| 1024 | 0.84 | 1.44 | 2.17 | 2.38 |
| 2048 | 0.77 | 1.37 | 2.10 | 2.30 |
| 4096 | 0.70 | 1.29 | 2.01 | 2.51 |
| 16384 | 0.66 | 1.14 | 1.77 | 1.87 |
| 32768 | 0.60 | | | 1.74 |
| 65536 | 1.08 | 1.36 | 1.63 | 1.46 |
| 131072 | | 1.02 | | 1.22 |
| 262144 | | | | 1.06 |

- **Series speedup:** 1.5–1.9× on 2 threads, 2.2–2.9× on 4 and 2.4–3.7× on 8, which p = 0.85 fits (1.74, 2.76, 3.90). The 8-thread figure is held down by the efficiency cores.
- **AGM speedup:** 1.02–1.05× wherever its products stay on the FFT, and 1.4–1.8× at 65536 limbs, where they run on the parallel NTT.
- **Where the AGM wins under the model:** on one thread from the cutoff; on 2 threads from 1M–2M limbs; on 3 from 2^29 limbs; and from 4 threads on, effectively never.
- **Unmodeled, at the top of the range:**
  - Past the FFT's limit the AGM gains threads the model doesn't credit, and the gap narrows to 1.02× on 2 threads at 131072 limbs and 1.06× on 8 at 262144, so the AGM may win beyond the measured sizes.
  - On one thread the same limit costs the AGM its lead at 65536 limbs (1.08), which the model doesn't see either.
- **Accuracy:** both methods returned bit-identical results at every size measured.
- **Static:** only measured on 8 CPUs, since macOS can't limit `available_parallelism`. There the series won by 2.3–2.5× from 1024 to 4096 limbs.
