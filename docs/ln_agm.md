# The theta–AGM logarithm: parameters and frames

The mathematics behind `ln_agm_dyn_at`/`ln_agm_static_at`, `theta_*` and `agm_*` in
`src/utils/trans.rs`. The code's layout is in `trans.md`.

## Notation

| Symbol | Meaning |
|---|---|
| p | target relative precision of ln s in bits; ε = 2^-p, split ε_θ + ε_A ≤ ε (default 2^-(p+1) each) |
| s | scaled argument, s = x·2^m > 1; L = log2 s; q = 1/s |
| Q, Q′ | theta nome Q = q⁴ and its complement, ln(1/Q)·ln(1/Q′) = π² |
| N | theta truncation: θ3 keeps k = 1..N, θ2 keeps k = 0..N (the code's const generic T) |
| n | AGM iterations |
| λ | the AGM's limit, AGM(θ2(Q)², θ3(Q)²) |
| R | cost of one theta term ÷ cost of one AGM step |

## Identity

    θ2(Q) = 2q·Σ_{k≥0} Q^(k(k+1))       θ3(Q) = 1 + 2·Σ_{k≥1} Q^(k²)
    ln s = π / (4λ)                      exact for all 0 < q < 1
    ln x = ln s − m·ln 2

**Modular structure.** The Jacobi imaginary transformation gives θ3(Q) = √(π/ln(1/Q))·θ3(Q′) and
θ2(Q) = √(π/ln(1/Q))·θ4(Q′), with ln(1/Q) = 4 ln s and ln(1/Q′) = π²/(4 ln s). One AGM step maps
the pair at nome r to the pair at r²: ((θ3² + θ4²)/2, θ3θ4)(r) = (θ3², θ4²)(r²). So after n steps
the pair sits at Q′^(2^n): the theta series is evaluated at Q and the AGM squares its way down
from Q′.

## The core relation

| Stage | Leading error | p-bit condition |
|---|---|---|
| Theta series truncated at N | 2·Q^((N+1)²) | (N+1)²·ln(1/Q) ≈ p·ln 2 (B) |
| AGM, n steps | 8·Q′^(2^n) | 2^n·ln(1/Q′) ≈ p·ln 2 (A) |

The θ2 tail is always smaller than θ3's, so (B) is set by θ3. Multiplying (A) by (B), s cancels:

    (N+1)²·2^n ≈ (p·ln 2 / π)²        n ≈ 2·log2(p·ln 2 / (π(N+1)))

s selects a point on that relation: N + 1 ≈ √(p/(4L)) and n ≈ log2(4p·(ln 2)²·L/π²). Doubling L
adds 1 to n and divides N + 1 by √2; **doubling N + 1 removes exactly 2 from n**. The extremes are
N = 0, L ≈ p/4 (Brent–Salamin, n ≈ 2·log2(0.22p)) and n = 0 with N + 1 ≈ 0.22p terms.

## The exact chain N → s → n

**Stage 1: N, once, by tuning.** Minimizing R·(N+1) + n gives N + 1 = 2/(R·ln 2), independent of p
and x. Needs p + 3 ≥ 4(N+1)².

**Stage 2: N, p → s.** The dropped θ3 tail T satisfies 2Q^((N+1)²) ≤ T ≤ 2Q^((N+1)²)(1 + 2Q^(2N+3))
for Q ≤ 1/16, a relative error of about 4Q^((N+1)²) on θ3². The AGM is homogeneous of degree 1 and
monotone, so input errors δ_a, δ_b move λ by at most max(δ_a, δ_b)
(∂ln λ/∂ln b = 1 − 1/ln(4b/a), ∂ln λ/∂ln a = 1/ln(4b/a)). Requiring 4Q^((N+1)²) ≤ ε_θ:

    L_min = log2(4/ε_θ) / (4(N+1)²)     = (p + 3)/(4(N+1)²) with the default split
    m = ⌈L_min⌉ − ⌊log2 x⌋              (or rounded up further)

**Stage 3: s, p → n, exactly.** a_n = λ·θ3(r_n)², b_n = λ·θ4(r_n)², r_n = Q′^(2^n). For 0 < r ≤ 1/8:

| Quantity | Exact form | Bounds |
|---|---|---|
| Gap (a_n − b_n)/a_n | 2θ2(r²)²/θ3(r)² | 8r(1 − 4r) ≤ · ≤ 8r |
| Error of a_n | θ3(r)² − 1 | 4r ≤ · ≤ 4r(1 + 2r) |
| Error of (a_n + b_n)/2 | θ3(r²)² − 1 | 4r² ≤ · ≤ 4r²(1 + 2r²) |

With the stopping condition C·r_n ≤ ε_A:

    n = max(0, ⌈log2(4·ln s·ln(C/ε_A)/π²)⌉)

C = 4 to return a_n, 8 to stop when the gap is below ε′; returning (a_n + b_n)/2 takes one root
fewer. The input enters only through ln s of the actual scaled value. An overshoot Δ in L raises n
by at most ⌈log2(1 + Δ/L_min)⌉.

**Window.** For fixed (N, n) every L in
[log2(4/ε_θ)/(4(N+1)²), 2^n·π²/(4·ln 2·ln(C/ε_A))] meets both conditions. Example: p = 65536,
N = 3, n = 24 gives 1024 ≤ L ≤ about 1314.

**Check** (p = 65536, iterate until the gap is below 2^-p): N = 0, 1, 2, 3, 4, 5, 7 took n = 28,
26, 25, 24, 24, 23, 22 against predictions 27.64, 25.64, 24.47, 23.64, 23.00, 22.47, 21.64. The
Stage 3 formula matched a direct AGM run in 160 of 160 cases (p from 256 to 16384, N = 0..9).

## Implementation

**q without a reciprocal.** The code passes theta q = X·2^-shift, so s = 2^shift/X and

    ln X = c·ln 2 − π/(4λ),     c = shift + 64b for X = x·2^(64b)

With shift = bits(x) + l, q is in [2^-(l+1), 2^-l) and c = ⌊log2 X⌋ + 1 + l.

**Frames** (`agm_frame`), for o output limbs and δ = |u − 1|'s lead λ:

| Quantity | Value | Why |
|---|---|---|
| f | o + 1, plus ⌈(λ + 2)/64⌉ when X is in [1/2, 2) | fraction limbs of T and the result: c·ln 2 and T cancel to \|ln X\| > 2^-(λ+2) |
| p | 64(f + 1) | relative error 2^-p in λ is absolute error T·2^-p, T < 2^64 |
| l | max(2, ⌈(p + 3)/(4(N+1)²)⌉) | Stage 2's L_min; theta needs q < 1/4 |
| g | f + 2 + ⌈l/64⌉ | the AGM's fraction limbs (below) |
| n | none | the AGM stops on its own gap, read from bit lengths each step |

*The AGM's frame.* Its smaller input after the first step is b1 = θ2θ3 ≈ 2q, and
∂ln λ/∂b1 ≈ 2^l/(l·ln 2). A unit at 2^-64g therefore moves T ≈ l·ln 2 by about 2^(l − 64g), so
64g ≥ 64f + l + O(1); two extra limbs absorb every step's rounding. Only b1 needs all l extra bits:
the steps from b2 on need about l/2 fewer.

**Free and replaced iterations.**

- The first pair is formed directly: a1 = (θ2² + θ3²)/2 = (θ2 + θ3)²/2 − θ2θ3 and b1 = θ2θ3, one
  square and one product in place of two squares, a product and a root.
- The AGM ends in the closed-form series of its limit,
  AGM(m + D, m − D) = m·(1 − u/4 − 5u²/64 − 11u³/256 − …), u = (D/m)², whose zeroth term is the old
  final (a_n + b_n)/2. Each step before it doubles the gap, so ending j steps early takes about
  2^(j−1)…2^j − 1 terms. `agm_dyn` ends 3 roots early and `agm_static` 2
  (`agm.<dyn|static>_tail_terms`).
- **Geometric means from half the product.** √(a·b) to ⌈t/2⌉ limbs needs only the product's top
  ⌈t/2⌉ + 2 limbs; reading them as padded moves the root by under 2^-33 units. Each step's buffers
  then fit out.len() + 2 limbs, which lets the static AGM run at N ≈ g + 2.

## Computing the powers

The series needs q^(n²) for n = 1..2N+1 (odd n to θ2, even to θ3). A power's window is about
1 − n²/(4(N+1)²) of the frame, which depends only on N, so the cheapest way to build the powers is
fixed at compile time per N (`theta_plan.rs`: a least-work `SERIAL` plan and a shortest-path
`PARALLEL` one). Squares cost about ⅔ of a product, and the plans mix both where the old recurrence
q^((n+1)²) = q^(n²)·q^(2n+1) spent two products a term: about 0.59× its work at N = 3.

The work per term pair stays roughly constant, so theta's cost grows linearly in N while the AGM
saves 2 steps per doubling of N + 1. Cheaper powers moved the best N from about 3 to 5–6, not
further.

## Choosing N, and series against the AGM (M1, 2026-11-07)

| | Best N | N = 3 | N = 5 | N = 6 | N = 8 |
|---|---|---|---|---|---|
| Dynamic, 1 thread, 1024–32768 limbs | 5 | 1.000–1.007 | 1.000 | 1.000–1.017 | 1.012–1.045 |
| Dynamic, 8 threads | 10–12 | 1.044–1.087 | 1.024–1.055 | 1.014–1.040 | 1.000–1.022 |
| Static, 1024 and 4096 limbs | 6 | 1.017–1.027 | 1.007–1.016 | 1.000 | 1.049–1.065 |

The knobs are `ln.dyn_agm_terms` = 5 and `ln.static_agm_terms` = 6. On several threads the parallel
plan's terms are nearly free beside the serial AGM, which moves the best N past 10, but there the
series wins anyway. Past N = 6 static's parallel plan no longer fits `THETA_STATIC_SLOTS`.

The AGM's time over the series' on a generic X at N = 5 (below 1 the AGM wins):

| Limbs | 1 thread | 2 threads | 4 threads | 8 threads |
|---:|---:|---:|---:|---:|
| 256 | 1.06 | 1.52 | 2.29 | 2.47 |
| 1024 | 0.84 | 1.44 | 2.17 | 2.38 |
| 4096 | 0.70 | 1.29 | 2.01 | 2.51 |
| 16384 | 0.66 | 1.14 | 1.77 | 1.87 |
| 32768 | 0.60 | | | 1.74 |
| 65536 | 1.08 | 1.36 | 1.63 | 1.46 |
| 262144 | | | | 1.06 |

The series' chunks and trees split across threads (1.5–1.9× on 2 threads, 2.4–3.7× on 8), while
the AGM is a chain of means (1.02–1.05× wherever its products stay on the FFT; 1.4–1.8× at 65536
limbs, where they run on the pooled NTT). On one thread the FFT limit costs the AGM its lead at
65536 limbs. Both methods returned bit-identical results at every size.

This table predates the tuner, which found the AGM ahead from about 32 limbs on one thread on the
M1. Dispatch reads two measured bounds, not a model of the speedup: `ln.<dyn|static>_agm_limbs` on
one thread and `ln.<dyn|static>_agm_par_until` on a pool (`trans.md`). An Amdahl fraction
(p = 0.85) scaling the one-thread cutoff was tried first and replaced.
