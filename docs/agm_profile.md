# agm_dyn: profile and optimization opportunities

A profile of the pure AGM, `agm_dyn`, from 64 to 16384 output limbs, with an assessment of two
proposed optimizations (a faster square root, and a series for the last geometric means) and the
other opportunities the profile points to. It covers the source as of 2026-11-02 (`c450a51`,
branch `AGM_Jacobie`). The profiling did not change any production code.

The AGM is a serial chain of identical steps, and the square root is the step's larger half:

- **Every iteration costs the same.** Early phase (b < a/2) or quadratic phase, each step is one
  high product and one `sqrt_approx_dyn` at the full frame, within about 2%. The AGM costs
  (n − 1) × (product + root).
- **The root is 54–72% of the AGM from 1024 limbs**, and its cost relative to a product grows with
  size: 1.2× at 1024 limbs, 2.1× at 4096, 2.5× at 16384.
- **The root is almost all division.** Divisions inside the root are 47% of the AGM at 1024 limbs,
  59% at 4096 and 64% at 16384 (about 89% of the root there).
- **Threads change nothing.** Dynamic products of these sizes run on the serial FFT, so the
  default 8-thread pool gives the same times as one thread.
- **The series tail works but saves little.** Replacing the last two roots with the AGM's
  closed-form series saved 4.0–5.6%, with no measurable loss of accuracy.

## Where time goes

Apple M1 (macOS 26.6), rustc 1.95.0, `prof` profile (thin LTO), one rayon thread. The default pool
gave the same times at 1024 and 4096 limbs. Timings are minima over repeated calls on warm
scratch buffers. Each size takes one input X with random limbs and its top bit set (no
cancellation), and the AGM gets the (a1, b1, n) that `ln_agm_dyn_at::<3>` builds for it:
`agm_frame`'s f, p, l and g, θ from `theta_dyn::<3>`, b1 = θ2θ3 and a1 = (θ2 + θ3)²/2 − b1, and n
from `ln_agm_core`'s formula. The one difference is that a1 and b1 come from exact full products
truncated to the frame, where `ln_agm_core` uses high products, so their last limbs can differ. The stage times
come from a copy of `agm_core` with timers, checked bit for bit against `agm_dyn`; its square root
is a timed copy of `sqrt_approx_dyn` and `zimmermann_sqrt_core`. The sampled profiles are samply
at 4 kHz, about 48,000 samples each, at 1024, 4096 and 16384 limbs.

### Product against root

| Out limbs | g | n | AGM | Product | Root | Means | Root ÷ product | Divisions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 69 | 15 | 42 µs | 37% | 61% | 1.3% | 1.65 | 41% |
| 128 | 134 | 17 | 140 µs | 44% | 55% | 0.9% | 1.27 | 43% |
| 256 | 264 | 19 | 0.50 ms | 52% | 47% | 0.5% | 0.90 | 39% |
| 512 | 524 | 21 | 1.9 ms | 55% | 45% | 0.4% | 0.81 | 39% |
| 1024 | 1044 | 23 | 5.4 ms | 46% | 54% | 0.3% | 1.18 | 47% |
| 2048 | 2084 | 25 | 16.3 ms | 36% | 64% | 0.2% | 1.80 | 56% |
| 4096 | 4164 | 27 | 45 ms | 33% | 67% | 0.1% | 2.07 | 59% |
| 8192 | 8324 | 29 | 113 ms | 31% | 69% | 0.1% | 2.26 | 61% |
| 16384 | 16644 | 31 | 273 ms | 28% | 72% | 0.1% | 2.54 | 64% |

"Divisions" is the share of the whole AGM spent in the root's `div_dyn` and `div_rem_dyn` calls.
At 256 and 512 limbs the root is cheaper than the product only because the product there is
`hi_mul_buf`, a schoolbook high product, up to `DYN_PARTIAL_MUL_CUTOFF` (312) output limbs; above it
`hi_mul_guarded_dyn` takes a full product of g-limb operands.

The per-step costs (µs), with the reference kernels at g limbs for scale:

| Out limbs | Product per step | Root per step | M(g) | S(g) | D(2g/g) | R(g) | √ (g+2 → g) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 14.9 | 13.4 | 21.3 | 9.2 | 24.7 | 20.1 | 12.4 |
| 1024 | 113 | 133 | 111 | 76 | 225 | 168 | 132 |
| 4096 | 569 | 1179 | 559 | 344 | 1156 | 904 | 1173 |
| 16384 | 2577 | 6550 | 2513 | 1736 | 5920 | 4639 | 6422 |

In units of M(g) at 4096 limbs: a square is 0.62, a reciprocal 1.62, a 2g/g division 2.07 and the
approximate root 2.10. At 16384 they are 0.69, 1.85, 2.36 and 2.56.

### The iteration chain

At 4096 limbs (n = 27, 26 roots), each step's product took 557–578 µs and its root 1174–1258 µs,
with no trend across the chain. The gap log2(a/(a − b)) of the pair entering each step:

| Steps | Gap (bits) |
| --- | --- |
| 1–10 | 0.0 (b < a/2: the early phase) |
| 11–17 | 0.4, 2.3, 7.3, 17.5, 38.1, 79.1, 161 |
| 18–23 | 325, 654, 1311, 2624, 5252, 10506 |
| 24–26 | 21015, 42033, 84068 |
| 27 (last mean, no root) | 168140 |

So 10 of the 26 roots run before the quadratic phase starts. They exist because q's lead l is
about P/64 at T = 3. At 256 limbs it is 6 of 18.

### Inside the root

As a share of the whole AGM, from the timed copy (the samples agree within a point):

| | 1024 limbs | 4096 limbs | 16384 limbs |
| --- | ---: | ---: | ---: |
| Top-level quotient-only `div_dyn` (n/2 by n/2) | 30.0% | 30.6% | 29.8% |
| Zimmermann core: `div_rem_dyn` | 17.2% | 28.6% | 34.0% |
| Zimmermann core: squares | 5.8% | 7.8% | 7.6% |
| Binomial leaves, normalization, copies | 0.9% | 0.4% | 0.3% |

Per root, the top-level division costs 0.67 M(g) at 1024 limbs, 0.96 at 4096 and 1.08 at 16384.
The core's divisions grow faster, from 0.38 to 0.90 to 1.24 M(g), as their sizes move from BZ and
Knuth into NR.

The divisions by stage (samples, inclusive, share of the whole AGM):

| | 1024 limbs | 4096 limbs | 16384 limbs |
| --- | ---: | ---: | ---: |
| NR quotient refinement (`nr_refine_quo`) | 21.5% | 30.4% | 34.1% |
| NR reciprocal chains (`nr_rcp_chain`) | 8.5% | 13.3% | 17.5% |
| NR remainder product (`nr_rem_finish`) | — | 6.9% | 10.7% |
| BZ with Knuth leaves (`bz_div_core_dyn`) | 12.8% | 8.1% | 1.5% |
| Plain Knuth (`knuth_div_rem_core`) | 5.0% | 0.8% | 0.2% |
| Error bands (`nr_err_band`, inside the first two rows) | 15.3% | 17.9% | 20.2% |

The quotient refinement is the Karp–Markstein step: a short product n·x, the error band's middle
product, and the correction product. `nr_rem_finish` computes the whole q·d product but uses only
its low d.len() + 1 limbs. At 1024 limbs every core division is still BZ or Knuth, which produce
the remainder in place.

At the kernel level, the FFT (`fft_core` and rustfft, inclusive) is 54% of samples at 1024 limbs,
75% at 4096 and 82% at 16384, and the FFT middle product alone is 14%, 17% and 20%. At 1024 limbs
the other hot spots are `knuth_loop` (17.8%, the BZ leaves) and `hi_mul_buf_by_rows` (14.2%, the
refinement's schoolbook short products at up to 312 limbs). Scratch-pool and allocation code, the
arithmetic means and `agm_core`'s own control are all under 1% from 1024 limbs.

### Accuracy slack

Against a reference AGM run from the same (a1, b1) with 3 more limbs and one more mean, the
result's error sits well below the last of the p bits `ln_agm_core` needs:

| Out limbs | 64 | 256 | 1024 | 4096 | 16384 |
| --- | ---: | ---: | ---: | ---: | ---: |
| l | 67 | 259 | 1027 | 4099 | 16387 |
| Error below p's last bit (bits) | 163 | 261 | 650 | 2187 | 8331 |
| l/2 + 128 | 162 | 258 | 642 | 2178 | 8322 |

The slack is l/2 bits plus the frame's two guard limbs. The AGM's own rounding starts at b2, whose
ratio a2/b2 is about 2^(l/2), so it is amplified by about 2^(l/2), not 2^l. Only b1's error
(theta's truncation and `ln_agm_core`'s first step, outside this measurement) needs all l of the
frame's extra bits.

## Proposal 1: optimize the square root

This is the right target: the root is 54–72% of the AGM from 1024 limbs, and about 90% of it is
division, whose cost relative to M grows with size.

**A different algorithm alone doesn't pay.** Reciprocal-square-root Newton with a Karp–Markstein
last step costs, by this machine's kernels (R ≈ 1.6 M, D ≈ 2.1 M at 4164 limbs), an estimated
2.4 M per root, against the 2.1 M Zimmermann measures. That agrees with the earlier NR sqrt
experiment. It would only win with FFT transform reuse (Bernstein's counts), which means exposing
the transforms from the multiplication layer.

**The approximate top level already skips what it can** (the low square, the remainder, the final
correction). Its quotient-only division is one half-precision reciprocal and one Karp–Markstein
refinement, 0.7–1.1 M(n), and inherent to the method.

What can be cut is in the recursive core:

1. **Form the root remainder from one wrapped square.** At a core level with a k-limb root, the
   division's remainder (`nr_rem_finish`, a full q·d product) and the separate q² square exist only
   to form X − s². That remainder fits in about k + 2 limbs, and X gives the high part of s². So
   s² mod (B^m − 1), m ≈ k + 2, from one cyclic FFT square of s, recovers X − s² exactly, and the
   core can call quotient-only `div_dyn`. The high part of s² is X's within one, which picks the
   right representative. That is 2 transforms in place of about 5, at about the same transform
   size. Estimated saving: about the `nr_rem_finish` share, 7% of the AGM at 4096 limbs and 11% at
   16384. It needs a new primitive, a product or square mod B^m − 1 (GMP's `mpn_mulmod_bnm1`, which
   GMP uses for the same purpose in its division remainders); the wrapped convolutions of
   `fft_mid_mul` are most of the plumbing. The same primitive would halve `nr_rem_finish` for every
   NR `div_rem`, since its high part is known from n.
2. **Pass reciprocals up through the recursion.** Every level divides by a prefix of the root, and
   the next level up divides by a longer prefix of the same root. Each NR division rebuilds its
   reciprocal from a one-limb seed (`nr_rcp_chain`), redoing what the level below already
   computed. Seeding each level with the one below would leave only its last refinement step,
   about half of the 13–17% in `nr_rcp_chain`: an estimated 5–8%. It needs a seeded entry into
   `nr_quo_est_dyn` and `div_rem_prepared_dyn`, and is the most intricate change here because of
   the guard limbs, the shift normalization and the BZ fallback.
3. **The mid sizes are a tuning question.** At 1024 limbs the core's balanced divisions (130–520
   limbs) go through BZ and Knuth (18%), and the refinement's short products are schoolbook up to
   312 limbs (14%). Whether NR, or the guarded high product, should start earlier for these shapes
   depends on the machine, and belongs with the planned tuning revamp rather than per-constant
   patches.

## Proposal 2: series for the last geometric means

Feasible, and accurate, but the measured gain is small. The prototype used the AGM's closed-form
tail rather than a series per mean. With m = (A + B)/2, d = (A − B)/2 and u = (d/m)²,

AGM(A, B) = m·(1 − u/4 − 5u²/64 − 11u³/256 − 469u⁴/16384 − 1379u⁵/65536 − …),

the inverse of K's series 1 + u/4 + 9u²/64 + …, since AGM(1 + ε, 1 − ε) = π/(2K(ε)). Every
coefficient is positive with a power-of-two denominator, so term i is N_i·V_i / 2^e_i with
V_i = d^(2i)/m^(2i−1), and it is needed only to p − 2i·s bits when A and B agree to s bits. The
prototype stops j roots early (K = n − 1 − j full steps), takes each V_i = V_(i−1)·V_1/m as one
division of truncated operands, and sums terms until one rounds to zero.

Time saved against the full AGM:

| Out limbs | Full AGM | j = 1 (1 term) | j = 2 (3 terms) | j = 3 (6 terms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 42 µs | 3.9% | 4.0% | −4.8% |
| 256 | 0.50 ms | 3.8% | 5.4% | 1.6% |
| 1024 | 5.4 ms | 3.5% | 4.0% | 1.7% |
| 4096 | 45 ms | 2.6% | 4.3% | 3.0% |
| 16384 | 271 ms | 2.8% | 4.5% | 3.2% |

At 4096 limbs one removed step is worth 1.75 ms, and the tail itself cost 0.51 ms (j = 1),
1.47 ms (j = 2) and 3.82 ms (j = 3), starting from gaps of 84069, 42033 and 21015 bits. In every
case its error landed on the same bit as the full AGM's, since the early-phase rounding dominates.
Stopping four or more roots early was slower at every size (and needed more than the prototype's
11 coefficients).

- **Per mean against closed form.** The proposed per-mean series, b' = m·√(1 − ε²) expanded, costs
  the same as the closed form for j ≤ 2: each replaced mean needs its own division by its own m.
  The closed form needs one 1/m for all of them, and its stopping rule is exact: the term count
  follows from the gap of (A_K, B_K), read from bit lengths, and the truncation is bounded by the
  next positive term.
- **Room left in the prototype.** One reciprocal of m plus a Horner pass at shrinking precision,
  instead of a division per term, should move the best j to 3, at an estimated 5–6%.
- **Cheaper inside ln.** `ln_agm_core` needs T = π/(4M) = (π/(4m))·F(u), F = 1 + u/4 + 9u²/64 + …
  (positive, power-of-two denominators again), and already ends in one division, π/M. With the
  tail it divides by m instead, and taking 1/m once also gives ε = d·(1/m) by a product, so the
  tail's own division disappears: about 2% more.
- **The ceiling is low.** Each removed root is worth 1/(n − 1) of the AGM (3.8% at 4096 limbs),
  and the tail's cost grows with every term, so the best case removes two or three roots.

## Recommendations

By expected gain on the pure AGM:

1. **Wrapped-square remainder in the Zimmermann core** (Proposal 1, item 1): an estimated 7–11%
   from 4096 limbs. It also speeds `sqrt_dyn`, `sqrt_only_dyn` and, in its general form, every NR
   `div_rem`.
2. **Closed-form tail** (Proposal 2): 4–5.6% measured, an estimated 5–6% with one reciprocal, and
   about 2% more folded into `ln_agm_core`. Low risk with existing primitives.
3. **Reciprocal chaining through the root's recursion** (Proposal 1, item 2): an estimated 5–8%,
   but the most complex.
4. **A narrower frame.** The slack above shows the steps from b2 on need about l/2 fewer bits, and
   l/2^(k−1) fewer after step k. Shifting a and b down by whole limbs after each early step (or a
   floating layout with separate exponents for a and b, each at p plus guard bits) saves up to
   ⌈l/64⌉ limbs per operand: at most about 1.6% at T = 3, more at smaller T. Small, but it needs
   its own error analysis, since b1's error does use all l bits.
5. **Multiple cores.** The chain is serial, and threads changed nothing here. The only way in is
   inside each multiplication: running a product's two forward FFTs in parallel, or a parallel
   transform. With the FFT at 75–82% of samples from 4096 limbs, this is the largest multi-core
   lever, but it lives in `mul.rs`, not in the AGM.
6. **Fewer iterations.** Outside the AGM itself, n ≈ 2·log2 P − 9, and the early phase's roots come
   from q's lead l ≈ P/(4(T + 1)²). Each doubling of T + 1 removes about two roots (7–8% at 4096
   limbs) in exchange for more theta terms, which, unlike the AGM, parallelize. That makes T a
   cheaper lever on several cores than on one.

## Reproduction

The measurements came from a throwaway example (not committed) built with `--profile prof` in a
detached worktree, run with `RAYON_NUM_THREADS=1`. It:

- builds the inputs as described under "Where time goes" (θ from `theta_dyn::<3>`, a1 and b1 from
  exact `mul_dyn`/`sqr_dyn` products truncated to g limbs, n from `ln_agm_core`'s formula on x's top
  64 bits);
- times `agm_dyn` as a whole, then a copy of `agm_core` with timers around `hi_mul_dyn`, the mean
  and the root (the root itself a timed copy of `sqrt_approx_dyn`'s padded Zimmermann path), and
  asserts the copy's output equals `agm_dyn`'s;
- for the tail, runs K full steps with the same kernels, then the series with `sqr_dyn`, `mul_dyn`
  and `div_dyn` on truncated operands, and compares both results with `agm_dyn` from the same
  inputs at 3 more limbs and one more mean;
- loops `agm_dyn` at one size for samply:
  `samply record --save-only --unstable-presymbolicate --rate 4000 -o agm.json.gz <bin> loop 4096 12`,
  with per-function self and inclusive shares read through the `.syms.json` symbol table.
