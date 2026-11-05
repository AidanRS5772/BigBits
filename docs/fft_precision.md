# 16-bit FFT precision: the round-off check, the residue check and the dispatch limits

The dynamic multiplication layer computes products, squares, middle products and squares mod
B^m − 1 on a floating-point FFT of 16-bit digits (`fft_entry`, `fft_sqr_entry`, `fft_mid_mul`,
`fft_sqr_mod_bnm1`). Each is exact only while every output coefficient's rounding error stays
below 0.5. Every entry now measures its own rounding and redoes a call on the NTT when the call
could be wrong. Products and squares can also check their results by residues, always in debug
builds. The four dispatch limits in `src/utils/mod.rs` only decide how often a redo happens. This note records the checks,
the evidence that they catch wrong results, how the limits were set, and how to remeasure them.
Measured 2026-11-02/04 on the Apple M1 (rustfft's NEON paths).

## The round-off check

`scale_and_round` scales and rounds the inverse transform's coefficients and returns the call's
`FftRounding`: `err`, the largest rounding distance max |x − round(x)|, and `peak`, the largest
rounded magnitude (NEON: `vabdq_f64`, `vabsq_f64` and `vmaxq_f64`; AVX: masked subtractions and
`_mm256_max_pd`; scalar for tails and the standard path). Before it reads a coefficient, each entry
trips when `err` passes `FFT_ROUNDOFF_LIMIT` (0.4, Prime95's convention) or `peak` reaches
`FFT_COEFF_LIMIT` (2^48). A tripped call's transform is discarded, `out` is left untouched and the
call is redone on the NTT:

| Checked entry | Redone by |
|---|---|
| `fft_entry` | `ntt_entry_dyn` |
| `fft_sqr_entry` | `ntt_sqr_entry_dyn` |
| `fft_mid_mul` | `ntt_mid_mul_dyn` (the same contract: within the documented low-limb error) |
| `fft_sqr_mod_bnm1` | `ntt_sqr_entry_dyn` into pooled scratch, then `fold_mod_bnm1` (not `sqr_dyn`, whose FFT square would round about as badly) |

The redo runs outside the `FFT_CACHE` borrow. `fft_reroutes()` (doc-hidden) counts redone calls
in the process, and the doc-hidden `fft_entry_raw`, `fft_sqr_entry_raw`, `fft_mid_mul_raw` and
`fft_sqr_mod_bnm1_raw` skip every check and return the `FftRounding` beside the FFT's own result,
for the probe and the tests. The middle product's check covers every coefficient of its transform,
not only the band it reads, so it trips more often than its band needs.

**Why 0.4 catches wrong results.** A wrong limb needs some coefficient off by e > 0.5, and that
coefficient then rounds to the wrong integer and measures 1 − e. One off by less than 0.6 measures
more than 0.4 and trips the check by itself. A wrong call escapes only if some coefficient is off by
0.6 or more while every other one stays under 0.4. Rounding errors are sums of many small errors
spread over the coefficients, so a distribution whose tail reaches 0.6 has coefficients between 0.4
and 0.5, provided the distance can be read that finely.

**Why the size gate.** It can't always be read that finely. Before the 1/N scaling, an inverse
transform output near N·c is a double, a multiple of ulp(N·c), so after scaling a coefficient's
error is a multiple of g = ulp(N·c)/N ≤ c·2^-52, then rounded to the doubles near c. For
c < 2^48, g is under 1/16, and readings fill [0, 0.5] in steps of at most 1/16. Past 2^48 the steps
coarsen. At N = 2^15·7 and c ≈ 2^49.7 (cyclic squares of m = 57344 limbs with digits at least
0xF000), g = 2^15/N = 1/7 and the doubles near c are 1/8 apart. The only readings are then 0, 1/8,
1/4 and 3/8: a coefficient off by 3/7 rounds right, one off by 4/7 rounds wrong, and both read
0.375. Without the gate, the first soundness run (below) found 264 such calls, wrong and untripped:
middle products at n = 57343 (N = 2^16·7) and cyclic squares at m = 57344, 81920 and 98304
(N = 2^15·7, 2^16·5, 2^17·3), all with digits at least 0xC000. Two of those shapes were inside
the dispatch limits then in force. The model g = ulp(N·c)/N, rounded to the doubles near c,
predicts every one of them, and every tripped call at the same sizes. Squares and products had no
misses: their coefficients rise from small at the ends to c at the middle, so the small ones, read
finely, still trip. Middle products and cyclic squares have coefficients of one size throughout.

Random operands stay below 2^48 up to the dispatch limits. Their largest coefficients there are
2^47.71 for products, 2^47.77 for squares and 2^47.998 for middle products. The gate sets the
cyclic limit (below), as every coefficient of a cyclic square sums the same 4m digit products.
Operands with digits near 2^16 reach 2^48 from about 18000 limbs, about where the old worst-case
limits were, and pay the FFT and then the NTT above that.

## The residue check

FFT products and squares (`fft_entry`, `fft_sqr_entry`) then check the result they would return by
its residues mod the NTT primes (`src/utils/verify.rs`, below): `Residues::of_with_top(out, top)`
against the operands' residues multiplied. A mismatch is redone on the NTT and counted in
`fft_residue_reroutes()` as well as `fft_reroutes()`; it means the round-off check let a wrong
result through, and it should stay 0. Debug builds always run it. Release builds run it when
`FFT_RESIDUE_CHECK` in `mod.rs` is set (`false` by default). It catches any wrong result whose
error fits in 188 bits, whatever the rounding read. Middle products and cyclic squares have no
such check, since their results omit an unknown part of the full product, and rely on the
round-off check alone.

## Cost

Release A/B on the M1, one thread, minimum of four alternating rounds of nine, against the same
tree without the check. Squares and products are within noise (−1.3% to +0.5%) from 1024 limbs to
the limits, and about 2% (0.25 µs) slower at 256 limbs. Tracking the peak costs nothing beside the
error. The first version cost 2–3% everywhere: each max accumulator was a loop-carried chain, which
two accumulators per quantity remove. A tripped call costs the FFT and then the NTT (below).

The residue check reads the operands and the result, 4n limbs for an n × n product, at 2.55 ns per
limb (`Residues::of`, below):

| Product | Residue check | FFT product | Share |
|---|---:|---:|---:|
| 256 × 256 | 2.8 µs | 13.9 µs | 20% |
| 1024 × 1024 | 10.6 µs | 69.7 µs | 15% |
| 4096 × 4096 | 41.9 µs | 318 µs | 13% |
| 16384 × 16384 | 166 µs | 1390 µs | 12% |
| 52920 × 52920 (the limit) | 534 µs | 9550 µs | 5.6% |

That is why `FFT_RESIDUE_CHECK` is off by default.

## Soundness

Two runs fed the raw entries operands whose results go wrong and counted the calls that came out
wrong without tripping (missed). The residue verifier judged products and squares, the NTT middle
products and cyclic squares.

**Run 1: worst-case operands where they go wrong, before the gate.** Seven kinds (all ones, a step,
and random digits all at least 0xF000, 0xFF00, 0xE000, 0x8000 or 0xC000) ran at every transform size
from where 10000 of them first gave a wrong limb (squares 144000, products 147456, middle products
134400, cyclic squares 92160) to about 1.5× the dispatch limits. Each random kind had 24 operands per
size: 128832 calls, 72922 of them wrong. The rounding distance alone missed 264 of them. All were
middle products and cyclic squares with coefficients past 2^49, the unreadable steps described
above, and the size gate trips every one. By largest coefficient (estimated from each kind's mean
digit):

| Largest coefficient | Calls | Wrong | Missed by the distance alone | Smallest wrong reading |
|---|---:|---:|---:|---:|
| [2^47, 2^48) | 9989 | 2 | 0 | 0.5 |
| [2^48, 2^49) | 52942 | 10037 | 0 | 0.4375 |
| [2^49, 2^50) | 56299 | 53364 | 216 | 0.375 |
| [2^50, 2^51) | 9602 | 9519 | 48 | 0.25 |

**Run 2: just below the gate.** With the gate in place and each call's largest coefficient
measured, the run covered two groups:
- Worst-case kinds at every size where a kind's coefficients reach 2^47.3–2^48.05: 256 operands per
  size and kind for squares and products, 128 for middle and cyclic squares, and one each for all
  ones and the step.
- Random operands where theirs reach 2^47.55–2^48.02, just past the limits: 1024 per size, or 256
  for middle and cyclic squares.

| Entry | Calls | Wrong | Wrong under the gate | Missed |
|---|---:|---:|---:|---:|
| Squares | 173478 | 2 | 0 | 0 |
| Products | 173478 | 1 | 0 | 0 |
| Middle products | 69158 | 9 | 2 | 0 |
| Cyclic squares | 59149 | 0 | 0 | 0 |

Wrong results below 2^48 are rare. In both runs only 4 worst-case calls (all middle products) were
wrong with coefficients below 2^48, and all read 0.5. Random operands gave no wrong result below
the gate in about 1.25M calls: the limit passes below, plus run 2's random group. Their 39764 wrong
results in the coarse pass, with coefficients just past 2^48, all tripped by distance alone, the
smallest reading 0.4375.

So the check rests on evidence, not proof. No wrong result below 2^48 has read under 0.4, but
nothing rules one out. For random operands at the largest sizes inside the limits, the data
bounds a miss to under about 1 in 10^10 per call: under 2.4e-6 wrong and under 6e-5 of those
missed. Extending the measured error tail puts it nearer 1 in 10^11 or below. The residue check
closes the gap for products and squares. Middle products and cyclic squares have no such backstop.

## The dispatch limits

With the check in place, the limits in `mod.rs` only decide how often a call pays for the FFT and
then the NTT. Each is the largest size below the first transform size where uniformly random
operands trip the check, either the distance or the gate, more than 1 in 2^10 times. The operands have every limb uniform and a
nonzero top limb, at the transform size's widest shape: balanced products, full-length squares,
`long[2n − 1] × short[n]` and m-limb cyclic operands. The probe ran in `rate` mode in two passes,
one thread per process:

1. Coarse: 1024 operands at every transform size from the old worst-case limits to about 4× them.
2. Fine: 2^14 operands at every size whose coarse pass had an error past 0.30, in ascending order.
   Each entry stopped at its first size with more than 16 trips (`STOP=16`).

Sizes whose 1024 coarse operands all stayed at or below 0.30 were not rerun. In the fine pass, none
of the 65 sizes whose error passed 0.30 in under 1/8 of their calls (1064960 calls in all) tripped
even once. A size that never passed 0.30 in 1024 calls is therefore far below 2^-10.

Those two passes measured the distance only. After the gate, a third pass ran 2^14 random operands
per size, counting either trip:
- every middle-product size from 500000 to 518616;
- every cyclic size from 240000 to 262144;
- the square and product limits.

Only the cyclic limit moved, from 65536 to 64800 modulus limbs.

| Entry | Limit | Transform N | Trips at the limit | First size past 2^-10 | Old limit |
|---|---:|---:|---|---|---:|
| Products (output limbs l + s − 1) | 105839 | 423360 | 0 in 2^14 | 428652: 17 in 12727 | 36863 |
| Squares (operand limbs) | 55296 | 442368 | 0 in 2^14 | 444528: 17 in 11323 | 18000 |
| Middle products (short limbs n) | 64826 | 518616 | 0 in 2^14 | 524288: 17 in 10875 | 16799 |
| Cyclic squares (modulus limbs m) | 64800 | 259200 | 11 in 2^14, all at the gate | 259308: 17 in 8845, the same | 23040 |

The new limits are 2.8–3.9× the old ones. Trip rates are not monotone in the size, so a size below
a limit can trip more than the limit itself. The most trips at or below each limit were:

| Entry | Trips in 2^14 | Transform N |
|---|---:|---|
| Products | 12 | 393660 = 2^2·3^9·5 |
| Squares | 13 | 428652 = 2^2·3^7·7^2 |
| Middle products | 5 | 470596 = 2^2·7^6 |
| Cyclic squares | 11 | 259200, the limit itself (the gate) |

Sizes past a first failure that round better stay on the NTT.

**Speed at the limits.** The trip-rate rule doesn't weigh the NTT's speed. Release, M1, random
operands, minimum of five alternating rounds, on one thread and on the default 8-thread pool:

| Entry, size | FFT, 1 thread | NTT, 1 thread | FFT, pool | NTT, pool |
|---|---:|---:|---:|---:|
| Squares, 18000 limbs (old limit) | 2.06 ms | 4.54 ms | 2.06 ms | 1.86 ms |
| Squares, 55296 limbs | 5.33 ms | 13.82 ms | 5.24 ms | 5.12 ms |
| Products, 36863 output limbs (old) | 1.98 ms | 5.86 ms | 1.97 ms | 2.22 ms |
| Products, 105839 output limbs | 9.58 ms | 21.47 ms | 9.63 ms | 6.76 ms |
| Middle products, n = 16799 (old) | 2.49 ms | 7.03 ms | 2.52 ms | 2.25 ms |
| Middle products, n = 64826 | 12.76 ms | 24.76 ms | 12.50 ms | 8.72 ms |
| Cyclic squares, m = 23040 (old) | 0.90 ms | 5.81 ms | 0.90 ms | 2.28 ms |
| Cyclic squares, m = 64800 | 3.24 ms | 19.43 ms | 3.21 ms | 7.74 ms |

On one thread the FFT is 1.9–6× faster at every limit. The FFT runs on one thread and the NTT
runs its three primes on the pool, so with 8 threads the NTT already matched the FFT for squares
and middle products at the old limits. It is about 30% faster at the new product and
middle-product limits. Only the cyclic square keeps a clear FFT lead (2.4×). A dispatch that
weighed the pool's size would switch earlier on many threads; that is not done.

Worst-case operands (digits near 2^16) trip well below the limits: they reach the size gate from
about 18000 limbs, and their errors passed 0.4 from about a quarter to a third of the limits in the
old search. Above that they pay the FFT and then the NTT, about 1.2–1.5× the NTT alone on one
thread. Their coefficients could be bounded before the transform: a cyclic square's mean
coefficient is (Σ digits)²/4m exactly. Such a bound could send them straight to the NTT; that is not
done.

## Why the FFT fails: precision, not arithmetic

A coefficient of an l × s product sums up to 4·min(l, s) products of 16-bit digits, so it can reach
about 2^34·min(l, s): 2^49 at 32768 × 32768. A double has 53 significant bits, which leaves about
4 bits below the units place for all of the transform's rounding error. Unsigned digits near their
maximum (all ones, or random digits all at least 0xF000) put most of the spectrum in one huge
zero-frequency term, and every rounding error scales with it. Uniformly random digits average a
quarter of the maximum product and round 3–4× better.

An independent plain rustfft complex convolution, without the library's real-FFT packing or
twiddle tables, has the same problem at the same sizes, with somewhat smaller errors (worst-case
squares):

| Transform N (operand limbs) | Library FFT square | Plain rustfft, same N | Plain, balanced digits |
|---|---:|---:|---:|
| 131072 (16384) | 0.19 | 0.16 | 0.03 |
| 190512 (23814) | 0.5, wrong | 0.31 | 0.06 |
| 222264 (27783) | 0.5, wrong | 0.375 | 0.09 |
| 262144 (32768) | 0.375 | 0.31 | 0.06 |

The library loses up to about a bit against the plain FFT at mixed-radix sizes (factors 3, 5, 7)
and matches it at powers of two; the random-operand trip rates show the same pattern, with
radix-3 and radix-7 sizes tripping where neighbouring powers of two don't. Recomputing every twiddle
of `FTTTwidleTower` exactly with `cis` changed nothing, so the tables are not the cause; the
half-length real-FFT packing and its recombination are the likely place, not yet confirmed.
Balanced digits in [−2^15, 2^15) remove the zero-frequency term and cut the worst case about 4×.

## History: the worst-case limits

`FFT_16BIT_CUTOFF` was 2^16 output limbs for products and middle products, and `dyn_sqr_dispatch`
compared it with the square's operand length, letting squares reach twice the product's size. Its
only accuracy tests used random operands. Against the exact NTT, worst-case operands
(`fft_worst_operand` in `src/tests/mod.rs`: all ones, a step, random digits all at least
0xF000, 0xFF00 or 0xE000) gave wrong results well inside those limits: all-ones squares from 23814
operand limbs and all-ones products from 26460 × 26460.

Before the check, the fix was a limit per entry just below the first transform size where 10000
worst-case operands gave a wrong limb: 36863 output limbs for products, 18000 operand limbs for
squares, 16799 short limbs for middle products and 23040 modulus limbs for cyclic squares. Those
limits had no margin, and the failure point kept moving down as the search grew, because
high-random operands give a spread of rounding errors (squares):

| Worst-case operands per transform size | First failing transform N | Operand limbs |
|---:|---:|---:|
| 8 | 175616 | 21952 |
| 26 | 157464 | 19683 |
| 200 | 157464 | 19683 |
| 1000 | 148176 | 18522 |
| 10000 | 144060 | 18007 |

A size's maximum error over n operands grows roughly like the maximum of Gaussian samples,
√(2 ln n). Middle products had the heaviest tail: their first failure moved from short length
n ≈ 22.6k (300 operands per size) to 16806 (10000), and their failures were sporadic. That search
could only ever give a limit that a larger search might break, and it sent random operands to the
NTT 2–3× earlier than they need. The check removes both problems.

## The residue verifier

`src/utils/verify.rs` checks c = a·b in linear time by residues mod the three NTT primes
P1, P2 and P3 (P = P1·P2·P3 ≈ 2^188). `Residues::of(x)` folds x from its low limb up with one
Montgomery reduction per limb per prime, r ← limb + r·B⁻¹, and `verify_mul`/`verify_sqr` compare. Any error D = 2^s·E with 0 < |E| < P is caught for certain, since 2 is invertible mod
each prime: one wrong FFT coefficient (±e·2^(16k)) or one wrong limb always is. A wider error passes
only if it vanishes mod all three primes at once; 2^16 has multiplicative order about 2^56 mod
each prime, so there is no periodic cancellation like mod B − 1, where errors one limb apart
cancel.

The fold's low half is the running residue r, so its carry is just [r ≠ 0], and each step takes two
multiplies where the general REDC takes three. A fold is a serial chain, so from 128 limbs x is
split into four segments per prime, 12 independent chains, recombined by Horner in W = B^seg. Powers
of B are Montgomery square-and-multiply. That runs at 2.55 ns per limb on the M1, down from 3.4 ns
with one chain per prime and the general REDC. That is about 8 cycles per limb. The likely limit
is the four compare-and-select steps per limb per prime (reduce the limb below p, the carry, the
final subtraction): 24 flag operations per limb, against the M1's three or so per cycle.

It can't check a middle product's band or a square mod B^m − 1, whose results omit an unknown part
of the full product, so the probe and the tests still compare those with the NTT. Besides the FFT
entries' residue check, `mul_dyn` and `sqr_dyn` assert every NTT result with it in debug builds.

## Remeasuring

`benches/probes/fft_precision.rs` runs operands through the raw entries at every transform size in
a range and reports, per size:
- the trips, either part, and `over`, the calls at the size gate;
- the errors' tail: calls past 0.30 and 0.35;
- the wrong results, and `under`, those below the gate;
- the missed ones (wrong and untripped, which must stay 0);
- the largest error, log2 of the largest coefficient (`peak`), and the smallest error among the
  wrong calls below the gate. Its header has the build command. `rate` mode takes uniformly
random limbs (the limits), and `worst` mode the worst-case kinds (`KIND=k` forces one; 6 and 7 are
random digits at least 0x8000 and 0xC000). `STOP=k` ends a size after k + 1 trips. Products and
squares are judged by the residue verifier, middle and cyclic squares by the NTT (`no-oracle`
skips it, for trip rates only).

A transform size's limbs follow from its entry's sizing: a square of n limbs takes
`find_fft_size(8n − 1)`, a product of O output limbs `find_fft_size(4O + 3)`, a middle product of
short length n `find_fft_size(8n + 4)`, and a cyclic square exactly 4m. The cyclic limit must be a
`bnm1_size`, since `bnm1_fft` only runs the cyclic FFT at such an m.

Rounding depends on rustfft's SIMD paths. On x86 (AVX) the trip rates, and so the best limits,
will differ; the check itself keeps the results exact either way.
