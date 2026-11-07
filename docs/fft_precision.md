# 16-bit FFT precision: balanced digits, the fold check, the residue check and the dispatch limits

The dynamic multiplication layer computes products, squares, middle products and squares mod
B^m − 1 on a floating-point FFT of 16-bit digits (`fft_entry`, `fft_sqr_entry`, `fft_mid_mul`,
`fft_sqr_mod_bnm1`). Each is exact only while every output coefficient's rounding error stays
below 0.5. Operands go in as balanced digits, so errors stay small. A call that reads above 0.25 runs
the fold check and is redone on the NTT if that fails. Products and squares can also check their
results by residues, always in debug builds. The four dispatch limits in `src/utils/mod.rs` bound both
speed and soundness. This note records the checks, the evidence behind them, how the limits were set,
and how to remeasure them. Measured on the Apple M1 (rustfft's NEON paths) unless noted.

**What's current.** The balanced digits, fold check, soundness and cost sections below were measured
2026-11-06 at the current sizes, with that day's limits. On 2026-11-07 the product, square and
middle-product limits rose to the most the 2^48 bound allows (see the dispatch limits), so "the limit" in
those sections means the old one for them (52920 × 52920 products, 55296-limb squares, n = 64826), and
products and squares now reach transform 524288, where the soundness sweep never ran them. The cyclic
limit was measured 2026-11-02/04 on unsigned digits at the old sizes and hasn't been remeasured.

## Transform sizes

`find_fft_size(n)` is the smallest 4·o·2^a at or above n with o in {1, 3, 5, 7, 9}, and `bnm1_size`
follows it (m = o·2^a). rustfft's NEON planner sends only 2^k and 3·2^k through its radix-4 kernel, and
everything else through a mixed-radix split with two transposes and a full twiddle pass, so per-point
cost falls in tiers. ns/(n log2 n) against a power of two in the same octave, median over 2^11–2^19 for
products and squares (all 651 7-smooth sizes from 2048 to 540000, warm, minimum of three shuffled
passes, one thread, M1):

| Odd part | Cost (products / squares) | Odd part | Cost (products / squares) |
|---:|---:|---:|---:|
| 1 | 1.00 / 1.00 | 27 | 1.49 / 1.46 |
| 3 | 1.00 / 0.99 | 35 | 1.35 / 1.35 |
| 5 | 1.21 / 1.19 | 45 | 1.33 / 1.30 |
| 7 | 1.18 / 1.18 | 49 | 1.42 / 1.40 |
| 9 | 1.21 / 1.20 | 63 | 1.36 / 1.28 |
| 15 | 1.22 / 1.18 | 125 | 1.79 / 1.73 |
| 21 | 1.36 / 1.34 | 243 | 1.72 / 1.67 |
| 25 | 1.50 / 1.49 | 729 | 1.93 / 1.80 |

A costly size just below a cheap one loses to it: 423360 (2^6·3^3·5·7^2) took 9.27 ms for a product,
and the 24% longer 2^19 took 6.94 ms. Time against the old rule, over log-uniform lengths from 2048 to
430000 digits:

| Set of odd parts | Products | Squares | Worst single length (products / squares) |
|---|---:|---:|---:|
| `{1}` | 0.862 | 0.884 | 1.69 / 1.67 |
| `{1,3}` | 0.707 | 0.730 | 1.19 / 1.22 |
| `{1,3,5,15}` | 0.725 | 0.742 | 1.16 / 1.15 |
| `{1,3,9}` | 0.690 | 0.711 | 1.09 / 1.14 |
| `{1,3,5,9}` | 0.688 | 0.708 | 1.03 / 1.07 |
| `{1,3,5,7,9}` | 0.689 | 0.708 | 1.00 / 1.03 |
| `{1,3,5,9,15,45}` | 0.740 | 0.754 | 1.15 / 1.14 |

The old rule's time was 1.46× (products) and 1.42× (squares) that of the fastest size at or above each
length; this set's is 1.006×. 15 is out because 15·2^k sits only 6.7% below 2^(k+4) at 1.2× the cost. 7
adds nothing on average, but it holds the worst length at today's time. Other entries' times follow:
0.735× over 12 operations from `mul_dyn` to `ln_dyn` and 0.80× for `exp_dyn`, with identical outputs.

Rounding depends on the class too: at the same operands 3·2^k reads about 1.5× its neighbours (mean
0.24 at 3·2^17 against 0.14–0.17 for 9·2^15, 5·2^16, 7·2^16 and 2^19, squares of 36000 random limbs, on
unsigned digits). That was already true of those sizes under the old rule, and the cause isn't known.

AVX plans its factors differently, as radix steps of one chain, so these tiers may not hold on x86;
`benches/probes/fft_sizes.rs` (`sizes` mode) measures rustfft's per-point cost by class on any machine.

## The stages around the transforms

Share of a call's time at n = 2^19 (products / squares): transforms 84% / 74%, recombine 6.6% / 13%,
accumulate 5.1% / 7.7%, rounding 2.4% / 3.8%, decomposing 1.5% / 0.9%, and setting up a warm call under
0.2%. At n = 4096 recombine is 12% / 21% and accumulate 9% / 12%. Three edits:

- `recombine` and `sqr_recombine` take two conjugate pairs per NEON step, with fused products. Their
  rounding differs slightly from the scalar loops', with no change in what the check reads. x86 keeps
  the scalar loops.
- `scale_and_round` multiplies by 1/n at power-of-two lengths (0.85× of that stage on the M1). That
  reciprocal is exact, so the result is bit-identical. Other lengths still divide, as an inexact
  reciprocal would move the distance the fold check's trigger reads.
- `fft_accumulate` adds each limb's four digits in two halves and reads only the one limb past `out`,
  rather than shifting digit by digit and scanning the padding (0.62× of that stage). Balanced digits
  made its coefficients signed since.

Together, these give 0.958× alone. With the sizes and tables, release A/B against the tree before them
(M1, one thread, minimum of three alternating runs, outputs identical at all 208 points from 400 limbs
to the limits): 0.667× geomean over 13 operations, from 0.585× (`mul_dyn`, 3:2 shape) to 0.79× (`ln_dyn`),
and at most 1.001× at any point. Below the FFT tiers nothing moves.

## Twiddle tables

`recombine` and `sqr_recombine` read e^(2πik/n) for k up to n/4 from a table per transform size, built
once per thread by `fft_twidles` with `cis`. Their cost is linear and a small part of a first call,
whose setup is about 85% rustfft's plans (5.3 ms of plans and 1.1 ms of table at n = 2^19). They replace
`FTTTwidleTower`, which kept one table per odd part at the finest size seen, made coarser sizes copy a
strided view on every call (4–6.6% of a product below the top size), and built finer ones by repeated
multiplication (worst error 4.5×2^-53 after three steps, with no change in the rounding read).

## Balanced digits

`decompose` and `sqr_decompose` write each operand as digits in [−2^15, 2^15): a digit of 2^15 or more
has 2^16 subtracted and carries 1 into the next one (`balance_limb` does a limb at once: x + bias +
carry, with each digit's top bit then flipped). The value is unchanged, and a coefficient's digit
products now mostly cancel instead of all adding up, which takes away the huge zero-frequency term that
every rounding error scaled with. An operand's top digit stays unsigned (`fft_absorb_top_carry`), at
most 2^16, so balancing adds no digit and no transform grows. `fft_accumulate_signed` carries the signed
coefficients, and a cyclic square folds a negative top carry back in. Middle products no longer add
`middle_correction`: coefficients below the band sum to under about 2^34 units of its lowest limb, in
either sign, so the band already holds the product's value. That can flip every limb of a run of zeros
or ones, so compare middle products by value.

The largest reading, unsigned against balanced, on the same operands:

| Operands | Shape | Unsigned | Balanced |
|---|---|---:|---:|
| Random | Products at the old limit (52920 × 52920) | 0.219 | 0.0007 |
| Random | Products, 49152 × 49152 (3·2^17) | 0.312 | 0.0016 |
| Random | Middle products at the old limit (n = 64826) | 0.188 | 0.0010 |
| Random | Cyclic squares at the limit | 0.188 | 0.0010 |
| Digits at least 0xC000 | Products, 52920 × 52920 | 0.5 | 0.0137 |
| Digits near −2^15, one sign | Products, 52920 × 52920 | 0.219 | 0.219 |
| Period 12, full size | Middle products at the old limit | 0.188 | 0.438 |

Random and high-digit operands read about 35–300× lower. The rows that don't improve are what balancing can't
help: digits of one sign at full size don't cancel, and repeating patterns concentrate into a few
frequencies. Their coefficients are at most 1/4 of the unsigned worst case's.

**Why there's no coefficient gate.** A coefficient's error is read in steps of up to c·2^-52, so past
2^48 the reading stops meaning much. The old check tripped there. A balanced coefficient sums at most
4·min(l, s) digit products of size at most 2^30, plus a little for the unsigned top digits, which stays
under 2^48 at every dispatch limit. A compile-time assert in `mod.rs` holds the limits to that, so the
limits now bound soundness as well as speed, and raising one past it needs a new answer to this. Past
each dispatcher's largest transform a checked entry goes straight to the NTT, without running a
transform: at N = 3145728 middle products read under 0.25 and came out wrong.

## The fold check

`scale_and_round` returns the call's `FftRounding`, whose `err` is the largest rounding distance
max |x − round(x)|. A call with `err` above `FFT_FOLD_TRIGGER` (0.25) runs `fft_fold_agrees`, which tests
the result itself; if that passes the result stands, however high it read, and if it fails the call is
redone on the NTT. A high reading only shows noise, and on the M1 most such results are right. A wrong
limb needs a coefficient off by more than 0.5, reading 1 − e, so a wrong result that skips the check
needs one off by at least 0.75 with every other within 0.25.

| Checked entry | Redone by |
|---|---|
| `fft_entry` | `ntt_entry_dyn` |
| `fft_sqr_entry` | `ntt_sqr_entry_dyn` |
| `fft_mid_mul` | `ntt_mid_mul_dyn` (the same contract: within the documented low-limb error) |
| `fft_sqr_mod_bnm1` | `ntt_sqr_entry_dyn` into pooled scratch, then `fold_mod_bnm1` (not `sqr_dyn`, whose FFT square would round about as badly) |

The redo runs outside the `FFT_CACHE` borrow. `fft_reroutes()` counts redone calls, and `fft_fold_checks()`
and `fft_fold_reroutes()` the fold check's runs and failures. The doc-hidden `fft_entry_raw`,
`fft_sqr_entry_raw`, `fft_mid_mul_raw` and `fft_sqr_mod_bnm1_raw` redo nothing and return the
`FftRounding`, fold verdict included, beside the FFT's own result, for the probe and the tests.

**The identity.** The transform computes V = a·b mod 2^(16N) − 1, whatever its digits are, where V sums
the N rounded coefficients at their digit positions. So V = a·b also holds mod any p dividing
2^(16N) − 1, which is what lets a middle product and a cyclic square check themselves: neither keeps a
whole product, so the residue check can't reach them. `FftFold::rem` takes four limbs a step with
precomputed B^k mod p, so only the accumulator chains: 0.76 ns a limb against `div_prim`'s 3.27.

**Products and squares** don't wrap, so any prime holds for them. `FFT_FOLD_PLAIN` is 3689348814741904247,
the largest safe prime below the kernel's bound of 2^64 / 5, so 2 has order (p − 1) / 2. Over 2^20 digits
no two positions share a weight or its negative, and no three unit errors cancel.

**Middle products and cyclic squares** need p to divide 2^(16N) − 1, so the weights 2^(16j) mod p repeat
at some length dividing N digits, and halfway through that length the weight is −1. The NTT primes can't
do this: 2 has order about 2^60 mod each, so they divide it at none of the sizes. Literature: Keller's
table of Fermat factors gives primes of order 2^(k+1), and shows none known for F20 or F22, so 2^21 and
2^23 can't be had that way; Prime95's SUMINP/SUMOUT is the same identity at x = 1. Nothing found publishes
a choice for this use, so the choice was made by search.

The search took every order d = o·2^t, o in {1, 3, 5, 7, 9}, t from 17 to 23, and every p = k·d + 1 with
k up to 2^35 below the kernel's bound: candidates sieved by primes below 2^13, tested by Montgomery
squarings, survivors checked by a deterministic Miller–Rabin. It rediscovered every known factor of F15–F19
and F21 in that range, and found none of order exactly 2^21 or 2^23, matching the table. 41 candidates of at
least 30 bits were then run against real wrong results, compared with the NTT: 446 at these sizes, nearly all
on unsigned digits, as balanced ones only went wrong past the limits, and 327 balanced ones up to N = 2097152.
The longer a prime's repeat, the worse it did:

| Weights repeat every | Wrong results tested | Passed by the prime |
|---:|---:|---:|
| N | 306 | 24.5% |
| N/2 | 345 | 9.3% |
| N/4 | 344 | 27.9% |
| N/8 | 555 | 0.5% |
| N/16 and shorter | 5991 | 1 (0.02%) |

Those errors are evenly spaced copies, such as limbs 1600, 11840, 22080 and 32320 at N = 163840, exactly N/4
apart: a smooth rounding wave goes wrong at each crest. If x^N = 1 and x^s ≠ 1, copies s apart sum to
x^t·(x^N − 1)/(x^s − 1) = 0, so a prime passes them unless its repeat divides s. A shorter repeat has its own
blind spot, opposite errors a repeat apart and equal ones half a repeat apart, but random pairs land there
only about 1 time in the repeat length. So each size takes the longest repeat up to N/16 digits, then the
most bits, from primes of at least 40 bits:

| Transform N | p | Bits | Weights repeat every, digits |
|---:|---:|---:|---:|
| 65536 | 2327042503868417 | 52 | 4096 (N/16) |
| 73728 | 2327042503868417 | 52 | 4096 (N/18) |
| 81920 | 2327042503868417 | 52 | 4096 (N/20) |
| 98304 | 2327042503868417 | 52 | 4096 (N/24) |
| 114688 | 2327042503868417 | 52 | 4096 (N/28) |
| 131072 | 2327042503868417 | 52 | 4096 (N/32) |
| 147456 | 1094462078977 | 40 | 9216 (N/16) |
| 163840 | 2327042503868417 | 52 | 4096 (N/40) |
| 196608 | 2327042503868417 | 52 | 4096 (N/48) |
| 229376 | 2327042503868417 | 52 | 4096 (N/56) |
| 262144 | 31065037602817 | 45 | 16384 (N/16) |
| 294912 | 31065037602817 | 45 | 16384 (N/18) |
| 327680 | 1753340313601 | 41 | 20480 (N/16) |
| 393216 | 31065037602817 | 45 | 16384 (N/24) |
| 458752 | 31065037602817 | 45 | 16384 (N/28) |
| 524288 | 31065037602817 | 45 | 16384 (N/32) |

On the wrong results at these sizes this table passed 0 of 446; the table of longest repeats passed 94. Sizes
outside it have no prime, and a call that fires goes to the NTT. No operand measured reads near 0.25 below
65536 (at most 0.078 there).

**Known limits.** Errors that repeat around the transform at a period no repeat length divides, such as
period 3, pass every wrap-compatible prime: `x^N = 1` makes them cancel. They came up past the limits with
period-3 and period-12 digits, where all candidates passed them, and not inside the limits. Sparse errors at
exactly the spacings above pass too, and anything else at a chance of 1/p, from 2^-40 to 2^-52 here.

## The residue check

FFT products and squares (`fft_entry`, `fft_sqr_entry`) then check the result they would return by
its residues mod the NTT primes (`src/utils/verify.rs`, below): `Residues::of_with_top(out, top)`
against the operands' residues multiplied. A mismatch is redone on the NTT and counted in
`fft_residue_reroutes()` as well as `fft_reroutes()`; it means the fold check and its trigger let a
wrong result through, and it should stay 0. Debug builds always run it, which also means tests of the
checked entries can't see most FFT bugs: `test_fft_raw_entries_are_exact_at_every_odd_part` tests the raw
ones. Release builds run it when `FFT_RESIDUE_CHECK` in `mod.rs` is set (`false` by default). It catches
any wrong result whose error fits in 188 bits, whatever the rounding read. Middle products and cyclic
squares have no such check, since their results omit an unknown part of the full product.

## Cost

Balanced digits, the fold check and removing the coefficient tracking together, release A/B against the
tree before them (M1, one thread, minimum of three alternating runs, 13 operations at 208 points from 400
limbs to the limits): 0.995× geomean, with every operation from 0.979× (squares, whose NEON recombine lost
some redundant scaling) to 1.001× and a worst point of 1.018×.
Outputs matched at every point but one `rcp_dyn` (19324 limbs), whose new value is the exact floor where the
old one's low limb differed, which the contract allows. Against the commit before this branch's FFT work the
whole tree is about 0.69×.

A fired fold check, on top of the FFT at the old limits (three runs): 4–7% on a product, 8–10% on a middle
product and 9–10% on a cyclic square. On one thread a redo costs about 2–3× the FFT on top of it, or up to 6×
for a cyclic square (the old-size timings below).

The residue check reads the operands and the result, 4n limbs for an n × n product, at 2.55 ns per
limb (`Residues::of`, below):

| Product | Residue check | FFT product | Share |
|---|---:|---:|---:|
| 256 × 256 | 2.8 µs | 13.9 µs | 20% |
| 1024 × 1024 | 10.6 µs | 69.7 µs | 15% |
| 4096 × 4096 | 41.9 µs | 318 µs | 13% |
| 16384 × 16384 | 166 µs | 1390 µs | 12% |
| 52920 × 52920 (the old limit) | 534 µs | 9550 µs | 5.6% |

That is why `FFT_RESIDUE_CHECK` is off by default.

## Soundness

**Same operands, old check and new.** Nineteen operand kinds (unsigned worst cases, random limbs, chirps,
and balanced worst cases: one sign at full size, random signs, periods 3 and 12) at every transform size
in the upper tiers, FILLS 1, 0.5 and 0.01, ratios from 1 to 16 for products, only shapes a dispatcher could
reach under the 2026-11-06 limits (so products and squares stopped at transform 458752), 8 operands each. The residue verifier judged products and squares, the NTT middle products and
cyclic squares.

| | Calls | Wrong | Redone on the NTT | Of those, right anyway | Fold checks run | Missed |
|---|---:|---:|---:|---:|---:|---:|
| Unsigned, 0.4 and the gate | 35112 | 2527 | 6727 | 4200 | – | 0 |
| Balanced, fold check | 35112 | 0 | 0 | – | 412 | 0 |

All 412 fold checks were on right results, and every fired kind was a balanced worst case: periods 3 and 12
(272) and one-sign digits (140). Random limbs never fired: at most 0.0026 over 51978 calls at the sizes from
16384 limbs to the limits.

**On x86 under Rosetta (SSE, not AVX2)** readings run higher. A narrower sweep (2352 calls) found 2 wrong middle products inside the limits,
reading 0.47 and 0.5 with digits of period 12, and the fold check caught both, with no miss. So "no wrong results inside the limits" is a NEON result, and AVX2 is unmeasured.

**Estimated misses.** A miss now needs a coefficient off by at least 0.75 with every other within 0.25, or a
wrong result the fold check passes. From each shape's largest-reading tail, per call, as an exponential fit (the
cautious one), counting only the size of the error and not the second condition:

| Operands | Old, error ≥ 0.6 under the gate | New, error ≥ 0.75 |
|---|---:|---:|
| Random, products or squares at the old limits | 10^-8 to 10^-13 | below 10^-1000 |
| Random, middle or cyclic at the old limits | too coarse to fit | below 10^-1000 |
| Random, 32768-limb operands | 10^-22 | below 10^-1000 |
| One-sign digits, 52920 × 52920 | 10^-11 | 10^-15 |
| One-sign digits, 55296-limb squares | 10^-10 | 10^-10 |

These extrapolate far past the data, and readings near 2^48 sit on a grid too coarse to fit many
structured shapes, so treat them as orders of magnitude at best. A rate says nothing about operands built to
defeat the check; only the residue check and the fold check speak to those.

## History: the unsigned-digit check

Until 2026-11-06 digits were unsigned, and a call was redone when its reading passed 0.4 (Prime95's
convention) or its largest coefficient reached 2^48. The evidence below is why the gate existed, and what the
dispatch limits were measured against.

**Why 0.4.** A coefficient off by e > 0.5 reads 1 − e, so one off by less than 0.6 trips by itself, and a
wrong call escapes only with one off by 0.6 or more while every other stays under 0.4.

**Why the gate.** Before the 1/N scaling, an output near N·c is a multiple of ulp(N·c), so a coefficient's
error is a multiple of g = ulp(N·c)/N ≤ c·2^-52, then rounded to the doubles near c. Below 2^48, g is under
1/16. At N = 2^15·7 and c ≈ 2^49.7 (cyclic squares of 57344 limbs with digits at least 0xF000), g = 1/7 and
the doubles near c are 1/8 apart, so the only readings are 0, 1/8, 1/4 and 3/8: a coefficient off by 3/7
rounds right, one off by 4/7 rounds wrong, and both read 0.375.

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

"Rare" only held for those widest shapes. A later sweep at the current sizes found unbalanced products wrong
well below 2^48 (1985 of 2247 wrong in 27648 calls, mostly at 3·2^k with ratios from 2 to 8); all tripped.

## The dispatch limits

**Products, squares and middle products** (2026-11-07). Their limits are the largest the compile-time
assert in `mod.rs` allows: 131066 output limbs, 65533 operand limbs and n = 65533. One-sign digits then
give coefficients up to 4·65533·2^30 ≈ 2^47.99, and one more limb could reach 2^48, past which a wrong
coefficient's reading can fall under the trigger (see the balanced digits section). So soundness sets these
limits, not speed. All three land on transform 524288 (2^19). They were 105839, 55296 and 64826, the
trip-rate limits below. Products and squares used to stop at transform 458752, so the soundness sweep
never ran worst-case operands for them on 524288; middle products already reached it.

A checked entry vouches for every size up to its dispatcher's largest transform, now 524288 for all
three. So a direct call to `fft_entry` or `fft_sqr_entry` with 65534–65536-limb operands (`fft_mid_mul`:
n up to 65535, as before) also runs the FFT, with one-sign coefficients up to about 2^48.

**The cyclic square** keeps its trip-rate limit (below). The next `bnm1_size`, 65536, puts 4m·2^30 at
2^48 and fails the assert by its margin of 8.

**Speed at the limits.** Release, M1, `benches/probes/fft_sizes.rs` in `entry` mode: the raw FFT against
the NTT on random operands of the widest shape, fastest of five, one process per point, on one thread
and on the default 8-thread pool (`RAYON_NUM_THREADS`). The FFT runs on one thread either way, and its
times on the pool matched these.

| Entry, size | FFT | NTT, 1 thread | NTT, 8 threads |
|---|---:|---:|---:|
| Products, 52920 × 52920 (old limit) | 6.4 ms | 21.5 ms | 6.8 ms |
| Products, 65533 × 65533 (limit) | 6.7 ms | 25.8 ms | 10.3 ms |
| Products, 73728 × 73728 | 9.0 ms | 26.5 ms | 8.5 ms |
| Products, 98304 × 98304 | 10.7 ms | 40.2 ms | 13.6 ms |
| Squares, 55296 limbs (old limit) | 4.1 ms | 13.7 ms | 5.0 ms |
| Squares, 65533 limbs (limit) | 4.1 ms | 17.7 ms | 8.1 ms |
| Squares, 73728 limbs | 5.6 ms | 18.2 ms | 6.7 ms |
| Middle products, n = 64826 (old limit) | 6.7 ms | 24.8 ms | 8.5 ms |
| Middle products, n = 65533 (limit) | 6.7 ms | 24.8 ms | 8.8 ms |
| Middle products, n = 73728 | 9.8 ms | 34.7 ms | 9.3 ms |
| Middle products, n = 98304 | 15.0 ms | 45.6 ms | 12.1 ms |
| Cyclic squares, m = 57344 (limit) | 2.0 ms | 17.1 ms | 5.9 ms |
| Cyclic squares, m = 65536 | 1.9 ms | 17.7 ms | 7.8 ms |

At the new limits the FFT is 3.7–4.3× faster than the NTT on one thread and 1.3–2× faster than the
NTT on 8 threads. Since the old table below, the FFT's product at the old limit went from 9.58 ms to
6.4 ms, so the pool's NTT no longer leads there; the two tie. Past 2^19 the 8-thread NTT ties or wins at
some sizes (products at 73728, middle products from 73728), so a further raise needs thread-aware
dispatch as well as a new answer to the 2^48 bound. Random operands read 0.0007–0.0027 at every size
here, and none came out wrong.

### History: the trip-rate limits

Until 2026-11-07 every limit in `mod.rs` was the one set when digits were unsigned, by how often a call
paid for the FFT and then the NTT. Each was the largest size below the first transform size where
uniformly random operands tripped the old check, either the distance or the gate, more than 1 in 2^10
times. Balanced random operands read about 0.001 at them, so as speed choices they were conservative. The
cyclic limit is still this one. The operands had every limb uniform and a nonzero top limb, at the transform size's widest shape: balanced products, full-length squares,
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
| Cyclic squares (modulus limbs m) | 64800 (now 57344) | 259200 | 11 in 2^14, all at the gate | 259308: 17 in 8845, the same | 23040 |

These limits were measured at the old transform sizes and were never remeasured. At the current sizes
they landed on 458752 (products and squares) and 524288 (middle products). 64800 is no longer a `bnm1_size`, so the
cyclic limit is the largest one below it, 57344 (7·2^13), which dispatches the same: the next one up,
65536, is past 64800 either way. A stage whose `bnm1_size(k + 1)` is 65536 (k from 57344 to 65535) now
stays classic, where k up to 64799 used to wrap.

The new limits were 2.8–3.9× the old ones. Trip rates are not monotone in the size, so a size below
a limit can trip more than the limit itself. The most trips at or below each limit were:

| Entry | Trips in 2^14 | Transform N |
|---|---:|---|
| Products | 12 | 393660 = 2^2·3^9·5 |
| Squares | 13 | 428652 = 2^2·3^7·7^2 |
| Middle products | 5 | 470596 = 2^2·7^6 |
| Cyclic squares | 11 | 259200, the limit itself (the gate) |

Sizes past a first failure that round better stay on the NTT.

**Speed at those limits, before 2026-11-05.** The trip-rate rule didn't weigh the NTT's speed. Measured
before the FFT work of 2026-11-05/06, at the old sizes. Release, M1, random operands, minimum of five
alternating rounds, on one thread and on the default 8-thread pool:

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

On one thread the FFT was 1.9–6× faster at every limit. The FFT runs on one thread and the NTT
runs its three primes on the pool, so with 8 threads the NTT already matched the FFT for squares
and middle products at the limits before those, and was about 30% faster at the trip-rate product
and middle-product limits. Only the cyclic square kept a clear FFT lead (2.4×). A dispatch that
weighed the pool's size would switch earlier on many threads; that is not done.

On unsigned digits, operands with digits near 2^16 reached the gate from about 18000 limbs and paid the FFT
and then the NTT above that. With balanced digits those operands read about 0.01 at the limits and don't
pay it.

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
of the tables then in use exactly with `cis` changed nothing, so the tables are not the cause; the
half-length real-FFT packing and its recombination are the likely place, not yet confirmed.
Balanced digits remove the zero-frequency term, which is why the entries use them now.

## History: the worst-case limits

`FFT_16BIT_CUTOFF` was 2^16 output limbs for products and middle products, and `dyn_sqr_dispatch`
compared it with the square's operand length, letting squares reach twice the product's size. Its
only accuracy tests used random operands. Against the exact NTT, worst-case operands
(all ones, a step, random digits all at least 0xF000, 0xFF00 or 0xE000) gave wrong results well inside those limits: all-ones squares from 23814
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
NTT 2–3× earlier than they need. Per-call checks removed both problems.

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
of the full product, so those rely on the fold check, and the probe and the tests compare them with the NTT. Besides the FFT
entries' residue check, `mul_dyn` and `sqr_dyn` assert every NTT result with it in debug builds.

## Remeasuring

`benches/probes/fft_precision.rs` runs operands through the raw entries at every transform size in
a range and reports, per size:
- `fires`, the calls that run the fold check, `needless`, those whose result was right, and `trips`, those
  it failed;
- the errors' tail: calls past 0.30 and 0.375;
- the wrong results, split into `by_fold` (caught), `blind` (it ran and passed them) and `quiet` (read too low
  for it to run), and `missed`, the last two together, which must stay 0;
- the largest error and the smallest among the wrong calls. Its header has the build command. `rate` mode
takes uniformly random limbs, `worst` mode the unsigned worst cases (`KIND=k` forces any kind in its
`operand`), and `sweep` the soundness run above, over `FILLS`, `RATIOS` and `KINDS`, with `REACHABLE=1`
leaving out shapes past the limits. `STOP=k` ends a size after k + 1 trips. Products and squares are
judged by the residue verifier, middle and cyclic squares by the NTT (`no-oracle` skips it).

`FILL=f` and `RATIO=r` set the shape, as operands can fill as little as 5/6 of a size and wide ratios read
differently. A transform size's limbs follow from its entry's sizing: a square of n limbs takes
`find_fft_size(8n − 1)`, a product of O output limbs `find_fft_size(4O + 3)`, a middle product of
short length n `find_fft_size(8n + 4)`, and a cyclic square exactly 4m. The cyclic limit must be a
`bnm1_size`, since `bnm1_fft` only runs the cyclic FFT at such an m.

Rounding depends on rustfft's SIMD paths, so run all of this on each machine: readings under Rosetta already
differ enough to give wrong results inside the limits. A new wrapped transform size needs its own prime.
