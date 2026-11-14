# 16-bit FFT precision: balanced digits, the fold check, the residue check, the limits

The dynamic layer computes products, squares, middle products and squares mod B^m − 1 on a
floating-point FFT of 16-bit digits (`fft_entry`, `fft_sqr_entry`, `fft_mid_mul`,
`fft_sqr_mod_bnm1`). Each is exact only while every output coefficient's rounding error stays below
0.5. Operands go in as balanced digits, so errors stay small; a call that reads above 0.25 runs the
fold check and is redone on the NTT if that fails; products and squares can also check residues.
The four limits in `src/utils/mod.rs` bound soundness as well as speed. Measured on the M1
(rustfft's NEON paths) unless noted. The code's layout is in `mul.md`.

**The user's priority is exactness over speed**: a dispatcher that silently returns wrong limbs is
worse than a slower tier. Validate any change to a float transform against worst-case operands and
an exact oracle, and count misses (wrong and untripped), not only totals.

**What is current.** The balanced-digit, fold-check, soundness and cost numbers were measured
2026-11-06 under that day's limits (products 105839 output limbs, squares 55296, middle products
n = 64826). On 2026-11-07 those three rose to the most the 2^48 bound allows, so products and
squares now reach transform 524288, where no worst-case sweep has run them. The cyclic limit was
measured on unsigned digits at the old sizes and hasn't been remeasured.

## Transform sizes

`find_fft_size(n)` is the smallest 4·o·2^a ≥ n with o in {1, 3, 5, 7, 9}; `bnm1_size` follows it
(m = o·2^a). rustfft's NEON planner sends only 2^k and 3·2^k through its radix-4 kernel and
everything else through a mixed-radix split, so per-point cost falls in tiers. ns/(n log2 n) against
a power of two in the same octave (products / squares, median over 2^11–2^19):

| Odd part | Cost | Odd part | Cost | Odd part | Cost |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 / 1.00 | 15 | 1.22 / 1.18 | 45 | 1.33 / 1.30 |
| 3 | 1.00 / 0.99 | 21 | 1.36 / 1.34 | 49 | 1.42 / 1.40 |
| 5 | 1.21 / 1.19 | 25 | 1.50 / 1.49 | 63 | 1.36 / 1.28 |
| 7 | 1.18 / 1.18 | 27 | 1.49 / 1.46 | 125 | 1.79 / 1.73 |
| 9 | 1.21 / 1.20 | 35 | 1.35 / 1.35 | 243, 729 | 1.72–1.93 |

A costly size just below a cheap one loses to it. Against the old rule (the smallest 7-smooth size)
the set {1, 3, 5, 7, 9} takes 0.69× the time for products and 0.71× for squares over log-uniform
lengths, within 1.006× of the fastest size at or above each length. 15 is out because 15·2^k sits
6.7% below 2^(k+4) at 1.2× the cost; 7 adds nothing on average but holds the worst length level.

At equal operands 3·2^k reads about 1.5× its neighbours' rounding (cause unknown). AVX plans its
factors differently, so these tiers may not hold on x86: `benches/probes/fft_sizes.rs` (`sizes`
mode) measures per-point cost by class on any machine, and its medians are what
`FFT_ODD_COST` holds per architecture.

## The stages around the transforms

Share of a call at n = 2^19 (products / squares): transforms 84% / 74%, recombine 6.6% / 13%,
accumulate 5.1% / 7.7%, rounding 2.4% / 3.8%, decomposing 1.5% / 0.9%.

- `recombine`/`sqr_recombine` take two conjugate pairs per NEON step with fused products (x86 keeps
  the scalar loops), reading e^(2πik/n) from `fft_twidles`, one exact `cis` table per transform size
  per thread. A first call's setup is about 85% rustfft's plans.
- `scale_and_round` multiplies by 1/n at power-of-two lengths (exact, so bit-identical). Other
  lengths divide: an inexact reciprocal would move the distance the trigger reads.
- `fft_accumulate` adds each limb's four digits in two halves and reads one limb past `out`.

The sizes, tables and these stages together: 0.667× geomean over 13 operations against the tree
before them, outputs identical.

## Balanced digits

`decompose`/`sqr_decompose` write each operand as digits in [−2^15, 2^15) (`balance_limb`: a digit
of 2^15 or more has 2^16 subtracted and carries 1 up). A coefficient's digit products then mostly
cancel, which removes the huge zero-frequency term every rounding error scaled with. An operand's
top digit stays unsigned, at most 2^16 (`fft_absorb_top_carry`), so no transform grows.
`fft_accumulate_signed` carries the signed coefficients; a cyclic square folds a negative top carry
back in. Middle products no longer add `middle_correction`: the band already holds the value to
about 2^34 units of its lowest limb, in either sign, so compare them by value.

Largest reading on the same operands, unsigned against balanced: random products at the old limit
0.219 → 0.0007; random middle products and cyclic squares 0.188 → 0.0010; digits ≥ 0xC000 0.5 →
0.0137. Two kinds don't improve: digits of one sign at full size (0.219, nothing cancels) and
repeating patterns (period 12: 0.188 → 0.438, energy in a few frequencies).

**Why there is no coefficient gate.** A coefficient's error is read in steps of up to c·2^-52, so
past 2^48 a reading stops meaning much. A balanced coefficient sums at most 4·min(l, s) digit
products of at most 2^30, which stays under 2^48 at every limit; a compile-time assert in `mod.rs`
holds the limits to that. Raising one needs a new answer to this, not only a measurement. Past each
dispatcher's largest transform a checked entry goes straight to the NTT without a transform: at
N = 3145728 middle products read under 0.25 and came out wrong.

## The fold check

`scale_and_round` returns `FftRounding`, whose `err` is max |x − round(x)|. Above
`FFT_FOLD_TRIGGER` (0.25) the call runs `fft_fold_agrees` on all N rounded coefficients; a pass
keeps the result however high it read, a failure redoes the call on the NTT (`ntt_entry_dyn`,
`ntt_sqr_entry_dyn`, `ntt_mid_mul_dyn`; for the cyclic square a full NTT square then
`fold_mod_bnm1`, since `sqr_dyn`'s FFT would round about as badly). A high reading only shows
noise, and most such results are right. A wrong limb needs a coefficient off by more than 0.5,
reading 1 − e, so a wrong result skips the check only with one off by at least 0.75 and every other
within 0.25.

**The identity.** The transform computes V = a·b mod 2^(16N) − 1, where V sums the rounded
coefficients at their digit positions. So V = a·b mod any p dividing 2^(16N) − 1, which lets a
middle product and a cyclic square check themselves (neither keeps a whole product, so residues
can't reach them). `FftFold::rem` takes four limbs a step: 0.76 ns a limb.

**Products and squares** don't wrap, so any prime holds: `FFT_FOLD_PLAIN` = 3689348814741904247,
the largest safe prime below 2^64 / 5. Over 2^20 digits no two positions share a weight or its
negative.

**Middle products and cyclic squares** need p | 2^(16N) − 1, so the weights 2^(16j) mod p repeat at
a length dividing N digits. `FFT_FOLD_WRAPPED` has one prime per size:

| Transform N | p | Weights repeat every |
|---|---:|---:|
| 65536, 73728, 81920, 98304, 114688, 131072, 163840, 196608, 229376 | 2327042503868417 (52 bits) | 4096 digits |
| 147456 | 1094462078977 (40 bits) | 9216 |
| 262144, 294912, 393216, 458752, 524288 | 31065037602817 (45 bits) | 16384 |
| 327680 | 1753340313601 (41 bits) | 20480 |

Sizes outside the table have no prime, and a call that fires there goes to the NTT (no operand
measured reads near 0.25 below 65536). A new wrapped size needs its own prime.

**How the primes were chosen: by real errors, not a formula.** Real wrong results are evenly spaced
copies (a smooth rounding wave goes wrong at each crest, e.g. limbs exactly N/4 apart). If x^N = 1
and x^s ≠ 1, copies s apart sum to zero, so a prime passes them unless its repeat divides s. Tested
on 446 real wrong results at these sizes:

| Weights repeat every | Passed by the prime |
|---:|---:|
| N | 24.5% |
| N/2 | 9.3% |
| N/4 | 27.9% |
| N/8 | 0.5% |
| N/16 and shorter | 0.02% (1 of 5991) |

So each size takes the longest repeat up to N/16, then the most bits, from primes of at least 40
bits; that table passed 0 of 446 (the longest-repeat table passed 94). A first scoring formula that
favoured the longest repeat was wrong.

**Known blind spots.** Errors repeating at a period no repeat length divides (period 3: seen past
the limits, not inside); opposite errors a repeat apart and equal ones half a repeat apart; anything
else at 1/p (2^-40 to 2^-52).

## The residue check

`fft_entry` and `fft_sqr_entry` then check the result by residues mod the NTT primes
(`verify.rs`, described in `mul.md`): a mismatch is redone on the NTT and counted in
`fft_residue_reroutes()`, which should stay 0, since it means the fold check let a wrong result
through. Always in debug builds; in release when `FFT_RESIDUE_CHECK` is set. It catches any wrong
result whose error fits 188 bits. Cost: 2.55 ns per limb over operands and result, 20% of the FFT
at 256 limbs, 12% at 16384, 5.6% at 52920, which is why it is off. Middle products and cyclic
squares have no such check.

Because debug builds always run it, tests through the checked entries can't see most FFT bugs: with
one coefficient off by 1 in every transform, every product test still passed. The raw entries are
what `test_fft_raw_entries_are_exact_at_every_odd_part` tests.

## Cost

Balanced digits and the fold check against the tree before them: 0.995× geomean over 13 operations,
worst point 1.018×. A fired fold check costs 4–7% of a product, 8–10% of a middle product, 9–10% of
a cyclic square; a redo costs 2–3× the FFT (up to 6× for a cyclic square).

## Soundness

Nineteen operand kinds (unsigned worst cases, random, chirps, balanced worst cases: one sign at full
size, random signs, periods 3 and 12) at every transform size in the upper tiers, fills 1, 0.5 and
0.01, ratios 1 to 16, shapes a dispatcher could reach under the 2026-11-06 limits, 8 operands each:

| | Calls | Wrong | Redone | Fold checks | Missed |
|---|---:|---:|---:|---:|---:|
| Unsigned digits, 0.4 trigger and 2^48 gate | 35112 | 2527 | 6727 | – | 0 |
| Balanced digits, fold check | 35112 | 0 | 0 | 412 | 0 |

All 412 fold checks were on right results from balanced worst cases. Random limbs never fired (at
most 0.0026 over 51978 calls).

**x86 under Rosetta (SSE, not AVX2)** reads higher: 2 wrong middle products inside the limits in
2352 calls (period-12 digits, reading 0.47 and 0.5), both caught. So "no wrong results inside the
limits" is a NEON result, and AVX2 is unmeasured.

**How to run a soundness check** (lessons, each from a run that missed something):

- Use narrow and unbalanced shapes, not only each size's widest: unbalanced products go wrong well
  below 2^48 (1985 of 2247 wrong in one sweep, mostly at 3·2^k with ratios 2 to 8; all tripped).
  Include lengths equal mod 32 and chirp operands, whose rounding collapses into one or two spikes
  at the operands' length boundaries.
- Raw 0x7FFF/0x8000 limbs balance to one sign (a carry persists): build signed worst cases from
  their digits.
- Count what each check alone is blind to, not only total misses. A fold mod B^r − 1 (r = 32) passed
  880K calls and then missed chirps of 524257 × 349505 limbs, whose ±1 errors land in one class and
  cancel. That is why the fold is mod a prime.
- Don't run the FFT and the NTT in one process near the limits on an 8 GB machine: the FFT peaks at
  2.15 GB at 2^25 digits.

## The dispatch limits

| Limit | Value | Transform |
|---|---:|---:|
| `FFT_16BIT_CUTOFF` (output limbs l + s − 1) | 131066 | 524288 |
| `FFT_SQR_16BIT_CUTOFF` (operand limbs) | 65533 | 524288 |
| `FFT_MID_16BIT_CUTOFF` (short limbs n) | 65533 | 524288 |
| `FFT_BNM1_16BIT_CUTOFF` (modulus limbs m) | 57344 | 229376 |

The first three are the largest the assert allows: one-sign digits then give coefficients up to
2^47.99. The cyclic limit is the largest `bnm1_size` below 64800, where random operands in unsigned
digits tripped the old check more than 2^-10 of the time; the next one, 65536, fails the assert by
its margin of 8. A sqrt stage with k from 57344 to 65535 therefore stays classic.

Raw FFT against the NTT, random operands, widest shape (`fft_sizes.rs`, `entry` mode):

| Entry, size | FFT | NTT, 1 thread | NTT, 8 threads |
|---|---:|---:|---:|
| Products, 65533 × 65533 (limit) | 6.7 ms | 25.8 ms | 10.3 ms |
| Products, 73728 × 73728 | 9.0 ms | 26.5 ms | 8.5 ms |
| Squares, 65533 limbs (limit) | 4.1 ms | 17.7 ms | 8.1 ms |
| Middle products, n = 65533 (limit) | 6.7 ms | 24.8 ms | 8.8 ms |
| Middle products, n = 98304 | 15.0 ms | 45.6 ms | 12.1 ms |
| Cyclic squares, m = 57344 (limit) | 2.0 ms | 17.1 ms | 5.9 ms |

At the limits the FFT is about 4× faster on one thread and 1.3–2× against the 8-thread NTT. Past
2^19 the pool's NTT ties or wins at some sizes, so a further raise needs thread-aware dispatch too.

## Why the FFT fails: precision, not arithmetic

A coefficient of an l × s product sums up to 4·min(l, s) products of 16-bit digits, about
2^34·min(l, s): 2^49 at 32768 × 32768, which leaves a double about 4 bits below the units place.
Unsigned digits near their maximum put most of the spectrum in the zero-frequency term. A plain
rustfft complex convolution has the same problem at the same sizes; the library's real FFT loses up
to about a bit more at mixed-radix sizes (not the twiddle tables, which were checked exactly;
probably the half-length real-FFT packing).

## History

1. **Until 2026-11-02:** `FFT_16BIT_CUTOFF` was 2^16 output limbs, checked only with random
   operands, and squares compared it with their operand length. All-ones squares were wrong from
   23814 limbs, all-ones products from 26460 × 26460, worst-case middle products from n ≈ 16800.
2. **Worst-case limits** (36863 / 18000 / 16799 / 23040): just below the first transform size where
   10000 worst-case operands gave a wrong limb. No margin: the failure point kept moving down as the
   search grew (√(2 ln n) for a maximum of Gaussian samples), and random operands went to the NTT
   2–3× too early.
3. **Per-call check on unsigned digits** (2026-11-04): redo past a reading of 0.4 or a coefficient of
   2^48. The gate existed because past 2^48 errors come in steps of ulp(N·c)/N: at N = 7·2^15 a
   coefficient off by 3/7 rounds right, one off by 4/7 rounds wrong, and both read 0.375 (264 wrong
   results read 0.375 or 0.25 in 128832 worst-case calls). Limits were set where random operands
   tripped more than 2^-10 of the time (105839 / 55296 / 64826 / 64800).
4. **Balanced digits, the fold check and the 0.25 trigger** (2026-11-06), then the three limits
   raised to the assert's maximum (2026-11-07) without a worst-case sweep on 524288.

Tried and not taken: a balanced MixedRadix plan (1.2–1.58× slower); a four-step parallel FFT (1.6×
on 8 threads at 2^23 points); a no-reorder kernel (0.57–0.63× at 2^18–2^19, but 2^k only, NEON
only, 1.2–1.4× worse rounding); scaling the short operand by a power of two (more noise); a
digit-energy scan before the transform (2.97 ns a limb, dearer than the residues and no better a
predictor). The user asked for no further checks that cost run time.

## Remeasuring

`benches/probes/fft_precision.rs` (build line in its header) runs operands through the raw entries
at every transform size in a range and reports per size: `fires` (fold checks run), `needless`
(result was right), `trips` (it failed), the errors' tail, and the wrong results as `by_fold`
(caught), `blind` (it ran and passed them) and `quiet` (read too low for it to run); `missed`, the
last two together, must stay 0. Modes: `rate` (uniformly random limbs), `worst` (unsigned worst
cases; `KIND=k` forces a kind), `sweep` (the soundness run, over `FILLS`, `RATIOS`, `KINDS`;
`REACHABLE=1` leaves out shapes past the limits). `STOP=k` ends a size after k + 1 trips. Products
and squares are judged by the residue verifier, the other two by the NTT.

A transform size's limbs follow from its entry: a square of n limbs takes `find_fft_size(8n − 1)`,
a product of O output limbs `find_fft_size(4O + 3)`, a middle product `find_fft_size(8n + 4)`, a
cyclic square exactly 4m. Rounding depends on rustfft's SIMD paths, so run this on each machine.
