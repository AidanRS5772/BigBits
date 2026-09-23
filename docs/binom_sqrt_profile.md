# Binomial square-root profile

Measured on 2026-09-19, against source commit `d655804`, on an Intel Core
i7-8750H (x86-64), pinned to logical CPU 2. Rust 1.94.0, Cargo's `prof`
profile: optimization level 3, thin LTO, debug symbols. Every measured sqrt
calls `binom_sqrt` directly, including sizes above the current cutoff.

The optimization priorities depend on operand size: improve the initial seed
for tiny roots, quotient estimation for medium roots, and the multiply/subtract
and add/subtract limb loops for large roots. The large-input bottleneck is
particularly clear: those limb loops account for about 94% of sampled CPU time
at 1,024 root limbs.

## Measurements

Root sizes are output limbs; a full input has twice that many limbs. These are
medians of three **unsampled** runs over 256 deterministic random inputs with
the high bit set. Timings include the input-buffer reset, but no allocation
inside the timed loop. Percentages come from separate CPU sampling runs and
use all process samples as their denominator.

| Root limbs | Time per operation | Initial seed, including its divisions | Quotient estimate/correction | Quotient multiply/subtract | Other limb add/sub helpers |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.112 µs | 84.5% | 0% | 0% | 0% |
| 2 | 0.209 µs | 88.2% | 0% | 0% | 0% |
| 4 | 0.408 µs | 44.3% | 24.1% | 4.5% | 4.1% |
| 8 | 0.881 µs | 19.8% | 33.4% | 9.2% | 6.8% |
| 16 | 2.100 µs | 8.6% | 36.0% | 17.1% | 10.7% |
| 32 | 5.430 µs | 3.5% | 30.0% | 23.3% | 17.2% |
| 64 | 14.931 µs | 0.9% | 21.4% | 37.4% | 20.2% |
| 128 | 46.250 µs | 0.5% | 15.1% | 45.9% | 23.5% |
| 256 | 143.683 µs | 0.1% | 9.4% | 53.2% | 27.6% |
| 512 | 503.901 µs | <0.1% | 5.4% | 61.0% | 29.0% |
| 1,024 | 1,898.770 µs | <0.1% | 2.9% | 66.6% | 27.6% |

Columns are disjoint and omit control flow, comparisons, final root correction,
and harness/runtime costs. Sample attribution includes inline frames when
available. Out-of-line add/sub helpers with multiple callers stay in their own
category; these samples do not identify a unique calling algorithm stage.

The per-digit multiplication touches root lengths 2 through `n-1`, for
`n*(n-1)/2 - 1` limb products when `n >= 2`. The near-quadratic runtime growth
and increasing multiply/subtract share agree with that structure.

Full and minimally padded inputs (`x.len() = s.len() + 1`) have similar costs.
For example, padded times were 2.136 µs at 16 limbs, 505.621 µs at 512, and
1,897.760 µs at 1,024. Virtual-padding moves are not a leading hotspot in this
range.

Independent input-copy controls cost about 7 ns, 13 ns, 78 ns, and 657 ns for
2, 16, 128, and 1,024 root limbs respectively. Harness reset costs are roughly
3.4% at two limbs and below 1% from 16 limbs onward.

## Where to optimize

1. **Improve the multiply/subtract kernel for large roots.**
   [`sub_mul_of_x86`](../src/utils/div.rs:103), reached through
   [`knuth_est`](../src/utils/div.rs:200), is the largest hotspot. Disassembly
   confirms one multiply plus carry/borrow handling, pointer increments, and
   loop control per limb. Benchmark two- or four-limb unrolling and a better
   carry/borrow schedule. A BMI2/ADX implementation is another candidate with
   CPU-feature gating and the existing portable paths retained. These are
   proposed experiments, not measured speedups. For scale, *halving* this
   kernel's cost would save about one third of total runtime at 1,024 limbs.

2. **Reduce the number of complete limb passes per root digit.**
   The loop in [`binom_sqrt_core`](../src/utils/sqrt.rs:132) reduces the current
   remainder, estimates/subtracts a quotient product, restores a root multiple
   for odd quotients, and applies a square correction. The add/sub helpers
   alone consume another 28–29% at 512–1,024 limbs. A sqrt-specific estimator
   that produces the root digit directly could combine some of those updates,
   instead of first constructing an exact division quotient and then halving
   it. This needs a bound on the initial estimate and careful carry, borrow,
   saturation, and virtual-padding tests. It is a larger algorithmic change
   than unrolling the existing kernel.

3. **Reuse a reciprocal for the medium-size quotient estimates.**
   The x86 `div` in [`div_rem_2_1_x86`](../src/utils/div.rs:15) alone accounts for
   28.4% of samples at 16 limbs, 16.6% at 64, and 11.8% at 128. The two highest
   root limbs passed to `knuth_est` remain fixed throughout one binomial sqrt.
   That makes computing a normalized high-limb reciprocal once and reusing it
   for each quotient estimate a promising experiment. Retain exact estimate
   correction and saturated-quotient handling. Its payoff declines for large
   roots: the same hardware-divide site is only 2.3% at 1,024 limbs.

4. **Simplify the initial two-limb root seed for small roots.**
   [`sqrt_4x2`](../src/utils/sqrt.rs:25) and the integer-square-root implementation
   dominate two-limb cases. Out-of-line 128-bit division support alone accounts
   for 42.0%. After `s_hi = floor(sqrt(x_hi))`, the remainder satisfies
   `0 <= rem <= 2*s_hi`, so `rem / s_hi` and `rem % s_hi` can be replaced with
   at most two compare/subtract steps. The subsequent `y / s_hi` has a
   one-limb quotient because `r0 < s_hi`; an explicitly bounded 128-by-64
   quotient/remainder operation is another candidate. The remaining standard
   `u128::isqrt` cost would still exist. This work has very little effect on
   large roots, where the entire seed is already below 1%.

5. **Consider zero-quotient and saturated-digit cases after the general loops.**
   Power-of-two inputs repeatedly compute zero quotient products. At 512 limbs
   their runtime is 340 µs, with 92.3% of samples in multiply/subtract despite
   those products being zero. A cheap zero-quotient path could remove that
   work; benchmark its extra branch on random inputs too. All-ones inputs
   stress saturation and correction: 512 limbs take 1,220 µs, with 24.6% in
   multiply/subtract, 47.8% in shared add/sub helpers, and 13.3% in root
   correction. These patterns warrant dedicated regression and benchmark cases.

Normalization is a secondary target. On inputs whose top limb is 1, explicit
normalization/denormalization accounts for about 12.5%, 6.1%, 2.8%, and 1.0% at
2, 16, 128, and 512 limbs. The binomial wrapper already skips that work for
normalized inputs. Timings across different input patterns are workload
comparisons, not measured speedups from an optimization.

## Follow-up: focus on roots below 64 limbs

For this range, the first experiment should be a cached reciprocal for the
fixed leading root limb. A standalone exact-division prototype is in
[`benches/probes/binom_sqrt_reciprocal.rs`](../benches/probes/binom_sqrt_reciprocal.rs).
It passed 200,048 deterministic boundary and randomized quotient/remainder
checks against native arithmetic. Five alternating timing rounds, pinned to
CPU 2, gave these medians:

| Root limbs | Estimates (`n-2`) | Hardware division chain | Reciprocal setup + division chain |
|---:|---:|---:|---:|
| 4 | 2 | 89.8 ns | 55.6 ns |
| 8 | 6 | 257.6 ns | 77.9 ns |
| 16 | 14 | 593.1 ns | 136.6 ns |
| 32 | 30 | 1,261.3 ns | 252.8 ns |
| 63 | 61 | 2,553.3 ns | 478.4 ns |

Each chain uses a fixed normalized divisor, includes inverse setup once, and
feeds each remainder into the next numerator to serialize the operations.
These are **division-only microbenchmarks**, not measured end-to-end sqrt gains.
Integrating the primitive may change register pressure, inlining, and correction
costs. Raw results are in `target/binom_sqrt_profiles/reciprocal_probe_serial.txt`.

For `B = 2^64` and normalized divisor `d`, the prototype stores
`inv = floor((B^2-1)/d) - B`. The corresponding full inverse `B+inv` gives a
quotient estimate at most one too small for a numerator below `d*B`. It uses
multiply-high operations followed by an exact remainder correction. The setup
fits one hardware division of `(!d, u64::MAX)` by `d`. This can replace the
division at the start of `knuth_est` without changing its subsequent correction
or multiply/subtract contract.

A second, more involved experiment is to specialize the consumer of that
estimate. The current sqrt computes an exact quotient `t`, subtracts `t*S`,
then adds `S` back when `t` is odd before using `t/2` as the root digit. A
sqrt-specific helper could round the estimate down to even **before** the
multiply/subtract, avoiding that parity-restoration pass. It must preserve an
explicit negative-remainder flag until square correction; the existing
`correct_sqrt` only detects underflow in its own square subtraction.

A deterministic integer model checked 256 normalized random radicands at each
of 4, 8, 16, 32, and 63 root limbs, verifying final roots and remainders against
Python integer square root. About half of the 28,928 root digits needed parity
restoration. The leading-two-limb quotient test adjusted roughly 29% of initial
estimates. Skipping that test produced too-large root-digit candidates in about
14–15% of steps. The full-divisor add-back was never needed in this random
sample, which does not establish that it is unnecessary on adversarial inputs.
Simply deleting quotient corrections is therefore a weaker first experiment
than reciprocal reuse or fusing the even-quotient update. A redesigned helper
needs its own correction bound and saturation tests.

## Assembly follow-up: combine multiply carry and subtraction borrow

The existing `sub_mul_of` loops keep multiply carry and subtraction borrow in
separate registers. They can instead maintain a single carry:

```text
p = d[i] * q + carry
win[i], borrow = subtract_with_borrow(win[i], low(p))
carry = high(p) + borrow
```

This carry fits a limb: inductively `carry <= q`. At the one boundary where
`high(p) == q`, `low(p) == 0`, so subtraction cannot add another borrow.
Finally subtract the combined carry from the overflow limb and return that
subtraction's borrow.

An x86-64 prototype replaces the per-limb `NEG b; SBB; SETC b` bookkeeping with
`SUB; ADC high, 0`. It needs no reciprocal or additional ISA feature. Five
rotating-order timing rounds on CPU 2 produced these medians:

| Divisor limbs processed by sub_mul_of | Existing instruction loop | Combined-carry loop | Time reduction |
|---:|---:|---:|---:|
| 2 | 12.786 ns | 11.344 ns | 11.3% |
| 4 | 17.918 ns | 15.403 ns | 14.0% |
| 8 | 26.319 ns | 22.391 ns | 14.9% |
| 16 | 49.342 ns | 40.267 ns | 18.4% |
| 32 | 85.114 ns | 73.681 ns | 13.4% |
| 63 | 153.829 ns | 134.012 ns | 12.9% |

These are isolated kernel-call timings including equal input-reset overhead,
not complete sqrt timings. Lengths here refer to each multiply/subtract's
divisor, not the final sqrt output. The baseline reproduces the existing
instructions with its modified `len` register correctly declared `inout`.

Six candidates, including two- and four-limb unrolling, passed 228,480
comparisons against a separate `u128` reference over lengths 1–70. Checks cover
output limbs, the overflow limb, the returned borrow, random inputs, zero and
maximal digits, and carry/borrow boundary combinations. Unrolling the original
loop alone regressed two- and three-limb cases by roughly 9–14%; its benefit at
32–63 limbs was only about 0–5%. Unrolling the combined-carry loop did not give
a consistent further improvement. Start with the simple combined-carry loop.

The AArch64 counterpart uses `SUBS` followed by `CINC carry, high, CC`, replacing
the separate borrow register and its `CMP`/`CSET` handling. The candidate
assembles with Clang's AArch64 target but has **not** been executed or timed on
ARM hardware. Pairwise `LDP`/`STP` and overlapping the independent next limb's
`MUL`/`UMULH` are further experiments for an ARM machine.

The production x86 assembly also declares `len = in(reg) len` while executing
`DEC` on it. That must be an input/output operand such as
`len = inout(reg) len => _`. Rust requires input-only registers to retain their
entry values on exit; this is a correctness issue, independent of the proposed
speedup. See the [Rust inline-assembly rules](https://doc.rust-lang.org/reference/inline-assembly.html#rules-for-inline-assembly).

Reproducers: [`binom_submul.rs`](../benches/probes/binom_submul.rs) and
[`binom_submul_aarch64.S`](../benches/probes/binom_submul_aarch64.S). Build/run
commands are at the top of each file. Raw timings are in
`target/binom_sqrt_profiles/submul_probe.csv`. Production assembly has not been
changed by these experiments.

## Follow-up: simplify the two-limb sqrt seed

An isolated probe of `sqrt_4x2` is in
[`binom_sqrt_seed.rs`](../benches/probes/binom_sqrt_seed.rs). It uses copies of
the seed arithmetic and the existing `correct_sqrt` helper; it does not change
production sqrt or division. The helper was linked from the earlier profiling
build, before the working-tree division assembly edits. No division-module
functions are called by this probe.

The most useful rewrite halves the numerator before dividing. Write `B = 2^64`,
`h = s_hi`, `r = x_hi - h*h`, and `a = x1`. The desired low root estimate is
`min(B-1, floor((r*B+a)/(2*h)))`. Because `r <= 2*h`, the ordinary case can use:

```rust
let (mut s_lo, u) = if rem == 2 * s_hi {
    (u64::MAX as u128, 2 * s_hi + *x1 as u128)
} else {
    let half = (rem << 63) | ((*x1 as u128) >> 1);
    let q = half / s_hi;
    let v = half % s_hi;
    (q, (v << 1) | ((*x1 & 1) as u128))
};
```

`half` fits in `u128` and `q` fits in `u64` when `rem < 2*s_hi`. The identity
`floor(floor(A/2)/h) = floor(A/(2*h))` preserves the estimate exactly, and
`2*v + (a & 1)` preserves its remainder coefficient. Saturation needs its own
branch because the unclamped quotient would be `B`. This removes the initial
`rem / s_hi` calculation, the three-way `c` dispatch, and the parity-times-root
multiplication. Assembly inspection confirms two division helper calls after
`isqrt` in the baseline, versus one on this rewrite's ordinary path.

The second useful change is a fixed-width correction: subtract `s_lo*s_lo`
from the low 128 bits, propagate the borrow through the third limb, and on
underflow decrement the root and add `2*new_root+1` in 129 bits. The probe's
`FIXED` branch contains this implementation. On x86 it compiles to direct
`sub/sbb` and `add/adc` chains instead of the generic slice helpers' assembly
loops and stack buffers. Explicitly narrowing the proven one-limb root values
also removes unnecessary high-half multiply terms, but its standalone timing
benefit was negligible.

Seed-only median times, i7-8750H, CPU 2, Rust 1.94.0, `rustc -O`:

| Variant | Random | Exact squares | Saturated |
| --- | ---: | ---: | ---: |
| Original | 200.8 ns | 191.6 ns | 186.3 ns |
| Compare/subtract for first quotient/remainder | 220.1 ns | 209.7 ns | 148.4 ns |
| Same, explicitly compute second remainder as `y-t*s_hi` | 182.9 ns | 171.6 ns | 135.8 ns |
| Halved numerator | 173.0 ns | 162.6 ns | 116.7 ns |
| Fixed-width correction only | 191.1 ns | 186.6 ns | 179.0 ns |
| Halved numerator + fixed-width correction | 164.6 ns | 159.8 ns | 110.5 ns |
| Same + narrow root values | 162.5 ns | 154.4 ns | 108.2 ns |

The straightforward comparison rewrite regressed on ordinary inputs: the
generated code called separate division and remainder helpers across its
branches. Explicitly expressing the second remainder as `y-t*s_hi` recovered
that loss, but the halved-numerator rewrite was still faster. Do not infer an
improvement from removing `/` in the source alone.

Each pattern cycles through 4,096 deterministic inputs, with five 100 ms
rounds per variant in rotating order after warmup. Full raw timings, including
min/max, are in `target/binom_sqrt_profiles/seed_probe.csv`. All variants passed
560,420 normalized inputs, including random values, exact squares and their
neighbors, root-width boundaries, and `rem = 0, h-1, h, h+1, 2*h-1, 2*h`.
Checks both compare root/remainder with the original and independently verify
`root^2 + remainder == input` and `remainder < 2*root+1`. The same checks passed
with Rust overflow checks enabled.

These are **seed-only**, x86 measurements. The roughly 19% random seed reduction
is not a 19% full-binomial improvement. Using the earlier sample shares gives
rough projections of 8% at four root limbs, 4% at eight, 2% at sixteen, and
below 1% at thirty-two. These projections need end-to-end measurements before
retuning cutoffs. One-limb roots do not call this seed. ARM has not been timed.

There is also a separate, unbenchmarked control-flow opportunity in
`binom_sqrt_est_reduced`. Its divisor has at least two limbs. If `c == 2`, the
previous exact remainder must have been `2*S`; reducing it leaves zero, so the
next division numerator consists only of the incoming single limb. It is
strictly less than `S`, hence `t == 0`. That branch can skip `knuth_est`, add
`S` twice to the existing numerator buffer, and return `u64::MAX`. For `c < 2`,
the quotient is simply `(c << 63) | (t >> 1)` and restoring the remainder
requires at most one `add_buf` when `t & 1 != 0`. This is especially relevant
to saturated patterns; it is not a measured general random-input speedup.

## Method and reproduction

Samply was blocked by this host's `perf_event_paranoid = 2`. The replacement
sampler uses `ITIMER_PROF` at 1,000 µs to record interrupted program counters
from the unchanged optimized executable. `addr2line` resolves source locations
and inline frames. It collects roughly 3,100–3,900 samples per case, including
warmup and complete-batch timing overshoot, with zero dropped samples. These
are approximate CPU-time shares, not hardware cycle, cache-miss, or branch-miss
counters. Out-of-line helper attribution and sampling granularity limit fine
distinctions between adjacent operations.

The main sweep contains 22 runs: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, and
1,024 root limbs, each full and minimally padded. At one limb the two shapes
coincide. Twelve further runs cover shifted, all-ones, and power-of-two full
inputs at 2, 16, 128, and 512 limbs. Random cases cycle through 256 operands;
the all-ones and power-of-two cases deliberately repeat their single pattern.

```bash
cargo build --profile prof --bench binom_sqrt_profile
# Substitute the executable path printed by Cargo, under target/prof/deps.
python3 scripts/python/profile_binom_sqrt.py \
  --binary target/prof/deps/binom_sqrt_profile-<hash> \
  --output target/binom_sqrt_profiles/random
python3 scripts/python/profile_binom_sqrt.py \
  --binary target/prof/deps/binom_sqrt_profile-<hash> \
  --sizes 2,16,128,512 --shapes full --patterns shifted,ones,power \
  --output target/binom_sqrt_profiles/patterns
cargo bench-utils -- 'zimmermann_sqrt/.*/binom/(16|64|256|512)$' --quick
```

Use an available logical CPU with `--cpu`; the default is 2. The collector
requires Linux, GCC, Python 3, and addr2line. The benchmark's `--mode copy`
measures the input-reset control loop. `--mode core` measures the normalized
binomial core directly; the main reported sweep uses `--mode binom`.

Raw address counts, resolved frames, timing logs, and JSON summaries are under
`target/binom_sqrt_profiles/{random,patterns}`. The input-copy controls are in
`target/binom_sqrt_profiles/controls.json`. These generated artifacts are local
build outputs; the benchmark and collection script make them reproducible.

The existing Criterion binomial cases also completed successfully. Their
central estimates were 1.696 µs, 11.350 µs, 129.890 µs, and 449.070 µs for
16, 64, 256, and 512 limbs. That benchmark uses a different operand/batching
workload and was not CPU-pinned, so its absolute timings are separate baselines.
No before/after optimization claim is made from either baseline.
