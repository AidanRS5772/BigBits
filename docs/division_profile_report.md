# End-to-end division profile report

Date: 2026-07-25

## Executive summary

The profiles confirm the multiplication-dominance hypothesis for
Newton-Raphson (NR), but not for the entire division stack:

- Transform-sized NR calls spend 96.4–99.9% of attributed CPU time in
  multiplication-owned work. Direct `div.rs` instructions are generally below
  1%.
- Schoolbook-middle NR calls still spend 92.2–95.9% in multiplication, except
  for the deliberately tiny dynamic reciprocal boundary at precision 9, where
  division setup/refinement is still 33.4%.
- Knuth calls spend 98.6–99.9% in division-owned work and effectively none in
  multiplication.
- Karatsuba-region Burnikel-Ziegler (BZ) calls are about 65% division and 35%
  multiplication. Transform-region BZ shifts toward multiplication: 68–69%
  for the dynamic FFT cases and 83–95% for the static NTT cases.

The main optimization target inside `div.rs` is
[`knuth_est`](../src/utils/div.rs#L302). On this AArch64 machine it accounts for
roughly 86–88% of end-to-end Knuth division. The compiler-supported 128÷64
operation around it contributes another 10–12%, much of it symbolized as
`<u128>::leading_zeros`. The same estimator and support code consume about 61–63%
of the Karatsuba-region BZ profiles because BZ reaches Knuth base cases.

The primary large-NR target is therefore not dispatch or wrapper cleanup. It is
the middle/short multiplication pipeline in `mul.rs`, especially transform
memory movement and NTT parallel scheduling. A division-specific opportunity
also exists in [`nr_rem_finish`](../src/utils/div.rs#L1417): it computes a full
`d * q_low` product even though it consumes only the low `d.len() + 1` limbs.

## Harness and method

The repeatable harness is [`benches/div_profile.rs`](../benches/div_profile.rs).
It calls the six production dispatchers:

- dynamic: `div_dyn`, `div_rem_dyn`, `rcp_dyn`;
- static: `div_static<N>`, `div_rem_static<N>`, `rcp_static<N>`.

It does not force internal algorithms and does not add profiling hooks to
`src/`. Operands are deterministic, trimmed, normalized, and use the rigid full
quotient shape. Quotient-only and reciprocal inputs are reused. `div_rem` must
restore its mutated numerator between iterations; that copy is separately
classified as harness work and stayed at or below 0.9% in every measured
`div_rem` case.

[`scripts/profile_division.sh`](../scripts/profile_division.sh) builds the
harness with the `prof` profile, records one Samply profile per case, and invokes
[`scripts/python/analyze_div_profiles.py`](../scripts/python/analyze_div_profiles.py)
to produce CSV and detailed Markdown summaries:

```bash
scripts/profile_division.sh

# One or more selected cases:
DIV_PROFILE_SECONDS=5 scripts/profile_division.sh \
    dyn_div_nr_ntt dyn_div_rem_nr_ntt dyn_rcp_nr_ntt
```

Raw profiles and generated summaries are written to
`target/division_profiles/`, which is already ignored with the rest of
`target/`.

The reported run used:

- Apple AArch64, 8 logical CPUs, macOS 26.5.2;
- Rust 1.95.0;
- Samply 0.13.1 at 1,000 Hz;
- 100 ms warmup plus 3 seconds of measured looping per case;
- the `prof` Cargo profile (`opt-level=3`, thin LTO, debug info);
- all Rayon workers, with stack weights based on Samply's per-thread CPU-time
  deltas.

The analyzer reports two related quantities:

- **`div.rs` direct**: the sampled instruction resolves directly to
  `src/utils/div.rs`.
- **division total**: direct `div.rs` plus compiler/runtime and limb-helper work
  whose nearest BigBits owner is division. This is the more accurate figure for
  inline assembly and compiler-generated 128-bit arithmetic.

Multiplication includes `mul.rs`, RustFFT/transpose, NTT, and runtime work below
a multiplication frame. Rayon worker/runtime samples in NTT-only cases are
charged to multiplication because the isolated process performs no other work.
Inlining can remove an intermediate multiplication frame, so ambiguous support
work is conservatively left with division; the reported division share is
therefore an upper bound in those cases.

## Why these sizes

The 29 cases cover all current production layers:

- divisor width 88 and 89 bracket the shared Knuth/BZ eligibility boundary;
- balanced 89-limb cases enter NR with schoolbook middle products;
- long-quotient 128-limb shapes enter BZ in the Karatsuba dispatch region;
- 2,048-limb dynamic balanced cases exercise NR/FFT;
- 65,536-limb dynamic balanced division and 65,536-limb reciprocal precision
  exercise genuine NR/NTT middle products;
- 2,048-limb static cases exercise NR/NTT;
- long-quotient dynamic/static cases exercise BZ in transform-sized regions;
- reciprocal precisions 8/9 and 100/101 bracket the dynamic/static Knuth-to-NR
  cutoffs.

The reciprocal profiles confirm that production dispatch contains only Knuth
and NR; no BZ reciprocal path is present.

## Attribution results

Percentages below are shares of attributed CPU time. Small differences below
about one percentage point should not be treated as cutoff-quality benchmark
evidence; the profiles are intended for hotspot attribution.

| Case | Production path | `div.rs` direct | Division total | Multiplication | Harness |
|---|---|---:|---:|---:|---:|
| `dyn_div_knuth` | Knuth, d/q 88/88 | 0.8% | 99.9% | 0.0% | 0.1% |
| `static_div_knuth` | Knuth, d/q 88/88 | 0.7% | 99.3% | 0.0% | 0.7% |
| `dyn_div_rem_knuth` | Knuth, d/q 88/88 | 0.7% | 99.5% | 0.0% | 0.5% |
| `static_div_rem_knuth` | Knuth, d/q 88/88 | 0.7% | 99.7% | 0.0% | 0.3% |
| `dyn_div_nr_school` | NR/school, d/q 89/89 | 4.4% | 5.1% | 94.9% | 0.0% |
| `static_div_nr_school` | NR/school, d/q 89/89 | 5.2% | 7.8% | 92.2% | 0.1% |
| `dyn_div_rem_nr_school` | NR/school, d/q 89/89 | 2.5% | 4.0% | 95.9% | 0.1% |
| `static_div_rem_nr_school` | NR/school, d/q 89/89 | 2.7% | 4.5% | 94.6% | 0.9% |
| `dyn_div_bz_karatsuba` | BZ, d/q 128/512 | 1.0% | 64.9% | 35.1% | 0.0% |
| `static_div_bz_karatsuba` | BZ, d/q 128/256 | 1.3% | 64.9% | 35.1% | 0.0% |
| `dyn_div_rem_bz_karatsuba` | BZ, d/q 128/256 | 0.6% | 64.9% | 35.0% | 0.1% |
| `static_div_rem_bz_karatsuba` | BZ, d/q 128/256 | 1.9% | 65.3% | 34.6% | 0.1% |
| `dyn_div_nr_fft` | NR/FFT, d/q 2,048/2,048 | 0.5% | 0.6% | 99.4% | 0.0% |
| `dyn_div_rem_nr_fft` | NR/FFT, d/q 2,048/2,048 | 0.9% | 1.1% | 98.7% | 0.2% |
| `dyn_rcp_nr_fft` | NR/FFT, precision 2,048 | 0.6% | 0.8% | 99.2% | 0.0% |
| `dyn_div_nr_ntt` | NR/NTT, d/q 65,536/65,536 | 0.3% | 0.3% | 99.7% | 0.0% |
| `dyn_div_rem_nr_ntt` | NR/NTT, d/q 65,536/65,536 | 0.1% | 0.2% | 99.7% | 0.1% |
| `dyn_rcp_nr_ntt` | NR/NTT, precision 65,536 | 0.1% | 0.2% | 99.7% | 0.1% |
| `static_div_nr_ntt` | NR/NTT, d/q 2,048/2,048 | 0.1% | 3.4% | 96.4% | 0.0% |
| `static_div_rem_nr_ntt` | NR/NTT, d/q 2,048/2,048 | 0.0% | 0.1% | 99.9% | 0.0% |
| `static_rcp_nr_ntt` | NR/NTT, precision 2,048 | 0.0% | 0.1% | 99.9% | 0.0% |
| `dyn_div_bz_fft_region` | BZ/FFT region, d/q 1,024/8,192 | 2.1% | 31.7% | 68.3% | 0.0% |
| `dyn_div_rem_bz_fft_region` | BZ/FFT region, d/q 1,024/4,096 | 2.1% | 31.2% | 68.7% | 0.1% |
| `static_div_bz_ntt_region` | BZ/NTT region, d/q 2,048/8,192 | 0.4% | 4.7% | 95.3% | 0.0% |
| `static_div_rem_bz_ntt_region` | BZ/NTT region, d/q 2,048/4,096 | 0.6% | 17.1% | 82.9% | 0.0% |
| `dyn_rcp_knuth_boundary` | Knuth, precision 8 | 14.6% | 98.6% | 0.0% | 1.4% |
| `dyn_rcp_nr_boundary` | NR/school, precision 9 | 16.8% | 33.4% | 65.2% | 1.5% |
| `static_rcp_knuth_boundary` | Knuth, precision 100 | 3.0% | 99.9% | 0.0% | 0.1% |
| `static_rcp_nr_boundary` | NR/school, precision 101 | 3.6% | 4.4% | 94.5% | 1.2% |

## Hotspots within the division stack

### Knuth and BZ base cases

On the 88×88 quotient profiles:

- dynamic `div`: `knuth_est` is 86.0% of total CPU time and the compiler's
  128-bit division support is another 12.2%;
- static `div`: `knuth_est` is 86.7% and 128-bit support is another 11.4%;
- dynamic/static `div_rem` show the same 88%/10–12% split.

`division_preflight`, the wrappers, normalization, and copying are collectively
well below 1%.

The Karatsuba-region BZ profiles spend 50–51% directly in `knuth_est` and
10–12% in its 128-bit support. [`div_3_2`](../src/utils/div.rs#L516) itself is
only about 1% of total time. In other words, the observed 65% division share is
mostly BZ's Knuth leaves, not recursive BZ bookkeeping.

At the larger dynamic BZ/FFT shape, `knuth_est` plus 128-bit support still
accounts for about 29% of the complete call, while multiplication accounts for
68%. Static BZ/NTT shifts further toward transform multiplication.

### NR quotient and reciprocal refinement

The relevant stages are
[`nr_err_band`](../src/utils/div.rs#L975),
[`nr_refine_rcp`](../src/utils/div.rs#L1075),
[`nr_rcp_chain`](../src/utils/div.rs#L1111), and
[`nr_refine_quo`](../src/utils/div.rs#L1243).

At 89 limbs, their own instructions account for only a few percent; the
schoolbook middle and short products beneath them account for 92–96%. At FFT
and NTT sizes, direct division work falls below 1% and multiplication rises to
about 99%.

The precision-9 dynamic reciprocal is the expected exception. It is so small
that generic slice/error-band work remains visible:

- `nr_err_band`: 7.8%;
- `nr_refine_rcp`: 5.6%;
- `nr_rcp_chain`: 2.6%;
- Knuth seed and division-owned support: the balance of the 33.4% division
  share.

This is a narrow boundary effect, not a large-input scaling problem.

No quotient-only NR profile showed `nr_exact_correction` as a material phase
for these deterministic normalized operands. The bounded correction/fallback
logic is not a current steady-state hotspot.

### NR remainder finishing

For dynamic 89-limb `div_rem`, inclusive phase attribution is approximately:

- quotient refinement: 44%;
- remainder finishing: 40%;
- reciprocal construction: 15%.

At 2,048 limbs it is approximately 50%, 21%, and 29%, respectively. Almost all
of each phase is multiplication. The direct loop and corrections in
`nr_rem_finish` are below 1%.

The important detail is structural: `nr_rem_finish` allocates
`d.len() + q_low.len() - 1` product limbs and calls the full multiplication
dispatcher, then reads only the low `d.len() + 1` limbs. A low truncated product
would directly reduce a substantial `div_rem` phase without changing the NR
error/correction model.

### Transform backends

Dynamic FFT profiles resolve primarily to RustFFT transpose/butterfly work,
`fft_core`, and buffer copies. The result is not sensitive to `div.rs` wrapper
behavior.

True NTT cases resolve to NTT butterflies/CRT, buffer copies, and Rayon/OS
synchronization (`__psynch_cvwait`/`swtch_pri`). The exact percentage assigned
to a terminal wait frame is approximate because a per-thread CPU delta can be
sampled after a worker finishes compute and enters the wait. The broad
conclusion is still robust: the time belongs to the parallel multiplication
stage, and direct division arithmetic is below 0.3% in the dynamic cases and
below 0.1% in most static cases.

## Optimization triage

1. **Optimize transform-backed middle/short multiplication for NR.**

   This owns 92–99.9% of every nontrivial NR profile and also dominates
   transform-sized BZ. Focus on the exact operand bands used by
   [`mid_mul_dyn`/`mid_mul_static`](../src/utils/mul.rs#L4239) and
   [`short_mul_dyn`/`short_mul_static`](../src/utils/mul.rs#L3585):

   - reduce decomposition, transpose, and result-copy traffic;
   - investigate reusing transformed `x` within quotient refinement where it
     participates in more than one similarly sized product;
   - tune NTT parallel granularity for these repeated, nested-sized products;
   - keep FFT/NTT plans, twiddles, and adequately sized scratch buffers hot
     across the full NR chain.

   This is the largest opportunity, though most implementation work belongs in
   `mul.rs`, not `div.rs`.

2. **Replace AArch64's repeated generic 128÷64 quotient estimate in
   `knuth_est`.**

   Precompute a reciprocal/inverse of normalized `d1` once per `div_buf_of`
   call, use `umulh`-based quotient estimation, and retain the existing bounded
   correction checks. Keep the x86-64 hardware `div` path. This attacks 86–88%
   of Knuth and roughly 61–63% of Karatsuba-region BZ, making it the highest
   value optimization actually inside `div.rs`.

3. **Add a low truncated multiplication for `nr_rem_finish`.**

   Compute `(d * q_low) mod B^(d.len()+1)` directly rather than materializing
   the full product. A schoolbook low-product kernel can halve the relevant
   coefficient work; a transform implementation can use a truncated/cyclic
   convolution strategy. This is narrower than item 1 but directly improves
   both dynamic and static `div_rem`.

4. **Retune BZ base/top-block decisions after optimizing `knuth_est`.**

   Current BZ Karatsuba profiles are dominated by Knuth leaves. Improving the
   leaf changes the best `BZ_CUTOFF` and potentially
   `BZ_TOP_PADDED_COST_SCALE`; tuning them beforehand would optimize against a
   cost structure that no longer exists.

5. **Specialize only the tiny dynamic reciprocal boundary if it matters to a
   real workload.**

   At precision 8, `reciprocal_divisor_window` and small-buffer movement are
   visible. At precision 9, generic `nr_err_band`/slice machinery is visible.
   A tiny fixed-size path or less generic band walker could help, but these
   calls already execute millions of times per second and the benefit
   disappears quickly with precision.

6. **Do not prioritize dispatcher, preflight, wrapper, or general setup
   cleanup.**

   Those layers are below the sampling noise floor in ordinary division and
   remain below 1–2% even in the smallest reciprocal profiles. The existing
   scratch reuse is working: allocation does not appear as a material
   steady-state hotspot.

## Validation guidance

Use this sampling harness to locate changed stacks, but use Criterion or the
cutoff tuner for performance decisions. In particular:

- compare before/after profiles with at least 5 seconds per affected case;
- benchmark Knuth and BZ separately after changing the AArch64 estimator;
- validate any low-product implementation against full `div_rem` identities;
- retune division/BZ cutoffs only after the underlying kernel changes;
- rerun on x86-64 before generalizing AArch64 conclusions.

## Validation status

- `cargo fmt -- --check`: passed.
- `cargo build --profile prof --bench div_profile`: passed.
- shell and Python harness syntax checks: passed.
- `cargo test test_div --features _bench_internals`: 53 passed.
- `cargo test`: 308 passed and 2 failed in the parallel full-suite run.
  `test_ntt_sqr_entry_dyn_parallel_branch` passed immediately when rerun alone,
  matching its documented full-suite flake. The unrelated deterministic
  `test_karatsuba_static_chunking_scratch_fallback` assertion still fails alone
  at `find_karatsuba_scratch(150, 50) > 199`; it was not changed as part of this
  division profiling work.
