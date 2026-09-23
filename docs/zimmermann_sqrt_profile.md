# Zimmermann sqrt: profiling and optimization opportunities

Analysis including the optimized binomial seed and merged-carry division
assembly. `ZIMMERMAN_SQRT_CUTOFF` is 17 root limbs. The initial profile tables
below precede the concurrent refactor that moved output handling into
`zimmerman_sqrt_entry`; the follow-up results are identified separately.
The profiling work did not change the production Zimmermann algorithm or cutoff.

The strongest general opportunity is division selection for medium operands.
For large operands, division's multiplication and FFT work dominates. Saturated
inputs offer a separate large gain by replacing the known low-root square with
its closed form. Avoiding zero-shift denormalization is a small, simple general
change with a larger payoff on sparse inputs.

## Where time goes

Intel i7-8750H, Rust 1.94.0, optimized `prof` build, CPU 2. Root widths are
shown below; full radicands contain twice as many limbs. Timing and sampling
are separate runs. These are warmed calls with reusable input/output buffers.

| Root limbs | Dynamic root+remainder time | Division callbacks | Squaring callbacks |
| ---: | ---: | ---: | ---: |
| 17 | 1.92 µs | — | — |
| 32 | 3.79 µs | 50.8% | 12.3% |
| 64 | 8.68 µs | 61.2% | 17.7% |
| 128 | 27.53 µs | 59.2% | 30.6% |
| 256 | 95.08 µs | 73.8% | 22.1% |
| 512 | 260.39 µs | 79.4% | 18.2% |
| 1024 | 567.13 µs | 79.7% | 18.2% |
| 2048 | 1259.61 µs | 80.6% | 17.9% |

Callback shares come from a separate instrumented probe. They include all
Zimmermann recursion levels, but exclude the division operations inside the
binomial leaf. Remaining time includes that leaf, wrapper/control/memory work,
and instrumentation. Timing instrumentation matters more at small sizes.
The wall times and callback percentages use different probes and must not be
combined to claim precise absolute stage costs.

PC sampling independently confirms the transition:

- At 64 limbs, approximately 64% of samples are in Knuth quotient estimation
  and multiply/subtract; squaring kernels account for about 17%.
- At 256 limbs, shared multiplication kernels account for about 52%, with
  another 15% in squaring kernels and 9% in FFT-related work.
- At 1,024 limbs, FFT-related work accounts for about 52%; at 2,048 it is 67%.
  These categories include conversion/reconstruction around the transforms.
- Warm scratch-pool/allocation code is under 0.2% in the sampled random cases.
  Direct sqrt control is below 1% above 64 limbs. They are low priorities here.

Padded inputs were also measured at every size; they have the same broad
bottleneck. Their extra copying did not become the dominant cost.

After the entry refactor, four fresh full-random profiles reproduced the same
bottlenecks: 64-limb quotient estimation plus multiply/subtract was 64.4%;
256-limb shared multiplication was 51.4%; FFT-related samples were 51.4% at
1,024 and 66.5% at 2,048. Unsampled production medians were respectively
8.74, 92.40, 566.26, and 1267.59 µs. The source snapshot and hashes are retained
as `sqrt_refactored.rs` and `metadata_refactored.json` in the artifact directory.

## 1. Select division algorithms for the actual sqrt operand shapes

Each recursive step divides by a root prefix. At even root widths, the top
split uses `lo=(n-1)/2`, so `d` has approximately `n/2+1` limbs and the quotient
body approximately `n/2-1`. Generic division currently switches these shapes
from Knuth to Newton–Raphson once the effective divisor exceeds 88 limbs.

The probe overrides only divisions with divisors wider than that existing
cutoff; smaller division calls retain their production dispatch. This keeps
the sqrt core unchanged and isolates the choice of division backend.

Controlled probe results for exact root+remainder (64 deterministic operands,
five rotating rounds; absolute times differ from the 256-operand profile):

| Root limbs | Current dispatch | Force Knuth above 88 | Force BZ above 88 |
| ---: | ---: | ---: | ---: |
| 256 | 105.21 µs | 78.45 µs | 87.04 µs |
| 512 | 288.89 µs | 238.69 µs | 285.80 µs |
| 1024 | 652.09 µs | 810.82 µs | 781.52 µs |
| 2048 | 1322.06 µs | 2749.93 µs | 1779.24 µs |

This initial sweep favors Knuth for medium remainder divisions. Forcing it
on larger operands is harmful; Newton wins at 1,024 and 2,048. BZ did not beat
the best alternative in this sweep. A subsequent hybrid probe retained Knuth
only for remainder divisions with divisor widths 89–257, preserving production
dispatch otherwise. Before the entry refactor, its measured reductions were
17.4% at 256 root limbs and only 1.4% at 512 for root+remainder, versus 25.4%
and 17.4% in the initial forced-backend sweep. The discrepancy makes the
512-limb result an unsuitable basis for choosing a final cutoff. The 257-limb
divisor limit is exploratory; input shape, output mode, code generation, and
run variability still need to be covered by cutoff tuning.

The requested output matters. At 256 limbs, forcing Knuth for the final
quotient-only division made approximate sqrt about 20% slower. Exact-root-only
still requests an exact division remainder to decide whether correction is
needed, so it follows a different cost curve from approximate-root mode.

### Follow-up against the refactored entry

The rebuilt probe uses `zimmerman_sqrt_entry` directly. The hybrid still
changes only remainder divisions with 89–257 divisor limbs, including those
inside child roots; larger divisions and final quotient-only divisions retain
production dispatch. Five rotating rounds on the same 64 inputs gave:

| Root limbs | Root+rem ordinary → hybrid | Root+rem reduction | Exact-root reduction | Approx-root reduction |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 100.24 → 79.06 µs | 21.1% | 25.3% | 1.1% |
| 512 | 260.01 → 238.61 µs | 8.2% | 12.4% | 8.7% |
| 1024 | 598.09 → 563.60 µs | 5.8% | 7.2% | 7.4% |
| 2048 | 1292.03 → 1249.47 µs | 3.3% | 5.2% | 4.2% |

These compare two instantiations of the same benchmark wrapper. A separate
public-production control was within about 3.5% of the ordinary wrapper.
Approximate-root timing ranges overlap at 256: treat its 1.1% as unchanged.
The strongest repeated result is the medium-size exact-output improvement;
the magnitude of the smaller gains varied between sweeps. This remains a
dynamic-path experiment, not a calibrated production cutoff for either
allocation model. Raw medians and ranges are in `stages_refactored.csv`.

## 2. Exploit the known saturated root digit

In `zimmermann_sqrt_core` and the outer entry, the saturated branch sets all
low root limbs to `u64::MAX`. The correction step subsequently calls a general
square for root+remainder output. That square is already known:

`(B^k - 1)^2 = B^(2k) - 2*B^k + 1`, with `B = 2^64`.

A benchmark callback constructs these limbs directly, retaining the existing
remainder subtraction/correction. Production can use the already available
`saturated` flag; the probe scans the callback input to avoid changing the
included core. All-ones inputs saturate at every recursive level.

| Root limbs | Ordinary squaring | Direct known square | Time reduction |
| ---: | ---: | ---: | ---: |
| 64 | 3.20 µs | 1.68 µs | 47.5% |
| 256 | 26.35 µs | 3.83 µs | 85.5% |
| 1024 | 119.06 µs | 12.45 µs | 89.5% |

These are whole root+remainder times from the refactored-entry follow-up,
not square-only timings. This is a measured special-case gain, not a
prediction for random inputs.
A further possibility is fusing subtraction of this known square directly
into the remainder instead of materializing it. A zero low-root block can
similarly bypass square/subtraction work; that extra shortcut is unbenchmarked.

## 3. Skip denormalization when the shift is zero

Both `sqrt_dyn_output` and `sqrt_static_output` unconditionally call
`sqrt_denormalization` for root+remainder. When `sh == 0`, `d == 0`, but
`add_mul` still walks the whole root before subtraction of a zero square.
An outer `if sh != 0` avoids this no-op work. The binomial wrapper already
uses that guard.

The measured change is small/noisy for ordinary random operands; sampled
cost is roughly 1–2% near the lower end and below 1% from 128 limbs upward.
Sparse inputs expose it more clearly. In the power-of-two probe:

| Root limbs | Current wrapper | Skip zero-shift denormalization | Time reduction |
| ---: | ---: | ---: | ---: |
| 64 | 2.64 µs | 2.51 µs | 4.6% |
| 256 | 4.70 µs | 4.21 µs | 10.3% |
| 1024 | 11.97 µs | 10.30 µs | 14.0% |

These timings were rerun against the refactored entry.

## 4. Reduce unused Newton remainder-product work

`nr_rem_finish` needs only the lowest `d.len()+1` product limbs, but allocates
`d_len + q_low_len - 1` limbs and calls a full multiplication before using
`prod[..w]`. For the nearly balanced division shapes produced by sqrt, almost
half the product limbs are discarded. A dedicated low-product implementation
could reduce this work while preserving the existing bounded quotient
correction. Simply passing a shorter output to `mul_dyn` would violate its
contract; this needs a real truncated-product path. This is source analysis,
not a measured optimization yet.

Exact-root-only mode offers a more ambitious related experiment: expose a
conservative remainder bound from NR's existing guard digits. Away from square
boundaries, that may certify the sqrt correction decision without constructing
the full exact division remainder. Ambiguous cases would keep the current
exact path. This requires a proof linking the bound to the sqrt correction;
requesting quotient-only division unconditionally is insufficient.

## 5. Optimize the shared multiplication/FFT support where samples land

At 1,024 dynamic root limbs, approximately 12% of all samples include
`fft_chunk_value`, another 2% are other accumulation code, and about 4% are
`scale_and_round`. At 2,048 these are roughly 15%, 2%, and 3.5%. Useful
experiments are explicit fixed-size four-coefficient reconstruction, fewer
variable shifts/carry operations, and fusing the round/store pass with
reconstruction. Correct coefficient and carry bounds must be preserved.

`mul_elem` and its schoolbook/Karatsuba callers are also substantial: about
39% of samples at 256 dynamic limbs and 24% at 1,024. Samples include the
inclusive-range iterator and indexing, as well as `mul_asm_x86`. Simplifying
that inner loop is worth benchmarking after division selection is addressed.
The samples do not establish a branch-misprediction count.

## Output modes and static allocation

Production timings for full random inputs:

| Root limbs | Dynamic root+rem | Dynamic exact root | Dynamic approximate root | Static root+rem | Static exact root | Static approximate root |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 8.68 µs | 7.47 µs | 7.56 µs | 9.87 µs | 7.92 µs | 7.74 µs |
| 256 | 95.08 µs | 77.05 µs | 53.24 µs | 101.71 µs | 79.75 µs | 53.55 µs |
| 1024 | 567.13 µs | 506.00 µs | 433.53 µs | 1278.07 µs | 1129.62 µs | 913.25 µs |

The current code already omits the final square for approximate roots and
usually for exact-root-only random inputs. Those are existing savings, not
new proposals. All recursive child roots still require exact remainders.

Static paths retain stack-backed scratch and avoid FFT. At 1,024 limbs they
spend roughly 52% of samples in shared multiplication kernels and 20% in NTT
kernels, rather than taking the dynamic FFT path. Large static buffers and NTT
scheduling merit their own measurements, but allocation was not the dominant
sampled bottleneck. Separate 1,024-limb timing controls with one Rayon worker
and with all 12 logical CPUs available still showed a substantial static vs
dynamic gap. The pinned results should not be treated as universal multicore
performance claims.

## Method, validation, and reproduction

The profiler records interrupted PCs with ITIMER_PROF at 1 ms and resolves
inline frames using `addr2line`. It does not collect hardware cycles, cache
misses, or branch-miss counters, and it cannot attribute every out-of-line
shared multiplication to its caller. Callback timers supplement that view.
Rayon runtime can appear in static NTT cases; all primary timing/profile
processes are pinned to one logical CPU. Warmup, complete-batch overshoot,
and sample overhead affect absolute sampling-run duration. Unsampled timings
are reported separately.

The stage probe includes the actual `src/utils/sqrt.rs` source, calls its
private core with benchmark-only callbacks, and leaves production code intact.
It validates each variant against production on full, near-full and minimally
padded shapes, normalized and shifted random values, all-ones and powers of
two, all three output modes, at 17–2,048 root limbs. The seven experimental
variants, including the hybrid dispatch and closed-form square, passed 8,064
differential root/remainder checks both before and after the entry refactor.
Approximate roots are checked against
their floor-or-floor-plus-one contract. These checks establish agreement with
production, rather than providing an independent mathematical oracle.

Artifacts:

- `benches/binom_sqrt_profile.rs`: additional Zimmermann modes, with a check
  that the requested root width reaches the production cutoff. Static modes
  use `N=next_power_of_two(2*root_len)`.
- `scripts/python/profile_zimmermann_sqrt.py`: PC sampling and unsampled times.
- `benches/probes/zimmermann_stages.rs`: callback timers and experimental variants.
- `target/zimmermann_sqrt_profiles/{main,modes,patterns}/`: per-case JSON, PCs,
  logs, and summaries. `stages*.csv` hold controlled-probe results;
  `thread_controls.json` holds threading controls. `refactored/` contains the
  four follow-up PC profiles; `stages_refactored*.csv` holds the rebuilt probe's
  results and `metadata_refactored.json` identifies its source and profile binary.

Example commands (substitute the actual executable and rlib hashes from
`target/prof/deps`):

```bash
cargo build --profile prof --bench binom_sqrt_profile
python3 -B scripts/python/profile_zimmermann_sqrt.py \
  --binary target/prof/deps/binom_sqrt_profile-BINARY_HASH \
  --output target/zimmermann_sqrt_profiles/main
rustc --edition=2021 -O -g -C lto=thin -C codegen-units=1 \
  benches/probes/zimmermann_stages.rs \
  --extern big_bits=target/prof/deps/libbig_bits-LIB_HASH.rlib \
  -L dependency=target/prof/deps \
  -o target/zimmermann_sqrt_profiles/stages_probe
taskset -c 2 target/zimmermann_sqrt_profiles/stages_probe \
  --sizes 256,512,1024,2048 --variants production,mirror,hybrid
```

The initial production sweep contains 40 cases and 86,773 PC samples. The
four refactored-entry profiles add 9,081 samples, totaling 95,854 across 44
profile cases. The sampler reported no lost samples in these runs.
