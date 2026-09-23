# Dynamic division cutoff retuning

This report records the first tuning pass with the shared cutoff. The subsequent
[separate-cutoff implementation](division_dispatch_split.md) supersedes its
selected constants and implements the strategy proposal below.

Measured on 2026-09-20 after the working-tree `sub_mul` optimization.
Hardware: Intel Core i7-8750H (x86-64). Toolchain: rustc 1.94.0,
LLVM 21.1.8; Cargo bench profile with optimization level 3, thin LTO,
and one codegen unit. Measurements use CPU affinity 2, the `powersave`
governor, and turbo enabled. Compiler work and benchmark processes run
sequentially.

The production selection strategy is unchanged. All cutoff constants remain
universal, without platform gates, as requested. Static-specific NR/BZ and
reciprocal constants are unchanged; the shared BZ constants also affect static
division. Both sides of each runtime comparison contain the optimized kernel:
the comparison measures the effect of retuning, not the kernel speedup itself.

## Method and scope

The existing `cargo bench-cutoffs` Criterion harness measures paired algorithms
on identical deterministic inputs with alternating pair order. Ratio benchmarks
encode a ratio of 1 as 1 ms; these values are **not operation latencies**.
The added `runtime` mode measures actual production dispatcher latency with
operand setup outside the timed region. Both operations use four deterministic
inputs per shape.

Added benchmark modes:

- `bz_base`: exactly one BZ split with Knuth children; independently checks
  quotient and remainder against Knuth on 16 normalized inputs before timing.
- `bz_top`: exposes the existing padded-top-block probe.
- `grid`: forced NR/BZ pairs at fixed quotient/divisor widths.
- `knuth`: forced NR/Knuth pairs at the same widths.
- `runtime`: production division, with Criterion saved-baseline support.

The shared-cutoff screening grid uses divisor widths
`48,64,80,96,128,160,224,384,787,1573,3109` and quotient/divisor width ratios
`0.25,0.5,1,2,4,8`: 66 shapes per operation and cutoff. Screening builds
use top-block scale 1.70 and dynamic reciprocal cutoff 11. Its 10 samples,
100 ms warmup, and 300 ms measurement target are for screening. Boundary
confirmation uses 30 samples, 750 ms warmup, and a 2 s target; the existing
BZ top-block probe uses 50 samples, 2 s warmup, and a 5 s target.

The versioned [grid data](division_dispatch_grid.csv) include mean ratios and
95% confidence intervals. [The fitter](../scripts/python/analyze_div_cutoffs.py)
searches the existing formulas, weighting each shape and operation equally.
It minimizes geometric cost relative to NR at each shape. Comparing shared
cutoffs this way assumes NR costs are stable across builds, so production
runtime validation is necessary. This is a measured compromise for this
corpus, not proof of a unique optimum for every operand distribution.

The supporting [boundary and Knuth probe data](division_cutoff_probes.csv)
also preserve mean ratios and 95% confidence intervals.

Raw Criterion estimates and logs are in `target/division_retune/` and
`target/criterion/` (unversioned build artifacts).

## Shared Knuth/BZ cutoff

`BZ_CUTOFF` controls both recursive BZ leaves and the dispatcher's mandatory
Knuth region. One split / Knuth measurements favor a leaf cutoff near 216:

| Divisor limbs | Ratio | 95% interval |
| ---: | ---: | ---: |
| 200 | 1.0172 | 1.0148–1.0200 |
| 208 | 1.0016 | 0.9998–1.0036 |
| 216 | 0.9996 | 0.9977–1.0021 |
| 220 | 0.9914 | 0.9896–0.9934 |
| 224 | 0.9855 | 0.9833–0.9876 |
| 232 | 0.9728 | 0.9714–0.9743 |

However, using 216 in the unchanged dispatcher forces Knuth where quotient-only
NR already wins. The rejected 216 candidate took 2.096× the original runtime
at `(d=96,q=48)` and 2.382× at `(192,96)`. Its quotient-only geometric runtime
was 1.024× on the regular corpus and 1.042× on the irregular corpus. The
leaf-only optimum is therefore unsuitable as the shared dispatcher cutoff.

Jointly refitting the existing cost formulas at smaller shared cutoffs gives:

| Shared cutoff | Quotient cost / NR | Remainder cost / NR | Joint cost / NR |
| ---: | ---: | ---: | ---: |
| 48 | 0.921156 | 0.776693 | 0.845846 |
| 64 | 0.928146 | 0.771338 | 0.846117 |
| 88 | 0.941425 | 0.759847 | 0.845777 |

The joint difference is below 0.05%, too small to distinguish reliably.
Retain **88**: reducing it helps quotient-only division but hurts remainder
division. Each family has its own fitted coefficients, as before.

## Reciprocal cutoff

NR / Knuth measurements around the dynamic reciprocal boundary:

| Precision (limbs) | Ratio | 95% interval |
| ---: | ---: | ---: |
| 9 | 1.2790 | 1.2528–1.3127 |
| 10 | 1.1257 | 1.1169–1.1361 |
| 11 | 0.9956 | 0.9922–0.9989 |
| 12 | 0.9817 | 0.9750–0.9903 |
| 13 | 0.9475 | 0.9292–0.9666 |
| 14 | 0.9186 | 0.9003–0.9401 |
| 15 | 0.7995 | 0.7958–0.8030 |
| 16 | 0.9096 | 0.9051–0.9153 |

Use Knuth through precision **11**, NR from 12. The advantage for NR at 11 is
below 1%, so retain Knuth there conservatively.

## Selected constants and boundary confirmation

| Constant | Before | After |
| --- | ---: | ---: |
| `BZ_CUTOFF` | 88 | 88 (retained) |
| `BZ_TOP_PADDED_COST_SCALE` | 0.295 | 1.75 |
| `DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF` | 1568 | 352 |
| `DYN_DIV_KARATSUBA_NR_BZ_CUTOFF` | 0.31 | 0.60 |
| `DYN_DIV_FFT_NR_BZ_CUTOFF` | 7.84 | 5.50 |
| `DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF` | 1280 | 192 |
| `DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF` | 0.58 | 24.0 |
| `DYN_DIV_REM_FFT_NR_BZ_CUTOFF` | 8.70 | 6.17 |
| `DYN_RCP_KNUTH_NR_CUTOFF` | 8 | 11 |

The unchanged dispatcher uses Knuth for `q == 1` or `d <= BZ_CUTOFF`.
Above that cutoff, it selects BZ when `d/q < linear_coefficient` below the
transition, or `log2(d)^2/log2(q) < transform_coefficient` above it. Otherwise
it selects NR, subject to the existing unsupported-shape fallback to BZ.
Here `q` is the logical quotient width including its possible high limb.

The grid places the quotient-only transition in the 320–384 gap; 352 is its
midpoint. Boundary probes refine the grid's linear coefficient from 0.75 to
0.60 and the transform coefficient from 5.542 to 5.50:

| Quotient-only boundary probe | Coefficient | NR / BZ |
| --- | ---: | ---: |
| Linear | 0.60 | 1.0113 |
| Linear | 0.75 | 0.8590 |
| Linear | 0.90 | 0.7411 |
| Transform | 5.40 | 1.0656 |
| Transform | 5.54 | 0.9614 |
| Transform | 5.70 | 0.9040 |

Linear probes use divisor centers 96, 128, 160, 224; transform probes use
224, 256, 384. Coefficients within gaps between measured shapes are not
uniquely determined by the grid, which is why boundary confirmation matters.

For remainder division, the grid's best transition of 352 caused a 47%
regression at `(d=224,q=112)` in production runtime validation. A transition
of **192** avoids sending that case to BZ and uses the near-optimal lower
transition regime (grid fit: 176–208, transform coefficient 6.1701).

BZ wins all sampled remainder shapes below that transition: the main grid's
NR/BZ ratios are 1.31–1.58 at divisor widths 96–160. Extra short-quotient probes
at widths 96, 128, and 160, down to eight quotient limbs, give ratios
1.43–2.32. Set the linear coefficient to **24.0**: for `max(q,d) < 192` and
NR-supported `q >= 8`, `d/q < 24`, so this retains BZ throughout the measured
small-size region. Shorter unsupported NR shapes already fall back to BZ.
This uses the existing formula and constants without changing the strategy.

Padded BZ / direct Knuth confirmation with shared cutoff 88:

| Top-block scale | Ratio | 95% interval |
| ---: | ---: | ---: |
| 1.65 | 1.0293 | 1.0258–1.0330 |
| 1.70 | 1.0070 | 1.0045–1.0097 |
| 1.75 | 0.9684 | 0.9652–0.9720 |

Select **1.75**, where padded BZ has a clear advantage. This is a coefficient
of the existing cost model, not a literal quotient/divisor width ratio.

## Final production runtime comparison

The final comparison reruns the saved original-constant executable immediately
before the tuned executable for each operation/corpus. Both contain the same
optimized `sub_mul` kernel and deterministic runtime benchmark. Each shape
uses 30 samples, 300 ms warmup, and a 1 s measurement target. The
[96-shape runtime data](division_cutoff_runtime.csv) include absolute mean
latencies and 95% confidence intervals for both versions.

Geometric mean of **after / before** runtime, with equal weight per shape:

| Corpus | Shapes per operation | Quotient-only | Remainder |
| --- | ---: | ---: | ---: |
| Regular | 32 | 0.9398 | 0.8851 |
| Irregular | 16 | 1.0013 | 0.9921 |
| All | 48 | 0.9599 | 0.9194 |

Across all shapes, geometric runtime falls **4.0%** for
quotient-only division and **8.1%** for remainder division.
These aggregate gains do not mean every shape improves. The largest
regressions are:

| Operation | Divisor limbs | Quotient limbs | Selection before → after | Runtime ratio |
| --- | ---: | ---: | --- | ---: |
| dyn_rem | 787 | 6296 | BZ → NR | 1.454 |
| dyn_rem | 1573 | 12584 | BZ → NR | 1.444 |
| dyn_div | 787 | 6296 | BZ → NR | 1.426 |

The irregular-size regressions correspond to NR/BZ timing discontinuities
that the existing two-regime formulas do not capture. For example, forced
NR/BZ quotient-only timing is 0.562× at `(d=384,q=3072)` but 1.448× at
`(787,6296)`, despite both having an 8:1 quotient/divisor ratio. The selected
constants favor the geometric mean on the documented corpus and retain this
tradeoff. Applications concentrated on the regressing shapes may prefer the
original coefficients; these measurements are not a claim of uniform speedup.

Small differences on unchanged paths can include code placement and CPU
frequency variation; the confidence intervals describe sampling uncertainty,
not every source of machine drift. The extra fresh-baseline run reduces that
risk compared with reusing baselines from the earlier exploratory scans.

Recompute the aggregate table with:

```bash
python3 scripts/python/analyze_div_cutoffs.py docs/division_cutoff_runtime.csv --runtime
```

## Evidence for a future selection-strategy change

Consider separating the recursive BZ leaf cutoff from the dispatcher's Knuth
threshold, and giving quotient-only and remainder division separate thresholds.
This proposal is **not implemented**. It would allow the faster Knuth leaf
kernel to be used inside BZ without forcing expensive top-level Knuth division.

Direct NR/Knuth quotient-only measurements illustrate the conflict:
`(d=32,q=16)` is near parity (0.991×), `(64,32)` favors NR (0.597×), and
`(192,96)` strongly favors NR (0.428×). But `(64,256)` favors Knuth (2.101×),
so quotient width matters too. Remainder division has a different crossover:
`(192,96)` still favors Knuth (1.263× NR/Knuth).

The 216 experiment above supplies end-to-end evidence that coupling these
choices is costly. A separate-threshold strategy deserves its own benchmark
pass; the present change preserves the existing strategy.

## Correctness checks

`cargo fmt -- --check` passes. `cargo test` reports **332 passed, 2 failed**,
including **all 51 division tests passing**. The two unrelated multiplication
failures remain unchanged:

- `test_empty_input_public_paths`: empty-input subtraction underflow at
  `src/utils/mul.rs:3695`.
- `test_karatsuba_static_chunking_scratch_fallback`: scratch-size assertion at
  `src/tests/test_mul.rs:2907`.

Division tests now derive capacities from `BZ_CUTOFF`, cover both sides of
the mandatory Knuth boundary, and size the static top-block fallback case
so it actually cannot accommodate the padded BZ buffer. The existing
production Knuth body is public for the benchmark; forced-stage helpers
remain in `benches/`.

## Reproduction

Build unpinned, then pin only the benchmark process. Run measurements
sequentially. For example:

```bash
cargo bench-cutoffs --no-run
DIV_TUNE_REGIME=bz_base DIV_TUNE_VALUES=200,208,216,220,224,232 \
  taskset -c 2 cargo bench-cutoffs -- --noplot
DIV_TUNE_REGIME=bz_top DIV_TUNE_VALUES=1.65,1.70,1.75 \
  taskset -c 2 cargo bench-cutoffs -- --noplot
DIV_TUNE_REGIME=reciprocal DIV_TUNE_FAMILY=dyn_rcp \
  DIV_TUNE_VALUES=9,10,11,12,13,14,15,16 \
  taskset -c 2 cargo bench-cutoffs -- --noplot
DIV_TUNE_REGIME=grid DIV_TUNE_FAMILY=dyn_div \
  DIV_TUNE_SCALES=48,64,80,96,128,160,224,384,787,1573,3109 \
  DIV_TUNE_VALUES=0.25,0.5,1,2,4,8 DIV_TUNE_SAMPLE_SIZE=10 \
  DIV_TUNE_WARMUP_MS=100 DIV_TUNE_MEASUREMENT_MS=300 \
  taskset -c 2 cargo bench-cutoffs -- --noplot
python3 scripts/python/analyze_div_cutoffs.py docs/division_dispatch_grid.csv
```

Repeat the grid for `dyn_rem`, and rebuild with shared BZ cutoffs 48, 64, and
88, with top-block scale 1.70 and reciprocal cutoff 11, to reproduce the
screening comparison. Restore the final top-block scale 1.75 afterward.
`knuth` uses the same fixed-shape
interface. The CSV stores Knuth ratios measured at widths 48, 64, 80, and 96.

The regular runtime corpus uses divisor widths
`96,192,224,384,768,1536,3072,4096` and quotient/divisor ratios `0.5,1,3,8`.
The irregular corpus uses widths `233,787,1573,3109` and ratios `1,2,4,8`.
Run both families (`dyn_div`, `dyn_rem`) before and after the constant changes:

```bash
DIV_TUNE_REGIME=runtime DIV_TUNE_FAMILY=dyn_div \
  DIV_TUNE_SCALES=96,192,224,384,768,1536,3072,4096 \
  DIV_TUNE_VALUES=0.5,1,3,8 DIV_TUNE_SAMPLE_SIZE=30 \
  DIV_TUNE_WARMUP_MS=300 DIV_TUNE_MEASUREMENT_MS=1000 \
  taskset -c 2 cargo bench-cutoffs -- --noplot --save-baseline before_retune
# After changing constants, repeat with --baseline before_retune.
```

Use a separate baseline name, `before_retune_irregular`, for the irregular
corpus. Criterion increases measurement duration when needed for sample count.
