# Separate division dispatch and recursive BZ cutoffs

This implements the strategy change authorized after the
[first tuning pass](division_cutoff_retuning.md). All constants are universal;
there are no platform gates. The optimized subtraction kernels are unchanged.

## Selection behavior

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `DIV_KNUTH_CUTOFF` | 88 | Mandatory top-level Knuth region, by effective divisor width |
| `BZ_CUTOFF` | 216 | Largest divisor width handled by a Knuth leaf inside BZ |
| `BZ_TOP_PADDED_COST_SCALE` | 1.60 | Padded BZ top-block cost coefficient |

The existing dynamic/static, quotient/remainder NR-versus-BZ cost-model
coefficients remain unchanged from the first tuning pass. Standard and high
quotient operations share the new top-level Knuth threshold, as they shared
the old threshold. Effective widths are measured after the existing operand
preparation and exact low-zero-limb factoring.

For a logical quotient width `q` and effective divisor width `d`:

1. `q == 1` or `d <= 88`: use Knuth.
2. Otherwise, evaluate the existing NR/BZ cost model.
3. If it selects NR and the shape supports NR, use NR even when `d <= 216`.
4. Otherwise, use Knuth directly for `d <= 216`, and BZ for `d > 216`.

Thus an NR choice at `(d=192,q=96)` remains NR. A BZ choice at
`(d=96,q=160)` goes directly to Knuth. BZ starts recursive splitting only above
216 limbs, with Knuth leaves at or below that width. An unsupported NR shape
uses the same Knuth/BZ fallback boundary.

NR's internal correctness fallbacks still use their existing BZ path. This
change concerns the initial performance dispatcher and BZ's recursive base
case, not the error-bound or exact-correction machinery.

The prior one-split benchmark placed the recursive crossover near 216 limbs.
Separating the decisions allows that larger leaf size without forcing every
89–216-limb division through Knuth. Keeping the NR eligibility threshold at
88 isolates this change; it does not claim that 88 is the optimal independent
NR threshold for every operation and operand shape.

## Validation

The public dispatcher boundary test now covers:

- Both sides of the 88-limb mandatory Knuth threshold.
- An NR choice and a BZ-to-Knuth choice below the recursive cutoff.
- Long-quotient BZ choices at widths 216 and 217.
- Unsupported NR quotient widths on both sides of the recursive cutoff.

All cases compare dynamic and static quotient/remainder results against
Knuth. `cargo test` reports **332 passed, 2 failed**, including all **51 division
tests passing**. The two unchanged, unrelated multiplication failures are
`test_empty_input_public_paths` and
`test_karatsuba_static_chunking_scratch_fallback`.

## Benchmark method

Measurements on 2026-09-21 use the existing `cargo bench-cutoffs` Criterion
infrastructure, Intel Core i7-8750H, CPU affinity 2, rustc 1.94.0, and the
repository bench profile. Builds and benchmark processes run sequentially.

The saved previous executable uses the optimized subtraction kernel and the
first-pass constants: shared cutoff 88 and top-block scale 1.75. Each runtime
family runs that executable immediately before the new executable. This
comparison isolates the differentiated cutoff, direct-Knuth bypass, and its
associated top-block coefficient from the already-completed first tuning pass.

The runtime corpus has 48 shapes per operation: divisor widths
`88,89,96,192,216,217,224,384,787,1573,3109,4096`, each with logical quotient/divisor
ratios `0.5,1,3,8`. Each shape has four deterministic operands, 30 samples,
300 ms warmup, and a 1 s measurement target. Operand setup is outside the timed
region. The result is an equal-weight geometric mean of per-shape ratios,
not a workload-frequency-weighted estimate.

The new `bz_knuth` mode directly compares the forced BZ wrapper with direct
Knuth on identical operands. Like the existing ratio probes, it encodes a
ratio of 1 as 1 ms; those reported durations are not operation latencies.
BZ top-block search windows now skip divisor widths below the recursive
cutoff, so the probe remains valid when the cutoff exceeds an early window.

## End-to-end results

The [96-shape runtime data](division_dispatch_split_runtime.csv) preserve mean
latencies, 95% confidence intervals, and the selected backend before and after.
Geometric mean of **new / previous retuned dispatcher** runtime:

| Corpus | Shapes per operation | Quotient-only | Remainder |
| --- | ---: | ---: | ---: |
| Boundary widths (88–224) | 28 | 0.9807 | 0.9503 |
| Larger widths (384–4096) | 20 | 1.0017 | 0.9864 |
| All | 48 | 0.9894 | 0.9652 |

Measured geometric runtime falls **1.1%** for quotient-only division and
**3.5%** for remainder division relative to the first-pass shared dispatcher.
These figures should not be added to the first report's improvements: the
corpus and baseline differ. Large quotient-only cases are essentially flat;
the gains are concentrated around the differentiated cutoffs.

The largest measured regressions are 4.5% for quotient-only `(d=384,q=384)`
and 4.1% for remainder `(d=224,q=224)`. Both retain NR selection. Small changes
on unchanged paths can include code placement, internal correctness-fallback
costs, and CPU frequency variation; sampling confidence intervals do not
capture every source of machine drift. The result is a modest aggregate
improvement, not a claim that every shape gets faster.

Recompute the table with:

```bash
python3 scripts/python/analyze_div_cutoffs.py docs/division_dispatch_split_runtime.csv --runtime
```

The fitting helper also accepts `--knuth-cutoff` for future independent
threshold sweeps; without that option it reproduces the historical shared-cutoff
fits. A split-cutoff fit requires NR/Knuth measurements wherever either
threshold can select Knuth.

## Direct Knuth and top-block checks

The [paired probe data](division_dispatch_split_probes.csv) confirm that calling
Knuth directly avoids the small BZ-wrapper overhead below the recursive cutoff:

| Operation | Shapes | Geometric BZ / Knuth | Range |
| --- | ---: | ---: | ---: |
| Quotient-only | 20 | 1.0082 | 1.0006–1.0225 |
| Remainder | 20 | 1.0072 | 0.9982–1.0206 |

These probes use divisor widths 96, 128, 192, and 216 and quotient/divisor
ratios 0.25, 0.5, 1, 4, and 8. The benefit of the direct bypass alone is modest;
the larger change is replacing unnecessary recursive splitting with Knuth.

Padded BZ / direct Knuth near the top-block model boundaries:

| Scale | Ratio | 95% interval |
| ---: | ---: | ---: |
| 1.55 | 0.9189 | 0.9159–0.9220 |
| 1.60 | 0.8789 | 0.8770–0.8809 |
| 1.65 | 0.8557 | 0.8537–0.8576 |

Scale 1.60, taken from the earlier 216-limb leaf experiment, remains a
conservative choice: padded BZ is clearly faster at its sampled boundary.
The corrected window filtering excludes crossings of the recursive cutoff
itself, so these boundary averages differ from the earlier exploratory probe.

## Reproduction

Build unpinned before running measurements on a single CPU:

```bash
cargo bench-cutoffs --no-run
DIV_TUNE_REGIME=bz_knuth DIV_TUNE_FAMILY=dyn_div \
  DIV_TUNE_SCALES=96,128,192,216 DIV_TUNE_VALUES=0.25,0.5,1,4,8 \
  DIV_TUNE_SAMPLE_SIZE=15 DIV_TUNE_WARMUP_MS=100 DIV_TUNE_MEASUREMENT_MS=400 \
  taskset -c 2 cargo bench-cutoffs -- --noplot
DIV_TUNE_REGIME=bz_top DIV_TUNE_VALUES=1.55,1.60,1.65 \
  taskset -c 2 cargo bench-cutoffs -- --noplot
DIV_TUNE_REGIME=runtime DIV_TUNE_FAMILY=dyn_div \
  DIV_TUNE_SCALES=88,89,96,192,216,217,224,384,787,1573,3109,4096 \
  DIV_TUNE_VALUES=0.5,1,3,8 DIV_TUNE_SAMPLE_SIZE=30 \
  DIV_TUNE_WARMUP_MS=300 DIV_TUNE_MEASUREMENT_MS=1000 \
  taskset -c 2 cargo bench-cutoffs -- --noplot --save-baseline shared_dispatch_div
```

Save the runtime baseline with the previous shared dispatcher, then repeat
with the new dispatcher and `--baseline shared_dispatch_div`. Repeat each
family-specific command with `dyn_rem`, using its own runtime baseline name.
Raw estimates and logs are in `target/division_retune/split_*.json` and `.log`.
