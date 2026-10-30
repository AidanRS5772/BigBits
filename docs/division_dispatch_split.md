# Separate division dispatch and recursive BZ cutoffs

This implements the strategy change authorized after the
[first tuning pass](division_cutoff_retuning.md). All constants are universal;
there are no platform gates. The optimized subtraction kernels are unchanged.

The values below, from 2026-09-21, are superseded by the
[retune after the Knuth rewrite](#retune-after-the-knuth-rewrite-2026-10-29) at
the end of this page; the selection rules themselves are unchanged.

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

## Retune after the Knuth rewrite (2026-10-29)

Knuth division got 1.6–3.9× faster:
- an asm 3/2 step;
- ADX multiply-subtract rows over a complemented window;
- a table-based 2/1 reciprocal;
- one-limb divisors through `div_prim`'s loop.

That moved every crossover that weighs Knuth against BZ or NR, including BZ's own leaves and the
reciprocal's Knuth/NR switch. Every constant below was remeasured on the new code, on the same
i7-8750H, pinned to CPU 2.

| Constant | Before | After |
|---|---:|---:|
| `DIV_KNUTH_CUTOFF` | 88 | 68 |
| `BZ_CUTOFF` | 216 | 140 |
| `BZ_TOP_PADDED_COST_SCALE` | 1.60 | 1.80 |
| `DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF` / `_KARATSUBA_NR_BZ_` / `_FFT_NR_BZ_` | 352 / 0.60 / 5.50 | 112 / 1.5 / 6.98 |
| `DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF` / `_KARATSUBA_NR_BZ_` / `_FFT_NR_BZ_` | 192 / 24.0 / 6.17 | 778 / 4.4 / 7.15 |
| `STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF` / `_KARATSUBA_NR_BZ_` / `_NTT_NR_BZ_` | 1664 / 0.65 / 9.485 | 842 / 0.75 / 11.66 |
| `STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF` / `_KARATSUBA_NR_BZ_` / `_NTT_NR_BZ_` | 3328 / 0.82 / 10.5 | 112 / 4.4 / 11.66 |
| `DYN_RCP_KNUTH_NR_CUTOFF` | 11 | 36 |
| `STATIC_RCP_KNUTH_NR_CUTOFF` | 100 | 36 |

### Static capacity

The static families were previously measured at the bench's fixed `N = 16384`. Static
drivers zero-fill their `[u64; N]` buffers on every call, about eight for NR and three for BZ,
so at that capacity small divisions mostly timed memsets. Static NR came out 3–10× slower than
BZ at every shape below about 400 limbs, which is roughly the ratio of the memsets. It is also
why the static reciprocal cutoff had been 100.

`DIV_TUNE_STATIC_SIZED=1` instead sizes `N` to the next power of two of the operand (at least 64),
as the [sqrt cutoffs](sqrt_cutoff_tuning.md#static-capacity) assume. Every static measurement
here uses it. The bench checks each static grid and runtime shape against `div_rem_dyn` before
timing it, and the runtime regime now also times `div_static`/`div_rem_static`. As with sqrt,
callers whose `N` far exceeds their operands would be better served by higher static cutoffs.

### BZ leaf and top block

The one-split probe (`bz_base`, `q = d`) drops below 1 from about 134 limbs. Up to about 176,
though, it is jagged by ±5% from one width to the next, because the half-width products land on
different multiplication paths. A first pass set `BZ_CUTOFF = 132`. The runtime check then showed
long quotients at `d = 133` losing 3–11% in every family. `bz_knuth`, which runs BZ against Knuth
for `q/d = 2, 4`, at every width from 141 to 200 gives:

| Widths | 133–140 | 141–158 | 159–174 | 175–200 |
|---|---|---|---|---|
| BZ / Knuth | 1.03–1.05, except 134 (0.98) | 0.95–1.03, mostly wins | 1.00–1.02 | 0.98–1.00 |

The summed log ratio is lowest with Knuth up to 140, at −0.371 against −0.281 for a cutoff of 174.
At `BZ_CUTOFF = 140`, padded BZ against Knuth on the top-block boundary is 1.098 at scale 1.60,
0.995 at 1.80 and 0.953 at 2.00, so the scale is 1.80.

### NR eligibility

NR against Knuth (`knuth` regime for the dynamic family, sized `grid` regime for the static one,
whose BZ side is Knuth below `BZ_CUTOFF`):

| d | 64 | 65 | 68 | 72 | 76 | 80 | 88 |
|---|---|---|---|---|---|---|---|
| dyn, q = d/4, d/2 | 1.06, 0.97 | 1.08, 0.95 | 0.99, 0.91 | 0.89, 0.91 | 0.89, 0.85 | 0.85, 0.84 | 0.75, 0.79 |
| static, q = d/4, d/2, d | 1.13, 1.02, 1.16 | 1.15, 1.00, 1.19 | 1.06, 0.96, 1.16 | 1.00, 0.94, 1.14 | 0.92, 0.89, 1.11 | 0.90, 0.87, 1.08 | 0.80, 0.86, 1.04 |

The static model sends `q = d` to NR (ratio 0.75). Each cutoff was scored over the shapes that
the models would hand to NR. That sum is positive at 65 and 68 and negative from 72, and every
cutoff from 68 to 71 ties on these widths. The smallest, 68, is used. A first pass at 64 made
static `d = 65` 14–19% slower.

### Cost models

The NR/BZ and NR/Knuth grid covered divisor widths 48–3109 against `q/d = 0.25–8`, 66 shapes per
family ([data](division_retune_grid.csv)). `analyze_div_cutoffs.py --knuth-cutoff 64` fit it
(`--family static_div`/`static_rem` for the static rows). The grid has no widths between 64 and
80, so the fit is the same at the final cutoff of 68. Likewise the grid ran at `BZ_CUTOFF = 132`,
and none of its widths lies between 132 and 140. Both remainder models return
the search's top Karatsuba ratio of 4.4. NR only wins those regions at quotients shorter than the
grid measured, so remainder division below the switch is effectively always BZ or Knuth.

### Reciprocal

NR / Knuth by precision (`reciprocal` regime, static at sized `N`):

| Precision | 30 | 32 | 34 | 36 | 38 | 40 | 44 | 48 |
|---|---|---|---|---|---|---|---|---|
| dyn | 1.020 | 1.117 | 1.019 | 1.039 | 0.981 | 0.968 | 0.921 | 0.888 |
| static | 1.064 | 1.150 | 1.070 | 1.031 | 0.968 | 0.972 | 0.936 | 1.000 |

Knuth handles precisions up to 36 in both models. At the old fixed `N = 16384`, static NR
didn't break even until about 200–256.

### Validation

The `runtime` regime ran the public entries (`div_dyn`, `div_rem_dyn`, `div_static`,
`div_rem_static`) on 96 shapes per family:
- divisor widths `48,64,65,80,88,96,112,132,133,160,216,217,384,787,1573,3109` against
  `q/d = 0.25,0.5,1,2,4,8`;
- static shapes up to 16384 limbs, which drops one static shape per family.

Each side was built with `-x86-branches-within-32B-boundaries` into its own target directory, and
the two binaries alternated per family. Pass 1 had `BZ_CUTOFF = 132` and `DIV_KNUTH_CUTOFF = 64`.
Pass 2 has the final constants and timed the after binary first. ([data](division_retune_runtime.csv))

| Family | Pass 1 | Pass 2 | Pass 2, d ≤ 217 | Pass 2, d > 217 |
|---|---:|---:|---:|---:|
| dyn_div | 0.953 | 0.960 | 0.962 | 0.956 |
| dyn_rem | 0.948 | 0.950 | 0.966 | 0.915 |
| static_div | 0.981 | 0.974 | 0.991 | 0.935 |
| static_rem | 0.887 | 0.879 | 0.883 | 0.873 |

These are geometric means of new / old runtime. The largest gains are long quotients at
d = 160–217 (0.54–0.59), which now take BZ instead of Knuth or NR. The rest:
- static remainders with q ≤ d at 96–217 (0.55–0.74), now Knuth instead of NR;
- large static divisions (0.43–0.62), now NR.

Shapes that regress in both passes, at 3–13% (two-pass means 5.5–10%), are compromises of the
models' feature. They change algorithm, and moving their boundary costs more at neighbouring grid
shapes:
- dyn (112, 112) and static (80, 80), (88, 88) now take Knuth and NR respectively;
- dyn remainder (384, 192) and static remainder (787, 394) now take BZ.

The four other shapes at 5–12% in pass 2 keep their algorithm and were 0.99–1.00 in pass 1.

### Reproduction

```bash
cargo bench-cutoffs --no-run
B=$(ls -t target/release/deps/cutoffs_bench-* | grep -v '\.d$' | head -1)
DIV_TUNE_REGIME=bz_base DIV_TUNE_VALUES=$(seq -s, 126 2 176) taskset -c 2 $B --bench --noplot division_tuning
DIV_TUNE_REGIME=bz_knuth DIV_TUNE_FAMILY=dyn_div DIV_TUNE_SCALES=$(seq -s, 141 1 200) \
  DIV_TUNE_VALUES=2,4 taskset -c 2 $B --bench --noplot division_tuning
DIV_TUNE_REGIME=bz_top DIV_TUNE_VALUES=1.6,1.8,2.0 taskset -c 2 $B --bench --noplot bz_top_block
# grid: per family dyn_div, dyn_rem (also with DIV_TUNE_REGIME=knuth), static_div, static_rem
DIV_TUNE_STATIC_SIZED=1 DIV_TUNE_REGIME=grid DIV_TUNE_FAMILY=static_div \
  DIV_TUNE_SCALES=48,64,80,96,128,160,224,384,787,1573,3109 DIV_TUNE_VALUES=0.25,0.5,1,2,4,8 \
  DIV_TUNE_SAMPLE_SIZE=10 DIV_TUNE_WARMUP_MS=100 DIV_TUNE_MEASUREMENT_MS=300 \
  taskset -c 2 $B --bench --noplot division_tuning
python3 scripts/python/analyze_div_cutoffs.py docs/division_retune_grid.csv --knuth-cutoff 64
python3 scripts/python/analyze_div_cutoffs.py docs/division_retune_grid.csv --knuth-cutoff 64 \
  --family static_div   # and static_rem
DIV_TUNE_STATIC_SIZED=1 DIV_TUNE_REGIME=reciprocal DIV_TUNE_FAMILY=static_rcp \
  DIV_TUNE_VALUES=$(seq -s, 30 2 52) taskset -c 2 $B --bench --noplot division_tuning
# runtime: the same command in a before and an after build, CRITERION_HOME kept apart
DIV_TUNE_STATIC_SIZED=1 DIV_TUNE_REGIME=runtime DIV_TUNE_FAMILY=static_rem \
  DIV_TUNE_SCALES=48,64,65,80,88,96,112,132,133,160,216,217,384,787,1573,3109 \
  DIV_TUNE_VALUES=0.25,0.5,1,2,4,8 DIV_TUNE_SAMPLE_SIZE=20 \
  DIV_TUNE_WARMUP_MS=200 DIV_TUNE_MEASUREMENT_MS=600 \
  taskset -c 2 $B --bench --noplot division_tuning
```

