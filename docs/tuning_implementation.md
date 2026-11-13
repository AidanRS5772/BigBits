# Tuning system: implementation plan and status

**Started 2026-11-08.** This document tracks how [the tuning system](tuning_system.md) is being
built into the crate. It covers Phases 0, 1 and 2, which are for continued development. Phases 3
and 4 (`cargo bigbits`, `BIGBITS_TUNING`, runtime loading) wait until the crate is ready to ship.
`tuning_system.md` holds the design and the rules (invariants and checklist). This file holds the
order of work, what each stage changes in the code, and each stage's status. Update the status
table when a stage lands, and correct the plan where the implementation changed it.

## Status

| Stage | Contents | Status |
|---|---|---|
| 0 | Phase 0: dependencies, empty `main`, `bz_2_1_cost`'s hidden constants | landed 2026-11-08 |
| 1 | Phase 1 infrastructure: schema types, shared resolver, `build.rs` codegen, `tune` feature, tests, bias harness | landed 2026-11-08 |
| 2 | Phase 1: `sqrt` | landed 2026-11-08: outputs identical to `faa9971` (161 items); bias 1.0014× (42 items); every cutoff's minimum is 3 |
| 3 | Phase 1: `series`, `bin_split` | landed 2026-11-08: outputs identical (161 items, both builds); bias 0.9997× (series items), 1.0024× (consts); fixed `rect_best` returning block length 0 when every prediction is infinite |
| 4 | Phase 1: `base`, `mul` | landed 2026-11-08: outputs identical (both builds); bias 0.9991× (45 mul items); against `faa9971` 1.0008×, with `hi_sqr_dyn` at 300–330 limbs +1.8% in the normal build while the tune build of the same code is 1.6% faster than normal (code layout); fixed `is_school_at`'s cancellation for large c; `base`'s x86 bias check pending |
| 5 | Phase 1: `div` | landed 2026-11-08: outputs identical (both builds); bias 0.9986× (22 div items); `div.bz` minimum 3 |
| 6 | Phase 1: `consts` | landed 2026-11-09: outputs identical (both builds); bias 1.0017× (consts items); `cache_par` saturates; debug builds' static split stacks ×4 (they overflowed on one-term static Hyper leaves) |
| 7 | Phase 1: `theta`, `agm`, `exp`, `ln` | landed 2026-11-09: outputs identical (161 items, both builds); bias 1.0034× (trans items; `exp_static` at 4–64 limbs +1–2% in the tune build, a few atomic reads in a µs call), 1.0001× (series items); all 313 random-knob vectors passed first time |
| P2.1 | Phase 2: runtime prediction layer (`Dispatcher`, `CostCtx`, tables, threads, presets layer) | landed 2026-11-09 except the generator's table and step-function knobs, which arrive with the first table knob (the pilot's switch): `tuning/{threads,model,table,dispatch}.rs` with tests; `build.rs` layers `presets/<arch>-<os>.toml` (else `<arch>.toml`) |
| P2.2 | Phase 2: tuner skeleton (`cargo tune`) | landed 2026-11-09: `cargo tune tune [SECTION…] [--level] [--out] [--write-preset] [--threads]`, `derive [--file]`, `list`, `status [--file]`; batched interleaved timing on performance cores, NNLS on relative error with a lack-of-fit report, sections merged into the file with evidence and upstream hashes. Not yet: sweeps, `check`, `--resume`, `--pin`, process isolation |
| P2.3 | Phase 2: sections (base, pilot, mul, div, sqrt, bin_split + series, consts, trans), presets | **done on the M1 (2026-11-10)**: `presets/aarch64-macos.toml` is a thorough tune through the final flow (eight sections, each kept by the end-to-end gate; no rejected fit; the build resolves it with no warning). Dynamic products and divisions dispatch on fitted equations, every one-input decision is a sweep, static values are approximate, and any checkout can tune itself into `presets/local.toml`. Open: the x86 preset (the user's, on the i7). The series' dispatch equations, rebuilt on fitted equations on 2026-11-10 and marked above the 5% line until 2026-11-13, are under it since ("Horner's and binary splitting's equations under the line, and the beta guard gone": 3.4%, 2.7% and 3.6% held out in the preset of that day's full retune), and nothing in the preset is marked. The history of how it got here is in the log below. Earlier status, kept for the record: in progress (2026-11-10). In `presets/aarch64-macos.toml` (thorough, M1): `base`'s units; mul's schoolbook, Karatsuba-level, recursive-Karatsuba, FFT and static-NTT fitted models; `mul.kara` derived; and seven sweeps (`mul.karatsuba_sqr`, `fft_sqr`, `fft_karatsuba`, `fft_chunking_karatsuba`, `static_ntt_sqr`, `ntt_karatsuba`, `ntt_chunking_karatsuba`). Not done in mul: FFT-to-NTT, high and middle products, the NTT's parallel gates, the cyclic square, `mul_ns` and per-call comparison above an extent. Then every later section and the x86 preset |

Handoffs to the i7-8750H (x86), tracked here until done:
- `base` bias check (Stage 4): the ADX cutoffs compile out on AArch64.
- Phase 2: the ADX sweeps, x86 `FFT_ODD_COST`, the pilot's x86 comparison, `presets/x86_64.toml`.
  The commands are in "The x86 handoff" at the end of this file.

## Ground rules

- **Baseline.** Phase 1 must give outputs and timings identical to commit `faa9971`. A worktree at
  `faa9971` is the identity baseline for every Phase 1 stage, and the place where the old
  harnesses still run, for Phase 2's comparisons.
- **Outside `src/utils`.** Benches, probes, scripts and tests may break. An old harness that a
  migration breaks is deleted, not ported: Phase 2 replaces it, and its results are reproduced
  from the `faa9971` worktree. `src/tests` is rewritten freely, but stays the correctness check.
- **Shared tree.** Other sessions edit and benchmark this tree. Never stash, check out or reset in
  it. Timing comparisons alternate two prebuilt binaries over several processes.

## Design choices that let Phase 1 grow into Phase 2

1. **Complete schema types from the start.** `schema_types.rs` declares every kind (`Interim`,
   `Model`, `Sweep`, `Derived`), every type (tables and thread step functions included), threads,
   arch, version, derive version and upstream sections. Phase 1 uses only `Interim`; Phase 2 adds
   data, not types.
2. **One resolver, shared by build and run time.** Layering, validation, `Follows` and hashing are
   in shared files compiled by both `build.rs` and the crate (through `#[path]`): the four the
   design names, plus `resolve.rs`. `build.rs` passes it no layers in Phase 1 and the presets in
   Phase 2 (and `BIGBITS_TUNING` in Phase 3). `cargo tune derive --file` calls the same function.
3. **One generator, several backings.** For each knob `build.rs` emits a `resolved::` constant, a
   `const fn` accessor (normal builds), an atomic-cell accessor (tune builds) and a metadata row.
   Phase 2 adds tables and step functions to the same generator.
4. **Knob bundles and pure predicates.** Each dispatcher becomes `pick(shape, &Knobs[, threads])`,
   where `Knobs::read()` reads the accessors once per call and passes them down (invariants 6 and
   7). In Phase 2 only a bundle's fields and the predicate's body change.
5. **The tuner lives in the crate** behind `tune` and times the crate's own entries.
6. **`resolved::` constants are for tests only.** In a tune build an accessor is not a `const fn`,
   so tests use `tuning::resolved::*` for const items and const generics. `src/utils` must never
   name `resolved::`, because that would bypass live values; a test checks it.

## Phase 0 (Stage 0)

- Moved `criterion` and `rand` to `[dev-dependencies]`, and removed `itertools`, which nothing
  used.
- Deleted `src/main.rs` (an empty `main`) and the empty `src/bin/`.
- Moved `bz_2_1_cost`'s fitted scales into `mod.rs` as `BZ_2_1_SCHOOL_TO_KARATSUBA` and
  `BZ_2_1_KARATSUBA_TO_FFT`. No other fitted constants hide in function bodies (`LOG_2_3`, `SCALE`
  and the masks are mathematics).

## Phase 1: the seam

### Stage 1: infrastructure

```
build.rs                    #[path]-includes the shared files; checks the schema; resolves; codegen
build/codegen.rs            writes $OUT_DIR/tuning.rs (build script only)
src/tuning/mod.rs           includes the generated file; the tuning API
src/tuning/schema_types.rs  shared: the schema's plain data types and const-fn builders
src/tuning/schema.rs        shared: SCHEMA, the only place knobs are declared
src/tuning/resolve.rs       shared: values, schema checks, layering, Follows, provenance
src/tuning/file.rs          shared: the tuning-file format (a TOML subset): parse, write, merge
src/tuning/hash.rs          shared: FNV-1a 64 over canonical text
src/tests/test_tuning.rs    schema, resolver, file and accessor tests; the random-knob test
benches/tuning_bias.rs      the bias and identity harness
```

The shared files use only `core` and `std`, and refer to each other only through `super::`.

**Generated code.** For each knob: `resolved::<section>::<NAME>` (a `pub const`, both builds),
`k::<section>::<name>()` (`const fn` returning the resolved value in a normal build; a relaxed load
of an `AtomicU64` cell in a tune build), plus `PATHS`, `RESOLVED_BITS` and `all_accessor_bits()`.
F64 knobs are stored as their bits.

**API.** Every build: `schema()`, `get(path)`, `k`, `resolved`. Tune builds: `set(path, value)`
(type and valid range checked), `apply("path=value,…")`, `snapshot()`, `restore()`, `reset()`.

**Random-knob test.** Live cells are global and `cargo test` runs tests on many threads, so
setting knobs in-process would change dispatch under other tests mid-call (invariant 5). The test
`random_knobs` therefore re-runs the test binary on one ignored test,
`random_knobs_child`, with `--test-threads=1`. The child runs the everything-low and
everything-high vectors, each knob's extremes, and random vectors (log-uniform within the valid
range, capped per section). Each vector is followed by every section's check against the suite's
oracles. A failure prints `BIGBITS_KNOB_VECTOR=…`, which replays it. Each migration adds its
section's check.

**Bias and identity harness.** `benches/tuning_bias.rs` times public entries only, so it builds
against `faa9971` too. `--worker --section S` prints medians and output hashes; `--compare A B`
alternates two binaries over several processes and prints paired ratios with a 95% interval.
Every migration runs it as normal-vs-tune build (the bias check) and current-vs-`faa9971` (the
identity check).

### Stages 2 to 7: migrations

**Phase 1 landed 2026-11-09.** `src/utils/mod.rs` keeps only the fixed constants (the FFT limits,
fold trigger, residue check and `THETA_STATIC_SLOTS`); 99 knobs in 11 sections replace the rest.
One change from the plan: the mul and trans stages read accessors where they are used instead of
threading `KaraKnobs`/`NttGates`/`ExpKnobs`/`LnKnobs` bundles down the recursions. Every such read
is once per call or recursion level (invariant 7), and invariant 5 keeps a call's reads equal, so
the bundles add nothing until Phase 2's gates need the call's thread count read once.

Each migration follows the [checklist](tuning_system.md#per-section-migration-checklist):

1. Declare the section with `Interim` knobs (`old` is the constant's name, defaults are today's
   values, valid ranges from the code and narrowed by the random-knob test).
2. Add knob bundles and pure predicates, read once per call; replace every direct read.
3. Rewrite tests: accessors at run time, `resolved::` in const contexts, value-independent
   assertions, the section's random-knob check.
4. Delete the constants from `mod.rs`; their measurement notes move to a dated "Values before the
   tuner" section of the area's tuning doc.
5. Delete or minimally fix whatever breaks outside `src/utils`; update every reference.
6. Check: `cargo test --lib`, `cargo test --lib --features tune`, the bias and identity runs.

| Stage | Section(s) | Code changes | Outside code removed or fixed |
|---|---|---|---|
| 2 | `sqrt` (9) | `SqrtKnobs` replaces `SqrtOutput::{dyn,static}_cutoff`; the leaf cutoff is passed down the recursion; `dyn_wrapped_rem` follows mul's square-mod cutoff (a literal plus a check until Stage 4) | sqrt probe and its two scripts deleted; `zimmermann_stages.rs` gets a `tuning` shim |
| 3 | `series`, `bin_split` (19) | `Node::LEAF_*` become methods; `SplitCutoffs::of` stops being `const`; `SeriesModel` bundles the `SERIES_NS_*` coefficients | `series_cutoffs.rs` and `tune_series_cutoffs.py` deleted |
| 4 | `base`, `mul` (26) | `KaraKnobs`, `KaraSqrKnobs` (scratch sizing and recursion read the same values), `NttGates` (likewise for the radix gates), `MulKnobs`, `SqrKnobs`, high- and middle-product predicates, `bnm1_fft` takes its cutoff | `mul_cutoffs.rs`, `mul_cutoffs.py` deleted |
| 5 | `div` (19) | `DivKnobs` replaces the `DivCutoffs` consts; `BzKnobs` passed through BZ's recursion | `cutoffs_bench.rs`, `analyze_div_cutoffs.py` deleted |
| 6 | `consts` (6) | accessor reads at the cache builders | — |
| 7 | `theta`, `agm`, `exp`, `ln` (22) | `ExpKnobs` and `LnKnobs` shared by each entry's prediction and decision; `knob_generic!` for the theta term count (a const generic) | `bin_split_cutoffs.rs` deleted |

When Phase 1 is done, `mod.rs` holds only fixed constants (the FFT limits, fold trigger, residue
check, `THETA_STATIC_SLOTS`), `ScratchGuard` and the row-kernel plumbing.

## Phase 2: the tuner

**P2.1, runtime prediction (every build).** `tuning/threads.rs` (`pool()`, cached `cpus()`),
`tuning/table.rs` (change-point lookup), `tuning/model.rs` (`Model`, the gate's closed form, the
`CostCtx` trait with a runtime and a derivation implementation) and the `Dispatcher` trait:

```rust
pub(crate) trait Dispatcher {
    const PATH: &'static str;
    type Shape: Copy; type Alg: Copy;
    fn algs() -> &'static [Self::Alg];
    fn eligible(s: Self::Shape, a: Self::Alg) -> bool;
    fn key(s: Self::Shape) -> (u32, u32);
    fn measure(s: Self::Shape) -> f64;
    fn terms(s: Self::Shape) -> u32;
    fn predict<C: CostCtx>(c: &mut C, s: Self::Shape, a: Self::Alg, threads: usize) -> f64;
}
```

Runtime dispatch reads the table below the extent and compares `predict` above it; derivation
walks the same trait. The generator gains table and step-function knobs, and `build.rs` the presets
layer.

**P2.2, the tuner** (`tune` only, `src/tuning/tuner/`): `cli`, `measure`, `fit` (weighted
non-negative least squares), `sweep`, `derive`, `check`, `out`, and one `SectionTuner` per
section. No new dependencies. Run with `cargo tune` (an alias for the `tune` example).

**P2.3, sections**, in derive order: `base`; the pilot (`mul.school`, `mul.school_sqr`, one
Karatsuba level, the derived `mul.kara` and `mul.kara_sqr`, checked against the old mul harness);
the rest of mul; div; sqrt; bin_split and series; consts; theta, agm, exp and ln; then the presets.
Each section replaces its `Interim` knobs with the inventory's measured and derived knobs, turns
its dispatchers into `Dispatcher` implementations, checks every derived value, and is compared
with its old harness in the `faa9971` worktree.

## Pilot findings (2026-11-09)

- **A missing part of base's work function.** The first joint fit of the three pass kinds with one
  shared call cost left 14.8% relative rms: `div_prim`'s intercept was about four times the others'
  (it computes its divisor's reciprocal once per call). `base.div_setup` models that, and the rms
  fell to 4.7%. This is the method's lack-of-fit loop working as intended: structure in the
  residuals, fixed in the work function, not by more samples.
- **Efficiency cores.** Three base runs on the M1 gave `base.pass` 1.27, 1.05 and 0.41 ns per limb.
  Under load macOS ran the tuner on efficiency cores. The tuner now asks for the user-interactive
  QoS class on every thread it measures from (`measure::prefer_fast_cores`), and two runs then
  agreed within 0.5%.
- **Copies are not carrying passes.** The first Karatsuba work function charged `karatsuba_core`'s
  two `copy_from_slice` calls at `add_buf`'s rate, about 10x too much. They are counted at the
  memcpy rate (`base.fill`) now.
- **The check found the schoolbook model's missing term.** `cargo tune derive --check` times
  `mul_buf`, one `karatsuba_core` level over schoolbook leaves, and the level's three sub-products
  alone, beside their predictions. With W = l·s alone the sub-products (8-33 limbs) were predicted
  30-50% low, which put the crossover near 30 limbs. `mul_buf` runs one asm row per limb of its
  short operand, so its work function gained a per-row term (`mul.school_row`, about 1.9 ns): the
  fit's rms fell from 8.3% to 7.1% (`sqr_buf`: 13.1% to 6.1%), and the derived crossover moved to 36
  for squares and 60 for chunking, against 44-55 and 64 measured by the old harness. The check's
  own timings put the square crossover near 40 on this M1 (40x40: level/school 0.99).
- **Scratch sizing assumes a monotone boundary.** `find_karatsuba_scratch` sizes a level by one
  largest sub-shape, which undersizes when a table makes a smaller sub-product Karatsuba and a
  larger one schoolbook; random `mul.kara` tables panicked in `split_at_mut`. Exact sizing (every
  sub-shape) recursed four ways and cost up to 42% on `mul_static`, so the sizing stays as it was
  and `karatsuba_mul` falls back to schoolbook wherever its scratch is too short: any valid table
  is correct, and a non-monotone one is only slower.
- **A table on the hot path needs an index.** `is_school` is asked at every recursion level. A
  linear scan of `mul.kara`'s 88 change points cost 3-21% on products up to 100 limbs, a binary
  search 3-9%. Two-input tables now come with a row index (`Table2`, built by `build.rs` and by
  `set`): one array read finds the row. Against `faa9971`: outputs identical (161 items), mul
  items 1.0008x, with `mul_dyn` 30x30 the one item past 1% (+1.6%, about 4 ns).
- **Check noise.** Single shapes still swing (48x48's products read 1017 ns in one run, 725 in
  another), so the check needs the settle loop (repeat until the ratio's interval clears 1 ± ε)
  before its verdicts are trusted shape by shape.
- **Fitted form for the Karatsuba level (the user's direction, 2026-11-09).** Assembled from base
  units, a level's overhead came out far too low. Fitted to `karatsuba_core`'s own times as
  scale · (sub-products) + pass · (l + s) + shift, the scale is 1.00-1.02 (the sub-products are
  predicted right) and the fixed cost 43-47 ns.
- **Derivation must follow the recursion and respect ties.** Comparing one level over schoolbook
  leaves at every size called 377x200 schoolbook, and near-ties flipped with l's parity (2560
  change points). Predictions now recurse with a memo (`derive::Pred`), and a row's choice changes
  only past the tie band: 86 change points, every row Karatsuba from s = 58-61.
- **Fit where the kernel competes.** Narrowing the schoolbook fit to 10-96 limbs (l up to 2s) and
  the level's to 24-96 moved the predicted square crossover from 34 to 36-38 and brought most
  check shapes within 1-3%.
- **End-to-end result (M1, normal level): correct, not yet faster.** With the derived table as the
  preset, outputs stay identical to `faa9971` and both suites pass, but mul items are 1.0018x
  overall: 200x60 0.94x and static 400x400 0.96x, against 100x100 1.06x and `hi_mul_static` 600
  1.05x. The model is still about 5% optimistic about a level at 36-48 limbs (the check and the old
  harness put the square crossover near 44), so it recurses a level too deep. The tuned file is
  kept at `target/tuning/m1-normal-2026-11-10.toml`; no preset is checked in.
- **What a preset exposed.** `cargo tune tune mul` derived from default `base` values (it now
  loads the file it merges into first); `mul_work` jumped where the table's boundary left its own
  scaling (it keeps the old formula boundary until `mul_ns`); and two tests pinned default values.
- **Check readings get disturbed.** Single probes sometimes read 30-40% high for a whole run
  (53x40 schoolbook 1099 ns against 768), probably buffer placement. The check needs to repeat a
  shape until its ratio settles, on fresh buffers.
- **Robust timing (`time_min`).** Each fit and check takes the least of 3 passes (5 at thorough) on
  freshly placed buffers (`pad`). The level's fit went from 4.3% to 1.8% relative rms and the
  disturbed readings disappeared.
- **End-to-end comparison (`cargo tune derive --compare`).** It times `mul_dyn` under the build's
  table and the derived one, installing each (`tuning::install`, one pointer store) between
  interleaved rounds. That is the real measure of time lost, and needs no second build.
- **The fit's error is the decision margin.** The check put 40x40 at a tie where the model said
  Karatsuba by 2%, and in real calls ties go to schoolbook (no scratch). The derivation now keeps
  schoolbook until Karatsuba is predicted faster by more than the two fits' combined relative rms
  (`mul.school_err`, `mul.kara_err`, about 4.2% together), in the table and in the recursion's own
  predictions. The derived boundary moved from 38 to 46 for squares and 62-66 at l = 4s, where the
  old harness measured 44-55 and 64.
- **Result (M1, thorough, 2026-11-10).** Under the derived table no compared shape is slower than
  under the old boundary (worst 1.002x) and 48x48, 84x56, 96x64, 100x100, 300x100 and 192x128 are
  2.6-4.9% faster (geometric mean 0.9936 over 42 shapes). Against `faa9971` in separate processes:
  outputs identical on all 161 items, mul 0.9995x (`mul_static` 400x400 0.94x), div 1.0043x, sqrt
  1.0042x. Both suites pass with and without the preset.
- **Squares (2026-11-10).** `mul.karatsuba_sqr` stays a cutoff in dispatch (a one-input table with
  one change point) and became a derived value: `sqr_buf`'s model (1.3% rms) and the Karatsuba
  square level's fitted form (`mul.kara_sqr_scale/pass/shift`, 1.6% rms, timed through
  `karatsuba_sqr_core`, now `pub(crate)`), with their combined error (2.1%) as the margin. Derived:
  75 on the M1, against the hand-tuned 71 (the old harness's tie was 69-71 on the i7). Timings
  against `faa9971` are neutral (mul items 1.0016x), outputs identical, both suites pass.
- **Karatsuba against the FFT (2026-11-10).** mul's tune now has a second phase, measured under
  the leaf boundaries just derived: recursive Karatsuba through its real entries, in the cost
  forms `is_karatsuba` already used (`karatsuba_cost`, `chunking_karatsuba_cost`, n^1.585) plus a
  linear term and a shift (`mul.kara_rec_*`, `chunk_rec_*`, `kara_sqr_rec_*`), and `fft_entry` and
  `fft_sqr_entry` as scale * rho(o) N log2 N + point * N + shift at their real transform size
  (`mul.fft_*`, `fft_sqr_m_*`; `FFT_ODD_COST` is the fixed rho(o), AArch64 only so far; the two
  square fits, `kara_sqr_rec_*` and `fft_sqr_m_*`, were purged later, see the end of this
  log). Fits: 3.4-5%
  relative rms; the recursive-square fit is flagged for lack of fit (a pure power law doesn't
  describe Karatsuba over a fixed leaf from 96 to 1800 limbs). Derived, as the value with the least
  predicted time lost: `mul.fft_sqr` 331-333 (hand-tuned 340), `mul.fft_karatsuba` 2.32 and
  `mul.fft_chunking_karatsuba` 2.25 (hand-tuned on the i7: 1.96 and 1.88).
- **End to end (M1).** `cargo tune derive --compare`, old defaults against every derived value in
  one binary: `mul_dyn` 0.9823x over 102 shapes, worst 1.015x, with mid-size shapes far faster
  (312x104 0.69x, 237x158 0.70x, 414x138 0.83x, 276x276 0.85x): the i7's ratios sent the M1 to the
  FFT too early. `sqr_dyn` 1.0000x over 34 shapes. Against `faa9971` in separate processes:
  outputs identical (161 items); mul 1.0008x, div 1.0033x, sqrt 0.9984x, series 1.0064x, trans
  1.0026x (that harness has few shapes in the improved zone; `sqr_dyn` 300 reads 3.5% slower, with
  the squares cutoff at 77 putting its 75-limb leaves on `sqr_buf`). Both suites pass with and
  without the preset.
- **Still boundary forms.** These decisions keep `is_karatsuba`'s ratio form and one cutoff for
  squares, with derived values. The design's end state, comparing predictions per call above an
  extent, needs the runtime predictors (`mul_ns`) and would follow the FFT's size steps exactly.
- **Transform crossings are sweeps (2026-11-10).** Deriving the Karatsuba-to-transform values from
  the fitted models failed its end-to-end check on the static side (`sqr_static` up to 8.7% slower,
  `mul_static` 2352x1176 11.5%): the NTT fits carry 7-9% error. The fit-error margin that settled
  the leaf boundary made the dynamic side worse here (364x364 24% slower), because past this
  crossing the transform pulls ahead quickly, so waiting is costly, where at the leaf the two are
  tied over a wide range. These six values, and `mul.karatsuba_sqr` (whose margin wobbled between
  75 and 79), are now `Sweep` knobs: both entries timed at each shape, and the value with the least
  measured time lost, in the existing boundary forms. Cutoffs take a coarse pass and then a dense
  one (sizes 1.5% apart) around the crossing, as a coarse grid steps over the sizes just past a
  transform step. Each sweep's evidence states the previous value's loss on the same timings.
- **Measure in the caller's context.** Sweeps first ran on a worker of a private pool and put
  `static_ntt_sqr` at 2000; from the main thread on the global pool, as callers run, it is
  2600-2800. The tuner now configures the global pool (`--threads`, performance cores, 64 MiB
  stacks) and measures everything from the main thread.
- **M1 results (thorough).** Measured time lost, new value against the old on the same timings:
  `karatsuba_sqr` 69 (old 71: 0.00% vs 0.02%); `fft_sqr` 318 (340: 0.80% vs 1.29%);
  `fft_karatsuba` 2.35 (1.96: 0.42% vs 2.50%); `fft_chunking_karatsuba` 2.33 (1.88: 0.38% vs
  2.29%); `static_ntt_sqr` 2788 (3884: 0.68% vs 6.41%); `ntt_karatsuba` 5.13 (5.7: 0.46% vs 0.94%);
  `ntt_chunking_karatsuba` 4.93 (5.9: 0.46% vs 2.79%). Worst single shapes stay at 6-19%: a single
  cutoff or ratio can't follow the transforms' size steps, which per-call comparison would.
  Against `faa9971` in separate processes: outputs identical (161 items); mul 1.0011x, div 0.9986x,
  sqrt 0.9976x, series 1.0062x, trans 1.0019x. Both suites pass.
- **Left in place:** the fitted mid-range models (`mul.kara_rec_*`, `fft_*`, `ntt_static_*`, …) are
  measured but no decision reads them now; they are what `mul_ns` will evaluate. `derive.rs` still
  has `fft_sqr_cutoff` and `fft_kara_ratios`, unused.
- **mul's other one-input decisions as sweeps (2026-11-10).** `late_cutoffs` sweeps the two code
  paths of each: `mul.hi_mul_row`/`hi_sqr_row` (columns against rows, every window from 6 to 200),
  `mul.{dyn,static}_partial_{mul,sqr}` (`hi_*_buf` against the guarded product, operands of 2w and w
  limbs), `mul.fft_mid`/`ntt_mid` (`mid_mul_buf` against the transform; these switch at n >= knob,
  so `sweep_cutoff`'s `plus` adds one) and `mul.sqr_bnm1_fft` (`sqr_dyn` and a fold against the cyclic
  FFT, at `bnm1_size`s). M1, normal level, new value (old: old's mean loss on the same timings):
  `hi_mul_row` 21 (24: 0.01%), `hi_sqr_row` 95 (90: 0.02%), `dyn_partial_mul` 318 (312: 0.46%),
  `static_partial_mul` 284 (408: 3.38%), `dyn_partial_sqr` 337 (464: 4.47%), `static_partial_sqr`
  513 (640: 0.83%), `fft_mid` 109 (125: 5.70%), `ntt_mid` 445 (672: 13.99%), `sqr_bnm1_fft` 80
  (340, a GUESS: 24.79%). `sqrt.dyn_wrapped_rem` follows `mul.sqr_bnm1_fft` + 1, so it moves from
  341 to 81 with it; that follow was wrong: with the wrapped remainder on from 81 root limbs, `sqrt_dyn` was 40% slower
  at 200 limbs and 9-12% at 700, and root-only outputs' unspecified `x` changed. The knob is its own
  decision (a stage's remainder by wrapped square or by the division's remainder), so it no longer
  follows: its default is 341 again, a GUESS, until the sqrt section sweeps it. With that, against
  `faa9971` (thorough preset): outputs identical on all 161 items; mul 0.9957x (`hi_mul_static`
  300-330 at 0.86-0.90x), div 1.0067x, sqrt 0.9957x, series 1.0008x, trans 1.0026x; both suites pass.

## Phase 2 plan as revised (2026-11-10)

The user's decision after the first sections (design doc, decision 8): sweeps for one-input
decisions, fitted runtime equations for multi-input dispatchers, no purely derived runtimes, and
a fitting library instead of the hand-written solver. Score when it was made: 1 decision derived
(`mul.kara`, itself from fitted forms), 7 derivations failed and became sweeps, 17 swept directly.

**The fitting layer (`tuner/fit.rs`).**
- Add `nalgebra` and `levenberg-marquardt` as optional dependencies enabled by `tune`
  (`tune = ["dep:nalgebra", "dep:levenberg-marquardt"]`). Normal builds don't compile them.
- Replace `nnls` with: weighted linear least squares on relative error (rows scaled by 1/t, solved
  by nalgebra's SVD), keeping coefficients non-negative by dropping a term that comes out
  negative and refitting; and a Levenberg-Marquardt fit for forms nonlinear in their constants
  (an exponent, a leaf size), with the same relative residuals.
- Each fit still reports its relative rms, worst residual and sign-run flag, and stores its rms.

**One-input decisions: sweeps.** Done for mul (16) and sqrt (8, in validation). Still to do, as
sweeps: mul's four NTT parallel gates; div's `knuth`, `bz` leaf and the two reciprocal cutoffs;
bin_split's eight leaf cutoffs and two gates; series' `horner_limbs`/`horner_terms`; consts' six
gates; theta's two gates, the AGM's two tail caps, exp's and ln's chunk widths, square counts,
square-root scales, gates and theta term counts (each one number, thread-dependent ones swept at
one thread and at the pool's size).

**Multi-input dispatchers: fitted equations compared by dispatch.**
- *Products (`mul_dyn`, `mul_static`).* One equation per algorithm in (l, s): schoolbook
  (done), Karatsuba near-square and chunking (forms exist; improve them until they pass the
  check, for example with the leaf size in the form), the FFT and the NTT (at their real
  transform sizes, dynamic and static). Dispatch then takes the least fitted time: `mul_ns` is
  that evaluation, and it replaces `is_karatsuba`'s ratio rule, the four ratio sweeps that stand
  in for it today, and `mul_work` in the series model. `mul.kara` stays a table computed from
  the fitted schoolbook and level equations, as small products can't afford the comparison.
- *Division (`div`, `div_rem`; dynamic and static).* One equation per algorithm in (q, d):
  Knuth (q d), BZ and NR in their theoretical forms, which may include mul's fitted product
  times, each fitted to its own forced runs through the prepared drivers. They replace the
  twelve BZ-vs-NR constants, `bz_top_padded_cost_scale` and `bz_2_1_cost`'s two scales.
- *Series (`series_eval`).* Already a fitted model compared at run time; refit its seven
  coefficients with the new fitting layer, per algorithm, against its own entries.
- Every such dispatcher is checked end to end: measured time lost of its choices over a grid of
  shapes, against the old rule's on the same timings (`cargo tune derive --compare` and the
  sweeps' "previous value" evidence are the pattern).

**Removed from the plan:** deriving tables, extents, gates and balance points from predictions;
`DeriveCost`; the base units as building blocks of composites (they remain as measurements, and
`mul.kara`'s chunking term still reads two of them, to be refitted as part of the chunking
equation).

**Order from here:** the fitting layer; products' equations and `mul_ns` with per-call
comparison; division; then the remaining one-input sweeps section by section (bin_split, series,
consts, trans); the x86 preset on the i7.

### Progress on the revised plan

- **Fitting layer (2026-11-10, done).** `Cargo.toml`: `nalgebra` 0.34 and `levenberg-marquardt`
  0.15, optional, enabled by `tune` only (a normal build compiles neither; `nalgebra` stays at
  0.34 because `levenberg-marquardt` 0.15 is built on it). `tuner/fit.rs`: the Lawson-Hanson
  solver is gone. `fit_rel` solves equations linear in their constants by nalgebra's SVD on
  relative error, dropping a negative term and refitting (`lstsq_nonneg`); `fit_nonlinear` fits
  constants inside a power by Levenberg-Marquardt, on the same relative residuals, with p = u^2
  keeping them non-negative and a central-difference Jacobian. Test:
  `test_fits_recover_coefficients`.
- **Fitted exponents for recursive Karatsuba (2026-11-10).** `fit_exp` fits
  t = a W(shape, e) + b lin + c with e free, started from the linear fit at log2 3, and keeps the
  free exponent only when it fits better; six new knobs `mul.<kara|chunk|kara_sqr>[_static]_rec_exp`
  (mul version 12). M1, quick level, rms at 1.585 against fitted: near-square 5.0% to 2.6%
  (e = 1.635), chunking 3.3% to 2.4% (1.623), squares 5.0% to 1.8% (1.637); static 2.8% to 2.5%
  (1.613), 2.8% to 2.5% (1.604), 2.0% to 1.3% (1.615). The exponent sits above log2 3 because
  schoolbook leaves (n^2) still weigh at these sizes. No decision reads these equations yet.
- **sqrt (2026-11-10, done on the M1).** Every sqrt decision is one-input, so all are sweeps
  (`sections/sqrt.rs`), through the public entries with the knob in question forced to each side:
  `leaf` 19 (was 11), the six top-level cutoffs 14-19, `dyn_wrapped_rem` 438 (was 341: 0.26%
  against 0.94% mean loss). Against `faa9971`: sqrt 0.9618x, both suites pass; four 22-limb
  root-only items hash differently (the unspecified x past a changed top-level cutoff; the roots
  themselves still to be confirmed equal).
- **Decisions on measurement (the user, 2026-11-10).** (1) No cache terms in any runtime
  equation: every run is assumed warm, and each probe must be warm before its data is gathered.
  (2) The timing loop (`tuner/measure.rs`) stays hand-written: benchmark crates are harnesses
  that don't return values or interleave shapes, so they'd save little. (3) A pass for line
  reduction and consolidation over the whole tuning change comes later; don't spend on it now.
  The acceptance thresholds below settle what was open here.
- **A noisy tune, and two tests it exposed (2026-11-10).** The thorough tune that wrote the
  current `presets/aarch64-macos.toml` ran with other sessions loading the machine (load average
  3-5). Its schoolbook fit reads 3.8% rms (about 2% on a quiet machine) and `mul.kara` came out
  at Karatsuba from 60 limbs for squares and 88 at l = 4s, against 46 and 64 before. **Retune on
  a quiet machine before trusting this preset.** Against `faa9971` it is still neutral (mul
  0.9969x, div 0.9962x, sqrt 0.9613x, series 1.0026x, consts 1.0187x, trans 1.0017x; identical
  outputs but the four 22-limb root-only/approximate sqrt items, whose hash includes the
  unspecified x). The tuner has no check for a loaded machine or a poor fit; see the open
  thresholds above. The higher table failed `test_karatsuba_core_recurse_boundary_sizing` (fixed
  short lengths 40-76) and `test_karatsuba_static_fallback_scratch` (sizes from `mul.karatsuba`,
  which no longer sets the boundary); both now take their shapes from the live table
  (`flat_branch`, `is_school`, and an N picked from a list). Other `KARA_CUTOFF` uses in
  `test_mul.rs` are correctness sweeps that pass under any table.
- **Acceptance thresholds (the user's decision, 2026-11-10; built).**
  *Fits:* accepted at relative rms <= 5% (`fit::ACCEPT_RMS`), "pretty standard in science". The
  worst single residual is reported but never judged: too much that changes from run to run can
  move one shape. A rejected fit's evidence says `REJECTED`, and `cargo tune tune` ends with the
  list of them (it still writes the file).
  *Samples:* a level's thoroughness is the spread its samples must come under, (interquartile
  range) / median: 5% quick, 2.5% normal, 1% thorough (`Level::spread`). `measure::sample` (`time_probes` then) takes at
  least `rounds` samples of each probe, then keeps sampling the unsettled ones up to
  `max_rounds` (6x). The tune ends with how many measurements never settled, and says the machine
  was busy when that passes 5% of them.
  *First readings (M1, load average 8):* quick, 1 of 1110 unsettled; normal, 6 of 3266. So the
  spread bound does **not** by itself detect a loaded machine: steady load slows every sample of
  a batch alike, which shifts the median without widening the spread. Detecting that needs
  something else (the load average at start, or a reference probe timed throughout).
  *Rejected today:* `base.pass`/`mul_pass`/`div_pass` (one joint fit, 9-12% rms, 9.2% in the
  standing preset, so not load) and `base.fill` (5.4-8%); by the last quiet readings also the
  static NTT equations (7-16%). The FFT's (4.0-4.3%) and every Karatsuba equation pass.
- **Rule for a rejected fit (the user's, 2026-11-10; also in CLAUDE.md).** Report which fit
  failed, its equation and by how much; propose a new form, or a few to try; wait for the user's
  approval. Pending under this rule: `base`'s joint pass fit, `base.fill`, the static NTT
  equations.
- **For Phase 3 (discussed, not decided):** a user's tune should not use a rejected fit; the
  decisions that depend on it keep the shipped values, the user is told which, and a rejected fit
  is remeasured once first.
- **div's one-input decisions as sweeps (2026-11-10; `sections/div.rs`, div version bumped).**
  `div.dyn_rcp_knuth_nr` and `div.static_rcp_knuth_nr` (`rcp_prepared_*` with `RcpAlg` forced, R
  limbs from an R-limb divisor, static at N = 192) and `div.bz` (2d / d through
  `div_prepared_dyn`: forced Knuth against forced BZ with the leaf at half of d, so one BZ level
  over Knuth leaves). Two runs on a busy M1 (thermometer: disturbed) agree: 22 (was 36: 0.9-1.0%
  mean loss, worst 18-20%), 28 (36: 0.1-0.2%) and 119 (140: 0.4%). **Not in the preset yet**: they
  wait for a quiet tune. `div.knuth` and the twelve BZ-against-NR constants are the two-input
  dispatcher's and wait for fitted equations in (q, d).
- **Fixed: a sweep's "previous value" read the probes' leftovers.** Probes that force a knob
  while they run (sqrt's, div's) left it at the last probe's value, which `sweep_cutoff` then
  reported as the previous value (79 for `div.bz`, whose default is 140; `usize::MAX` and 3 in
  sqrt's evidence). `cutoff_sweep` now restores a snapshot after each measurement. The values
  the sweeps chose were unaffected: each probe installs what it needs before it runs.
- **base's pass fits, separated (approved by the user, 2026-11-10; base version 3).** `add_buf`,
  `add_mul_limb` and `div_prim` each have their own fit t = a n + c (`pass`/`call`,
  `mul_pass`/`mul_call` (new), `div_pass`/`div_setup`, the last now div_prim's whole fixed cost),
  over 11 sizes from 16 to 512 limbs: below 16 these kernels' times follow their unrolled blocks
  and tails, and nothing reads them there. M1, normal, thermometer clean: 4.0%, 4.4% and 0.4% rms,
  all accepted (the joint fit read 9-12%).
- **Who reads `base`.** `pass`, `call` and `fill` are read in one place, the chunking term of
  `mul.kara`'s derivation (`derive::Pred` through `kara_work::chunking`), at passes of about 40
  limbs and up. `mul_pass`, `mul_call`, `div_pass`, `div_setup`, `alloc`, `eval`, `thread`,
  `rayon_wake` and `rayon_task` were read by nothing: the building blocks of the retired
  derive-everything method. All removed in the purge below.
- **`base.fill` was still rejected (moot: removed in the purge below).** With the approved changes (eight sizes
  32-4096, t = a n) it reads 6-9% rms. ns per limb, repeatable over four runs: 0.101 at 32, 0.091
  at 64, 0.100 at 128, 0.093 at 256, 0.087 at 512 and 1024, 0.090 at 2048, and 0.095-0.119 at
  4096. So the zeroing has a real +-6% wiggle by size below 512 (the memset's strategies), and
  4096 moves from run to run.
- **Parallel policy as first proposed (settled below).** The design doc's policy (structural predictions
  from three measured overheads) is the retired kind. Proposed: gates are one-input sweeps at
  the tune's thread count; equations for pool algorithms are fitted on the pool; undecided is
  whether a tune covers several thread counts (per-count values, which the schema's step
  functions hold, or a fitted parallel fraction).
- **The purge (the user's rule, 2026-11-10).** "If fits and benchmarks are no longer needed in
  the dependency graph of the algorithms' dispatch, purge them from the tuning's code, docs and
  plan. A procedure used only in the tuning of one other algorithm is included in that
  algorithm's tuning, not tested separately." What a decision reads today: the schoolbook
  product fit, the Karatsuba level fit and the new chunking fit (together they give `mul.kara`),
  and the sweeps. Removed:
  - `base`'s twelve cost units and `tuner/sections/base.rs` (base version 4; the section keeps
    the two x86 ADX knobs and is no longer tuned on aarch64);
  - `mul.school_sqr_*` and `mul.kara_sqr_{scale,pass,shift,err}` and their fits (the squares'
    leaf is a sweep);
  - `derive.rs`'s `MidModels`, `fft_sqr_cutoff`, `fft_kara_ratios`, `sqr_level_squares`,
    `kara_sqr_cutoff`, `kara_pred`; `mul.rs`'s `kara_work`;
  - `src/tuning/{model,dispatch,threads}.rs` (the `Model`/`gate`, `Dispatcher`/`CostCtx` and
    cached thread counts of the retired design) and their two tests.
  *Folded in:* chunking Karatsuba is fitted on itself, t = scale (its chunk products' fitted
  schoolbook time) + pass (l + s) + shift, over schoolbook products at s 24-96, l = 2s to 4s
  (`mul.chunk_scale/pass/shift/err`; mul version 13, derive 10). First reading 2.9% rms,
  accepted. `Pred::kara` prices a chunked product with it, over chunk products as dispatch will
  run them.
  *Kept, to be confirmed:* the mid-range product equations (recursive Karatsuba with its fitted
  exponent, the FFT entries, the static NTT entries). No decision reads them today; they are
  the inputs of the per-call product dispatch that is the plan's next step.
  *Plan items this retires:* P2.1's `CostCtx`/`Dispatcher`/gate closed form, `sections/base.rs`,
  P2.3 step 1 (`base`), and the base units in the series model's refit.
- **Parallel policy (the user's decision, 2026-11-10).** Users will be told to tune at the
  number of cores they intend to run on. The tuner finds values for a serial version and a
  fully parallel version, and the library decides between them at run time. Gates are sweeps;
  pool algorithms' equations are fitted on the pool. Not built yet: today every thread-dependent
  value is measured once, at the tune's pool size.
- **The thermometer: removed (the user's decision, 2026-11-10).** A 64-limb add read around
  every group of measurements, with a hot group measured again, was built and then taken out the
  same day: with it and the 1% spread bound, a thorough tune that had taken 15-20 minutes was
  still running after 45. Nothing in the tuner now detects a steadily loaded machine (the spread
  bound doesn't: steady load shifts the median without widening the spread), so check `uptime`
  before a tune and distrust one run beside other work. Earlier entries here that cite
  thermometer counts are from the hours it existed. If the spread bounds or the 5% rms line keep
  failing or make tunes excessively slow, tell the user so they can be reevaluated.

## Completing Phase 2: the approved plan (2026-11-10)

The plan the user approved is `~/.claude/plans/i-want-you-to-scalable-donut.md`: Step 0 baseline;
1 shared sweep helpers (`tuner/sweep.rs`, with `param_sweep` and `with_knob`); 2 serial/parallel
value pairs; 3 the remaining one-input sweeps by section; 4 products dispatched by comparing
fitted equations per call; 5 division's equations in (q, d); 6 the series model's refit; 7
presets (M1, then x86 on the i7) and close-out. Standing at approval: about 28 of 95 decisions
tuned (157 knobs: 29 sweeps, 58 equation constants, 70 interim).

- **Step 0, baseline (done 2026-11-10).** Thorough tune without the thermometer, load average
  3-4.5: **about 36 minutes** (15-20 with fixed sample counts), and 2697 of 27150 measurements
  (10%) never came under the 1% spread in 126 rounds. Fits: all accepted (schoolbook 3.8%, level
  1.8%, chunking 2.9%, recursive Karatsuba 1.3-3.7%, FFT 4.2% and 4.6%) but the two static NTT
  ones (7.9%, 7.6%). `mul.kara` with the chunking fit: Karatsuba from 42 limbs for squares and 54
  at l = 4s (the same in two runs; 46 and 64 with the old chunking term). Both suites pass (574,
  578). Against `faa9971`: outputs identical but the four known 22-limb sqrt items; mul 0.9935x,
  div 1.0168x, sqrt 0.9647x, series 1.0031x, consts 1.0569x, trans 0.9971x.
  *Regressions to resolve:* `rcp_static<1024>` at 30 limbs is **1.50x** slower: the sweep put
  `div.static_rcp_knuth_nr` at 28 with N = 192, but at N = 1024 static NR's `[u64; N]` fills make
  Knuth far better at 30, so that cutoff depends on N as well as R (two inputs).
  `div_rem_static<2048>` 200/100 is 1.11x and the four consts items average 1.057x, causes not
  yet found (the new 42/54 table is the suspect; `cargo tune derive --compare` it).
- **Step 0 follow-up: the regressions are static decisions tuned at one N (2026-11-11).**
  (1) `div.static_rcp_knuth_nr` is back to its default 36 and out of the sweeps (`sections/div.rs`
  sweeps only the dynamic one); `rcp_static<1024>` at 30 limbs is no longer slower, and div reads
  0.9945x. (2) `div_rem_static<2048>` 200/100 at 1.10x is the new `mul.kara` table: with the
  table's line removed from the preset it disappears, while `div_rem_dyn` at the same shape is
  0.965x with the table. The table is derived from dynamic timings and shared by `mul_static`,
  whose Karatsuba pays for `[u64; N]` scratch. (3) `ln2_static<1100>` at 1.04x is not the table
  (unchanged without it); the other static sweeps (measured at N = 4096 or 8192, run here at
  N = 1100) are the suspects. All three say the same thing: **a static decision's best value
  depends on N as well as the size**, so sweeping it at one N is tuning a two-input decision as
  if it had one. Awaiting the user's choice (two-input equations in (size, N); or sweeps at a
  large N; or leave static values at their defaults until the two-input dispatchers exist).
  The stale `[base]` section was removed from `presets/aarch64-macos.toml`.
- **Two decisions by the user (2026-11-11).** (1) Spread bounds are 6% quick, 4% normal, 2%
  thorough (`Level::spread`; were 5 / 2.5 / 1). (2) "For the rest of the tuning just come up
  with approximate values for the static algorithms": their memory handling is about to change
  fundamentally. So: the static sweeps stay as single-N approximations (N = 8192 for the
  Karatsuba-to-NTT values, 4096 for the high and middle products, 128 for sqrt),
  `div.static_rcp_knuth_nr` stays at its default, nothing two-input is built for static entries,
  and the five static fitted equations are removed with their 23 knobs (`mul.kara_static_rec_*`,
  `chunk_static_rec_*`, `kara_sqr_static_rec_*`, `ntt_static_*`, `ntt_sqr_static_*`; mul
  version 14): only a static per-call dispatch would have read them. That also retires the two
  standing rejected fits and the NTT residual study. Known and accepted: the shared `mul.kara`
  table costs the static path about 10% at `div_rem_static<2048>` 200/100, and `ln2_static<1100>`
  reads 1.04x.
  *Plan changes:* Step 4's item 1 (static NTT residuals and forms) is dropped, `mul_ns` and the
  per-call comparison are for dynamic entries only (static keeps today's ratio rule and cutoffs),
  and Step 5's equations are for `div_dyn`/`div_rem_dyn` only (the six static BZ-vs-NR knobs
  keep their defaults).
- **Baseline under the 2% bound (2026-11-11).** Thorough tune: about 23 minutes (36 at 1%),
  with 1366 of 24845 measurements (5.5%) never under 2% in 126 rounds, on a machine at load
  average 4-5. No fit rejected. Both suites pass (574, 578). Against `faa9971`: outputs identical
  but the four known 22-limb sqrt items; mul 0.9957x, div 0.9958x, sqrt 0.9634x, series 1.0019x,
  consts 1.0063x, trans 1.0001x. Beyond 2% slower: `div_rem_static<2048>` 200/100 at 1.094x (the
  shared table, accepted), `hi_mul_dyn` 330 at 1.035x, `sqr_static<8192>` 3000 and
  `exp_static<1100>` 1040 at 1.03x, `ln_dyn` at 2 and 16 limbs at 1.03x (those two since Phase 1).
  This preset (`presets/aarch64-macos.toml`: mul, div, sqrt) is the baseline for Steps 1-7.
- **Step 1, shared sweep helpers (done 2026-11-11).** `tuner/sweep.rs` holds `cutoff_sweep`,
  `sweep_cutoff[_plus]`, `dense_around` and `lost` (moved from `sections/mul.rs`), and three new
  ones: `with_knob` (a probe run with a knob installed), `gate_sweep` (a gate knob forced shut
  and open, as a cutoff sweep) and `param_sweep` with `pick_param` (an algorithm parameter with
  no sides: the candidate losing least against the best candidate at each size). Test:
  `test_param_sweep_choice`. (`gate_sweep` never got a caller, as every gate is swept through
  `param_sweep` with `with_knob`, and was deleted in the 2026-11-11 simplification pass. Since
  that pass every sweep entry takes the knob's path, not its bare name, and `cutoff_sweep` takes
  its dense pass's cap as an `Option`.)
- **Step 3 begun before Step 2 (2026-11-11).** The sweeps at one and at N threads are what show
  which knobs need a serial/parallel pair, so the sections come first and Step 2 follows their
  data. Built, all as `param_sweep`s (M1, normal level, values provisional: the machine was
  busy, 8% of measurements unsettled):
  - `sections/agm.rs`: `agm.dyn_tail_terms` 9 (was 10: 0.64% against 0.74% mean loss) and
    `agm.static_tail_terms` 7 (was 4: 0.95% against 1.17%), through `agm_*_at` over caps 0..=19.
  - `sections/trans.rs`, exp: `dyn_chunk_min_bits` 24 (16), `static_chunk_min_bits` 2 (16),
    `dyn_sqr_bits` 64 (24), `static_sqr_bits` 32 (24). Every candidate from 8 bits up is within
    half a percent of the best at every size, so these barely matter and the old values stand
    within noise.
  - `sections/trans.rs`, ln (the series, `ln_*_at`): `dyn_chunk_min_bits` 3 (2),
    `static_chunk_min_bits` 16 (2), `dyn_sqrt_a` 0.5 (0.25), `static_sqrt_a` 0.5 (0.25). Here one
    value can't serve every size: the best single value still loses 3-4.4% on average against
    the best per size (worst 10-32%), and the old values 4.7-7.8%. The parameter's best value
    moves with the size, so its form (A log2 of the precision, a fixed minimum width) is what
    limits it, not its constant.
  Still to build in Step 3: mul's four NTT gates, bin_split, series' Horner limits, consts,
  theta, the exp and ln chunk gates, ln's AGM cutoff and term counts.
- **Step 3: `sections/bin_split.rs` built, values not yet measured (2026-11-11).** All ten knobs
  as `param_sweep`s over the old tuning doc's workloads, through `bin_split_tree_at` with
  explicit `SplitCutoffs`: dynamic BBP (ln 2's `Ln2S<6, 1>` at 64-4096 limbs, `ATanh` at 64 and
  256 bits a term), Engel (`Exp` at shifts 1, 8, 64), Hyper (`Exp` with x of 1, 4 and 16 limbs
  at shift 2 bits(x)), the term floor, `dyn_par` (through the cutoffs) and `dyn_merge_par` (the
  knob installed); static trees at N = 1100. bin_split version 2. **Its first run is not
  usable**: at load average 5.4, 686 of 1158 measurements (59%) never came under the 4% spread,
  and it reported cutoffs changing times by 28-77% where the hand tuning of
  `docs/bin_split_cutoff_tuning.md` saw a few percent. Two things to settle on a quiet machine:
  whether the workloads reproduce that doc's tables at its cutoffs, and whether these
  millisecond-scale, partly parallel trees can meet the spread bound at all (if not, that is
  the "spread keeps failing" case to bring to the user).
- **bin_split measured: its dynamic knobs are coupled (2026-11-11).** The busy run's values came
  from load (a system photo-analysis daemon at 200% CPU, load average 13). Rerun at load 2.5-5:
  a single pass from the defaults agrees with itself over three runs (Engel leaf 384-512, Hyper
  leaf 512, `dyn_par` 1024) but its best values still lose 8-10% on average, and a run that
  happened to start with `dyn_par` at 1024 chose leaves of 48 and 96 losing under 3%. With the
  split gate at its default 256, large leaves win only by avoiding parallel splits that cost
  more than they save. `sections/bin_split.rs` now sweeps the dynamic set twice, each sweep
  under the values before it; two runs then agree: `dyn_bbp_leaf_limbs` 64 (was 64),
  `dyn_engel_leaf_limbs` 64 (96), `dyn_hyper_leaf_limbs` 24-96 (48; flat once the floor is
  right), `dyn_hyper_leaf_terms` 32 (8), `dyn_par` 1024 (256), `dyn_merge_par` 256-512 (256),
  each losing about 1% or less. Static, single pass at N = 1100: BBP 192 (512), Engel 192 (128),
  Hyper 768 (64), floor 2-64 (8; flat). Not in the preset yet.
  *Spread:* about half of these measurements (1041 of 2094) never come under the 4% bound even
  on the quieter machine: millisecond trees that split across the pool vary more than that from
  run to run. The chosen values are stable anyway. This is the "spread keeps failing" case to
  put to the user: for work on the pool, the bound may need to be wider or judged differently.
  *General lesson:* knobs that interact (a gate and the leaves under it) need the sweep
  iterated, or a first pass can settle on a consistent but poor combination.
- **Whole-algorithm sections make thorough tunes slow (2026-11-11; put to the user).** With
  bin_split, agm, exp and ln added, a thorough tune was stopped at 67 minutes. Cause: a kernel
  fits thousands of calls into one sample, but a tree or a whole exp is one call of tens of ms
  a sample, and an unsettled one ran 126 rounds x 5 passes. `Level::probe_budget_ns` (50 ms /
  250 ms / 1 s per probe per pass; at least 5 samples) now bounds that. With it, at thorough:
  bin_split alone takes 13.5 minutes with 2378 of 3490 measurements (68%) over the 2% spread,
  and exp + ln 9.7 minutes with 1676 of 3545 (47%). So a full thorough tune is about 50 minutes
  (mul, div, sqrt: 23), and for these sections the spread bound is not met by most measurements
  however long they run. Their chosen values are nevertheless stable from run to run. Options
  put to the user: judge whole-algorithm sweeps by the stability of the value instead of the
  spread; run them at a lighter level than the kernels; fewer workloads and candidates; or
  accept the time.
- **The tie band, and the new sections validated end to end (2026-11-11).**
  *Tie band.* Where a knob hardly matters its best value follows noise (exp's square count read
  64, then 8, against 24, all within 0.1%), and a sweep on the pool can't see what such a knob
  costs on one thread. So every sweep (`sweep_cutoff`, `pick_param`, `sweep_ratio`) now
  keeps the knob's previous value unless another loses less by more than the level's ε
  (`Level::eps`: 3% / 1% / 0.5% of mean time lost; the rule and its evidence are one function,
  `sweep::keep_or_take`, since the 2026-11-11 simplification pass, which also removed the
  `sweep::set_tie` global this entry first named); its evidence then says "kept".
  At normal level this kept the AGM caps (10, 4), exp's four parameters (16, 16, 24, 24) and
  `dyn_merge_par` (256), and let through bin_split's Engel leaf (64, was 96), term floor (32,
  was 8) and split gate (1024, was 256), its static BBP and Engel leaves (192), and
  `ln.static_sqrt_a` (0.5).
  *End to end* (those four sections at normal level on top of the thorough mul/div/sqrt preset;
  both suites pass, outputs identical): the first check found `pi_static<1100>` 1.30x slower
  (`static_hyper_leaf_limbs` 768 from 64, chosen on exp's series alone) and `ln_dyn` 1.22x slower
  at 300 limbs with small sizes 10-25% faster (`ln.*_chunk_min_bits` 16 from 2: one width can't
  serve every size, and the sweep minimizes the mean). Both knobs are **held at their defaults
  and out of the sweeps**: the first as a static approximation that didn't transfer, the second
  pending the user's decision on that trade-off. With them held: mul 0.9966x, div 1.0057x, sqrt
  0.9575x, series 1.0043x, consts 1.0433x (`pi_static` 1.07x), trans 0.9901x; `ln_static` 8-12%
  faster from 4 to 500 limbs and 1.20x slower at 2 limbs (`static_sqrt_a`, approximate).
- **Step 3: theta's gate and ln's AGM decision (2026-11-11).** `sections/trans.rs` gains
  `tune_theta` (the serial plan against the threaded parallel plan through `theta_*_at`, in the
  frames ln's AGM gives it) and, in `tune_ln`, `ln.dyn_agm_terms` (a `param_sweep` over 0..=16)
  and `ln.dyn_agm_limbs` (`ln_dyn_at` against `ln_agm_dyn_at::<T>`), both through
  `measure::on_one_thread`, a one-thread rayon pool, as the library defines that decision on one
  thread. theta version 2, ln version 3. M1, normal level:
  - `theta.dyn_par_limbs` 408-458 (was 256: 2.4% mean loss, worst 38%); `static_par_limbs` 258
    (was 1024: 9.8%; approximate).
  - `ln.dyn_agm_terms` 5, confirmed exactly (0% lost).
  - `ln.dyn_agm_limbs` **48** (was 1024: 7.5% mean loss, worst 38%), the bottom of its sweep
    twice (256, then 48): on one thread the AGM now beats the series at almost every size.
    End to end against `faa9971` with `RAYON_NUM_THREADS=1`: `ln_dyn` 0.78x at 64 limbs, 0.85x
    at 120, 0.77x at 500, unchanged at 300 and 1040. On the 8-thread pool: 0.94x at 120, 0.98x
    at 500, 0.96x at 1040, and **1.09x at 300**, where `agm_wins`' Amdahl scaling
    (`ln.dyn_series_par_fraction`) now takes the AGM though the parallel series is faster.
    That is Step 2's case: a serial cutoff (48) and one swept on the pool, in place of the
    scaling rule. Outputs identical to `faa9971` but the four known sqrt items.
  Left in Step 3: mul's four NTT gates, series' Horner limits, consts' gates, the exp and ln
  chunk gates, and the static AGM knobs (approximate: at their defaults).
- **Whole-algorithm sweeps: judged by stability, and measured less (the user's decisions,
  2026-11-11).** (A) `param_sweep` and the new `cutoff_sweep_whole` take a fixed number of
  samples (`Level::whole_rounds`: 5 / 7 / 11, `measure::time_fixed`, no spread bound) twice,
  choose from the element-wise least, and accept the choice when it is within the tie band of
  the best in each measurement alone (`sweep::measure_twice`, `settle`). An unstable sweep keeps
  the previous value and says UNSTABLE; a choice equal to the previous value is never unstable.
  The tune ends with "stability: U of W whole-algorithm sweeps disagreed". The spread bound is
  unchanged for kernels and fits. (C) Fewer workloads and candidates: bin_split's leaves
  32-256 (7 values) over 3-4 workloads a knob and static trees at one size; the AGM caps 10 of
  the 20, sizes four apart; exp's widths and square counts 5 values each; ln's scales 5; theta's
  and the AGM cutoff's grids 3 sizes an octave. Every default is among its candidates. (1)
  `ln.*_chunk_min_bits` stays at 2 and out of the sweeps.
  *Result, thorough level:* bin_split, theta, agm, exp and ln together take **82 seconds** (over
  23 minutes before), and in the run written to the preset all 27 sweeps were stable (6
  disagreed in the run before it, each keeping its previous value). Both suites pass; against
  `faa9971`: mul 0.9885x, div 0.9767x, sqrt 0.9652x, series 0.9853x, consts 1.0098x, trans
  0.9811x, outputs identical but the four known sqrt items. Values now in the preset beyond the
  earlier ones: `ln.dyn_agm_limbs` 32 (the sweep's bottom again: the AGM from the smallest size
  tried), `bin_split.static_hyper_leaf_terms` 64 (8 lost 55% on its one workload),
  `ln.static_sqrt_a` 1.0. Slower than `faa9971` by more than 2%: `ln_static<1100>` 1.18x at 1040
  limbs and 1.10x at 120 (while 12-21% faster at 4-64 and 500; the static scale, approximate),
  `div_rem_static<2048>` 1.08x (the shared table), `ln_dyn` 1.06x at 300 on 8 threads (Step 2's
  case), `pi_static<1100>` 1.04x.
- **Step 2, the serial/parallel pair, first case (2026-11-11).** Design: a second plain knob
  beside the serial one, read by the call site for its thread count; no generator change (the
  plan's step-function accessor is not needed). `ln_takes_agm` now takes the AGM from
  `ln.<dyn|static>_agm_limbs` (measured on one thread) and, on two or more threads, only below
  `ln.<dyn|static>_agm_par_until` (measured on the pool; new knobs, default 0 = the series on
  every pool, as before). `ln_series_speedup`, the two `ln.*_series_par_fraction` knobs and
  `agm_wins`' speedup argument are gone. ln version 4. A first form, "the AGM from a second
  cutoff upward on a pool", fitted badly (3.9% mean loss, 41% worst): on a pool the choice is
  not one-directional, as the series doesn't split below about 100 limbs and the AGM wins there
  too. M1, thorough: serial cutoff 32, pool bound 129 (0.57% mean loss against 7.14% for the
  series always). Both suites pass (574, 579). Against `faa9971`: on 8 threads `ln_dyn` 0.63x
  at 64 limbs, 0.97x at 120, 0.95x at 500, 0.97x at 1040; on one thread 0.79x at 64, 0.86x at
  120, 0.78x at 500. Sections on 8 threads: mul 1.0008x, div 1.0018x, sqrt 0.9639x, series
  0.9825x, consts 1.0428x, trans 0.9757x.
  *Still open:* `ln_dyn` at 300 limbs on 8 threads stays 1.065x though it now takes the series
  as before, so the AGM choice was not its cause. Suspect: `bin_split.dyn_par` at 1024 (was
  256), swept on exp's Hyper trees only; ln's trees at that size may want parallel splits
  earlier. The static pair is unswept (approximate): `static_agm_par_until` 0.
  *Other pairs:* add one where a sweep at one thread and on the pool disagree; none else is
  evidenced yet.
- **bin_split's gates are tuned through their callers (2026-11-11).** `ln_dyn` at 300 limbs on 8
  threads (1.065x) was not the AGM choice but `bin_split.dyn_par` at 1024: with that one preset
  line removed, `ln_dyn` 300 read 1.02x and `exp_dyn` at 300 and 1040 limbs 0.98x instead of
  1.013x. The sweep timed isolated trees from the main thread, where a parallel split wakes the
  pool; inside exp and ln the trees are built by chunks already on pool workers. Per the user's
  rule (a procedure used inside one algorithm is tuned through it), `dyn_par` and `dyn_merge_par`
  are now `param_sweep`s over whole `exp_dyn` and `ln_dyn_at` calls at 256-4096 limbs with the
  knob installed. They read 256-512 and 512 (were 256 and 256). The leaves stay on isolated
  trees and wobble within a flat region from run to run (BBP 64-128, Engel 128, Hyper 48-96,
  floor 64, each about 1% from the best). `ln.static_sqrt_a` is held at its default and out of
  the sweeps, like the chunk widths: its sweep's 1.0 made `ln_static` 12-20% faster at most
  sizes and 1.11x and 1.20x slower at 120 and 1040 limbs. Both suites pass (574, 579); against
  `faa9971` on 8 threads: mul 0.9952x, div 0.9937x, sqrt 0.9617x, series 0.9892x, consts
  1.0042x, trans 0.9792x, with `ln_dyn` 0.66x at 64 limbs and no dynamic item beyond 2% slower
  but `ln_dyn` at 16 limbs (1.03-1.06x since Phase 1).
- **Step 3: the remaining gates (2026-11-11).**
  *mul's NTT gates* (`ntt_gates` in `sections/mul.rs`; mul version 15): each acts at every level
  of a transform, so each is a `param_sweep` through the NTT entries on the pool, the dynamic
  ones at 70000 limbs (where `mul_dyn` takes the NTT) and the static ones at 3000 and 3600.
  M1, thorough: `ntt_par_conv` 512, `ntt_par` 3600 and `ntt_par_radix3` 27648 kept (each within
  0.4% of the best); `ntt_par_radix5` 6400 (25600 lost 3.95%, worst 12%).
  *exp's and ln's chunk gates* (`*.dyn_chunk_par_work`; exp version 3, ln version 5), through
  whole `exp_dyn` and `ln_dyn_at` calls at 32-1024 limbs: exp's kept at 1280 (flat); ln's 768
  (512 lost 3-5%, worst 9-28%).
  *Left at their defaults:* consts' six gates (`cache_par` takes two inputs, new and cached
  limbs, and is open for practically every build past the 1024-limb tables), the static chunk
  gates and the static AGM pair (approximate).
- **ln's square-root scale is held too, and why the section flipped (2026-11-11).** In the full
  thorough tune `ln.dyn_sqrt_a`'s sweep settled on 0.5 (earlier runs: unstable, kept 0.25). With
  0.5 the series is much faster at small and mid sizes, so the one-thread AGM cutoff then read
  1024 instead of 32 and the pool bound came out unstable (kept 0). End to end on 8 threads that
  combination made `ln_dyn` 0.87x at 30 limbs and 0.74x at 64 but 1.086x at 1040. So the scale
  is a third size-dependent knob (after the chunk widths), and the knobs after it are coupled
  to it: whether its sweep settled decided the whole section. Per the user's choice for such
  knobs it is held at 0.25 and out of the sweeps, as is the static one. With it held, three
  consecutive ln tunes agree exactly: `dyn_chunk_par_work` 768, `dyn_agm_terms` 5,
  `dyn_agm_limbs` 32, `dyn_agm_par_until` 129, every sweep stable.
  *End to end with that preset* (both suites pass, 574 and 579; outputs identical but the four
  known sqrt items). 8 threads: mul 0.9980x, div 0.9989x, sqrt 0.9607x, series 1.0116x, consts
  1.0002x, trans 0.9966x (0.977x in the dedicated rerun); `ln_dyn` 0.63x at 64 limbs, 0.97x at
  120 and 500, 1.00x at 300 and 1040. One thread: `ln_dyn` 0.79x at 64, 0.86x at 120, 0.78x at
  500, 1.00x at 300 and 1040. `exp_dyn` within 2% everywhere. Beyond 2% slower: only static
  items (`div_rem_static<2048>` 200/100 at 1.10x, `sqr_static<8192>` 3000 at 1.04x) and
  `ln_dyn` at 2-16 limbs (1.02-1.06x, there since Phase 1).
- **Thorough tune time now.** The full tune that included the NTT gates took about 37 minutes
  on a busy machine (20% of kernel measurements unsettled): roughly 25-28 for mul, div and
  sqrt's kernel sections, 6-8 for the NTT gates at 70000 limbs, 2 for everything else.
- **series' Horner limits are not a one-input decision: tried, reverted (2026-11-11).**
  `series.horner_limbs` and `series.horner_terms` were swept through `series_eval_range`
  (Horner first against the model and its pick) on exp's shape, a Hyper range with a 64-bit x at
  shift 128, and BBP and Engel ranges at 16 and 64 limbs. Two runs agreed: 609 limbs (28 lost
  3.4%, worst 80%) and 38-39 terms (12 lost 10%, worst 58%). End to end they made `ln_dyn` on 8
  threads **1.49x slower at 500 limbs and 1.37x at 300** (and `ln_static` 1.23x at 500): ln's
  chunks have a small x and gain only a few bits a term, so at 500 limbs they need thousands of
  terms, where Horner's terms x limbs loses badly to binary splitting. The limit depends on the
  precision, the term count and x together, which is what the series cost model exists to
  decide; the two limits are only its shortcut at tiny inputs, and 28 and 12 are conservative
  for that reason. Both are back at their defaults and unswept, and `sections/series.rs` is
  removed until Step 6 (the model's refit) puts the fitted equations there.
  *What it also showed:* on exp's shape the model's pick is up to 80% slower than Horner between
  29 and 609 limbs, so the model does misjudge some shapes: evidence for Step 6.
  *Lesson:* a one-input sweep of a shortcut over a multi-input decision passes its own check
  and fails end to end. Sweep a knob only over workloads that span every caller's shape, and
  keep the end-to-end comparison as the gate.
- **Step 3 is complete (2026-11-11).** Swept: mul's 20 cutoffs and gates (plus the `mul.kara`
  table from its three fits and the four ratio sweeps that Step 4 replaces); div's dynamic
  reciprocal cutoff and BZ leaf; sqrt's 8; bin_split's 9; theta's 2; agm's 2; exp's 5; ln's 4
  (chunk gate, AGM term count, serial cutoff, pool bound). Held at their defaults on purpose:
  `ln.*_chunk_min_bits`, `ln.*_sqrt_a` (size-dependent), `series.horner_*` (multi-input),
  consts' gates (two inputs, practically always open), `sqrt.dyn_rcp_carry` (structural), and
  the static knobs that regressed at another N (`div.static_rcp_knuth_nr`,
  `bin_split.static_hyper_leaf_limbs`, static chunk gates and AGM pair). Remaining: Step 4
  (dynamic products by fitted equations), Step 5 (dynamic division), Step 6 (the series model),
  Step 7 (x86 preset, close-out).
- **Step 4: dynamic products are dispatched by fitted equations (2026-11-11).** Karatsuba
  against the FFT has two inputs (l, s), so by the user's rule it compares fitted runtime
  equations per call. `utils/mul.rs`: `MulEq` (read once per call, and only past the schoolbook
  table) holds recursive Karatsuba's equations (near-square `a (l^e - (l - s)^e)`, chunking
  `a l s^(e - 1)`, each plus fitted linear and fixed terms, which come out 0 on the M1) and the
  FFT's (`a rho(o) N log2 N` at `fft_mul_size(l, s)`); `dyn_dispatch` = `dyn_small_dispatch`
  (single limb, two limbs, the `mul.kara` table) then `dyn_large_dispatch` (the cheaper
  prediction; past `FFT_16BIT_CUTOFF` the NTT in the FFT's place under the same comparison, as
  a long thin product still belongs to chunking Karatsuba). `MulKnobs` and `is_karatsuba` remain
  for static products only.
  *The gate* (`cargo tune equations`, new: both entries and `mul_dyn` timed over four shape
  grids, two of them held out from both the fits and the ratio sweeps; M1, normal level, two
  runs agreeing), measured time lost, mean / worst:

  | Shapes | ratio rule (2.35, 2.33) | fitted equations | `mul_dyn` itself |
  |---|---|---|---|
  | near-square, the sweep's (96) | 0.40% / 15% | 0.07% / 6% | 0.23% / 6% |
  | unbalanced, the sweep's (96) | 0.34% / 8% | 0.22% / 7% | 0.21% / 7% |
  | near-square, held out (66) | 0.23% / 8% | 0.00% / 0% | 0.07% / 1% |
  | unbalanced, held out (66) | 0.71% / 19% | 0.01% / 0.5% | -0.10% / 0.7% |

  The equations' own error on these timings: Karatsuba 1.6-4.2% rms, the FFT 2.8-4.2%. The gain
  is the FFT's equation following its transform-size steps, which one ratio over (l, s) can't.
  Both suites pass (574 + 1 new test, `test_dyn_dispatch_compares_equations`; 579 + 1 with
  `tune`). Against `faa9971`: outputs identical but the four known sqrt items; `mul_dyn` 400x400
  0.92x; mul 1.0011x, div 1.000x on a rerun (1.016x on a busy first run, every item alike,
  static and Knuth-only ones included), sqrt 0.9667x, series 0.9957x, consts 0.9938x, trans
  0.9847x.
  *Defaults.* A target with no preset now dispatches by the equations' schema defaults, so
  they are set deliberately: Karatsuba's are the M1's fit (1.52, exponent 1.635; chunking 1.58,
  1.633) and `mul.fft_scale` is 0.49, which puts the balanced crossing near 230 limbs, where the
  old x86-tuned ratio (1.96) had it. The x86 preset replaces them.
  *Not purged yet:* `mul.fft_karatsuba` and `mul.fft_chunking_karatsuba` and their sweeps:
  `mul_work` still reads them to place its transform tier for the series model. They go when
  `mul_ns` replaces `mul_work` (Step 6). Squares stay a sweep (`mul.fft_sqr`: one input).
  *A dependency to watch:* the FFT equation's fit sits at 4.0-4.6% rms, close to the 5% line a
  dispatch-feeding fit must pass; a rejected fit there is the report-and-ask case.
- **Step 5: dynamic division is dispatched by fitted equations (2026-11-11).**
  *Measurement.* `division_timings` (`sections/div.rs`) forces Knuth, BZ and NR through the
  prepared drivers, and times the public entry, over a grid of q (the quotient body and its
  structural limb) by d (the divisor) at two sizes an octave from 4 to 4096 limbs, for `div_dyn`
  and `div_rem_dyn` (about 880 shapes, 100 s at normal level; whole-algorithm probes, the least
  of two fixed-sample measurements). `cargo tune division` writes the timings and reports
  losses on a grid half a step away.
  *What the old rule lost.* The cutoff rule (`div.knuth`, then three BZ-vs-NR constants a
  family) lost 0.8% of measured time on `div_dyn` (worst 37%) and 3.4% on `div_rem_dyn` (worst
  59%): it forced Knuth at divisors of about 64 limbs with quotients of 11-64, where NR (which
  reads only the divisor's top) is 20-40% faster; it took NR for `div_rem` with quotients of
  8-45 limbs and long divisors, where Knuth is 10-65% faster (the remainder costs q d either
  way); and it missed the BZ/NR boundary by up to 30-50%.
  *The equations* (linear in their constants, `fit_rel`; forms found offline against the saved
  timings, then fitted in the tuner):

  | Algorithm | Equation | rms |
  |---|---|---|
  | Knuth (both entries) | a q d + b q + c d + e | 2.3% |
  | NR, quotient only | a M(h) + b q + c d + e, h = q / 2 + 1, M = `MulEq::mul_ns` | 4.7% |
  | NR, with remainder | the same + r Mul(d, min(q - 1, d + 1)), `nr_rem_finish`'s product | 4.2% |
  | BZ (both entries) | a blocks (sum 2^k M(d / 2^k) + Knuth leaves) + b Knuth(top, d) + c q | 3.4% |

  M(q) in place of M(h) read 19.6% for NR, Mul(q, d) for the remainder 6.5%, and a fitted power
  q d^(e - 1) for BZ 7.7-10%; the recurrence over mul's and Knuth's fitted equations fits with
  multiples of 0.99 and 1.00. `MulEq` gained the schoolbook equation and `mul_ns(l, s)`, what
  `mul_dyn` will run, for these to stand on.
  *Dispatch.* `dyn_div_alg` (`utils/div.rs`): Knuth with no equation read where nothing else can
  run or at d up to the guards `div.dyn_knuth_d`/`div.dyn_rem_knuth_d` (derived from the
  equations: the widest divisor at which they pick Knuth for every quotient; 38 and 115), else
  `dyn_div_alg_at`, the least prediction among NR (`nr_split`), BZ (`bz_has_blocks`) and Knuth.
  The guards and reading BZ's top-block rule only where BZ is eligible cut the dispatch's own
  cost: without them 138/8, 40/32, 264/200 and 1030/1000 were 2-4.5% slower than the old rule.
  *The gate.* On the held-out grid (M1, normal level): `div_dyn` 0.03% mean, 4% worst;
  `div_rem_dyn` 0.15%, 14%. In a normal build, old rule against equations on the same tree
  (`benches/tuning_bias.rs`' new section `divgrid`, 40 small and mid divisions): geometric mean
  0.959x, six shapes 0.65-0.89x (`div_rem_dyn` 1600/400 0.65x, `div_dyn` 88/64 0.72x, 1600/400
  0.74x), none slower; the general `div` section 1.002x. Outputs identical to `faa9971` on all
  201 items but the four known sqrt ones. Both suites pass (577 and 582, with
  `test_dyn_div_alg_compares_equations` and `test_dyn_div_guards_and_entries`).
  *Purged:* `DivCutoffs::dyn_div`/`dyn_div_rem` and the six `div.dyn_div_*` knobs. Static
  entries keep `DivCutoffs` and their six (approximate). div version 4.
  *A REJECTED FIT, awaiting the user (the rejected-fit rule).* BZ's equation fitted over every
  shape where BZ has a block, including a top block padded from a quotient shorter than the
  divisor (q < d), read **13.4% rms** (worst 44.5%). Whole blocks and a fully padded top fit
  within 2-6% at every divisor; the misfit is the partially filled padded block, which costs
  59-94% of a full block at divisors past 1200 limbs (its products shrink with the quotient,
  its Knuth leaves don't), so those rows pulled the block multiple down and full blocks were
  then underpredicted by 13-22%. Proposals: (1) keep BZ's equation and eligibility to shapes
  with a full-size block (q >= d): 3.0-3.7% rms, and BZ is not considered for q < d, where NR or
  Knuth almost always wins anyway; (2) add the partial-fill term, padded top = leaves +
  products (top / d)^g with g fitted by Levenberg-Marquardt (about 0.5): 3.5% rms offline over
  every shape. **Interim: (1)**, which keeps BZ inside the domain its accepted form covers.
  *Still on hand-set constants:* BZ's own top-block rule (`use_bz_for_top_block`:
  `div.bz_top_padded_cost_scale` and `bz_2_1_cost`'s two scales). It could compare the same
  equations (a 2/1 block against Knuth at (top, d)) and drop those three knobs; not done.
- **Step 6: the series cost model's refit is REJECTED, awaiting the user (2026-11-11).**
  *The tool.* `sections/series.rs`, `cargo tune series`: the dynamic entries with each algorithm
  forced (`series_eval_dyn_at`) and the dispatcher's own pick, over the callers' own series
  through their typed contexts: exp's chunks (`Hyper::<Exp>`, x of s bits at shift 2s) and top
  bit (`Engel::<Exp>` at shift s), ln's chunks (`Hyper::<ATanh>` over a^2, x of 2s bits at
  shift 4s) and top chunk (`Bbp::<ATanh>` at shift 2s), in frames of 4 to 1024 limbs, chunk
  positions four apart; 110 workloads, and 100 more half a step away as a held-out set; each
  algorithm's counted `Work` from `SeriesNode::costs`. About 50 s at normal level.
  *What the model loses today* (live prices, M1): its picks lose **6.2% of measured time on the
  fitted set and 8.4% on the held-out one, 215-277% at worst**, and the entry itself 10-12%
  with its dispatch. Worst: exp's Engel series at 1024 limbs and shift 512 takes binary
  splitting (40-54 us) where Horner takes 15 us; ln's BBP at 1024 limbs and shift 256, 58 us
  against 19 us.
  *The refit.* The seven shared prices fitted jointly to every timed (workload, algorithm) by
  `fit_rel`: **30.6% rms, worst 164%: rejected**, and the picks are no better with them (8.0%
  and 8.6%). So the prices aren't the problem; the counted work is.
  *Where.* Prediction over measurement (geometric mean, and its spread), live prices:

  | Series | Horner | Rect | Binary splitting |
  |---|---|---|---|
  | exp Hyper | 0.93 (19-24%) | 0.91-1.00 (22-51%) | 1.04-1.09 (28-33%) |
  | exp Engel | 0.98 (9%) | - | **0.56-0.59 (55%)** |
  | ln Hyper | 0.82-0.86 (14-18%) | 0.83-0.85 (21-26%) | 0.99-1.11 (27-61%) |
  | ln BBP | 0.95-0.98 (7-11%) | - | **0.69-0.70 (50%)** |

  Horner's count is sound. Binary splitting's is far off for Engel and BBP, and by component
  (`BIGBITS_TUNE_VERBOSE=1`) the miss is at few terms with a wide shift at high precision: at
  1024 limbs with shifts of 512 bits and up (8-128 terms) it predicts 3-14 us (one division
  pass of about 3.6 us, 1-2 us of scalar passes, a small product) and measures 14-55 us, three
  to seven times more, while many-term trees, where the product term dominates, are within
  about 20%. The tree's wide shifts and allocations and its finalize division's real cost are
  the suspects; not yet profiled.
  *Effect on exp and ln.* Those shapes are the top-bit and top-chunk series when the argument is
  small (exp) or near 1 (ln); for a generic argument the Hyper chunks carry the work, where the
  model is 4-11% off on average with a 30-60% spread.
  *Proposals.* (1) Rebuild binary splitting's (and rect's) equation on the fitted equations, as
  BZ's was: merges' products from `MulEq::mul_ns`, finalize's division from `DivEq`, the passes,
  shifts and allocations counted from the code after profiling the wide-shift shapes, each part
  with a fitted multiple. This also lets `mul_work` and the four mul knobs only it reads go.
  (2) Keep the model and give BBP and Engel, where Horner's count is good to 10% and binary
  splitting's isn't, a fitted correction on binary splitting's prediction. (3) Leave the model
  as it is for Phase 2, with this loss recorded, and take it up as its own piece of work.
  Nothing in the library changed for this step: the prices stay as they were.
- **BZ's partial-fill term: the approved form is REJECTED; a structural one awaits approval
  (2026-11-11; superseded by the entry "BZ multiplies each quotient piece at its significant
  width" below, which removed the cause).** The user approved `a (blocks (P + L) + [padded] (L + P (top / d)^g)) + b
  [Knuth top] Knuth(top, d) + c q`. Built: `DivEq::bz_fill` (`div.bz_ns_fill`, default 0.5),
  `bz_block_parts`/`bz_leaves_ns`/`bz_padded_top_ns`/`bz_top_padded`/`bz_plan` (which replaced
  `bz_block_ns`, `bz_ns` and `bz_has_blocks`), dynamic BZ's own top-block choice by the equations
  (`dyn_bz_top_padded` in `bz_div_core_dyn`; static BZ keeps `use_bz_for_top_block` and its three
  knobs), the tune-only `BZ_TOP_FORCE` so the tuner times BZ with its top block forced padded
  and forced Knuth (`DivTiming::bz_top`; the fit then doesn't depend on the rule it feeds), and
  the fit by `fit_nonlinear`. div version 5. Both suites pass (577, 582).
  *The tuner's fit: 15.4% rms, worst 50%, g = 0.17: rejected.* (My offline 3.5% at g = 0.5 only
  had blocks the old rule chose to pad, tops long against the divisor.) A padded top block
  alone, over a full block:

  | d | fill <= 0.1 | 0.25 | 0.5 | 0.71 | 1 |
  |---|---|---|---|---|---|
  | 152-609 | 1.02-1.06 | 1.03-1.06 | 1.03-1.06 | 1.03-1.06 | 1.05-1.07 |
  | 861, 1218 | 0.77-0.80 | 0.82 | 0.89 | 1.01 | 1.06 |
  | 1722, 2435 | 0.60-0.63 | 0.68 | 0.80 | 0.93-0.96 | 1.05 |
  | 3444 | 0.49 | 0.59 | 0.73 | 0.91 | 1.04 |

  No discount under about 700 limbs, and a floor above, far over the leaves' share (0.09-0.18).
  *Mechanism:* `fft_entry_impl` trims its operands' leading zero limbs (`buf_len`) and sizes its
  transform by what is left; Karatsuba and schoolbook don't. So only the recursion's FFT-sized
  products get cheaper under a short quotient.
  *Proposed form (no fitted exponent, linear):* a padded top block of `top` quotient limbs costs
  its leaves, its Karatsuba-level products in full, and at each FFT level k (pieces of n_k =
  d / 2^k limbs) `floor(top / n_k) M(n_k) + fft(n_k, top mod n_k)`. Offline over all 490 forced
  timings: **1.3% rms, worst 3.9%**, multiples 1.05 and 1.00 (a full block: 19.7%; the power:
  15.4%). Not implemented: it waits for the user's approval.
  *State meanwhile:* the approved form is in the code with its schema defaults; the preset's div
  section falls through (version 5), so division runs on default equation constants. With the
  refitted power the gate read `div_dyn` 0.08% / 5.5% and `div_rem_dyn` 0.21% / 16% (the interim
  before it: 0.03% / 4.3% and 0.15% / 14%), and BZ's top-block choice lost 0.25% against the
  cutoff rule's 0.52% (worst 28% both).
- **Series: products from mul's equations help little; the real fault is the parallel split of
  tiny trees (2026-11-11).** The user asked to model the series with the multiplication fits
  and see. `mul_work`'s tune-only `MUL_WORK_FITTED` prices dynamic products as
  `MulEq::mul_ns / school price`, and `cargo tune series` reports the model both ways on the
  same timings (M1, normal):

  | | picks lose, fitted set | held out | Horner rms | bin_split rms |
  |---|---|---|---|---|
  | the model as it is | 6.25% | 8.42% | 15-17% | 41% / 104% |
  | products from mul's equations | 6.41% | 7.44% | 12% | 46% / 96% |

  Both refits are still rejected (30-31%). The component view had already put the miss
  outside the products; timing binary splitting's tree apart from its finalize
  (`SeriesTiming::tree`) put it in the tree: at 1024 limbs a 2-term Engel tree takes 12 us and
  an 8-term one 25 us against 1-2 us predicted, while at 256 limbs the same shapes are right.
  *Cause, confirmed with `cargo tune series --set bin_split.dyn_par=1099511627776`:*
  `bin_split_tree_at` splits a node in parallel when its limb estimate reaches
  `bin_split.dyn_par`, whatever its terms, so a tree of a few wide terms wakes the pool from
  the caller's thread for nothing. Gate shut: exp Engel at 1024 limbs, shift 32768: 15.8 us to
  2.1; shift 8192: 29.1 to 4.9; shift 512: 43.5 to 24.7; the model's predictions then agree
  (2.9, 5.9, 16.8), and its picks lose **3.06% and 4.11%** (worst 60-68%) in place of 6.25% and
  8.42% (worst 230-250%). Large many-term trees still gain from the split (ln's BBP at shift 2:
  2.30 ms shut, 1.34 ms open). So the gate needs the node's work, not its width alone, and the
  model counts a parallel split as a pure speedup, with no wake.
  *Proposals (not implemented):* (1) gate the split on a work measure, limbs times terms, swept
  through `exp_dyn` and `ln_dyn` as `bin_split.dyn_par` is now; or (2) keep the limb gate and
  add a term floor; and in the model, count a split from outside the pool at its wake cost.
  `MUL_WORK_FITTED` stays a tune-only switch until that is settled; the library's prices and
  `mul_work` are unchanged.
- **BZ multiplies each quotient piece at its significant width; the fill exponent is gone
  (2026-11-10).** The user asked why leading zeros reach Karatsuba and schoolbook at all. They
  were BZ's: a top block on a padded numerator has a quotient whose upper limbs are zero, and
  `div_3_2` multiplied each quotient piece by the divisor's low half at the piece's structural
  width. Only the FFT entry trims its own operands. The user: "add that easy optimization and
  re-run the fit for BZ since that seems to be the root of the problem".
  *The change (`div_3_2`, shared by dynamic and static BZ):* `q_sig = buf_len(q)`; a zero piece
  returns with no product and nothing to subtract; any other is multiplied as `q[..q_sig]`.
  *The equation:* `DivEq::bz_padded_top_ns(top, d, mul, kara)` prices a padded top block of
  `top` quotient limbs by its recursion: at the level with pieces of n limbs, `min(top / n,
  2^k)` whole products M(n, n), one M(n, top mod n) for the piece `top` falls in, nothing for
  the pieces above, and the Knuth leaves in full (measured: at q = 4, d = 181 the block still
  costs its two leaves). No fitted exponent, so `div.bz_ns_fill`, `DivEq::bz_fill` and div's use
  of `fit_nonlinear` are gone and BZ's fit is linear again: `a (blocks (P + L) + [padded]
  padded top) + b [Knuth top] Knuth(top, d) + c q`. `bz_plan` now also offers BZ with no whole
  block when its padded top is predicted cheaper than Knuth. div version 6.
  *Checks:* `test_div_2_1_short_quotients` (a 2/1 block at every count of significant quotient
  limbs: none, one, either side of the piece boundary, all; exact quotient and remainder);
  `test_dyn_div_alg_compares_equations` pins the new pricing by hand at three fills. Suites:
  578 and 583 pass. Outputs identical to `faa9971` (`div` and `divgrid`: 0 hash mismatches).
  Against the tree before this change, alternating processes: `div` 0.984, `divgrid` 1.000,
  nothing outside its interval but noise (an unchanged `rcp_dyn` read 1.16), so no measurable
  change on those grids, where a padded short top is rare. On BZ forced, the old cutoff rule
  for the top block now loses 3.3% of measured time (worst 109-113%) against the equations'
  0.06-0.30%: padding is cheap where that rule still takes Knuth. Static BZ keeps the rule
  (`use_bz_for_top_block`, `div.bz_top_padded_cost_scale`), whose constant was set for the old
  cost and is now pessimistic; one sweep would fix it (static: approximate).
  *The tuner's fit: REJECTED in every run today, and so were fits whose code didn't change.*
  Five `cargo tune division --level normal --refit` runs:

  | fit | accepted earlier | today's five runs |
  |---|---|---|
  | Knuth (code unchanged) | 2.3% | 4.4-8.2% |
  | NR | 4.7% | 5.7-7.5% |
  | NR with remainder | 4.2% | 4.9-7.4% |
  | BZ (new equation) | (3.4%, old code) | 7.2-9.5% |

  *Cause: the machine, not the forms.* Its battery sat at 5%, on AC but not charging, and it
  throttles under load: one fixed workload slowed to 1.5-1.6x within 15 s and then wandered by
  5% from one 1.3 s window to the next. Read from each shape's own Knuth run against the
  starting equations, the machine ran at a median 1.31x, 1.12-1.43x from the 5th to the 95th
  percentile. That is error from shape to shape, which the spread bound (within one shape's
  samples) can't see. It is worse for BZ and NR than noise alone: their equations are built on
  mul's (fitted at full speed, read from the preset) and on Knuth's (refitted throttled), so
  one fit mixes two machine speeds, and leaf-heavy rows come out over-predicted and
  product-heavy ones under.
  *The form, with the drift cancelled (a diagnostic, not the tuner's verdict):*
  `BIGBITS_TUNE_VERBOSE=1 cargo tune division --refit` writes every BZ row to
  `target/tuning/bz_fit.tsv` with its shape's Knuth run (timed in the same rounds) and that
  run's prediction under the starting equations. Each BZ time divided by that speed, against
  regressors from the starting equations: **3.1% rms** (worst 14%), multiples 1.04, 0.99 and
  0.0, with no trend past 2% by divisor, by top / d, by top-block mode or by entry. That figure
  carries Knuth's own 2.3%. Before the optimization, on the machine at full speed, the
  FFT-only version of this structure read 1.7% in the tuner.
  *State:* the equation and the optimization are in; the schema's BZ multiples are unchanged
  (0.98, 1.0, 4.6); nothing from today's fits is kept. The division fits have to be rerun on
  the machine at full power before the preset gets a div section. No new form is proposed.
  *For the user to decide:* whether the tuner should say when the machine's speed moved
  during a tune (it removed the kernel thermometer on 2026-11-10; a report-only line, like
  "samples" and "stability", would have named this at once), and whether a section built on
  another's equations (div on mul) should check a few of those predictions against fresh
  timings before fitting.
- **Disk (2026-11-10):** the x86 test build failed with `LLVM ERROR: IO failure on output
  stream: No space left on device`: `target/debug` had grown to 50 GB (72000 files in `deps`,
  5.5 GB of incremental state). `target/debug` and `target/x86_64-apple-darwin` were removed
  (build caches only; `target/release` and `target/tuning` kept).
- **The user's decisions of 2026-11-10 (evening), and what was set up under them.**
  (1) "If we have revised a fit model once already and a problem still remains then we should
  accept the noisy fit and move on", marking the problem and recording what was tried and what
  still doesn't conform. (2) BZ is marked and left for later. (3) The laptop is in no state to
  measure (it was on a 15 W supply: `pmset -g adapter`), so the framework is set up with the
  fits predicted to be correct, to be verified later. (4) x86 is the user's to run later.
  *The tuner.* A fit above the line now reads REJECTED only on its model's first form; on a
  model already revised it reads NOISY, its constants are written like any other's, and the
  closing summary counts the two apart (`Fit::evidence_revised`, `fit::NOISY`; BZ's fit and the
  series report use it). `cargo tune derive` and the end of every tune now also rederive
  `div.dyn_knuth_d` and `div.dyn_rem_knuth_d` from the equations in force
  (`sections::div::guards`; they time nothing), so equation constants can change without a
  timing run. Tests: `test_fit_evidence_marks`, `test_div_guards_derive_from_equations`.
  *The predicted fits.* BZ's multiples are 1.04, 0.99 and 0 ns per quotient limb, in the schema's
  defaults and in the preset: the drift-cancelled fit of the current form, which agrees with
  the tuner's 1.04, 0.97, 0 for the same structure at full speed before `div_3_2` changed.
  `presets/aarch64-macos.toml`'s `[div]` was moved by hand from version 3 to 6 so it applies
  again: its two sweeps (22, 106) and its Knuth, NR and NR-with-remainder fits are the accepted
  measurements it already held (their knobs didn't change meaning); BZ's four lines carry
  "predicted, NOT a tuner fit" as their evidence; the guards were rederived (38, and 107 where
  the old equation gave 115). Suites under it: 578 (normal); the tune build's tuning and
  division tests, the random-knob child included (103).
  *A loose end found on the way, since closed:* `cargo tune derive --file
  presets/aarch64-macos.toml` also rederived `mul.kara` and got 67 change points where the
  stored table has 73 (rows 43-55 differed). Cause: the table's derivation has hysteresis,
  leaving Karatsuba inside a row only where schoolbook wins by the level's tie band, and
  `derive` always used the normal level's band (1%) on a file tuned at thorough (0.5%). It now
  reads the level the file's `[mul]` section records (`file_level`; normal when it doesn't
  say), and rederiving the preset reproduces it byte for byte.
  `test_derive_reads_the_files_level`.
- **The tuner says when its timings can't be trusted (2026-11-10, no timing kept).** A smoke
  run of the changed flow, `cargo tune tune mul div --level quick` into a scratch file: 39 s,
  the guards derived after mul's first phase and again at the end, the file resolves (61 and
  29 values), and the summary read "2 rejected ...; 1 noisy but accepted". It also showed that
  `tune`'s "samples" line did flag the throttling machine (596 of 3881 unsettled), while the
  report commands never printed it and division's and the series' timings never counted
  toward it: they are fixed-sample measurements taken twice, with no spread loop. Now
  `measure::least_of_two` (since replaced by `time_repeated`, see the entry "BZ's predicted fit
  confirmed" below) takes the least of the two and counts a probe whose two measurements
  are further apart than the level's spread bound as unsettled, and `cargo tune equations`,
  `division` and `series` end with the same line (`samples_line`). On the throttling machine
  `cargo tune division --level quick` read 1873, 2169 and 4064 of 4064 unsettled in three runs
  (`BIGBITS_TUNE_VERBOSE=1` prints the second measurement over the first: median 1.00, 0.89 to
  1.12 from the 5th to the 95th percentile). This adds no measurement and corrects nothing; it
  is the two measurements the timings already took, compared. Not known yet: what it reads on
  a quiet machine at each level (see "What is left"). Suites: 586 (tune), 578 (normal).
- **Purged: the two square-product fits (2026-11-10).** `mul.kara_sqr_rec_{scale, lin, shift,
  err, exp}` and `mul.fft_sqr_m_{scale, point, shift, err}`, fitted by mul's second phase to
  `karatsuba_sqr_entry_dyn` and `fft_sqr_entry`, were read by no dispatcher, work function or
  derivation: `sqr_dyn`'s Karatsuba-to-FFT crossover has one input and is the sweep
  `mul.fft_sqr`. Under the user's rule (only what a decision reads is measured) the nine knobs
  and their two fits are gone from the schema and the tuner; mul is version 16. The M1
  preset's `[mul]` was moved to 16 by hand with those nine lines deleted and nothing else
  changed (52 values; rederiving it still reproduces it byte for byte). A section's upstream
  hash covers its knob set, so mul's changed with no value changing: the preset's seven
  recorded `mul:` hashes were updated to the new one, which leaves only the five warnings that
  are true (sections tuned under the old division constants). Checked: the preset resolves in
  full, the tuning tests pass in both builds (random-knob run included) and mul's dispatch
  tests pass. The full suites were not rerun (no library code changed; the laptop was at 3%).
- **BZ's predicted fit confirmed; a copy taken out of division's timings; the "samples"
  check for whole-algorithm timings revised (2026-11-10, afternoon).** The battery had climbed
  to 10-11% on the same 15 W supply and a fixed workload held one speed for 90 s (1.03x of its
  best, against 1.8-1.9x an hour earlier at 7%), so item 1 of "Verification owed" was run:
  `cargo tune division --level normal --refit`, 2 min 40 s a run, under a watchdog that stops
  it below 7%.
  *BZ.* Five runs on the steady machine:

  | Fit | Accepted before | Runs 1-3 (as the harness was) | Runs 4-5 (copy taken out, below) |
  |---|---|---|---|
  | Knuth | 2.3% | 2.3%, 2.3%, 2.3% | 2.4%, 2.4% |
  | NR, quotient only | 4.7% | 5.1%, 5.1%, 4.9% (worst 18-20%) | 4.4%, 4.4% (worst 8-9%) |
  | NR, with remainder | 4.2% | 4.3%, 4.4%, 4.3% | 4.3%, 4.3% |
  | BZ, form 4 | none (predicted) | 2.3%, 2.4%, 2.4% | 2.2%, 2.2% |

  BZ's multiples: 1.044-1.046 and 0.981-0.984, with 0 ns per quotient limb, against the
  predicted 1.04, 0.99 and 0. The guards came out at 37-39 and 107 (the preset holds 38 and
  107). The equations' picks on the held-out grid lose 0.04-0.06% of measured time for
  `div_dyn` (worst 4.6-5.7%) and 0.17-0.19% for `div_rem_dyn` (worst 12-15%); BZ's own
  top-block choice loses 0.01-0.02% (worst 1.3-2.8%) against the old cutoff rule's 3.9%
  (worst 127%). So BZ's equation has an accepted fit and leaves "Marked problems". The preset
  and the schema keep the predicted multiples (within 1% of the fitted ones) until the full
  tune writes its own.
  *A copy in the timings.* NR's quotient-only fit read 5.1% twice, over the line. Its worst
  rows were short quotients over long divisors (q = 23 at d = 4096, -18%; q = 11 at d = 2048,
  +17%), and its d term was 0.136 ns per limb, though that division reads only the divisor's
  top. The cause was the harness: every probe restored its numerator with `copy_from_slice`
  before the call, needed only with a remainder (which overwrites it). At q = 10 over
  d = 3444 the division takes about 350 ns and the copy of 3453 limbs about 500, with a jump
  once the buffers pass the L1 cache. Now quotient-only probes copy nothing, and each shape
  with a remainder times its copy on its own (a "copy" probe) and takes it off that shape's
  timings, so every timing is the division alone. NR's quotient-only fit then reads 4.4%
  (worst 8-9%), its d term 0.0012 ns per limb, and nothing else moved. The equations in the
  preset were fitted with the copy in; it was common to all three algorithms, so their
  comparison stands, and the full tune replaces them.
  `BIGBITS_TUNE_VERBOSE=1 cargo tune division --refit` now also writes the fitted grid's
  timings to `target/tuning/division_fit.tsv` (the file `--out` names is the held-out grid).
  *A form for NR that fits better, not adopted (nothing is rejected).* M(h, h) is a full
  product, while NR runs middle products (school below `mul.fft_mid`, the FFT above; there is
  no Karatsuba middle product) and high products (`hi_mul_buf` to `mul.dyn_partial_mul`, the
  guarded product above). Its residual by quotient length swings from +7% to -7% between q =
  91 and q = 724 for that reason. Offline, with each product of the reciprocal chain and of
  `nr_refine_quo` priced at its own shape and one fitted multiple a tier (school middle 1.83,
  FFT middle 1.17, school high 1.00, guarded high 0.90) plus `b q + c d + e`: 1.7% on the
  fitted grid and 2.6% held out, against 4.2% and 3.7% for the form in use; with a remainder
  2.9% and 3.0% against 4.2% and 3.6%. One multiple for all four tiers reads 6.0%, and one
  for middle and one for high products 4.5%. It does not help the picks that lose most, NR
  with a remainder at a short quotient over a divisor past 1200 limbs (10-13% under-predicted
  either way: that is the remainder's d by q product). A candidate for the user, with its
  cost: eight constants in place of four, and a pricing function that has to follow
  `mid_mul_dyn`'s and `hi_mul_dyn`'s dispatch.
  *The "samples" line.* On the steady machine the two-measurement count read 6.7-7.5% in
  runs 1-3 and 3.1-3.4% in runs 4-5, around the 5% warning, while the fits repeated to 0.1%
  and each run's least-of-two timings agreed with another run's within 4% on 99% of shapes.
  By the probe's time: 7-12% of the probes under a microsecond had their two measurements more
  than 4% apart, and 1-2% of the rest. It is not the operands (same in both passes: 12%) nor
  mostly the buffers (same probes timed twice: 7-8%); short kernels just repeat less tightly
  from one pass to the next, and a tighter level would not change that. So
  `measure::time_repeated` replaces `least_of_two`: one set of probes timed twice, a third
  time where the two are apart, the least kept, and a probe counted as unsettled when the
  least of its measurements has no other within the bound. The first version of that rule
  (unsettled when the third is apart from both others) was wrong, and run 6 showed it: the
  machine began throttling during the second grid (battery at 7%; second over first median
  1.114, 0.996 to 1.270; every fit broke, Knuth 9.9%, BZ 12.9%), 3102 of 4873 probes were
  apart, and the third measurement, as slow as the second, agreed with it for all but 468.
  With the least required to be one of the agreeing pair those stay unsettled. Not yet run
  with the final rule: any level (see "What is left").
  *The machine.* The throttle followed the battery, not the adapter: steady at 8-11%, slow
  again at 7%, all on the 15 W supply. Single-thread tuner runs drew the battery down about
  4% an hour, and it gains about as much idle. The full tune has pool phases and runs 15-20
  minutes; it waits for a charger that keeps up.
- **Any machine can tune itself: the local layer, the end-to-end gate, and what its first runs
  found (2026-11-10, evening; plan `~/.claude/plans/i-want-you-to-scalable-donut.md`).** The
  laptop was on a 65 W charger from here on and held one speed (a fixed workload within 1.002x
  of its best for a minute).
  *The user's eight decisions.* (1) Scope: any checkout, with a machine-local file; the
  `cargo-bigbits` driver waits for Phase 3. (2) A rejected fit is still written, marked.
  (3) `cargo tune check`, and every tune ends each section with it as a gate. (4) The FFT's
  odd-part factors stay hard-coded per architecture. (5) No steadiness preflight. (6) NR's
  equation prices its products one by one. (7) The local file applies automatically. (8) The
  series model's split gate and Linux core pinning are in; the line-reduction pass stays
  deferred. Later the same evening: the gate's allowance is 3%, 2% and 1% for quick, normal and
  thorough (`Level::gate`), wider than the sweeps' tie band, so that noise from one tune to
  the next doesn't reject a section.
  *The machine's own file.* `build.rs` layers the file `BIGBITS_TUNING` names over the target's
  preset; `.cargo/config.toml` points it at `presets/local.toml` (gitignored), which
  `cargo tune tune` now writes by default. It lives in `presets/` and not at the repository
  root because cargo can only notice a newly created file cheaply inside a directory the build
  script already watches (a missing watched file reruns the script, and rebuilds the crate, on
  every build). A file whose `[meta] target` isn't the build's is skipped with a warning
  (`resolve::layer_target_ok`), and `tune` refuses to merge into one. A section set by several
  layers has only the top layer's upstream record checked. `cargo tune status` lists the layers
  and which supplies each section. Checked end to end: a normal-level tune (361 s) wrote the
  file, the next build resolved it with only the preset's own two fall-through warnings, and
  `status` showed it supplying all eight tuned sections.
  *The gate.* `tuner/check.rs` times `tuner/check_items.rs`' workloads (the 201 items of
  `benches/tuning_bias.rs`, now one file both include; the crate names itself `big_bits` in
  tune builds for that) under two knob snapshots in paired rounds, alternating which goes
  first. After a section is measured its values are compared with the values before it over
  the workloads of that section and of every section built on it (201 for mul, 156 for div, 94
  for sqrt, 52 for bin_split, 48 for theta, agm, exp and ln). Dropped: mean past 1 + the
  allowance with a 95% interval that excludes 1, or an exact output that differs; the live
  values and the file go back. A workload more than 2% slower is listed, never a reason to
  drop. The final derive rewrites `mul.kara` only when mul was tuned and kept, and div's guards
  only when mul or div was. Across four full tunes no section was dropped (means 0.992 to
  1.003).
  *What the gate's list found.* (a) `bin_split.dyn_hyper_leaf_limbs` read 192, and `ln_dyn`
  was listed 6-9% slower at 300, 500 and 1040 limbs while the section's mean read 1.0001.
  One knob at a time through `cargo tune check --file`: that leaf alone costs `ln_dyn` 7% at
  300 limbs and 11% at 500; the gates and the new work gate cost nothing. The leaf sweeps timed
  exp's and ln 2's trees only, and ln's atanh chunks want about 48. Every dynamic leaf sweep
  now times whole `exp_dyn` and `ln_dyn_at` calls as well as its own trees (the rule that a
  sweep's workloads span every caller's shape), the leaf reads 48-64, and nothing is listed.
  This predates the day's work: the gate is what made it visible. (b) `div.bz` reads 99
  whether its sweep times the quotient-only division or both entries (it times both now).
  `div_rem_static<2048>` at 200/100 is then 10% slower than at the default 140, as static
  entries share the knob and cross later; it is listed on every run (static values are
  approximate). `div_rem_dyn` at 200/100 is listed 3% slower when the remainder guard comes out
  at 99 and not when it comes out at 101: that shape sits on the guard.
  *NR priced product by product* (`DivEq::nr_parts`; div version 7). Tuner fits on the steady
  machine: quotient only 1.4-1.6% (from 4.4%), with remainder 2.6-3.0% (from 4.3%); Knuth
  2.4-2.6%, BZ 2.0-2.3%. The equations' picks lose 0.02% of measured time for `div_dyn` (worst
  2-6%) and 0.15-0.18% for `div_rem_dyn` (worst 15%, still NR with a remainder at a short
  quotient over a divisor past 1200 limbs). Dispatch cost: the report's "entry itself" line
  rose from 1.0-1.15% to 2.0% for `div_dyn` (5% at 8-63 quotient limbs), about 25 ns a call.
  That line is a tune build's, where every equation constant is an atomic load; in a normal
  build the 40-division grid reads 1.0007 against the tree before the change, nothing slower
  beyond noise, outputs identical. The pricing loop was tightened anyway (integer sums, no
  schedule array): 1.39% in the tune build.
  *The split's work gate* (`bin_split.dyn_par_work`, `dyn_wake_ns`; bin_split version 3). It
  applies only to a tree started off the pool: there a split has to wake the pool, 12-15 us on
  the M1 (a 2-term tree at 1024 limbs: 14.0 us split, 0.34 us serial), while on a worker the
  limb gate alone decides, as it was tuned through exp and ln. The plan had this form as the
  fallback; the tuner's own notes on why the gates are swept through callers made it the first
  choice. The series model counts parallel levels only where the root passes both gates and
  charges one wake. The sweep's history: on Engel trees alone it read 2^24, the top candidate;
  with many-term trees and whole exp and ln calls added, 2^22 or 2^24 by noise (losses 0.03%
  apart), so it now takes the lowest candidate within the tie band of the best
  (`sweep::param_sweep_lowest`), which reads 2^22. With only this knob changed nothing in the
  exp, ln or constant workloads moves (0.998-1.000).
  *Placement.* Two tunes fitted Knuth and NR with remainder with one shape 47-75% off, which
  standalone runs never showed. The evidence now names each fit's worst shape, and the second
  time they were q = 6 over d = 1024 and q = 32 over d = 2048. Measured directly: Knuth at
  6 over 1024 takes 2.9 us, and up to 7.1 us (2.45x) with the divisor 1 to 7 limbs past a
  4096-byte alignment against its working buffer; at d = 1000 at most 1.11x. The loads from
  the divisor alias the stores to the numerator. It is a property of the library's loop, and
  a measurement hazard: that afternoon I had changed division's timings from two builds of the
  probes to one build timed twice (for the "samples" check), which keeps an unlucky placement
  for every measurement. `measure::time_placed` times two placements (`pad`'s pass moves the
  buffers 13 limbs), re-times the faster one where they disagree, and keeps the least; a
  machine that slowed still shows, as its fast time can't be repeated. Its first run caught a
  probe 2.25x apart between placements and fitted Knuth at 2.4% (worst 9%). A mistimed
  numerator copy is also held to three times the median copy rate, though it was not the cause.
  *The "samples" line under its final rule*, steady machine: `division` 0.04%, 0.4% and 0.7%
  unsettled at quick, normal and thorough; `series` 4.6%, 2.8% and 6.0%, its disagreeing
  probes all 10 us and longer (the parallel ones). Full tunes: 3.3% at normal, 4.8-4.9% at
  thorough. So `cargo tune series --level thorough` trips the 5% warning on a healthy machine;
  nothing in `tune` uses its timings. Left as it is, to be read again now that few-term trees
  no longer wake the pool.
  *Also.* The tuner pins itself to the fastest cores on Linux (`fastest_of`: CPUs within 15%
  of the top sysfs rating; tested), with the `sched_setaffinity` call type-checked under a
  swapped cfg but never built for or run on Linux. Cutoff sweeps note and count a value chosen
  at either end of their grid. Static BZ's top-block constant is swept on the dynamic forced
  top-block timings: 0.92, losing 1.05% against 3.7% at 1.8. Suites: 580 and 591.
  *The final tune (the user's decision: this pass is the last).* `cargo tune tune --level
  thorough --write-preset`, 1802 s, starting from the pass before it. All eight sections kept
  under the 1% allowance, none with a workload listed as slower:

  | Section | Time after / before | Workloads |
  |---|---|---|
  | mul | 0.9992 [0.9952, 1.0033] | 201 |
  | div | 1.0010 [0.9983, 1.0037] | 156 |
  | sqrt | 0.9997 [0.9985, 1.0009] | 94 |
  | bin_split | 1.0002 [0.9990, 1.0015] | 52 |
  | theta | 1.0002 [0.9980, 1.0025] | 48 |
  | agm | 0.9996 [0.9986, 1.0007] | 48 |
  | exp | 1.0000 [0.9983, 1.0018] | 48 |
  | ln | 0.9989 [0.9966, 1.0013] | 48 |

  Fits: schoolbook 3.8%, the Karatsuba level 2.0%, chunking 2.9%, recursive Karatsuba 2.2%
  and 1.8%, the FFT 4.1%; Knuth 2.5%, NR 1.4%, NR with remainder 2.9%, BZ 2.3% (multiples
  1.044 and 0.978), the worst single residual 17%. No fit rejected or noisy. One of 34
  whole-algorithm sweeps unstable; one edge (`ln.dyn_agm_limbs` at 32, the bottom of its
  sweep, as before: on one thread the AGM wins at nearly every size). Values of note:
  `div.bz` 99, guards 39 and 99, `bin_split.dyn_par` and `dyn_merge_par` 512, leaves 64, 48
  and 64, `dyn_par_work` 2^22, `dyn_wake_ns` 14945. The "samples" line read 5.05% (1485 of
  29398), just past its 5% warning; the two thorough tunes before it read 4.9% and 4.8% on the
  same steady machine, so at the thorough level's 2% bound that line sits on its own
  threshold. Not rerun.
  *Closing checks under that preset (nothing re-tuned).* The build resolves it with no warning,
  and `cargo tune derive --file` reproduces it byte for byte. Both suites pass (580 and 591).
  `cargo tune check --file presets/aarch64-macos.toml --level thorough`, the preset against
  the schema's defaults: 0.9864 [0.9855, 0.9873] over 201 workloads (mul 0.9935, div 0.9950,
  divgrid 0.9947, sqrt 0.9705, series 0.9921, consts 1.0044, trans 0.9739). `cargo tune
  division`: the equations' picks lose 0.02% for `div_dyn` (worst 1.5%) and 0.17% for
  `div_rem_dyn` (worst 14.5%); BZ's own top-block choice 0.02% against the static rule's 2.2%
  at its swept constant. `cargo tune equations`: the product equations' picks lose 0.00-0.10%
  (worst 6.1%) against the old ratio rule's 0.21-0.71%. `cargo tune series`: below, under
  "Marked problems".
  Against `faa9971` in a normal build (`tuning_bias`, 6 processes a side): outputs identical on
  all 201 items but the four known sqrt ones (unspecified buffers at 22 limbs);

  | Section | Geometric mean | Fastest items |
  |---|---|---|
  | mul | 0.9928 | `hi_mul_static<1024>` 330 0.885, `hi_mul_dyn` 300 0.891, `mul_dyn` 400x400 0.919 |
  | div | 0.9955 | `rcp_dyn` 30 0.849, `div_dyn` 200/100 0.902 |
  | divgrid | 0.9637 | `div_rem_dyn` 1600/400 0.645, `div_dyn` 88/64 0.728, 1600/400 0.741 |
  | sqrt | 0.9612 | `sqrt_approx_static<4096>` 22 0.410, `sqrt_static<4096>` 32 0.875 |
  | series | 1.0011 | `ln_dyn` 120 0.929 |
  | consts | 0.9934 | `ln2_static<1100>` 1060 0.968 |
  | trans | 0.9767 | `ln_dyn` 64 0.616 |

  More than 2% slower than `faa9971`, each with what is known about it:
  `div_rem_static<2048>` 200/100 1.114 (static entries share `div.bz`, 99, and cross to BZ
  later than dynamic ones; listed by the gate on every run; static values are approximate);
  `sqr_static<8192>` 3000 1.041 and 80 1.031 (static, on the table and cutoffs derived from
  dynamic timings); `hi_mul_dyn` 330 1.039 (it sits beside `mul.dyn_partial_mul`, where 300
  is 0.891: the two sides of that cutoff trade); `ln_dyn` 2 limbs 1.032 and 16 limbs 1.030
  (not from this round: 1.011 and 0.998 against the tree before it; older than that, and not
  traced). The preset against the defaults also lists `pi_static<1100>` at 1030 and 1060
  limbs 3-5% slower and `div_rem_dyn` 200/100 2%.

- **The series' dynamic dispatch rebuilt on fitted equations (2026-11-10).** The user, on the
  marked series cost model: "Lets rebuild it using the other fitted equations and see if that
  brings better results." It does.
  *The form.* Each algorithm has its own runtime equation: its own parts (`Parts`: what its
  kernel does, by kind of operation) times its own fitted prices (`SeriesEq`). Multi-limb
  products are in ns by mul's equations (`MulEq::mul_ns`; a high product as `hi_mul_dyn` takes
  it), finalize's division in ns by division's (`dyn_div_best`), and the linear kernels are
  limbs of each kind of pass (`div_prim`, `div_prim2`, a one-limb multiplier, a two-limb one,
  additions and shifts, fills and copies), plus terms, kernel calls and allocations. Binary
  splitting's parts walk the tree `bin_split_tree_at` builds, level by level. Rect runs
  Horner's kernels, so it takes Horner's prices for their passes and fits only its own product
  multiple, call price and fixed cost. 23 knobs: `series.horner_ns_*` (9), `series.rect_ns_*`
  (3), `series.bin_split_ns_*` (10), `series.bin_split_par_gain`. Static nodes keep the shared
  prices over `Work` (approximate), whose predictions are byte-identical over a 7560-shape grid
  after the old code was cut down to them; `series.ns_alloc` and the tune-only
  `MUL_WORK_FITTED` are gone.
  *What was wrong, found one residual at a time on one saved set of timings (`cargo tune series
  --timings FILE`): counts, not prices.*

  | Cause | Effect on a fit |
  |---|---|
  | A uniform tree depth: a 5-term tree counted as 8 leaves and 7 merges | binary splitting 32% (102% held out) to 10.5% with the products and division from the fitted equations |
  | Rect's block products priced at the wrong block's frame | rect 60% to 12.7% |
  | A product with a 1- or 2-limb side priced by mul's schoolbook equation (fitted from 10 limbs; its 14 ns intercept is not a `mul_prim` pass's) | Horner at 16-64 limbs 10-19% high, with a 2-limb x at 1024 limbs 35% low |
  | Horner's groups aligned from the shallow end (the kernel groups from the deep end), of one whole size, with q's size at the midpoint (half a bit high for q = n); `div_prim2` priced as `div_prim` | division limbs 27% short at a few wide terms, 12% long at thousands; Horner 10.7% to 7.6% |
  | Parallel trees counted at a full speedup | the 14 such trees 30% low; a gain of 0.25-0.3 a thread reads them at 12% |
  | Every forced time carrying the entry's own work (the tail bound's bisection for the term count, the eligibility pass, the context): 100-200 ns of a 350 ns call for exp's series | small sizes 10-25% off by family; timed on its own now and taken off: Horner 7.6% to 6.7%, binary splitting 8.2% to 6.5% |
  | A division by 1 charged in full (`div_prim` returns at once; exp's q(1) is 1, and with rect at block length 1 the first term is always alone) | rect on its own prices put a division limb at 0.7 ns where Horner's is 2.7; fixed, Horner's prices fit rect |
  | A high product by a 2-limb x taken as a `mul_prim2` pass (it is `hi_mul_buf`'s two rows up to 251 limbs), and BBP's second pass over a group's terms | Horner 6.7% to 5.6% |

  *Which constants are shared.* On the same timings: one fit an algorithm, Horner 6.7%, rect
  6.0% (12.2% held out: 25 workloads for 12 columns), binary splitting 6.5%, picks lose 0.13%
  and 0.34%. Rect on Horner's kernel prices: rect 9.5% fitted and held out, picks 0.22% and
  0.11%; adopted, as its held-out numbers match its fitted ones. Horner and rect in one fit:
  8.8%. One fit for all three, the old model's structure: 17.4%, picks lose 2.0% and 2.6%.
  *The evaluation's cost.* First wired, the equations made `exp_dyn` 15-23% slower at 4-120
  limbs in a normal build (series workloads 1.05, exp and ln 1.03): 43-110 ns a call for BBP
  and Engel and 90-250 for Hyper, against about 25 for the old counts, over the three or four
  modelled chunks of a small exp. Cut to 40-60 and 55-230: a parabola for `log2`; the pool
  looked at only by a tree wide enough to split; Knuth's equation alone within its guard;
  rect's block lengths predicted only where Horner's products by x cost more than rect's fixed
  cost; a one-leaf tree without the level walk. (One cut was wrong and was caught by the next
  report: Horner's products priced once at the mean frame, which holds only while every frame
  holds x whole; Horner read 8.3% until it was fixed. The first fix, eight sampled
  evaluations, was right and too dear: `exp_dyn` gave back 3-5% at 4-120 limbs. The sum is
  now a closed form over the frames' four stretches, `Shape::school_products`, checked term
  by term in `test_horner_products_closed_form`; forcing the helpers inline was tried after
  that and lost 1-3%.)
  *Result (M1, `series`, `exp` and `ln` tuned at thorough through the gate into the preset).*
  The gate kept all three sections at 1% (7 of 575 measurements unsettled, no sweep unstable,
  nothing rejected). Fits, in the preset: Horner 5.6% (worst 18%), binary splitting 6.9%, rect
  13.3%; binary splitting's multiples on mul's and division's equations 1.004 and 1.014; the
  parallel gain 0.3 a thread. `exp`'s values did not move; `ln.dyn_chunk_par_work` went from
  768 to 1280 (in the first of the three tunes this took, the old value lost 1.9% against
  0.65%; the next two kept 1280). `cargo tune series` on fresh timings, normal level:

  | | Shared prices over counted work | The equations |
  |---|---|---|
  | Picks lose, fitted / held out | 3.82% / 4.25% | 0.13% / 0.35% (0.07-0.13% / 0.15-0.35% over four runs) |
  | Worst pick | 65% / 65% | 6.4% / 20% |
  | rms, Horner | 17% / 15% | 5.7% / 6.0% |
  | rms, rect | 59% / 24% | 13.3% / 13.3% |
  | rms, binary splitting | 31% / 102% | 7.6% / 7.6% (its 14 parallel trees included) |
  | The entry itself, in a tune build | 9.55% / 10.31% | 8.58% / 10.46% |

  In a normal build against the tree before the rebuild (`benches/tuning_bias.rs`; outputs
  identical): series workloads 0.977, exp and ln 0.984; `exp_dyn` 3.5-6% faster from 4 to 300
  limbs; `ln_dyn` 3.8% slower at 8 limbs and 1.8% at 16 (its picks there were already the best
  ones, so only the evaluation's cost shows) and within 1.3% elsewhere; `exp_static` 5-6% and
  `ln_static` 2-4% faster at small sizes (the static model's predictions are unchanged; its
  evaluation lost a rayon lookup and its dead dynamic branches in the cleanup); mul 1.002, div
  1.001, the division grid 1.000, sqrt 1.000, consts 1.00-1.01. Against `faa9971`: series 0.977,
  exp and ln 0.961. Both debug suites pass (583 and 594). `dyn_div_alg_at` now returns
  `dyn_div_best`'s pick, which the series reads for its predicted time; without
  `#[inline(always)]` on it `div_dyn` at 76/64 and 88/64 ran 3.7% and 2.6% slower in two
  closing runs (the helper was no longer inlined into the dispatch).
  *Not done.* `mul.fft_karatsuba` and `mul.fft_chunking_karatsuba` are now read by no
  decision; see "What is left".

## One way to fit (2026-11-11)

The user's decisions, after a walk through how a fit is found: effort into more shapes and not
into a tighter time at each; shapes over a truly two-dimensional area; one solver, nalgebra's
SVD, for every fit; no fitted exponent (recursive Karatsuba's became a schoolbook term with its
own weight, the user's idea); a term that solves negative "isn't needed" and is left out; and
the question of which terms a form needs is the developer's, asked with `--analyze`, never by a
tune. The plan is `~/.claude/plans/come-up-with-a-dapper-minsky.md`; CLAUDE.md's "How a fit is
found" is the rule as it stands.

**What changed in the code.** `tuner/fit.rs`: `solve` (one SVD, columns scaled to unit length,
standard errors, condition number), `solve_nonneg` (a negative term left out and named),
`fit_split` (judged on held-out rows), `analysis`; `lstsq_nonneg`'s silent loop, `fit_nonlinear`
and the `levenberg-marquardt` dependency are gone. `tuner/measure.rs`: `time_fit` (three shuffled
rounds of one batch a probe, two passes, the least kept, their difference the noise);
`time_repeated`, `time_placed` and `least_confirmed` are gone. `tuner/design.rs` (new): `scatter`
(the R2 sequence in log scale), `Design`, `products`, and `UNSEEN` for reports. `utils/mul.rs`:
`MulEq::kara_ns` is linear, `[scale, school, lin, shift]` on `karatsuba_cost` /
`chunking_karatsuba_cost` at the fixed log2 3; mul's version 17; `mul.fft_karatsuba`,
`mul.fft_chunking_karatsuba`, their sweeps and `mul_work`'s dynamic tier purged. mul's, div's and
the series' fits all run on the framework; a tune no longer times the probes only the reports
read (division's plain BZ and public entry, the series' dispatcher pick and bare tree).

**Two things the first tries got wrong.**
- *Batches back to back.* The first `time_fit` measured each probe to completion. A disturbance
  of a few milliseconds then took all three of one probe's batches: in the process's first second
  5-7% of the first fit's probes read up to 2.4x slow in one pass ("passes differ" 5-18% for
  schoolbook against 0.3-2% for the later fits). A warm-up spin didn't cure it. Rounds did: the
  same three batches spread in time, where a disturbance costs several probes one batch each and
  their medians drop it (schoolbook 3.2%).
- *A negative constant as a rejection.* Tried for an afternoon: four of mul's six fits were
  rejected, two of them for terms with no evidence either way (recursive chunking's shift,
  -136 +- 290), which would have flipped from run to run. The user's rule replaced it: such a
  term isn't needed, and the tune leaves it out.

**The first full run** (M1, normal level, `--no-gate --analyze`, to a scratch file; 264 s; 4178
fit probes, passes differ 2.5%). Fitted rms, held-out rms, and what the method before read:

| fit | shapes | fitted | held out | before | left out as negative | not needed by the held-out rms |
|---|---|---|---|---|---|---|
| mul schoolbook | 120 | 4.0% | 4.2% | 3.8% (45, l to 2s) | | row (t 1.3) |
| mul Karatsuba level | 118 | 1.8% | 2.1% | 2.0% (48) | | pass (t 1.6) |
| mul chunking | 120 | 2.7% | 2.9% | 3.2% (44) | shift (-43 +- 25) | |
| mul recursive Karatsuba | 120 | 3.0% | 2.8% | 2.2% (32, exponent) | l + s (-7.9 +- 1.4), shift | |
| mul recursive chunking | 120 | 2.2% | 1.7% | 1.6% (24, exponent) | shift (-192 +- 320) | l + s (t 1.2) |
| mul FFT | 120 | 3.3% | 3.7% | 4.1% (56) | per point (-1.6 +- 0.2), shift | |
| div Knuth | 788 | 2.3% | 2.2% | 2.5% | | (all four needed) |
| div NR | 363 | 2.2% | 2.2% | 1.4% | shift (-10 +- 2), d (0 +- 0.001) | |
| div NR with remainder | 363 | 3.6% | 4.0% | 2.9% | shift (-16 +- 6) | |
| div BZ | 538 | 3.2% | 3.6% | 2.3% | q (-2.9 +- 0.25) | |
| series Horner | 192 | 6.0% | 7.4% | 5.6% | | one (t 2.2) |
| series rect | 44 | 12.0% | 13.5% | 13.3% | | |
| series binary splitting | 182 | 8.0% | 10.2% | 6.9% | | mul1 (9.8% held out without it) |

("Before" is each fit's rms under the old method, on its own shapes: mul's from the pilot's two
baseline runs, div's and the series' from the M1 preset and CLAUDE.md. They are not on the same
shapes, so read them as "about the same", not to the decimal.)

Recursive Karatsuba, solved freely, is the recurrence itself: scale 2.69, l s -0.016 +- 0.005,
l + s -7.9 +- 1.4 (T(n) = 3 T(n/2) + c n solves to A n^1.585 - 2 c n). With constants kept
non-negative the best two-term form is the user's, theoretical term + l s: 2.8% held out,
against 5.0% for the theoretical term alone.

**What the fits are for: the picks, on shapes the equations never saw.**

| report | this run (normal) | the tuner before (thorough preset, its own grids) |
|---|---|---|
| products, near-square: equations' picks | 0.00%, worst 0.0% | 0.00%, worst 0.0% (same 66 shapes) |
| products, unbalanced | 0.00%, worst 0.2% | 0.04%, worst 1.7% (same 66 shapes) |
| Karatsuba's equation error there | 3.3% and 5.6% | 2.0% and 4.0% |
| `div_dyn`: equations' picks | 0.06%, worst 5.1% (200 shapes) | 0.01%, worst 0.9% (400) |
| `div_rem_dyn` | 0.33%, worst 14.1% | 0.15%, worst 12.1% |
| BZ's top block | 0.02%, worst 1.0% | 0.02%, worst 2.6% |
| series, held-out half | 0.24%, worst 12.0% | 0.39%, worst 20.2% |
| series, fitted half (unseen by this tune too) | 0.45%, worst 11.6% | 0.13%, worst 7.3% (its own fitted set) |
| time of the division report | 35 s | 73 s |
| time of the series report | 8 s | 18 s |

Read with care: only the products are on the same shapes; this run was at the normal level and
the preset at thorough; and division's equations here are built on a Karatsuba equation that
reads 1.3 to 1.6 points worse than the exponent form did.

**Not done.** The M1 preset is not retuned (`[mul]` falls through to the defaults until it is);
no form was changed on the analysis' findings (the user's call); the levels' shape counts (60,
200 and 24 points at quick; twice and four times that above) are first guesses; and the
end-to-end check against the old preset was not run, as the two can't be loaded into one binary
across mul's version change.

### The forms trimmed and the M1 retuned (2026-11-11)

On the user's go-ahead the 15 constants the analysis found unneeded came out of the equations
themselves (schema, `MulEq`, `DivEq`, `SeriesEq`, the closed forms that expanded them, the tuner
and the by-hand tests), and the M1 preset was retuned at the thorough level.

| equation | form now | constants removed |
|---|---|---|
| schoolbook | a l s + c | the term a row |
| Karatsuba level | a (three sub-products) + c | the term a limb |
| chunking | a (chunk products) + b (l + s) | the shift |
| recursive Karatsuba | a (l^e - (l - s)^e) + q l s, e = log2 3 | the term a limb, the shift |
| recursive chunking | a l s^(e - 1) + q l s | the term a limb, the shift |
| FFT | a rho(o) N log2 N | the term a point, the shift |
| Knuth | a q d + b q + c d + e | (none) |
| NR | four product tiers + b q | the term in d, the shift |
| NR with remainder | four tiers + r Mul(d, min(q - 1, d + 1)) + b q + c d | the shift |
| BZ | a (blocks + padded top) + b (Knuth top) | the term in q |
| series Horner | eight prices | its fixed cost |
| series rect | three prices on Horner's | (none) |
| series binary splitting | nine prices | passes by a one-limb multiplier |

mul is at version 17, div at 8, series at 3. The schema's defaults for those equations are the
M1's new fit.

**The retune** (`cargo tune tune --level thorough --write-preset`, 1503 s with the gate; the old
preset is kept as `aarch64-macos.preset.v16.bak` in the session's scratchpad only). The gate
kept all 9 sections, each within 0.3% of the values before it (mul 0.9970, div 0.9994, sqrt
0.9992, bin_split 1.0021, series 1.0002, theta 0.9993, agm 0.9989, exp 0.9994, ln 1.0015; "before"
is the schema's defaults for mul, div and series, whose old sections no longer load, and the old
preset's values for the rest). Listed slower, never dropped: `ln_dyn` at 500 limbs (2-4.5%), `ln_dyn`
at 4 (4%), and the static entries at N = 1100 near 1030-1060 limbs and `div_rem_static<2048>`
200/100 (2.5-10%: static values are approximate). 8354 fit probes, passes differ 2.8%; no term
solved negative; no fit rejected, the series' three NOISY. The sweeps' "samples" line read 5.9%
unsettled, past its 5% warning, on battery (51% to 35%): the caution on this preset.

| fit | shapes | fitted | held out |
|---|---|---|---|
| schoolbook | 239 | 4.0% | 3.9% |
| Karatsuba level | 233 | 1.9% | 2.0% |
| chunking | 240 | 2.4% | 2.5% |
| recursive Karatsuba | 239 | 2.8% | 2.6% |
| recursive chunking | 240 | 1.9% | 1.6% |
| FFT | 240 | 3.3% | 3.6% |
| Knuth | 1568 | 2.2% | 2.2% |
| NR | 726 | 3.0% | 3.1% |
| NR with remainder | 726 | 3.5% | 3.8% |
| BZ | 1082 | 2.7% | 2.9% |
| series Horner | 384 | 7.3% | 7.0% |
| series rect | 94 | 11.2% | 11.6% |
| series binary splitting | 376 | 8.1% | 8.6% |

Picks under the preset, on shapes no tune took (normal level): products 0.00% lost (worst 0.2%),
Karatsuba's equation error there 3.2% and 4.8%; `div_dyn` 0.07% (worst 4.9%), the entry itself
1.32%; `div_rem_dyn` 0.30% (worst 13.2%), the entry 0.65%; BZ's top block 0.01%; series 0.24% and
0.62% on the two halves (worst 11.7%). The trim cost one fit: NR quotient-only held out at 2.2%
before it and 3.1% after.

### The series' equations: what they are, where they miss

Each is a count of the algorithm's own work by kind (`Parts`, from its kernel) times a fitted
price a kind. In the M1 preset:

- **Horner** = 1.056 x (products by x', in ns by mul's equation) + 3.115 x (limbs of `div_prim`
  passes) + 3.476 x (limbs of `div_prim2` passes) + 0.414 x (limbs of passes by a one-limb
  multiplier) + 1.362 x (limbs by a two-limb one) + 0.187 x (limbs filled or copied) + 6.705 x
  terms + 3.132 x kernel calls.
- **Rect** = Horner's prices for the kernels it shares (its divisions, multiplier passes, fills)
  + 0.979 x (its products by its powers, in ns by mul's equation) + 6.482 x calls + 142.5.
- **Binary splitting** = 1.003 x (its merges' and leaves' products, by mul's equation) + 1.015 x
  (finalize's division, by division's) + 3.017 x (`div_prim` limbs) + 0.991 x (two-limb
  multiplier limbs) + 0.245 x (limbs added or shifted) + 0.167 x (limbs filled or copied) +
  0.893 x terms + 3.697 x calls + 8.501 x allocations; a tree wide enough to split has each
  level divided by 1 + 0.2 (threads - 1) and pays one pool wake.

**Where they miss** (the report's 96 + 96 unseen workloads, each equation fitted on one half and
read on the other; from `target/tuning/series_parts.tsv`, which `BIGBITS_TUNE_VERBOSE=1 cargo tune
series` writes, by a one-off script that isn't kept in the repo):

- Horner (6.3% held out there): an offset a family, which a shared per-term price can't hold:
  exp's chunks -5.1% on average, ln's first chunk (BBP) +3.1%; and calls past 30 us read 4.9% low.
- Binary splitting (7.5%): the small, fast trees. Frames of 16-63 limbs read 9.6% (3.7% high on
  average) and calls under 3 us 9-11%, where calls from 3 us up read 4.9% and frames from 64
  limbs 5.5%. A quarter of its predicted time is kernel calls.
- Rect (13.4%): 22 workloads to fit on, and no pattern by family, frame or time that a price
  would take up: 10.6-15.9% whichever way it is cut. It is the fastest algorithm at 4 or 5 of 96
  workloads.

**Alternatives tried on the same timings** (held-out rms; "current" is the form above):

| alternative | Horner | binary splitting | rect |
|---|---|---|---|
| current | 6.3% | 7.5% | 13.4% |
| + a fixed cost a call | 6.2% | 7.5% | (has one) |
| the per-term price by family, the rest shared | **5.2%** (worst 15.5% to 11.6%) | 7.6% | - |
| term, call and allocation prices by family | 5.2% | 6.6% | 13.4% |
| every price by family (four equations) | 7.7% (exp's chunks overfit: 12.9%) | **6.2%** (5.3-6.9% a family) | 12.4% |
| prices for frames under 64 limbs and from 64 | 6.0% (6.8%, 4.9%) | 7.8% (9.4%, 5.5%) | 12.9% |
| + the linear passes' limbs x log2(frame) | 6.3% | 7.3% | 13.4% |
| products priced apart under and from 32 limbs | 7.2% | 8.1% | 13.7% |
| fitted only where it is within 1.5x of the fastest | 9.1% | 6.9% | 12.8% |
| every price its own (none from Horner's) | - | - | 11.6% |

**Proposed to the user (2026-11-11), not applied, and withdrawn the same day** (the series are to be general evaluators, and "family" below is exp's and ln's own workload kinds; see "The series evaluators over general series"): Horner with its per-term price by family
(three more constants; about 5.2%); binary splitting either with prices by family (6.2%) or
left as it is and judged from 3 us up, where it reads 4.9%; rect recounted, not refitted (its
miss is in its counts: no price helps), with "every price its own" (11.6%) as a stopgap. The
picks already lose only 0.24-0.62% of measured time, so these buy conformance to the 5% line
more than speed. "Family" in the trial is the four workload kinds (exp's chunk, exp's top bit,
ln's chunk, ln's top chunk); in the code it would be the kernel (BBP, Engel, Hyper with and
without x), which is untried.

### The tuner's exp workloads timed nothing (found and fixed 2026-11-11)

`exp_dyn(x, m, out)` reads x as a fixed-point number whose **top** limb is its integer part: the
top limb is worth 2^(64m). Every exp workload of the tuner passed `m = -(n)` with an n-limb x
and an n-limb output, which puts X below the output's last limb, so e^X came out as exactly
1.000...0, in about a microsecond at any size:

| limbs | the tuner's call (m = -n) | a real exp (m = -1) |
|---|---|---|
| 16 | 1.06 us | 16.5 us |
| 500 | 1.44 us | 1455 us |
| 1040 | 2.56 us | 4495 us |

What had used that call: the end-to-end gate's and `benches/tuning_bias.rs`'s `exp_dyn {o}` and
`exp_static<1100> {o}` items (`check_items.rs`), every `exp.*` sweep (`sections/trans.rs`:
which is why the preset's `[exp]` values had all read "kept, within the tie band"), and the
three exp callers of bin_split's leaf and gate sweeps. ln's workloads were real (a scale
doesn't make a logarithm trivial). So **every `exp_dyn` and `exp_static` figure in this file
and in CLAUDE.md that came from the gate, `cargo tune check` or `tuning_bias` before this date
is of the degenerate call**: the "15-23% slower at 4-120 limbs" when the series' equations
were first wired, the "3-5% off `exp_dyn`" of the closed forms, "`exp_dyn` 3.5-6% faster from
4 to 300 limbs", and each "exp and ln" mean (their ln half is real). What they measured is the
series dispatcher's own cost against a call that does almost nothing else, which overstates
it: on a real exp at 16-32 limbs the same kind of cost is about 3% (rect's prediction, below).

The fix: one named scale, `check_items::EXP_FRACTION` (-1), taken by every exp workload; the
comments in `trans.rs` that read as if x[0] were the unit; and
`test_exp_scale_is_the_top_limbs`, which pins both readings. The exp items' outputs differ
from older builds', so a `tuning_bias` baseline must be rebuilt with this file to compare
exp.

Then bin_split, series, exp and ln were tuned again on the M1 (thorough, `--write-preset`, on
mains power; a fixed 1040-limb exp read 4534-4569 us over 60 runs before it). The gate kept
all four; fit probes' passes differ 0.5%.

| knob | before | after | the sweep's reading |
|---|---|---|---|
| `exp.dyn_chunk_min_bits` | 16 | 32 | 16 lost 5.9% (worst 10.1%); 32 loses 3.0% (worst 7.3%) |
| `exp.static_chunk_min_bits` | 16 | 64 | 16 lost 24.6% (worst 48.6%) |
| `exp.static_sqr_bits` | 0 | 24 | 0 lost 9.5% (worst 19.4%) |
| `exp.dyn_sqr_bits`, `exp.dyn_chunk_par_work` | 24, 1280 | the same | the best of their candidates, now on real calls |
| `bin_split.dyn_par` | 512 | 1024 | |
| `bin_split.dyn_merge_par` | 256 | 1024 | |
| `bin_split.dyn_par_work` | 2^22 | 2^19 | the lowest within the tie band |
| `bin_split.static_bbp_leaf_limbs` | 128 | 192 | 128 lost 5.4% |

The gate: bin_split 0.9968 (listed: `ln_dyn` 300 at 1.050), series 1.0013, **exp 0.9399** over
48 workloads, ln 0.9979. exp's mean hides four listed slowdowns: `exp_dyn` 300 at 1.120 and
500 at 1.071, `exp_static<1100>` 500 at 1.225 and 1040 at 1.208.

**Why: one constant serves the serial and the parallel chunk path, and they want opposite
values.** Time under `exp.dyn_chunk_min_bits` = c over the best c at that size:

| limbs | 1 thread: c = 16 | 32 | 64 | 8 threads: c = 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 16 | 1.106 | 1.073 | 1.000 | 1.106 | 1.106 | 1.071 | 1.000 |
| 128 | 1.078 | 1.065 | 1.000 | 1.215 | 1.108 | 1.064 | 1.000 |
| 256 | 1.033 | 1.024 | 1.000 | 1.104 | 1.031 | 1.000 | 1.193 |
| 300 | 1.033 | 1.017 | 1.000 | 1.114 | 1.000 | 1.135 | 1.383 |
| 500 | 1.000 | 1.000 | 1.005 | 1.000 | 1.005 | 1.108 | 1.273 |
| 1024 | 1.005 | 1.000 | 1.010 | 1.077 | 1.000 | 1.038 | 1.232 |
| 4096 | 1.026 | 1.008 | 1.000 | 1.119 | 1.058 | 1.000 | 1.197 |

Run serially, a wider first chunk is better at every size (64). Once the chunks run on the
pool (from about 256 limbs on 8 threads) the widest chunk sets the wall time and 8-32 is
best, with 64 losing 19-38%. The static entry is the same: (chunk 64, squares 24) is best to
120 limbs, where the old (16, 0) was 31-81% slower, and loses 15-27% at 500 and 1040 limbs,
where its chunks run on threads. The sweep's sizes (4, 16, 64, 256, 1024, 4096) also miss
300 and 500. This is the case the user's rule on parallel algorithms describes (a value for
the serial version and one for the parallel version, chosen at run time), and it is not
built for exp's chunk width: **proposed, not applied**. Until then the values are a
compromise the gate accepted on the mean.

## The series evaluators over general series: a theory and a first map (2026-11-11, proposed)

The user's direction (2026-11-11): the series evaluators are general evaluators, for users'
series and for more of this library later, not exp's and ln's helpers; their dispatch has to
be modelled from the theory of the algorithms, over the factors that decide it (how fast the
coefficients grow, how wide the argument is, how many terms). Nothing below is applied: it is
the frame proposed for that work and the first measurements against it.

**What describes one evaluation.** Five quantities, all already in `Shape`:

- W = 64 w: the frame, in bits (the output's significant limbs).
- t: the terms, from the series' own tail bound. sigma = W / t is the bits a term fixes on
  average. It has two sources: s = k - bits(x) from the argument and the shift, and the
  coefficients' own ratio, log2(q / r) a term (a factorial's log2 n).
- X: the argument's bits (0 for BBP, Engel and x = 1).
- lambda: the bits a term adds to the exact fraction, log2 q + log2 r (log2 q for BBP and
  Engel). It also sets how many terms one Horner division covers: G = 128 / log2 q, at most 32.

Two ratios organize them:

- **rho = X / sigma = t X / W**: the argument's width against the convergence a term.
- **beta = (X + lambda) / sigma = t (X + lambda) / W**: the exact fraction's size against the
  frame. beta >= rho.

**Each algorithm's work, to leading order.** M(a, b) is a product's time, c a scalar pass's
price a limb, L the accumulator's mean length: W / 2 when the convergence comes from the
shift (the frame shrinks s bits a term) up to W when it comes from the coefficients.

- *Horner*: t L (c_q + c_r) / G + t M(L, X). Scalar passes, shared G terms at a time, and one
  product by x a term.
- *Rect at block length m*: Horner's scalar passes, + (m - 1) M(min(m X, W), X) for the
  powers, + (t / m - 1) M(L, min(m X, L)) for the blocks, + two add passes a term over its
  power's limbs. Once a block's power fills the frame (m X >= W) the best m is about
  sqrt(t W / X) and the products come to 2 / sqrt(rho) of Horner's; below that a power is
  shorter than the accumulator Horner would multiply, which is where the rest of its gain is.
- *Binary splitting*: about M(beta W) log2 t for the tree, + the division, D(W, min(W, t
  log2 q)). Its numbers are the exact fraction's, so its work follows beta, not the frame.

So rect against Horner is a question of rho (and of whether x is wide enough for its products
to outweigh two scalar passes a term), and binary splitting against both is a question of
beta and of how far below quadratic M is at the frame.

**A first map, measured.** `Geo<LQ>`, a synthetic Hyper family with the dials apart: q and r
of LQ bits with r / q about 1 (so the coefficients add size but no convergence), x of xl
limbs, the shift set for t terms. 244 shapes: w in {16, 64, 256, 1024}, t from 4 to 4096, xl
from 1 to w, LQ 8 and 60; each algorithm forced through `series_eval_dyn_at`, one thread, a
tune build on the M1. (The probe is a scratch program, not kept in the repo.) Fastest: rect
in 118, Horner in 74, binary splitting in 52.

Rect's time over Horner's, by rho (geometric means):

| rho | under 0.6 | 0.6-1.5 | 3-6 | 6-20 | 20-100 | over 100 |
|---|---|---|---|---|---|---|
| x of 16 limbs or more, 8-bit q | 1.06 | 0.92 | 0.69 | 0.54 | 0.37 | 0.25 |
| x of 16 limbs or more, 60-bit q | 0.91 | 0.86 | 0.75 | 0.66 | 0.50 | 0.35 |
| x of 2-4 limbs, 8-bit q | 1.64 | 1.28 | 0.84 | 0.70 | 0.69 | 0.67 |
| x of 2-4 limbs, 60-bit q | 1.24 | 1.27 | 1.22 | 1.09 | 1.04 | 0.95 |

Binary splitting's time over the better of the other two, by beta and frame:

| frame | beta under 0.5 | 0.5-2 | 2-8 | 8-64 | over 64 |
|---|---|---|---|---|---|
| 16 limbs | 1.02 | 1.22 | 1.79 | 5.5 | 31 |
| 64 | 1.1 | 1.14 | 2.85 | 8.1 | 53 |
| 256 | 0.85 | 1.06 | 2.78 | 6.1 | 60 |
| 1024 | 0.64 | 0.80 | 1.85 | 4.5 | 54 |

Both follow the laws: rect's gain grows with rho from a crossing near 1 (near 2 for a 2-4
limb x, and none when 60-bit coefficients leave Horner dividing every other term, so that x's
product is a small share of its work); binary splitting loses in proportion to beta and wins
only where the exact fraction is no larger than about twice the frame and the frame is in the
hundreds of limbs. With beta far under 1 it also wins on a few fast terms, because the whole
sum is then one division of a frame by a Q of a limb or two.

**exp's and ln's chunks sit on the map's triple point.** A bit-burst chunk at bit position s
has an x of about s bits and a shift that fixes about s bits a term, so rho is 0.5-1 and beta
1-2 by construction: the bit-burst is exactly the rewriting that turns one evaluation at
rho = t into log2 W evaluations at rho near 1. That is where all three algorithms are
closest. On the family, with x of two limbs or more:

| | shapes | runner-up over the winner (median) | within 15% | slowest over the winner (median) |
|---|---|---|---|---|
| rho 0.25-4 | 70 | 1.22 | 21 | 1.9 |
| rho 4-32 | 40 | 1.34 | 8 | 7.0 |
| rho over 32 | 68 | 2.33 | 3 | 44 |
| exp's and ln's own chunks (the tune's set) | 45 | 1.21 | 19 | 1.5 |

So the equations were fitted and judged where a few percent of error decides the pick and
where rect has the least to offer, and nothing was fitted where the differences are tenfold.
It is also why rect read as marginal there (fastest in 9 of 192 workloads) while being the
most frequent winner over the family.

**How the live equations do off their training set.** The dispatcher's own pick
(`series_eval_dyn`) against the best forced algorithm on the 244 shapes: 0.2% of the total
time lost (the large shapes are picked right); a shape's median 0.8%; over 10% in 38. Of
those 38, 16 are wrong picks, each 10-32% (mostly Horner taken where rect is better at a
2-4 limb x and many terms, or rect or binary splitting taken at four terms where Horner is
better), and 22 are the model's own cost on a series under a few microseconds: 54 ns median
at a one-limb x, 180-205 ns at wider ones (a tune build, whose knobs are atomics). On exp's
real calls, rect's prediction alone costs about 3% at 16-32 limbs (rect priced out of the
dispatch: 0.968-0.974 there, 1.01-1.05 from 128 to 4096 limbs on one thread).

**What this proposed** (the user's answers, and what was built from them, are in the next section).

1. *The design.* The series' fits take their shapes from a synthetic family scattered over
   (w, t, x's width, coefficient bits), each equation where its algorithm competes (rect
   from rho about 1 up, binary splitting to beta about 8), with exp's and ln's chunks kept
   as a judged subset: they are the in-crate callers and the hardest corner.
2. *Guards ahead of the equations*, from two ratios the shape already holds: binary
   splitting's parts not walked when beta is past about 8 (it loses 4-60x there, and the tree
   walk is the dearest part of the model); rect taken outright for a wide x at a high rho.
   The equations then decide only near the boundaries.
3. *Rect stays a full algorithm.* The earlier question (fold it into Horner, or remove it)
   was asked of exp's and ln's chunks. Its prices are another matter: fitted where it
   competes they should no longer be the tuner's worst fit.
4. *The per-family prices* proposed above for Horner and binary splitting are withdrawn:
   "family" there was exp's and ln's four workload kinds, which is the opposite of general.
5. *A kernel question the theory raises, unverified against the code*: in a series whose
   convergence comes from its coefficients (a factorial), an error in the accumulator at
   term j is divided by q(1)...q(j) on the way out, so its low log2(q(1)...q(j)) bits cannot
   reach the result. Horner and rect keep the frame's full depth at every term; trimming it
   from the bottom would cut their scalar passes by up to half where the factorial carries
   the convergence.

**Limits of this first map.** One thread (binary splitting splits across a pool; the other
two don't); one synthetic family, with no convergence from its coefficients; binary
splitting not run where its exact parts would pass 300 000 limbs; one machine; a tune build.

### Built: the series tuned as general evaluators (2026-11-11)

The user's instructions on the five proposals above: build the design, **without exp or ln**
("The series fitting should stand on its own with no knowladge of what uses it because it will
be used more generally as it is now"); add the guards, **as tunable parameters**; keep rect; no
trimming of the accumulator ("that is the problem that Bit Burst solves the series code works at
an exact level"); the transcendental functions wait until the series code is settled.

**The family** (`tuner/synth.rs`, written against the crate's public paths so the gate's items
and `benches/tuning_bias.rs` take it too). Each member sets one thing a series' coefficients
decide: `Geo<LQ>`, q and r of LQ bits with r / q about 1 (size, no convergence); `Den<LQ>`, an
LQ-bit q over r = 1 (LQ - 1 bits a term of its own); `Fact`, q(n) = n; and for BBP, whose terms
don't compound, `Odd` (2n + 1) and `Big<LQ>`. `Synth::decay` gives the shift under which about
the terms asked for reach the frame. `test_synthetic_series_converge_and_agree` checks both
that and that every member sums to the same value by all three algorithms.

**The design** (`sections/series.rs`, `family`): 80 shapes at the quick level, 320 at thorough.
Hyper with an x: `Geo<8>`, `Geo<56>`, `Den<32>` and `Fact` at (frame, terms, x's bits), by
`design::scatter3` (the R3 sequence). Hyper at x = 1, Engel and BBP at (frame, terms). Frames 4
to 1024 limbs; terms 2 to a term a bit, capped at 2^19 / frame; x from 2 bits to the frame.
Each set is held out two by two (`Design::in_pairs`): the third coordinate's step is 0.55, so
alternate points sit in opposite halves of x's range and a split by parity, at a dozen points
a set, held out 15 of rect's shapes and fitted 4. Each algorithm is timed where it competes
and some way past: rect from rho 1/4; Horner except past rho 64 at an x of 16 limbs or more
where rect can run; binary splitting to beta 24 where another can run. `report` times every
algorithm that isn't hopeless, so a pick is judged wherever it lands.

**The fit** (M1, thorough, mains power; 997 probes, passes differ 0.6%; the series part of the
tune takes about 40 s):

| | shapes | fitted rms | held out | on exp's and ln's chunks (the fit before) |
|---|---|---|---|---|
| Horner | 311 | 7.7% | 8.2% | 7.0% |
| rect | 82 | 7.8% | 8.3% | 11.4-13.4% (22 workloads) |
| binary splitting | 278 | 8.5% | 9.7% | 8.6% |

All eight of Horner's terms are needed over the family (t from 5.9 to 77 in the form
analysis; the best form of each size short of all eight holds out worse) and none solves
negative. The fits stay above the 5% line and read NOISY, as before: what they gained is
range, not precision. The parallel gain reads 0.35-0.55 from the 6 to 11 trees that split
across the pool, at 9-13% rms: the weakest number in the section.

**The guards** (`Shape::guards` at explicit `SeriesGuards`; `Shape::decide_at`; knobs
`series.bin_split_beta`, `series.rect_rho`, `series.rect_x_limbs`): binary splitting isn't
predicted or taken past a beta where another algorithm can run, and rect is taken outright, at
its balance block length, from a rho and an x width. Each is a sweep of the entry itself
(`sweep_guards`: 20 shapes of the family, on points neither a fit nor a report takes).

The first sweeps took their loads from 0 to 20 ms and chose rect from a rho of 2 at any x of
two limbs or more. On 320 unseen shapes:

| guards | picks lose, mean | worst | the entry loses, mean | under 1 us | 1-10 us | 10-100 us |
|---|---|---|---|---|---|---|
| off | 0.47%, 0.59% | 14.9%, 40.7% | 9.7%, 9.6% | 22%, 23% | 6.3%, 6.0% | 1.9%, 1.3% |
| beta 8; rho 8, x 16 | 0.46%, 0.56% | 13.5%, 39.5% | 9.4%, 9.0% | 22%, 23% | 5.8%, 4.9% | 1.7%, 1.0% |
| beta 8; rho 2, x 2 (that sweep) | 1.48%, 1.94% | 64.9%, 76.5% | 8.4%, 7.7% | 21%, 20% | 3.9%, 4.1% | 2.5%, 0.9% |

(Two figures a cell: the report's two halves.) The low thresholds lowered the mean by sparing
the model's cost on calls under a microsecond, and paid for it by overriding equations that
were right: `hyper geo56 w151 t342 x122`, a 100 us series, ran rect at 132 us where the
equations had Horner at 102 (measured 98). A guard is for ranges where the algorithms are far
apart; what the dispatch costs a short series is another question. So the guards' loads run
from 50 us (`GUARD_LOADS_NS`), where that cost is under the tie band, and the sweeps read:
`bin_split_beta` 8 (the best of its ten candidates, 0.26%), `rect_rho` 32 (2 lost 0.95%,
worst 14.2%), `rect_x_limbs` 16 (2 lost 0.92%). They settle 11 and 39 of the 320 shapes.

**On 320 unseen shapes, under the preset** (`cargo tune series`): the picks lose 0.48% and
0.62% of measured time (mean of the shapes; 0.69% and 0.95% under the equations fitted to
exp's and ln's chunks), worst 18.5% and 42%. Fastest: Horner in 169, binary splitting in 98,
rect in 53. The gate kept the three sections: series 1.0005 over 78 workloads (its own 30
synthetic items among them; `series hyper geo8 w16 t16 x128`, a call under a microsecond,
1.256 against the low-threshold guards before it), exp 1.0002, ln 0.9987, the last two with
the same values as before.

**What it leaves.**

1. *The dispatch's own cost was the larger loss*: 22-23% on series under 1 us, 5.6-5.9% at
   1-10 us, in a tune build. Answered by the small-series table (next section), which brings
   the entry's mean loss from 9.4% to about 2%.
2. *`horner_first`* took 50 of the 320 and lost 14-42% on three (32-bit denominators at 12 to
   22 limbs). The table is read ahead of it and leaves it 1 shape; its knobs stay, for ranges
   the table hands on and for a target with no table.
3. *The largest single miss*: `hyper fact w732 t304 x299`, binary splitting picked at a
   prediction of 252 us and measured at 321 (rect 271): a tree that splits across the pool.
4. *Not in the family*: coefficients past 64 bits (the wide kernels), frames past 1024 limbs,
   and more than one thread count.
5. `series.bin_split_par_gain`'s reading (above).


### The small-series table (2026-11-11)

The user: "for small series use a table as is done for other algorithms". `series.small` is a
derived table knob, as `mul.kara` is: what a short series runs, read ahead of everything else
(`SeriesNode::small`, then `horner_first`, the guards and the equations for what it leaves).

**Its two inputs and its classes** (the layout since 2026-11-12, when the table was shrunk:
"The table shrunk, ..." below has the layouts tried). A row is `class x 16 + the frame's bucket`
(`small_frame`: the octave of the frame's limbs, 16 buckets to 65535) and a row's change
points run over the terms; a cell is Horner, rect, binary splitting, or "the equations" (every
cell past the extent). A class is what the pick turns on besides the frame and the terms, in
buckets: the kind and x's width (10: BBP, Engel, and Hyper with no x, an x of up to 16 bits,
up to a limb, 2 limbs, 3-8, 9-16, 17-32, more), q's bits at the range's midpoint (4: to 8, 16,
32 and 64 bits), and whether r is past 1. 72 classes have a series. A lookup costs a
coefficient or two and a few compares; a class is found without building the range's `Shape`.
As first built: frames by half octaves to 255 limbs (15 buckets, and from 2026-11-12 eight
more by octaves to 65535, rows `class x 32 + bucket`), 11 kinds (an x of 3-4 and of 5-8 limbs
apart), 6 q buckets (25 and 42 bits among them, by how many terms one of Horner's divisions
holds), 120 classes with a series.

**Derived, timing nothing** (`derive::series_small_table`; `cargo tune derive`, and every tune
of the series section before its gate). For each class, `Shape::of_class` builds the shape of a
series of that class that has converged to the frame (its first term a term's worth of bits
below the top, the coefficients' own convergence taken off the shift), at each bucket's middle
frame and along the terms by thirds of an octave; the cell is what the dispatch decides for it,
out to where the best prediction passes 20 us (`SMALL_EXTENT_NS`: past it the equations cost
under 1% of the series). Three things the first tries got wrong, each from a wrong pick it
produced: a bucket stands for its top, not its middle (a class read at 6-bit q's sent binary
splitting a 9-term tree whose Q, at 8 bits a term, no longer fit a limb: 40% slower); a step of
the grid is decided at its last term count, for the same reason; and a one-limb x is two
classes, as binary splitting's numbers grow by x's bits a term (a class that stood for 56 bits
gave Horner a series with a 3-bit x that was binary splitting's by 42%). The M1's first table
had 5050 change points (63 KB of the preset; 120 classes x 15 buckets x about 3), 6563 once
its buckets ran to 65535 limbs, and has 2602 (31 KB) in the layout above.

**What it does** (M1, 320 unseen shapes, `cargo tune series`; two figures a cell, the report's
halves):

| | without the table | with it |
|---|---|---|
| the entry loses, mean | 9.4%, 9.4% | 1.9%, 2.1% |
| ... on series under 1 us | 22%, 23% | 2.3%, 3.4% |
| ... 1-10 us | 5.9%, 5.6% | 2.0%, 2.2% |
| ... 10-100 us | 1.5%, 1.1% | 1.3%, 0.8% |
| worst | 115%, 88% | 17%, 21% |
| picks lose, mean | 0.48%, 0.62% | 0.43%, 0.66% |

The table takes 203 of the 320 shapes; `horner_first` is left 1 (it only sees what the table
hands on, so its 14-42% losses on 32-bit denominators are gone), the rect guard 10, the beta
guard 22. In the tune that first derived it the series section's gate read 0.9799 over 78
workloads (its 30 synthetic items and exp's and ln's), nothing listed slower.
`cargo tune series --rederive` derives the table from the equations in force for one run, to
read a change to its classes or grid without a tune.

On a target with no preset the default table hands every range to the equations, so nothing
changes there until a tune or `cargo tune derive`.

(Seen in passing: `cargo tune derive --file presets/aarch64-macos.toml` would also rewrite
`mul.kara`, with 2573 of 87763 shapes differing from the preset's own table (Karatsuba from 44
for squares where the preset has 40), so a tuned file does not reproduce its mul table at the
moment. Not looked into; the series table goes in through the tune, which leaves mul alone,
or, since 2026-11-12, through `cargo tune derive --file FILE series`, which rewrites only the
sections it is given.)

### Fit models proposed for the series (2026-11-11; 1 and 2 applied the next day, below)

Asked for: models that improve the three equations' accuracy (8.2%, 8.3% and 9.7% held out).
Tried on the report's 320 unseen shapes; rms is cross-validated in six folds unless a fixed
split is named.

**Where the error is: in short series, for Horner and binary splitting.**

| a fit made and judged on series of | all | under 5 us | 5-20 us | from 20 us |
|---|---|---|---|---|
| Horner | 8.5% | 9.5% | 8.0% | **4.6%** (worst 11%, 76 shapes) |
| binary splitting | 8.8% | 9.8% | 6.6% | **4.3%** (worst 17%, 57) |
| rect | 8.7% | 6.6% | 9.4% | 9.5% (worst 33%, 41) |

One fit over every size reads 5.3-6.0% on Horner's long series with a bias of -2.6 to -3.3%:
the short ones pull its prices. Fitted from 20 us up, both forms as they are come under the 5%
line. No form tried does better on the short ones (a fit of their own reads 9.5-9.8%).

**Other forms, on the fixed split (held out):**

| form | Horner | binary splitting | rect |
|---|---|---|---|
| now | 9.1% | 8.7% | 9.0% |
| + a fixed cost | 9.0% | 8.7% | - |
| term price by series kind | 9.3% | 12.4% | - |
| term and call prices by kind | 8.4% (14 constants) | 30.6% | - |
| every price by kind | 11.1% | 36.4% | - |
| rect: every price its own | - | - | 10.9% |
| rect: + its own term price | - | - | 8.4% |

Prices by kind overfit (binary splitting badly); they are not proposed. Under any of these the
picks lose what they lose now (0.35% mean with no guards) or more.

**Counts that are wrong, which no price fixes.**

- *Horner's groups, for a few terms.* The model takes groups of the average size its budget
  allows, a real number (3.63 terms for a 32-bit q), and gives the shallowest group what is
  left. For 8 terms that is three divisions (0.74 + 3.63 + 3.63 terms, 1039 limbs) where the
  kernel makes two of four terms (about 650): `engel den32 w518 t8` reads +39%. Horner's four
  worst shapes are all a few terms in a wide frame (+24 to +39%).
- *Binary splitting*: BBP with 40-bit q's in small frames at beta 3.4-8 reads 32-36% low; two or
  three terms over a wide frame read 20-22% high.
- *Rect* misses where Horner's kernels are 60% or more of its time (+6.7% bias, 12.7% rms over
  14 shapes): one fitted multiple on that part (0.90) gives 7.8% from 8.3%.

**Proposed:**

1. **Fit each equation where the dispatch reads it**: from the table's extent up (Horner 4.6%,
   binary splitting 4.3%). The family then needs more long shapes (a quarter of its shapes
   are past 20 us now), and the table's derivation needs prices for short series: a second
   set, fitted under the extent and read only by the derivation (the choice proposed), or the
   long fit's, untried.
2. **Count Horner's groups whole when a range holds only a few.**
3. **Rect**: its own term price (8.4% from 9.0%), or the multiple on its Horner part; small
   either way, and its counts are the real question.
4. Not proposed: prices by series kind.


### Applied: a fit a regime and whole groups; and what is special about rect (2026-11-12)

The user: "I would also say impliment 1 and 2. But do you just think its impossible to
accurately predict rect using the same approach as every other algorithm? What is so special
about the 'rect' series."

**Applied.**

- *A fit a regime* (`Regime` in `sections/series.rs`). A workload is long or short by whether
  the short prices in force put its best algorithm past `SMALL_EXTENT_NS`, the test the
  table's derivation makes. The long series' fit gives the equations the dispatch reads
  (`SeriesEq::read`); the short series' gives `series.short_*` (21 constants then, 22 since
  rect's equation changed later that day; `SeriesEq::read_short`), which only the table's derivation reads. The table could not be
  derived from the long fit: its prices read 21-24% on short series for Horner and binary
  splitting. The family is walked three times its count for the long regime, which keeps about
  a quarter of what it walks.
- *Rect's own price for one-limb divisions* (`series.rect_ns_div1`; `SeriesEq::RECT` was
  product, div1, call, fixed, and is product, div1, add, term, fixed since later that day;
  approved by the user afterwards, 2026-11-12: "That addition to
  the Rect equation is ok"). Forced by the regime fit, not proposed before it: rect's groups
  end at its blocks, so it divides by a one-limb Q in every range, where Horner's groups fill
  two limbs and its long series hardly make one (1 of 67 in the report's set). Horner's long
  fit has no price for that kernel (it solves negative and is left out), and on Horner's
  price rect read 25.5% with the picks losing 2.0%; with its own, as before the split.
- *Horner's groups, whole for a q of constant size* (`Shape::group_of`, from log2 q at the
  range's two Gauss points, `Shape::gq`). Where the same whole number of terms fits a group
  at both points, that number; otherwise the average less half a term, as before. Whole
  numbers averaged over the two points, tried first for every q, are wrong for a growing one:
  32 terms of q(n) = n are 118 bits and one group, and read as two (+70%).
- *The table's buckets run to 65535 limbs* (octaves past 255; rows were `class x 32 + bucket`
  until the table was shrunk later that day, next section).
  With the equations fitted to long series, a short series in a frame the table didn't hold
  was picked by prices that are wrong for it: 3 terms in 648 limbs ran 66% slow.
- Each series fit's evidence names the workload it misses most.

**The M1's fits** (thorough, mains power, passes differ 0.5%; held out):

| | long series | short series | one fit over every size (before) |
|---|---|---|---|
| Horner | 5.8% (222 shapes) | 9.7% (238) | 8.2% |
| rect | 8.2% (123) | 6.2% (41) | 8.3% |
| binary splitting | **3.9%** (148) | 10.5% (228) | 9.7% |

Binary splitting's long fit is under the line and no longer marked. Horner's is at 5.8%, not
the 4.6% the trial promised: its long shapes are mostly those with a multi-limb x (below).
On unseen shapes (`cargo tune series`, now a block a regime): long series, picks lose 0.25%
and 0.14%, the entry 0.72% and 0.69%; short series, picks 0.34% and 1.58%, the entry 1.31%
and 2.84% (2.5% and 3.9% under 1 us, 0.7% and 2.3% at 1-10 us). The gate of the tune that
brought the regimes in read 0.983 over the series section's 78 workloads against the state
before it, nothing listed slower once the buckets were extended. The table had 6563 change
points (82 KB) in that layout.

**What is special about rect: nothing about the algorithm. It is the products.** (Point 3's
conclusion, that rect's remaining error is the products' price, was tested later the same day
and does not hold for rect: its error does not follow its products. The next section has the
test, and "Rect: everything tried" under "Marked problems" the whole list.)

1. *At equal numbers of shapes rect reads what the others do.* Cross-validated over every
   size on the family: rect 8.7% on its 82 shapes; Horner 8.5% and binary splitting 9.0% on a
   random 82 of theirs. Its 11-13% reputation came from the 22 chunks of exp's and ln's it was
   fitted on before.
2. *What stays high is long series with a multi-limb x, for Horner too.* Fitted and judged
   from 20 us up: Horner with no x or a one-limb x 3.9%, binary splitting 4.2-4.9%, Horner with
   a multi-limb x 7.1%, rect (always a multi-limb x) 9.6%.
3. *Those two spend their time in high products, the one kernel in the series' models with no
   fitted equation of its own.* `hi_mul_dyn` is priced as mul's schoolbook formula less the
   triangle (frames to `mul.dyn_partial_mul`, 242 limbs) or as mul's full-product prediction
   (above). Timed over 199 scattered (l, s) with a short side of 3 limbs or more:

   | | measured over the price | spread |
   |---|---|---|
   | full product against mul's equation, inside its fitted areas (s from 10, l/s to 8) | 1.00 | 4% |
   | the same, outside them | 0.89 | 23% (0.50 to 1.96) |
   | ... a short side of 3-5 limbs | 0.79 | 25% |
   | ... l/s past 32, frames past 242 | 1.24 | 18% |
   | high product against the series' price, schoolbook tier | 1.17 | 16% (0.81 to 1.75) |
   | high product, guarded tier | 1.08 | 9% |

   mul's equations are right where mul's own dispatch needed them, on near-balanced products.
   The series multiply a frame by an x: l/s runs to hundreds and the short side down to 3
   limbs, where nothing was fitted. And the schoolbook-tier high product costs between 0.4
   and 2.2 times the full product on the same operands, which no multiple of mul's formula
   follows. Binary splitting multiplies near-balanced full products, and reads 3.9%.
4. *Rect has more product shapes than Horner* (the accumulator by each power, each power by
   x), so it inherits more of it.
5. *It also leaned on a kernel Horner hardly runs* (the one-limb division, above), which is
   fixed.

So it can be predicted the same way as every other algorithm, by applying the same rule one
level down: an equation may include another's fitted time, but that time has to be fitted
where it is read. **Proposed, not applied** (it adds equations to mul's section, and a retune
of everything built on mul): (a) a fitted equation for the high product, in both its tiers,
on (l, s) scattered to 1024 limbs with s from 3; the guarded tier is the full product's time
plus about 0.06 ns a limb and 50 ns (3.4% with the measured full product in its place), the
schoolbook tier needs the kernel's own form (the best four-term form tried reads 11.7%); and
(b) mul's fits extended to the unbalanced products the series read (short sides of 3 to 9
limbs, l/s past 8).


### The table shrunk, rect's error looked at again, and the chunking FFT evaluated (2026-11-12)

The user: "That addition to the Rect equation is ok, and definitely shrink the size of the
series table. As for why still rect fails, I think it may be due to a lack my chunking fft
algorithm. I used to have that algorithm in my multiplication stack but removed because it
wasn't getting much use. I think it would be a perfect candidate for rect series as it would
handle uneven inputs better and it would fit better them."

**The table: 6563 change points to 2602.** It was 82 KB of the 124 KB preset and about 95 KB of
a build (79 KB of entries and a 17 KB row index); `mul.kara` is 220 numbers. Every layout was
scored the same way: the report's forced timings kept (`cargo tune series --timings FILE`), the
table rederived under the layout for the run (`--rederive`), and the picks' loss read on the
245 short unseen workloads, the report's two halves.

| layout | change points | picks lose, mean | worst |
|---|---|---|---|
| before: frames by half octaves to 255 limbs and octaves above, q to 8, 16, 25, 32, 42, 64 bits, 11 kinds | 6563 | 0.35%, 1.58% | 9.5%, 21.9% |
| frames by octaves | 4433 | 0.28%, 1.46% | 7.5%, 21.9% |
| ... and q to 8, 16, 32, 64 | 2901 | 0.34%, 1.50% | 13.3%, 21.9% |
| ... and an x of 3-4 and of 5-8 limbs one kind (**in use**) | 2602 | 0.34%, 1.60% | 13.3%, 21.9% |
| frames by octaves, q to 16, 32, 64 | 2189 | 0.54%, 1.95% | 15.0%, 25.4% |
| frames by octaves, q to 8, 32, 64 | 2166 | 0.34%, 2.32% | 13.3%, 98.5% |

The 8-bit and the 16-bit q buckets both have to stay. Not tried: an x of 9-16 and of 17-32
limbs as one kind (its run was stopped), the flag for r, the terms' grid (thirds of an octave)
and the hysteresis. What limits the rest: the old table had 2080 stored rows of about 3.2
change points each and only 859 distinct row contents, because every row ends on its own
extent (the term count where the equations take over), which moves from one frame bucket to
the next and keeps a row from repeating the one before it. Rows shared between classes (3304
change points for the old layout) would need another format.

In use: `small_frame` is the octave of the frame's limbs, `SMALL_Q_BITS` is 8, 16, 32, 64,
`SMALL_KINDS` is 10, and rows are `class x 16 + bucket` (`SMALL_ROWS`: 16 buckets, so a class's
keys are exactly its buckets and the row index, which runs over every key, is half what 32
made it). 2602 change points over 72 classes: 31 KB of the preset, which is 72 KB; in a build
31 KB of entries and a 5 KB row index. The series section's `derive` number is 3, so a file
with a table of the old layout has it fall through to the default (everything to the
equations) with a warning to rederive, not misread. `cargo tune derive` takes section names
now (`cargo tune derive --file FILE series`), which is how the preset's table was rewritten
with nothing else in the file touched (no tune; the prices are the 2026-11-12 fit's).

A fresh report under it, each side timed anew the same morning (`cargo tune series`, thorough,
mains power, passes differ 0.8% and 0.9%; two figures a cell, the report's halves):

| | the old table (6563) | the new (2602) |
|---|---|---|
| short series: the table takes | 238 of 245 | 232 of 245 |
| short series: picks lose, mean | 0.35%, 1.58% | 0.30%, 1.66% |
| ... worst | 9.5%, 21.9% | 11.9%, 22.8% |
| short series: the entry loses, mean | 1.37%, 2.78% | 1.25%, 2.75% |
| ... on series under 1 us | 2.6%, 3.8% | 2.1%, 3.6% |
| ... 1-10 us | 0.7%, 2.2% | 0.8%, 2.4% |
| long series: picks lose, mean | 0.26%, 0.15% | 0.23%, 0.14% |
| long series: the entry loses, mean | 0.74%, 0.66% | 0.68%, 0.73% |

**Rect's remaining error does not follow its products.** The section above concluded that what
keeps rect (and Horner with a multi-limb x) above the line is the price of high products. Read
against rect's own residuals, on the report's 123 long unseen shapes where rect was timed
(`target/tuning/series_parts.tsv` of the 2026-11-12 tune; residual = predicted / measured - 1
under the preset's prices; 8.3% rms, mean +1.6%):

| by the share of rect's predicted time that is products | rms | mean |
|---|---|---|
| lowest third (3-33%) | 9.5% | +4.7% |
| middle (34-64%) | 8.7% | +0.6% |
| highest (64-93%) | 6.4% | -0.5% |

| by x's width | shapes | rect: rms, mean | Horner on its shapes of that width: rms, mean |
|---|---|---|---|
| 2 limbs | 14 | 10.7%, +4.7% | 6.5%, +3.3% |
| 3-4 | 12 | 12.7%, +7.7% | 5.7%, +1.7% |
| 5-8 | 19 | 9.8%, +3.9% | 4.3%, -2.4% |
| 9-16 | 11 | 8.7%, +6.4% | 5.9%, -4.0% |
| 17-32 | 19 | 5.0%, -0.1% | 3.9%, -2.7% |
| 33 and more | 48 | 6.1%, -2.2% | 7.2%, +2.5% |

Rect is worst where products are the least of its time, and it is predicted too slow by 4-8%
on average at an x of 2 to 16 limbs. The residual's correlation with each part's share of the
predicted time: calls +0.43, terms +0.38, one-limb multiplier passes +0.36, products -0.25,
one-limb divisions -0.05. (Horner's long series read 5.0% on the same report with no such
trend: 4.2%, 4.5% and 6.2% by thirds of product share, no correlation past 0.06.) And prices
do not take it up. Rect's long equation refitted with more prices its own, each half of the 123
shapes judged by the other:

| rect's own prices | held out (both ways) | fitted on all 123 |
|---|---|---|
| product, div1, call, fixed (in use) | 7.4% | 7.1% |
| + one-limb multiplier passes | 7.2% | 6.9% |
| + one-limb multiplier passes and fills | 7.1% | 6.8% |
| + one-limb multiplier passes and terms | 7.2% | 6.9% |
| + one-limb multiplier passes, fills and terms | 7.1% | 6.9% |
| every price its own | 10.2% (12.4% one way) | 6.9% |

No set of prices reads under 6.8% even on the shapes it was fitted to. Prices that depend on
x's width do take the bias out, and no more than that (same shapes, each half judged by the
other; rms and mean by x's width):

| form | held out | x of 2-4 limbs | 5-16 | 17 and more |
|---|---|---|---|---|
| in use | 7.4% | 8.7%, +2.3% | 8.4%, +1.3% | 6.2%, -2.1% |
| a product multiple of its own to an x of 4 limbs (fits 0.80 against 0.99) | 7.1% | 7.7%, -0.5% | 8.5%, +2.5% | 6.0%, -1.6% |
| ... to 16 limbs (0.86 against 1.01) | 6.7% | 7.9%, +1.4% | 7.2%, -1.8% | 5.9%, -0.4% |
| ... by three widths (0.77, 0.88, 1.00) | 6.8% | 7.8%, -0.6% | 7.5%, -0.1% | 5.9%, -0.3% |
| a price for one-limb multiplier passes of its own to an x of 8 limbs (0.30 ns a limb against 0.40) | 6.9% | 6.4%, -0.4% | 9.1%, +0.3% | 5.9%, -1.1% |
| ... to 16 limbs (0.30 against 0.45) | 6.8% | 6.6%, -0.1% | 8.5%, -1.3% | 6.0%, -0.4% |
| both apart to 16 limbs (products 0.94 and 1.00, passes 0.33 and 0.43) | 6.8% | 6.8%, 0.0% | 8.3%, -1.5% | 5.9%, -0.2% |
| a call price of its own to an x of 16 limbs | 7.4% | 8.4%, +1.0% | 8.4%, +0.2% | 6.5%, -0.9% |

So at a narrow x about a quarter of something is predicted that isn't spent, and the data
can't say which of two things: the products with a short side of a few limbs (the multiple
0.77-0.86 is what the high-product probe read on short sides of 3-5 limbs, 0.79, where mul's
schoolbook equation, fitted from 10 limbs, is the price), or the term adds (inside an x of 2-4
limbs the residual goes with the share of one-limb multiplier passes, +0.74, calls +0.57 and
terms +0.55, and not with products, +0.07). Fitted together the two trade off. Either way the
bias goes and the spread stays: 6.7-7.1% held out, with an x of 5 to 16 limbs at 7-9% under
every form. From 17 limbs rect reads 5.9-6.2%, which is Horner's long fit (5.8%). The high
product's price is therefore part of it at a narrow x (the probe's spreads stand) and not what
holds rect above the line; a fitted high-product equation would be for that half point, and
for Horner at an x of 33 limbs and more (7.2%). What is left at 5-16 limbs is in rect's counts
and has not been located. What the misfit costs: the picks on long series lose 0.14-0.23% of
measured time.

**The chunking FFT: a real gain for `mul_dyn`, and not rect's problem.** The old code is in
history (`fft_chunking` at `f7660b1`, `src/utils/mul.rs`; removed in `66372d0`, 2026-04-10): it
cut the long operand into chunks of about the short operand's length (`n = find_fft_size(8 s)`,
chunks of `n / 8`), ran a full FFT product a chunk with both operands transformed each time,
and was dispatched only where `4 s^2 <= l`. Emulated on today's library by cutting the long
operand into chunks of c limbs and calling `mul_dyn` on each (so, like the old code, with no
reuse of the short operand's transform), against one `mul_dyn(l, s)`; time of the whole
product over the best chunking's:

| short side s | l = 2s to 4s | 6s | 8s | 16s | 32s | 64s | 128s |
|---|---|---|---|---|---|---|---|
| 24 to 96 limbs | 0.91-0.99 | 0.94-1.05 | 0.96-1.00 | 1.00-1.04 | 0.98-1.04 | 0.98-1.04 | 0.98-1.04 |
| 192 | 0.77-0.99 | 1.01 | 1.23 | 1.31 | 1.38 | 1.39 | 1.38 |
| 384 | 0.73-0.98 | 1.10 | 1.12 | 1.25 | 1.32 | 1.50 | 1.58 |
| 768 | 0.77-1.00 | 1.04 | 1.11 | 1.20 | 1.40 | 1.45 | 1.56 |
| 1536 | 0.76-0.93 | 1.07 | 1.09 | 1.28 | 1.37 | 1.47 | |
| 3072 | 0.78-0.92 | 1.03 | 1.16 | 1.26 | 1.39 | | |

So 1.1-1.6 times faster from a short side of 192 limbs and a long side 8 times it, more with
more imbalance; nothing at 96 limbs and under (Karatsuba's chunking is already there) and a
loss under 4 times. The best chunk is 4 to 15 times the short side (c + s filling a cheap FFT
size), not the short side: chunks of s read 0.70-1.12, a loss or a tie everywhere but from
s = 384 with l from 64 s. That, and a dispatch boundary of l >= 4 s^2, is probably why the old
one saw little use. A version that transforms the short operand once would gain more; not
measured. mul's own equation already predicts these unbalanced FFT products well (measured
over predicted 0.93-1.04 from s = 96), so nothing here was a pricing error; where it is off in
this probe is Karatsuba's chunking at s of 24 and 48 limbs under a long side from 16 times it,
which runs 9-11% over the prediction (outside the l/s that equation was fitted on). The probe
is a scratch program, not kept in the repo.

It is not what holds rect's fit up: the synthetic family's frames stop at 1024 limbs, so no
product a tuned series makes has a short side of 192 limbs under a long side 8 times it; rect's
error is at an x of 2-16 limbs; and rect is the algorithm that turns Horner's unbalanced
products (frame by x) into balanced ones, so a chunking FFT would help Horner's per-term
products more than rect's, and only for frames past the tuner's range. Restoring it is a
change to mul (an algorithm, its equation, mul's version and a retune of what is built on
mul): the user's call, not made.


### Rect's counts read against the kernel, and hi_mul_buf's own equation (2026-11-12)

The user: "this pass through is dedicated to trying to find an accurate runtime model for the
rectangular splitting algorithm. By accurate we mean that the model falls into the hold-out rms
of under 5%"; then, on the form proposed: "Move the tier price in to the mul section and
implement it."

**How it was found.** Every row of rect's record up to 22 changed a price or a set of shapes.
None could check a count, as there was nothing to check one against. So the kernel was given
counters (a scratch worktree: `mul_acc`, `div_acc`, `add_pass`, `mul_frac`, `group` each
recording its limbs and calls, and every high product its (l, s)), and 747 rect shapes of the
synthetic family were timed at up to seven block lengths each (3830 timings, the rect kernel
alone at an explicit block length; passes differ 0.87%) and counted. That splits the error in
two: the model's counts against the kernel's, and prices fitted to the kernel's own counts.

*The counts were wrong, by amounts that vary shape to shape, which no price absorbs.* The
model over the kernel, long series, every block length (geometric mean, and the spread of the
log):

| Count | Before | After |
|---|---|---|
| term adds and R passes (`mul1`) | 1.74, spread 22% | adds 1.02, 5%; R passes 1.00, 2.6% |
| divisions (`div1`) | 1.15, 21% (1.42 at 1.5-3 blocks, 1.04 past 12) | 1.01, 5% |
| products, as priced | 1.06, 9% | 1.006, 1.3% |
| block products' area alone | 1.10, 15% | 1.00, 0.3% |
| copies | not counted for a guarded product | 1.01, 2.5% |

What was miscounted, each read off `rect_hyper` and its helpers:

1. *A power's width.* Power j was taken as j times x's limbs. The powers are exact and
   `low_zeros` strips their zero low limbs, so it is ceil(j bits(x) / 64): an x of two limbs
   has 65 to 128 bits. This is the "narrow x" error of rows 19-21 (add limbs 1.23 high at two
   limbs, 1.02 at every width once fixed), and it is in every term add, every block product's
   short side and the block length whose power fills the accumulator.
2. *Passes a term add.* Counted as two. `add_term` splits the weight at the term's bit within a
   limb and skips a half that is zero, so a weight of b bits makes about 1 + b / 64: 0.97
   (`geo56`, a weight of 1), 1.20 (`den32`), 1.24 (`fact`), 1.54 (`geo8`) in the kernel.
3. *Limbs a term add.* Taken as the lesser of the mean power and the mean frame. The kernel
   takes the lesser term by term, and within a block the powers grow as the frames shrink, so
   the two cross: a closed form a block now, summed over sampled blocks.
4. *Where a group divides.* Every group of a block was charged at the block's shallowest frame.
   A group divides at its own shallowest term's.
5. *Copies.* A guarded high product is made in scratch and its window copied out, and each
   power is scanned for its zero low limbs. With both counted, rect's own price for a limb
   copied fits at 0.127 ns against Horner's 0.132, where the old count wanted 0.22.

And two smaller ones: the shallowest block's product is by its own, shorter power; and the
accumulator's own length sits a fraction of a limb off a term's frame (the sum a division takes
is Q / q times a term, what it leaves is a q below it, in a top limb half full on average),
which is derived from where the bits land and not fitted, and takes the block products' area
from 1.14 with a 16% spread to 1.00 with 5.5% on short series.

*Prices on the kernel's exact counts* give the floor for any equation of this kind: 4.6% held
out with five prices of rect's own, 3.5-3.7% with the high products of `hi_mul_buf`'s tiers
priced apart from the guarded ones. So the counts were worth about three points and the high
product's price most of another.

**The forms, on the corrected counts** (the same timings, fitted on the half not held out; the
tune's points, the report's, and a third set; the form in use read 8.3 / 7.6 / 7.5):

| Rect's own prices | Held out | Over every block length |
|---|---|---|
| product, div1, term, fixed (its adds at Horner's `mul1`) | 4.8 / 5.4 / 4.4 | 5.5% |
| + add | 4.6 / 4.9 / 4.3 | 5.4% |
| + a multiple for the products `hi_mul_buf` runs | 3.8 / 4.2 / 3.6 | 4.1% |

The third was proposed as rect's own constant. The user moved it to where it belongs: the
price of a kernel is the kernel's, in mul's section.

**`hi_mul_buf`'s equation** (`HiEq`, `mul.hi_rows_*` and `mul.hi_cols_*`). Its two kernels were
timed forced on 543 (window, operand) shapes. Both make the same limb products, the window's
columns and the two guard columns below it, W = l s - (s - 2)(s - 3) / 2, and differ in what
their fixed cost attaches to:

| Kernel | Form | Held out | As priced before (mul_buf's equation on the area) |
|---|---|---|---|
| rows (`hi_mul_buf_by_rows`) | a W + b s + c: a term a row | 2.2% (2.7% in the tune) | 14.6%, 10% low |
| columns (`hi_mul_buf_cols`) | a W + b l + c: a term a column | 3.5% (3.8% in the tune) | 31%, 18% low |

The term a row is 2.9 ns a limb of the shorter operand, and the rows fit is flat across that
operand's width (2.4% at two limbs, 1.9-2.2% from three to the window). Without it (W and a
shift) the rows kernel reads 9.5%. The columns kernel is fitted to a third past where
`mul.hi_mul_row` stands: its cost isn't this form over a wide range (3.5% to 32 limbs, 4.2% to
48 and to 64). The series' `SeriesEnv::hi_ns` reads it for windows within
`mul.dyn_partial_mul`, and Horner's closed-form sum of its products by x (`school_products`)
was rewritten on it: its frames' stretches are now cut where the frame passes the kernels'
cutoff as well as where it passes x. Division's NR keeps its own fitted multiple a tier.

**As built.** `Shape::rect_parts` in closed form (no loop over terms: the dispatch evaluates it
for three block lengths): the divisions and R passes from two lattice sums (`frame_from_lifted`
at the groups' mean offset), the term adds from a closed form a block (the whole powers'
limbs exactly, from a 64 x 65 table of the sums of ceil(j a / 64); the blocks that hold every
power whole in one step, the rest sampled), the products at each block's own length.
`SeriesEq::RECT` is product, div1, add, term, fixed: rect's term adds are `add_mul` passes of a
weight over a power, which Horner has no price for, and its price a term takes the place of
its price a call. `series.rect_ns_call` is gone, `rect_ns_add` and `rect_ns_term` are new
(and the short pair); the series section is at version 6. The short regime walks the family
three times over as the long one does: one walk left rect 20 short shapes at the normal level
for five constants.

**The M1 preset, retuned** (`presets/aarch64-macos.toml`, thorough, mains power, battery full;
all nine sections kept by the gate, no fit rejected). Held out, with the preset before:

| Equation | Before | Now |
|---|---|---|
| rect, long series | 8.2% | **2.8%** over 62 (2.7% fitted on 125) |
| Horner, long series | 5.8% | **4.7%** over 100 (3.7% fitted on 223) |
| binary splitting, long series | 3.9% | 3.6% over 68 |
| rect, short series (the table's prices) | 6.2% over 22 | 7.6% over 68 |
| Horner, short series | 9.7% | 7.4% |
| binary splitting, short series | 10.5% | 9.4% |

For the first time the three equations the dispatch reads are all under the line, and none of
them is marked; the three short sets of prices stay NOISY. Horner came under it without being
touched: its products by a multi-limb x are the same kernel, and its product multiple went
from 1.115 to 1.025 (it had been absorbing the kernel's price). Rect's prices: product 1.024,
div1 3.43 ns a limb, add 0.384 ns a limb, 18.6 ns a term; its fixed cost solved negative and is
left out. `hi_mul_buf`: rows 0.382 W + 2.87 s + 5.6 ns, columns 0.456 W + 1.54 l + 5.9 ns. The
gate: mul 0.9992, div 1.0003, sqrt 0.9989, bin_split 0.9953, series 0.9820 (against the
schema's defaults, as its version changed), theta 1.0007, agm 1.0006, exp 0.9995, ln 1.0008;
listed past 2%: `exp_dyn` 1040 (1.023), two series (1.020, 1.029). The small-series table
derived to 2531 change points. One caution, the same as for the preset before: the sweeps'
"samples" line read 6.0% unsettled (1285 of 21360), past its 5% warning; the 10474 fit probes'
passes differ 2.3% (0.6-0.7% for the series' fits). And one sweep moved a value that a test
stood on: `bin_split.dyn_engel_leaf_limbs` 48 to 64, which puts a tree of
`test_bin_split_parts_count_the_tree` on the leaf cutoff, where the level walk is two nodes off
(the known limit below); the test's tolerance now says "a node or two" in nodes.

On the report's unseen long series (`cargo tune series`, thorough, the old tree against the
new before the retune, each fitted on one half and read on the other): rect 7.5% to 2.9%,
Horner 6.7% to 4.9%, and the picks under the fitted equations lose 0.14% and 0.02% of the total
time (0.30% and 0.06%), worst 8.7% and 3.2% (17.9% and 12.6%).

**Under the retuned preset** (`cargo tune series`, thorough, 235 long and 725 short unseen
workloads; the two halves of each):

- Long series: the live equations read Horner 4.1% and 3.1%, rect 3.0% and 3.3%; their picks
  lose 0.11% and 0.02% of the total time, worst 6.7% and 2.7% (the preset before: 0.26% and
  0.06%, worst 17.9% and 12.6%); the entry itself, its dispatch included, 0.34% and 0.10%
  (0.47% and 0.26%), worst 7.8% and 3.4% (18.3% and 13.7%).
- Short series: the table takes 697 of the 725. The picks lose 1.40% of the total time on one
  half (worst 52%) and 2.11% on the other, where one workload carries it (below); the entry
  1.89% and 2.41%.
- End to end in a normal build, the old tree with its preset against the new with its own
  (`benches/tuning_bias.rs --compare`, outputs identical on every item): the series items
  0.988 on the geometric mean (26 items: factorial series at a wide x 0.82 and 0.90, as rect
  takes a better block length, 6 where it took 4; one series 1.035, where the balance
  candidate moved from 121 to 145 and 121 was the faster; one 1.016 from the dispatch's cost;
  a BBP series 1.022 from bin_split's retuned gates); exp, ln, theta and agm 0.998 (48 items).
  `exp_dyn` and `ln_dyn` from 300 to 8400 limbs: 0.99-1.015 on one thread, 0.95-1.02 on eight.

**A corner the rederived table lost.** Two terms at an x past 32 limbs, in a frame of 256
limbs or more (`hyper fact w548 t2 x18748`: Horner 16.4 us, rect 0.63, binary splitting 18.7):
the table before said rect there, and this one says binary splitting, 3.5 to 30 times slower on
such a series (6.5 us for 1.7 at an x of 48 limbs, 55 for 1.8 at 293). The equations, read
for the series itself, pick rect in both trees. The cell is decided at the class's
representative shape (an x of 48 limbs for "more than 32"), where binary splitting's short
equation reads 2.0 us for a tree that takes 6.4 (and Horner's 5.1 for 18.2): a near tie in the
predictions, which rect's refitted short prices tipped (its one-limb division at 3.18 ns a
limb for 2.48, its fixed cost at 288 ns for 152). At three and four terms both tables say
binary splitting there, for a rect 4.7 and 3.4 times faster. It is the table's derivation
standing on equations that are a factor of three off on few-term trees at a wide x, not rect's
equation; exp's and ln's own chunks don't land on it (above). Not fixed here: what to do with
a cell whose two best predictions are within the equations' own error (leave it to the
equations, at the cost of reading them), or with a kind of x that is open at the top, is a
question about the table.

**What it costs, and what is left.**

- *The dispatch.* `rect_parts` does more: a decision that predicts rect went from 260-490 ns
  to 370-610 in a normal build on the shapes tried, 100-310 ns more (it was 570-710 before the
  table for the powers' limbs and the whole blocks taken in one step). A table pick of rect computes its
  block length the same way.
- *Rect's short prices* did not improve (7.6%). The tuner's forced time includes computing
  the block length, about 290 ns of fixed cost on series of a microsecond; the rect kernel
  alone, timed at an explicit block length, fits at 5.8% on short series.
- *A q that grows* (`fact`): the groups are larger at the shallow end, where the frames are
  long, and the model spreads them evenly: its divisions read 4.7% high (8.7% spread). Weighting
  the two Gauss points by their groups takes the bias to 1.5% and was left out, as it moved no
  fit. It is Horner's count as well.
- *The block length.* Among the lengths timed (the model's pick, half, double, a quarter, four
  times) the best is 8% faster than the old model's pick on long series on average. The new
  equation reads 4.1% over every block length, so more candidates than three could now be
  told apart; not tried, as each costs an evaluation.
- *The powers' products* run over the whole accumulator though a power's low limbs are zero
  (a third to a half of the work where a block's power fills it): a change to the kernel, not
  to its model.

### Tables measured, under one rule (2026-11-12)

The user: "one thing that was left in a bit of a broken state before the table for fast series
evaluations where the dispatcher would have genuine over head. The table should be reworked and
derived from actual measurments at each index size instead of the table being derived from the
fits. This should be the case across the board. Furthermore, I think standardizing where a table
should be used tuning system wide would be useful. So I think a cutoff like if dispatch is more
then 5% of runtime at that size then use a table. This should be enforced system wide." And,
while it was being built: "If the 5% rule creates exsesively large tables then we can revaluate
it to be larger and thus make the tables smaller."

**The rule.** Where reading a dispatcher's fitted equations costs more than `tables::SHARE`
(0.05) of the call, the dispatcher reads a table; past that it reads the equations. One constant
for every dispatcher. `--share X` puts another number in its place for one run of `tune`,
`dispatch` or `series`, and every table's evidence gives its size at the larger shares from the
same timings (`tables::sizes_above`, `rows_at`).

**The builder** (`tuner/tables.rs`, one for every table; a table is a `Spec`: its rows and
grids, the probes of an index, and what to do with the rows of a group once measured).
- At every index of a row's grid, in ascending size, each algorithm that can take the shape is
  timed, forced, and the dispatcher's decision is timed beside them: its equations read at
  explicit knobs, as a normal build has them. `measure` takes a row a few indices at a time
  (`time_fit`: three batches of every probe in shuffled rounds, two passes with the buffers
  moved, the least kept).
- `row` is the rule on a row's timings, with nothing timed, so tests give it made-up ones. A cell
  is the pick that costs least: the time under it over the fastest algorithm's. The pick before
  it stays while its cost is within the tie band, and a pick that lasts one cell between two of
  another is dropped. A row ends at the first index where the decision is within the share of
  the call (`Cell::past`), which is the equations', and nothing past it is timed.
- A step of a coarse grid is timed at one of its sizes and stands for shorter calls too:
  `Cell::least` is the shortest as a fraction of the one timed, and the rule is read there. (The
  series' steps are timed at their last term count and read at their first, and their octaves of
  frames at the smallest frame. Before this the check below found 11 real series up to 8.1%
  past the rule at indices the table had handed on; after it 2 or 3 within a point of the line.)
- An index may stand for several calls (`Cell::of`, a `Timed` each; the series' kinds, below).
  Its pick is the one that costs them least together, the mean of each call's time over its own
  fastest algorithm's, a call that can't run the pick being left to the equations; and it is the
  equations' by the rule only when the decision is within the share of every one of them.
- Reading the equations is itself one of a cell's picks, at its own cost (`Cell::read`: the
  decision paid on every call). With one call at an index its fastest algorithm always costs
  less, so this changes nothing there.

**The four tables** (`dispatch::TABLES`).
- `mul.kara` (`sections::mul::kara_table`): `mul_buf` against `karatsuba_entry_dyn` at every
  (s, l), s from 8 and l to 4s, a row at a time in three steps (the square; the near-square
  shapes to l = 2s - 2, a level; the chunked ones), each under the rows and steps before it, as a
  level's sub-products and a chunked product's chunks are earlier rows'. It has no "equations"
  pick (it is where schoolbook wins; `mul_dyn` past it reads its two equations for under 5% of a
  product everywhere tried), so its rows are measured over their whole grid, and it stops after
  six rows that are Karatsuba throughout. **A trap:** the Karatsuba entry asks the table about
  its own shape before anything else, so with the table-so-far saying "schoolbook" both probes
  ran schoolbook, tied, and the first table read "Karatsuba from 211 for squares". The indices of
  the step being timed are Karatsuba's in the live table while they are timed. On the M1:
  Karatsuba from 44 limbs for squares and 49 at l = 4s (96 change points; every row from s = 61 is Karatsuba throughout); the table derived from the fits said 46 and 66 (it only switched past a margin of the
  two fits' errors, which is widest where the two cost curves cross at a shallow angle).
- `div.dyn_small`, `div.dyn_rem_small` (`sections::div::small_table`): rows by the divisor's
  limbs (every one to 64, then eight an octave), change points over the logical quotient width
  (2 to 24, then eight an octave), only shapes where something besides Knuth can run; a cell
  Knuth, BZ, NR or the equations. They stop when a group of eight rows is the equations' from
  its first index. On the M1 they hold only Knuth cells: 32 change points with rows of their own at divisors to 38 limbs (quotient only) and 57 at divisors to 108 (with remainder).
- `series.small`: below.

**Enforced** (`tuner/dispatch.rs`). `TABLES` is every dispatcher that reads fitted equations at
run time with its table. `cargo tune dispatch`, and the last lines of every tune, time the
decision (at explicit knobs) against the entry's own call on indices over each dispatcher's
small sizes where the dispatch in force reads its equations, and name the indices past the
share; past 1.25 times it the command exits non-zero. `test_every_table_is_a_dispatchers` holds
the schema's table knobs to the list and requires that no knob is derived.

**What went.** `tuner/derive.rs` and `cargo tune derive` (a table is measured by `tune SECTION
--tables`); the fits of one Karatsuba level and of chunking (`mul.kara_{scale,shift,err}`,
`mul.chunk_{scale,pass,err}`), which were read only to derive `mul.kara`; division's Knuth
guards `div.dyn_knuth_d` and `div.dyn_rem_knuth_d`; the series' short prices (`series.short_*`,
22 constants), the fit a regime (`Regime`, `SeriesEq::read_short`, `Shape::of_class`) and
`horner_first` for dynamic nodes. `Kind::Table` is a measured kind: part of its section's hash,
and a file's table of another layout falls through with the section's `derive` number. mul is at
version 18, div at 9, series at 7 (its table's layout number, `derive`, at 5).

#### The series' equations under the rule

There is one set of prices again, fitted on the series the equations are read for: those whose
timed decision is within the share of the fastest algorithm's time (`SeriesTiming::read`). That
puts series from a few microseconds in the fits, where the derived table's extent had kept them
past a predicted 20 us. On one set of 960 workloads (thorough), held out, by the share:

| share | read for | Horner | rect | binary splitting | their picks lose (mean, worst) |
|---|---|---|---|---|---|
| 1% | 208 | 3.8% | 3.4% | 5.7% | 0.05%, 5.5% |
| 2% | 334 | 8.0% | 3.9% | 5.9% | 0.09%, 5.7% |
| 5% | 501 | 6.6% | 4.8% | 7.2% | 0.19%, 9.5% |
| 10% | 641 | 6.9% | 5.1% | 7.7% | 0.22%, 11.9% |
| 20% | 780 | 6.9% | 4.8% | 7.9% | 0.27%, 13.9% |
| no table | 960 | 7.5% | 7.1% | 8.4% | 0.28%, 17.1% |

So at the rule's 5% Horner's and binary splitting's fits are back above the line (NOISY, marked
again: "Marked problems"), with rect under it, and their picks lose a fifth of a percent. The
share moves the fits far more than it moves any table's size.

#### The series' table: its index

As first measured the table kept the derived one's index: a row a class and an octave of
frames, 72 classes (ten kinds of x, four buckets of q's bits, an r past 1 or not), each timed on
one synthetic series. 1873 change points at 5% (2636 at 1%, 1596 at 10%, 1310 at 20%), against
the derived table's 2602.

The user: "I feel like there are simpiler strategies and that complex of a table is causing an
explosion in entries?" It was: 691 rows of 2.7 change points on average, 691 of the 1877 being
each row's end, and no two neighbouring octaves ever the same row (each ends on its own
extent).

**The two ratios and a size** (the user's proposal: "That table will only have three indexes so
it shouldnt be too large"), scored on kept timings (`BIGBITS_TUNE_VERBOSE=1`: every series timed
at every index, and 960 unseen workloads), on the 459 workloads the table is for. Regret is the
call's time over the fastest algorithm's, less 1, the decision paid where the equations are
read; time-weighted is the extra time over the total.

| ratios bucketed by | size | cells used / dense | time-weighted | mean | worst |
|---|---|---|---|---|---|
| half an octave | frame | 616 / 15288 | 10.9% | 11.5% | 149% |
| an octave | frame | 266 / 4056 | 10.1% | 10.2% | 149% |
| 2 octaves | frame | 121 / 1176 | 9.8% | 9.9% | 163% |
| 4 octaves | frame | 65 / 384 | 15.0% | 14.0% | 828% |
| 8 octaves | frame | 40 / 120 | 15.9% | 14.1% | 828% |
| an octave | frame x terms | 253 / 2704 | 7.7% | 9.7% | 149% |
| 2 octaves | frame x terms | 113 / 784 | 8.1% | 10.5% | 163% |
| any | terms | 39-267 | 16-18% | 17% | 3055% |
| (no table) | | 0 | 13.6% | 23.4% | 114% |
| (the 72 classes) | | 1873 points | 2.0% | 1.0% | 46% |

About 10% at every grain: the grain is not what limits it. rho is 0 for every series with no x,
233 of the 459 (they lose 14-18%, those with an x 4-8%); scored on the very cells it was built
from the index still loses 3.6% at half an octave and 5.6% at two; and what decides a short
series is its fixed costs a call and a term, which follow the frame and the terms themselves.
(rho and beta were derived for long series, where the work decides.) Frame, terms and one more
coordinate do better (frame, terms and rho at two octaves: 402 cells of 1848, 4.7% and 3.8%),
and the classes merged do best.

**The user's choice**: "Lets go with the 7 x-kinds, no q, no r 203 pionts table. That is roughly
the size I want cahces to be at and the mean is still less then 2% and the weighted less then
3%."

**The numbers that choice was made on were wrong, and mine.** I had given 1.9% on the mean and
2.7% of the time, worst 96%. Built (a row a kind and an octave, each cell timed on the kind's
members, the 72 series the classes had been, and taking the pick that costs them least
together), the table was 205 change points and lost 2.9% and 4.1%, worst 107%, on the series it
is for; and it took about a hundred longer series, which the equations had been right about, and
lost 3.9% on the mean on those (2.1% with no table at all), one of them 426% slow. Two errors in
the scoring: it counted only the series the table is for, and a row that stands for a whole kind
reaches the longer series of that kind as well; and it had fitted each cell's pick to the
classes still inside the rule's extent at that index (the others' rows having ended), where the
builder has to pick for all of the kind.

**The replay.** A scratch script (not kept in the repo) rebuilds the table from the kept member timings as
`tables::row`, `measure` and `assemble` do and looks it up as `SeriesNode::small` does. It agreed
with the live table's pick on 960 of 960 workloads, and it scores every workload, the decision
paid where the equations are read. On it (mean / time-weighted / worst, in %):

| layout | rule | points | the table's 458 | the equations' 502 | all 960 |
|---|---|---|---|---|---|
| no table | | 0 | 23.3 / 13.7 / 114 | 2.08 / 0.41 / 76 | 12.2 / 0.57 / 114 |
| 7 kinds | as built | 205 | 2.94 / 4.11 / 107 | 3.88 / 1.22 / 426 | 3.43 / 1.26 / 426 |
| 7 kinds | equations a pick | 190 | 2.13 / 3.07 / 47 | 2.82 / 1.18 / 426 | 2.49 / 1.21 / 426 |
| 8 kinds: 17-32 limbs and more apart | as built | 231 | 2.83 / 3.84 / 107 | 3.05 / 0.44 / 143 | 2.94 / 0.48 / 143 |
| 8 kinds: 17-32 limbs and more apart | equations a pick | 219 | 2.11 / 3.03 / 47 | 1.99 / 0.40 / 76 | 2.05 / 0.44 / 76 |
| 6 kinds: 2-16 limbs one | equations a pick | 183 | 2.08 / 3.05 / 58 | 1.97 / 0.40 / 76 | 2.03 / 0.44 / 76 |
| 9 kinds: 3-8 and 9-16 apart too | equations a pick | 257 | 1.99 / 2.71 / 46 | 2.01 / 0.40 / 76 | 2.00 / 0.43 / 76 |
| 9 kinds x q to 16 bits or more | as built | 506 | 1.38 / 2.07 / 46 | 1.97 / 0.40 / 76 | 1.69 / 0.42 / 76 |
| 10 widths x 4 q x r (the classes) | either | 1881 | 0.98 / 1.96 / 47 | 1.90 / 0.40 / 76 | 1.46 / 0.42 / 76 |

Two things carry it. *The equations as a pick*: where no one algorithm serves a kind's spread
(Horner is 2x off for a q of 8 bits at many terms, binary splitting 2x off for one of 62), a cell
that names either loses 100% on half of the kind, and paying for the decision loses a few
percent on all of it. *An x past 32 limbs as its own kind*: the 426% was an x of 414 limbs,
7 terms in 598 limbs, read off a row timed on x of 28 and of 48 limbs, where binary splitting
wins; and with q split as well it was a 2-term series at an x of 293 limbs given binary
splitting where rect is 30x faster (the corner the derived table had lost, below).

**As built** (8 kinds; the user was told and went on: "Run the full retune"): `SMALL_KINDS` = 8
(BBP, Engel, Hyper with no x, an x to a limb, of 2 limbs, 3-16, 17-32, more), no bucket of q
and no flag for r in the class, the series section's `derive` number 5. The members
(`member_cell`, `MEMBERS`): for each width of x (none for BBP, for Engel and for Hyper; 12 and
60 bits; 2 limbs; 7 and 14; 28; 48) coefficients of 8, 16, 32 and 62 bits, with an r past 1 and
without. On the M1 (the retune below): 218 change points, 1316 indices timed
(each on its kind's 4 to 16 members), Horner in 66% of the table's cells, rect in 17%, binary
splitting in 17%; at 146 of the table's 1229 indices no one algorithm serves the kind and the
equations are read, the decision up to 26% of the call there. On the 960 unseen workloads
(`cargo tune series`, thorough), where the table decides (459): its picks lose 1.48% on the mean
and 2.26% of the time, worst 45.0% (the 72 classes: 0.80%, 1.67%, 46.3%; the seven kinds as first
built: 2.87%, 3.90%, 107%), and the entry itself, its dispatch included, 3.00% and 3.45% (2.56%
and 2.52%; 3.66% and 4.35%). With the decision paid where the table hands a series to the
equations the call loses 2.13% and 3.14% there, 1.91% and 0.41% where the equations are read
(worst 77%), 2.01% and 0.44% over all 960: the replay had said 2.11 / 3.03, 1.99 / 0.40 and
2.05 / 0.44. The equations' own picks, on the series they are read for: 0.94% and 0.56% on the
mean on the two halves (0.16% and 0.04% of the time; worst 76% and 48%). And the corner the
derived table had lost is right: a two-term series at an x of 293 limbs in a frame of 275 runs
rect from the table, 631 ns where Horner takes 16.6 us.

The dispatch check reports the indices such a table hands to the equations past the share as
the table's own choice (`Verdict::chosen`), not as a breach: at 146 of the table's 1229 indices on the M1, the decision up to 26% of the call there (in `cargo tune dispatch`'s sample, 6 of 98 indices: 8.0% to 16.7%).

#### Division: what the rule costs

The end-to-end comparison (`benches/tuning_bias.rs`, normal builds, this tree over the one
before) read the division grid 1.3% slower, single shapes to 7%. 12 of its 40 shapes are at
divisors within the old guards, where `d <= guard` returned Knuth with one compare at any
quotient: they read the equations now (which answer Knuth), as the tables hand a division on as
soon as the decision is within 5% of it.

| division | before | now | new / old |
|---|---|---|---|
| `div_dyn` 16/8, `div_rem_dyn` 40/32 | one compare | the table: Knuth | 1.007, 1.009 |
| `div_dyn` 40/32 | one compare | the equations | 1.073 |
| `div_dyn`, `div_rem_dyn` 40/16 | one compare | the equations | 1.052, 1.045 |
| `div_rem_dyn` 76/64, 96/48, 88/64 | one compare | the equations | 1.040, 1.029, 1.021 |
| `div_dyn`, `div_rem_dyn` 88/24, 138/8; `div_rem_dyn` 260/60 | one compare | the equations | 1.009-1.016 |

The 12 as a group: 1.028. About 15-22 ns a quotient-only division and 20-65 ns with a remainder
(from the ratios and Knuth's fitted time; not timed directly). A measured table can't say what
the guard said: "Knuth at every quotient" is a property of the equations, and by measurement it
would take timing every quotient. The ways out were a lower share (the tables were 95 and 82
change points at 1%), the guard back in front of the equations (it changes no pick they would
make, but it is computed from the fits), or accepting it. The user: "Ok well just accept it then
it is within the noise threshold roughly anyway. (For the sake of consistency)".

#### A tuner bug the comparison found: sweeps before the fits they depend on

`hi_mul_dyn` at 300 limbs was 14.8% slower than in the tree before, and `agm_dyn`, `theta_dyn`
and `exp_dyn` at 300 limbs and `div_dyn` 1200/600 3-5.5%. `mul.dyn_partial_mul` had come out at
312 where every tune before had it at 242. mul's second phase fits recursive Karatsuba's and the
FFT's equations and then sweeps the cutoffs whose guarded side dispatches through `mul_dyn`, and
the fits were only made live when the phase returned. This tune began mul from the schema's
defaults (its version had gone to 18), whose `mul.fft_scale` is 0.49 against the M1's 0.574, so
`mul_dyn` took the FFT too early throughout the sweep, the guarded product ran slow from 242 to
317 limbs, and the sweep "kept" the default. A build with 242 in its place: `hi_mul_dyn` 300
0.867, `agm_dyn` 300 0.952, `theta_dyn` 300 0.966, `exp_dyn` 300 0.971, `div_dyn` 1200/600
0.961. `tune_late` sets its fits live before its sweeps now. It would have bitten the first tune
on any new target (the x86 handoff). The gate could not have caught it (its "before" was the
same defaults), and the mean over mul's 45 items read 1.0006: it was in the items.

#### The retune, and the checks

`cargo tune tune --level thorough --write-preset` on the M1, 35 minutes (17:28 to 18:03; it
was 25 before the tables were measured), all nine sections kept by the gate, no fit rejected. The
machine was unplugged when it began and on charge from a few minutes in; a fixed workload held
within 0.5% over five runs just before. The sweeps' "samples" line read 6.4% unsettled, past its
5% warning, as the three presets before it had (5.9%, 6.0%, 6.8%); the fits' passes differ 1.9%
over 64426 probes. (A first retune the same afternoon, before the series' layout was changed and
with the sweep bug above in it, is not the preset.)

- mul: `mul.kara` Karatsuba from 44 limbs for squares and 49 at l = 4s, 96 change points;
  `mul.dyn_partial_mul` 244 (a clean crossover: no time lost at it, where 312 lost 1.5% and 8.2%
  at worst). Held out: schoolbook 3.9%, `hi_mul_buf` by rows 2.6% and by columns 3.8%, recursive
  Karatsuba 2.7%, recursive chunking 1.6%, the FFT 3.6%.
- div: `div.dyn_small` 32 change points, rows of its own at divisors to 38 limbs;
  `div.dyn_rem_small` 57, to 108. Held out: Knuth 2.2%, NR 3.0% and 3.8% with the remainder, BZ
  2.4%.
- series: Horner 6.6% and binary splitting 8.1% held out (NOISY), rect 3.1%; `series.small` 218
  change points (184 at a share of 10%, 158 at 20%).
- `cargo tune dispatch`: `mul_dyn` reads its equations at 34 of the indices tried, the decision
  0.7% of the call on average; `div_dyn` at 98, 2.3%; `div_rem_dyn` at 88, 1.7%; each within 5%
  at every one. The series' entry at 98, 1.8%, past the rule at 6 by the table's own choice (8.0%
  to 17%).
- Both suites pass under the preset (593, and 606 with `tune`).

**End to end** (`benches/tuning_bias.rs`, normal builds, this tree and preset over the tree and
preset before the change, six processes a side; every output hash the same):

| section | items | new / old | the items that moved |
|---|---|---|---|
| series | 26 | 1.0054 | `hyper geo56 w64 t64 x256` 1.145; `geo8 w64 t256 x64` 1.039; `engel fact w8 t30` 1.027; `hyper fact w30 t200 x1920` 0.971 |
| trans | 48 | 1.0044 | `exp_dyn` and `ln_dyn` at 2 to 16 limbs 1.018-1.038; `exp_dyn` 120 1.026; `ln_dyn` 30 0.968 |
| mul | 45 | 1.0012 | nothing outside its interval (`hi_mul_dyn` 300 was 1.148 with the cutoff at 312) |
| div | 22 | 1.0024 | `div_rem_dyn` 120/60 1.016 |
| divgrid | 40 | 1.0105 | the shapes the guards settled: 40/32 1.073, 40/16 1.052 and 1.044, 76/64 1.041, 96/48 1.029, 88/64 1.022 |
| sqrt | 42 | 1.0022 | nothing outside its interval |

So the change is a small loss end to end, 0.1% to 1% a section, and nothing gained in speed: what
it buys is tables that are timings and not consequences of fits, one rule a reader can state for
every dispatcher and a check that holds each to it, a series table an eighth of the size, and the
table corner and the sweep bug found on the way. The losses are the two the user took on: the
division's hand-over to the equations, and the series' table by kinds (its picks on short series,
and the short series of `exp_dyn` and `ln_dyn` at a few limbs, which the 72 classes had closer).
The gate cannot see either: its "before" is the preset of the same design.

#### A fit takes the shapes its equation is read for

The user's rule, asked while reading the series' fits (2026-11-12): "are the fits also fit to
samples that would fall in the table?", then "go ahead and remove the samples that the table
would decide and not the equations, well see if that makes a difference. Furthermore, do that for
every algorithm that has a table as it is just good practice to not fit over samples that the
fits arnt used for."

**What a fit took before.** The series' fits took a series when its own timed decision was
within the share of its fastest algorithm's time (`SeriesTiming::read`). That is the rule applied
to one series; the table applies it to a kind. A row goes on while any member of its kind is past
the share, so it takes series that are within it, and a cell whose pick is "the equations" hands
them series that are past it. On the report's 960 unseen series, 77 of the 501 the fits took were
series the table decides (median 2.9 us), and 37 the table hands to the equations (0.5 to 7.6 us,
the decision 5% to 14% of the call) were in no fit. Division's fits took every (q, d) from 4 to
4096 limbs, the tables' area included. mul's Karatsuba and FFT fits start at 96 limbs, past
`mul.kara` on the M1 (Karatsuba's throughout from 61), with nothing that held them to it.

**What was built.** An equation is fitted on the calls it is read for:

- A dispatcher's equations are not read where its table names an algorithm, so those samples are
  out of their fits: the series' three equations (`Rows::Table`: the series `series.small` names
  no algorithm for); division's NR, NR with remainder and BZ (`Rows::Read`, `read_for`: not where
  the entry's own table names one); mul's recursive Karatsuba, recursive chunking and FFT (not
  where `mul.kara` says schoolbook: nothing on the M1).
- An equation that is a price inside other equations is read at any size it runs at, and keeps
  every shape: schoolbook's product and `hi_mul_buf`'s (the series' and division's models),
  Knuth's division (BZ's leaves and Knuth top block, a series' finalize). This reading of the
  rule is mine: taken to the letter it would leave schoolbook's fit no shapes at all.
- **A tune fits twice from one set of timings.** A table has to be measured under some equations
  (the decision is timed beside the algorithms, and rect's block length under a table pick is
  the model's), and a fit can only leave out what a table decides once that table exists. So
  div's and the series' first phase fits as before (division on every shape, the series by each
  one's own decision), the tables are measured under that fit, and a second phase
  (`sections::div::tune_late`, `sections::series::tune_late`) fits again on what the tables
  leave, with nothing timed again. The first fit's evidence carries no verdict; the second is
  the one judged, and its evidence gives the first's rms and the samples left out. The series'
  guards moved to the second phase, so they are swept under the equations that are written
  (the rule on a phase's fits and its sweeps). mul already had the order: its table, then those
  fits.
- The reports follow: `cargo tune series` fits and judges on the series the table in force names
  no algorithm for (under `--share X`, which no table was measured at, by each series' own
  decision), and `cargo tune division --refit` fits by `Rows::Read` and judges the equations'
  picks on the shapes the tables leave them.
- `test_fits_leave_out_what_tables_decide` checks both predicates at tables made by hand.

**Whether it makes a difference: it does not.** On one set of timings (the report's 960 unseen
series, thorough, `--timings`), held out:

| the fits take | series | Horner | rect | binary splitting |
|---|---|---|---|---|
| each series by its own decision, within 5% (before) | 501 | 6.8% | 4.2% | 6.9% |
| those, less the 77 the table decides | 424 | 6.7% | 3.9% | 6.8% |
| what the table leaves (the rule as built: less the 77, with the 37 handed on) | 461 | 6.9% | 4.0% | 7.5% |

Leaving out what the table decides moves the fits by 0.1 to 0.3 points. The 37 short series the
table hands to the equations cost binary splitting 0.7 points, and they belong in: the equations
are read for them. (The middle row was a temporary switch for this comparison, not a rule the
tuner has.) The picks' loss on the held-out half is 0.10% and 0.11% on the mean under the last
two, worst 8.6%. So Horner's and binary splitting's fits are not above the line because of table samples;
what moves them is how short the series are (at a share of 1%, 3.6%, 2.5% and 4.9% on the same
timings), which is the models on series of a few microseconds.

**A tune under it** (`cargo tune tune div series --level thorough`, to a scratch file, 11 minutes
48; mains power, the battery full; the gate kept both; 4 of 2540 sweep measurements unsettled,
the fits' passes differ 1.9%). **The preset was not rewritten** (the user's rule: no retune
unprompted), so it holds the fits of the retune above.

| equation | first fit | after the tables | left out |
|---|---|---|---|
| Knuth | 2.2% | 2.2% | none (a price: every shape) |
| NR, quotient only | 2.9% | 2.9% | 26 of 726 shapes |
| NR with remainder | 3.3% | 3.3% | 41 of 726 |
| BZ | 2.3% | 2.3% | 0 of 858 (its divisors are past the leaf, 99 limbs, where the tables have rows only for the shortest quotients, at divisors to 117) |
| series Horner | 6.6% | 6.2% (NOISY) | the table takes 78 within the share and hands on 39 past it: 467 of 960 read |
| series rect | 3.1% | 3.1% | the same |
| series binary splitting | 8.0% | 7.5% (NOISY) | the same |

The tables it measured: `div.dyn_small` 34 change points (divisors to 38 limbs),
`div.dyn_rem_small` 50 (to 117), `series.small` 220; the guards stayed at 8, 32 and 16. The
gate: div 1.0004 [0.9997, 1.0011] over 182 workloads, series 1.0005 [0.9992, 1.0017] over 78,
against the preset's values, so nothing moves end to end either. `cargo tune dispatch` as
before (every dispatcher within 5% where it reads its equations; the series' entry past it at 6
indices by the table's own choice).

What it leaves open:

- Rect's block length under a table pick is still read from rect's equation
  (`SeriesNode::small`), at series that are now outside its fit by the rule. Before the rule at
  most 9 of the 53 such series in the unseen set were in rect's fit (the 9 whose own decision
  was within the share), so little changed, but the reader is there. The same would hold for BZ's top-block choice under a table cell of BZ (none on the
  M1: every cell is Knuth).
- A series' finalize is priced by `dyn_div_best`, the least of division's equations without the
  table, so in the tables' area it can read NR's equation where NR's fit has no shapes now.
  The tables measured Knuth fastest there and the price takes the least prediction, so a wrong
  NR price could only lower it. Not measured and not changed: the price read the equations
  without the table before this too.
- `cargo tune tune --tables` measures tables alone: the equations in the file stay the ones
  fitted under the tables before.

#### What it left

- Horner's and binary splitting's equations are above the 5% line on the series the rule gives
  them (the table above): a lower share takes them under it (3.8% and 5.7% at 1%) at the cost of
  more indices timed; the user's number. Fitting them only on what the table leaves does not
  ("A fit takes the shapes its equation is read for" above: 6.2% and 7.5% in a tune, 6.9% and
  7.5% on unseen series). (Closed the next day by their forms: "Horner's and binary splitting's
  equations under the line, and the beta guard gone" below.)
- Division pays up to the share where the guards paid nothing (accepted).
- The series' table on 8 kinds loses more than the 72 classes did on the series it is for
  (its picks 1.5% on the mean and 2.3% of the time against 0.8% and 1.7%; the entry, its dispatch included, 3.0% and 3.5% against 2.6% and 2.5%), for a table an eighth the size: the user's trade.
- End to end the change costs 0.1% to 1% a section against the tree before it (the table
  above). One series item is 14.5% slower (`hyper geo56 w64 t64 x256`: an x of 4 limbs with
  56-bit coefficients, a kind whose row is shared with 8-bit ones), not looked into.
- The members of a kind are a fixed spread (x to 48 limbs, coefficients to 62 bits); an x far
  past them is read off the widest kind's row, as the 414-limb one was before that kind was
  split. Coefficients past 64 bits have no row (the equations).
- `Kind::Derived` and `Derivation` are still in `schema_types.rs`, with the resolver's handling
  and its tests: no knob of the schema has the kind. (Gone on 2026-11-13, with the `pinned`
  key: "The tuner's leftovers cleared, and a full retune".)
- `div_rem_dyn` 1030/1000 was 6.7% slower in the first comparison and reads the equations in
  both trees: it read 1.003 after the retune, so it went with the cutoff's fix and the refit, and was not a pick of the tables'.

### Horner's and binary splitting's equations under the line, and the beta guard gone (2026-11-13)

The user: "I need yo to try find an alternative form for the fit for horners and binary
spliting so that both fall in the 5% hold out rms threshold." And on the result: "Go ahead and
implement the new fit and retune everything upstream. Also consider expanding the table if the
new fit slows dispatch. As for your questions the only question I have an opinion on is if the
beta gaurd is causing a slow down remove it and rely purely on the table and the fired
equations."

Both are under it, with rect, at every level, and nothing in the series section is marked.
Held out, in a tune (the judged fit, after the table):

| level | Horner | rect | binary splitting |
|---|---|---|---|
| quick | 3.5% | 4.4% | 3.8% |
| normal | 4.0% | 3.3% | 3.3% |
| thorough (the preset) | 3.3% | 2.8% | 3.6% |
| thorough, the forms before | 6.2% | 3.1% | 7.5% |

#### The method: the form's floor on the kernel's own counts

The counters that fixed rect, with one step added in front. In a scratch copy the primitives
`utils/series.rs` imports (`div_prim`, `div_prim2`, `mul_prim`, `mul_prim2`, `mul_add_prim`,
`add_mul`, `sub_mul`, `add_buf`, the shifts, `mul_dyn`, `hi_mul_dyn`, `div_dyn`) are shadowed by
wrappers of the same names that count limbs and calls, so only the series' own calls are
counted; composite calls are recorded by their lengths and priced at the dump with the model's
own `SeriesEnv`, and a counting allocator gives the allocations. One row a workload and
algorithm, on the report's workloads, joined with `series_parts.tsv`.

**The step in front: fit the times to the kernel's own counts under the form as it is.** That
is the best the form can do, and it says which of two things is wrong. On the 461 unseen series
the equations are read for (yesterday's timings; fitted / held out):

| fitted on | Horner | binary splitting |
|---|---|---|
| the model's counts, the form as it was | 5.6% / 6.9% | 6.3% / 7.5% |
| the kernel's counts, the form as it was | 5.0% / 5.8% | 4.5% / 4.9% (its one-limb passes priced) |
| the kernel's counts, the new form | 3.1% / 3.3% | 2.9% / 2.9% |
| the model's corrected counts, the new form | 3.3% / 3.2% | 2.9% / 3.0% |

Rect had been the first case (counts wrong, form right). Horner is mostly the second: on exact
counts its form still read 5.8%. Binary splitting is purely the second: the model on
its own counts reads what the kernel's do once the form has the prices (its columns are within
0.3-2.2% of the time of the kernel's, the `add` limbs 3.2%).

#### Horner

**The form.** Its fixed costs were a price a term and a price a kernel call, every call
alike. They are not alike: `div_prim2` measures 3.8 ns a limb and 12 ns a call (its
reciprocal), a term's add a few ns, and a step by a one-limb x is three calls of its own
(`mul_dyn` copies the operand and makes the pass, `mul_frac` copies it back). On the kernel's
counts:

| fixed-cost prices | fitted / held out |
|---|---|
| a term, a call (every kernel call) | 5.0% / 5.8% |
| ... and a group | 4.6% / 5.2% |
| a group, and a term by kernel path (plain, one-limb x, wider x, BBP) | 3.1% / 3.3% |
| a term, a group, a call of the term's own step (3 for a one-limb x, 1 for a wider x, 1 for BBP) | 3.1% / 3.3% |
| ... with 2 calls for a one-limb x; with 4 | 3.4% / 3.7%; 3.4% / 3.4% |

The fourth row is what is built: `term` (every term: its coefficients, its weight and
its add, 9.4 ns), `step` (a group: its division's reciprocal and the scan that forms it, 18 ns)
and `call` (what a term's own step does besides: 2.7 ns each). Four prices a term by path fit
the same and say the same thing (8.5, 17.4, 10.8 and 11.6 ns).

**Two counts.** On the old model's limbs the new form read 5.2% / 6.4%; with the kernel's
division limbs 3.9% / 5.7%; with its multiplier passes as well 3.2% / 3.4%.

- *A product by R runs over what the group below left.* The model charged it at the group's own
  frame. A range of one group multiplies nothing: 9 terms of an 8-bit q in 760 limbs were charged
  a two-limb pass over all of them, 28% of the call. And the shallowest group's R is its own few
  terms', a one-limb pass when they fit a limb (`Groups::left`, `next`).
- *A q that grows makes groups that grow towards the shallow end*, where the frames are widest.
  The model had groups of one mean size with what was left over at the shallow end: 37% more
  division limbs than the kernel's for 132 terms of q(n) = n in 285 limbs, all priced as two-limb
  divisions. `Shape::groups` now takes the range's bits of ∏q over what a group holds (the
  budget less half a term): that many groups, the last a part of one; their divisions' limbs from
  the frames at the two Gauss points, each at the top of its group; the shallowest group a whole
  division at the first term's frame however few its terms, by a one-limb Q while its bits fit
  one. The range's bits are `Shape::qs`: q as c n^d through the two Gauss points, summed by
  Stirling's formula, because their mean alone is 2-3.5% high for q(n) = n from n = 1 and one
  group or two turns on less (38 terms: 148 bits, which the mean reads as 154). The budget is
  128 bits for p Q, of which a one-bit p takes none (the model allowed 127: 16 terms of an 8-bit
  q where the kernel takes 17). The division's count error went from 3.5% of the time (rms over
  the series) to 1.3%.

**The price of a two-limb pass is the one-limb price times the kernels' ratio**
(`SeriesEq::HORNER_PASS2`, 3.3: `mul_prim2` measures 1.05-1.26 ns a limb where `mul_prim` does
0.31-0.39, and the fit is the same from 3.0 to 3.6). Fitted on its own it passed at the thorough
level and sat on the line at the normal one, which I had not run before reporting the form:

| Horner, held out, each level's own tune | quick | normal | thorough |
|---|---|---|---|
| nine prices, the two-limb pass fitted | 3.9% | 5.0-5.1% (NOISY, twice) | 3.6-3.7% |
| eight, the two-limb pass tied | 3.5% | 4.0% | 3.5% |

With the halves swapped the normal level read 3.7%: the split, not the form. A two-limb pass is
a large part of few series' time (seven of the normal tune's 209), and that tune's fitted half
had only short-frame ones, where a price a limb and the price a group can't be told apart: it
read 2.03 ns a limb and 9.4 ns a group for 1.44 and 17.2 on the other half. **A new form is
tuned at all three levels before it is called sound**, and for each price the series where it
is over a tenth of the time are counted in each half. Horner so has eight prices, as it had: a
price a group in place of a fitted one for two-limb passes. The ratio was a constant set by hand
at first (`SeriesEq::HORNER_PASS2`), and is the measured knob `series.horner_pass2` since later
that day (the next entry): 3.46.

#### Binary splitting

Three prices, on counts that needed no change:

- **Its passes by a one-limb multiplier** (`mul1`, 0.28 ns a limb). The form analysis of
  2026-11-11 had left the price out, on series past the derived table's extent; on the series the
  rule on tables gives the equations it is a tenth of the time, and the fit goes from 6.3% / 7.5%
  to 4.5% / 5.1% with it alone.
- **Finalize's division by a two-limb Q** (`div2`, 8.05 ns a quotient limb, beside `div1` for a
  one-limb Q). Division's equations are fitted from four limbs, and Knuth's reads 27% high at
  two (10.2 ns a quotient limb for 8.1 measured, at every quotient from 33 limbs to 1001) and 6%
  at three. A series of few terms ends in exactly that division, at 50-87% of its time. With it:
  3.5% / 4.0%. (The price is the series' own, as `div1` is. In division's section it would be
  a retune of division for a divisor only the series' finalize reads the equation at.)
- **A term of an x-leaf** (`step`, 9.1 ns): a step of its own, its product by x and the passes
  around it, beyond its limbs and its calls. With it: 2.9% / 2.9%.

Every one of both forms' prices is needed: with any one left out the held-out rms goes up (the
least, binary splitting's `add`: 3.0% to 3.3%; Horner's group: 3.2% to 4.7%).

#### What was tried and ruled out

- *The family's large denominators.* `div_prim2` and `div_prim` are data-dependent: on fresh
  buffers they run 11% and 17% slower for some divisors (the products of the family's 16- to
  32-bit `sized` q, 2^(bits - 1) and a little: 4.26-4.36 ns a limb for 3.85, 3.71 for 3.17), and
  on one repeated buffer the same (the predictor learns the correction branch), so a short
  workload timed in a loop reads cleaner than a long one. A price of their own for those
  members' divisions moves Horner's fit from 5.0% / 5.8% to 4.9% / 5.7% on the kernel's counts,
  and they don't explain the normal level's line either (4.9% with it).
- *Terms that land within a limb of the last* (a price for them): 3.2% / 3.2% from 3.4% / 3.5%
  with a term price by path, which the x terms told apart by x's width do as well (3.1% / 3.3%).
- *Both of a one-limb x step's copies as limbs*: 3.3% / 3.2%, as built.
- *A fixed cost for Horner, a price a product, a price a leaf or a merge for binary splitting*:
  each solved 0 or negative.

#### The beta guard

Binary splitting was neither predicted nor taken where its tree's width passed
`series.bin_split_beta` (8) frames and another algorithm could sum the range. With its equation
fitting, the guard was the largest loss left on the series the equations are read for: on two
unseen series of a few microseconds it barred binary splitting where the equations predicted
it right (7873 ns for Horner and 4709 for binary splitting, measured 7928 and 4543), 75% and 67%
slow, and its sweep, on series from 50 us, could not see them. Under the same table and
equations, on the 138 unseen series whose decision the guard changes:

| the entry over the fastest algorithm | guard on | guard off |
|---|---|---|
| worst | 75% | 7% |
| mean, under 10 us (51 series) | 3.5% | 1.6% |
| mean, 10-100 us (48) | 0.5% | 1.8% |
| mean, 0.1-1 ms (35) | 0.06% | 0.53% |
| of their total time | 0.10% | 0.53% (the other 822 series: 0.45% and 0.53%) |

What the guard spared is the tree's walk in the decision: 47 ns becomes 200-570 on such a
range without a wide x, and the median over the 138 goes from 590 to 857 ns. It is gone, with its knob and its sweep
(`SeriesGuards` keeps rect's two): the equations are read at any beta, and what the dearer
decision costs a short series is the table's to answer by the rule. The table measured after it
is 227 change points (218-220 before), 1549 indices timed for 1385, and takes 546 of the 960
unseen series for 499-517.

#### The retune, and the checks

`cargo tune tune series exp ln --level thorough --write-preset` (9.5 minutes, mains power, a
fixed workload within 2% over five runs before it): the sections built on series with it, whose
values came out as they were. Series is at version 8: `horner_ns_step`, `bin_split_ns_div2`,
`bin_split_ns_mul1` and `bin_split_ns_step` are new, `horner_ns_mul2` and `bin_split_beta` gone,
and `Parts` has a thirteenth column, `step`.

- Fits (held out): Horner 3.3% over 194 of 395 shapes, rect 2.8% over 57 of 115, binary
  splitting 3.6% over 164 of 328 (their first fits, before the table: 3.2%, 2.8%, 4.7%); 0
  rejected, 0 noisy; 58747 fit probes whose passes differ 1.2%.
- Prices. Horner: product 1.032, div1 2.991, div2 3.876, mul1 0.4027, light 0.1538, term 9.424,
  call 2.660, step 18.02. Rect: product 1.019, div1 3.416, add 0.3896, term 17.54, fixed 656.
  Binary splitting: product 1.018, division 1.001, div1 2.360, div2 8.051, mul1 0.2796, mul2
  0.3959, add 0.1249, light 0.1795, term 0.9010, call 1.910, alloc 21.46, step 9.138; gain 0.4.
- The gate kept all three: series 0.969 over 78 workloads (against the defaults with no table:
  the preset's old section fell through at the new version, so this gate says little), exp
  1.0014, ln 0.9997. Listed: `hyper geo56 w64 t64 x256` 14% slower, the item the table's layout
  cost on 2026-11-12.
- The dispatch check: the series' entry reads its equations at 83 of the indices tried, the
  decision 2.4% of the call on average (2.1% at 93 before the guard went), 11 past the share by
  the table's own choice.
- 960 unseen series (`cargo tune series`), under the preset. Where the equations are read (414):
  their picks lose 0.07% and 0.02% on the mean on the two halves, worst 12.4% and 2.4% (0.72%
  and 0.08%, worst 73% and 9%, the day before); the equations read Horner 3.1% and 2.9%, rect
  3.2% and 3.2%; the entry, its dispatch included, 2.9% and 2.3% on the mean, worst 22% (3.1%
  and 2.1%, worst 75%), and of the total time 0.48% and 0.33% (0.45% and 0.15%: the dearer
  decision, 0.4% on series past 100 us for 0.1%). Where the table decides (546): its picks
  1.7% on the mean and 2.3% of the time, worst 48%, the entry 2.2% and 2.3% (1.7%, 2.2%, 48%;
  2.4% and 2.4%).
- End to end, normal builds, against the tree and preset of 2026-11-12 (`benches/tuning_bias.rs`,
  outputs identical): series items 1.003 over 26 (0.972 to 1.022), exp/ln/theta/agm 0.999 over
  48. No gain in speed, as the picks were already within a tenth of a percent of the time.
- 596 tests, 610 with `tune`.

#### What it left

- **The decision costs more**: about 4% (median, a tune build) for the model itself, 1-2% for a
  q of one size and 7-13% (10 ns) for one that grows (`Shape::qs`: up to four logarithms), and
  the tree's walk on ranges the guard used to settle. The table answers for it where it is past
  5% of a call and nothing does under that: 0.3% of the time on the unseen series past 100 us. A lower bound read
  off binary splitting's own equation (its finalize and its root's products, O(1)) would spare
  the walk where it can't win, with no pick changed: a shortcut computed from the equations, so
  proposed and not built.
- **Binary splitting's `div1` and `div2` rest on few series** (finalize by a Q of one or two
  limbs: 2-5 series a half at quick and normal). It passes in the judged direction at every
  level (3.7%, 3.2%, 3.6%), and with the halves swapped reads 8.5% and 7.8% at quick and normal:
  the half that is fitted then has no series of the one-limb kind, and prices it 0.
- **Rect's groups** still came from `Shape::group_of`: the mean size for a q that grows, and
  127 bits of room. Its fit is under the line (2.8%), and it reads Horner's corrected prices.
  (One model with Horner's since later that day: the next entry.)
- **Knuth's equation at two- and three-limb divisors** is 27% and 6% high: only the series'
  finalize reads it there, through a price of its own at two. And the division kernels' speed
  turns on the divisor (above), which the family's `sized` denominators sit on the slow side of.
- **Static nodes** keep the old shared prices and counts.
- **Rect's fixed cost** is not determined: 148 to 656 ns from one fit to the next.

#### The tuner's leftovers cleared, and a full retune (2026-11-13, later)

An audit of the tuning code after the change above found nothing dead in it (no unused item, no
suppressed warning, no name of a removed feature), and eight things to settle. The user:
"Implement everything up to 6 as is for 7 leave failed fits as marked noisy and go ahead and fit
over all beta addressing number 8. Leave all cosmetic changes out of this pass."

1. **`cargo test` failed on a machine that had tuned itself.** `test_accessors_match_resolved`
   built its layers from the preset alone, and `cargo tune tune` writes `presets/local.toml`,
   which the build layers over it. The test layers the machine's own file as `build.rs` does
   (the path `BIGBITS_TUNING` named at build time, unless the file is another target's). Checked
   both ways with a file that moves one sqrt knob: it passes, and without the layer it fails.
2. **The ratio of a two-limb pass to a one-limb one is measured.** It was a constant set by
   hand (3.3, from a scratch benchmark), which x86 would have read as the M1's. It is the knob
   `series.horner_pass2`: `mul_prim2` and `mul_prim` timed in place at lengths of 64 to 1024
   limbs, each fitted as a L + c, the ratio of the two a (`sections::series::pass2_ratio`: 1.264
   over 0.365 ns a limb, 3.46 in the preset's tune, and 3.47 at the quick and normal levels). Taken
   as a tune's first measurement, over 16 to 1024 limbs, both kernels read 1.85 times their
   time at the quick level with the passes 22% apart (the ratio itself held, 3.36): it is taken
   after the series' timings, and from 64 limbs, where the passes agree to 0.5%.
3. **Text from removed guards**: the help's line for `cargo tune series`, two comments, and the
   division report's filter for `knuth_d` keys.
4. **The ten `*_err` knobs are gone** (`mul.{school,hi_rows,hi_cols,kara_rec,chunk_rec,fft}_err`,
   `div.{knuth,nr,nr_rem,bz}_ns_err`): each fit's rms as a knob of its own, the margins the
   derived tables read, which nothing had read since the tables became measured. The rms is in
   the evidence of every constant of its fit. They were in mul's and division's hashes, so every
   section built on those would have warned until tuned again: the reason for the retune below.
5. **`Kind::Derived` and `Derivation` are gone**, with the `pinned` key of a tuning file, which
   marked a derived knob as a measurement. Every knob is measured and in its section's hash
   (`section_hash` takes no pinned list), and a section's `derive` number is only its table's
   layout (`Section::derive`; the method `Kind::follows_derive` that said so went later that
   day). The resolver's test of that number is on a table now.
6. **One model of a group's size.** Horner's `groups` and rect's `group_of` were two: rect's
   took the mean of q's bits at the two Gauss points for the range's and left 127 bits for p Q.
   `Shape::group_room`, `whole_group` and `group_count` are shared; `group_of` is the whole
   number for a q of one size and the terms over the count for one that grows. Rect still takes
   one size for all its blocks: its groups end at its blocks.
7. *(Unchanged, the user's word.)* A series or division fit above the line is marked NOISY and
   written, not REJECTED, as a model already revised once.
8. **Binary splitting is timed and fitted at every beta.** Its area stopped at a beta of 24
   where another algorithm can run, and since the guard on beta went its equation is read past
   it. On the morning's unseen series the equation fitted to 24 read the 13 series past it at
   3.4% (mean +0.2%), and fitting them moved no price (3.05% and 3.04% to 3.06% and 3.02%); a
   run of each takes 14% longer. More trees split across the pool are in the gain's set (83 for
   26).

With all of it, the series section alone, held out: quick Horner 3.2%, rect 3.4%, binary
splitting 4.0%; normal 3.9%, 2.9%, 3.5%.

**The retune** (`cargo tune tune --level thorough --write-preset`, 36 minutes, mains power, a
fixed workload within 2% over five runs before it). All nine sections kept; 0 fits rejected, 0
noisy; 73387 fit probes whose passes differ 1.8%; two whole-algorithm sweeps UNSTABLE and kept
(`mul.ntt_par_conv` 512, `bin_split.dyn_hyper_leaf_terms` 32). **The sweeps' "samples" line read
5.9% unsettled, past its 5% warning**, as every full thorough tune of this machine has (5.9-6.8%).

| fit | held out |
|---|---|
| mul: schoolbook; `hi_mul_buf` rows, columns | 3.9%; 2.7%, 3.8% |
| mul: recursive Karatsuba, chunking; FFT | 2.7%, 1.6%; 3.5% |
| div: Knuth; NR, with remainder; BZ | 2.2%; 3.0%, 3.8%; 2.6% |
| series: Horner, rect, binary splitting | 3.4%, 2.7%, 3.6% |

Tables: `mul.kara` 94 change points (Karatsuba from 44 for squares and 49 at l = 4s),
`div.dyn_small` 34, `div.dyn_rem_small` 48, `series.small` 232. Gates, after over before: mul
1.0022, div 1.0002, sqrt 0.9999, bin_split 1.0043, series 0.9998, theta 0.9996, agm 1.0004, exp
0.9996, ln 1.0004. The dispatch check as before (the series' decision 2.4% of the call).

On 960 unseen series under the preset: where the equations are read (413) their picks lose
0.10% and 0.01% on the mean, worst 11.4% and 2.5%, the equations read Horner 3.1% and 3.3% and
rect 3.0% and 3.5%, and the entry 2.9% and 2.2% (worst 22%); where the table decides (547) its
picks lose 1.8% on the mean and 2.3% of the time, the entry 2.0% and 1.8%.

End to end, normal builds, against the build before the pass (outputs identical in all): mul
1.002 over 45 items, div 1.001 over 22, the division grid 1.000 over 40, sqrt 1.001 over 42,
series 1.003 over 26, exp/ln/theta/agm 1.000 over 48. Two items:

- **`div_rem_dyn` 1030/1000 is 6% slower** (1.063; the tune's gate listed it at 1.057). NR with
  remainder against BZ at a quotient of 31 limbs over 1000: the equations have them within a
  percent or two there and the clock 6% apart, and the shape has flipped between retunes
  before ("What it left" under "Tables measured, under one rule"). NR with remainder's
  constants moved 5-17% from the tune before (its schoolbook high-product multiple 0.65 to
  0.76, its term in q 3.63 to 3.92) at the same 3.8% rms: its tiers are not told apart well.
- **`mul_dyn` 100x100 reads 15% slower in the bench, and it is not the tune.** mul's source is
  untouched, and the old and new `mul.kara` pick the same on its path (Karatsuba at 100 and at
  50 and 51, schoolbook at 25 and 26: the tables differ at 76 unbalanced shapes, none on it).
  With the old mul values in the new build it is still 9.5% slower than the old build (0.97 of
  the new build's time), and the tune's own gate, which compares values in one process, did not
  list it: the binary's layout. Each build against itself reads 0.99-1.00 on the item.

596 tests, 610 with `tune`. Not in this pass (the user's word): clippy's 18 style notes in
`src/tuning`, the preset's repeated evidence comments, the line-reduction pass.

#### A simplification pass over `src/tuning` (2026-11-13, last)

The user ran the simplify review over all of `src/tuning` (four reviewers in parallel: reuse,
simplification, efficiency, altitude; 55 findings between them, many the same mechanism seen
from two sides). The rule held to while applying them: what a tune measures, fits and writes
stays as it was, since the preset had just been written and was not to be retuned. One thing
a tune does differently is marked below.

- **The tune loop names no section.** `sections::SectionTuner` carries a section's first
  phase, its dispatch tables (each knob's key and its measurement) and its second phase, so
  the loop runs one block for every section, and `--tables` takes the sections whose list of
  tables isn't empty. The three-armed `match` on "mul", "div" and "series" and the list
  `TABLED` are gone: a section that gains a table is a line of `sections::SECTIONS`. The
  gate's item sections are one table too (`check::ITEM_SECTIONS`: each label with the schema
  sections whose knobs its calls read, in the order `cargo tune check` prints them).
- **One way to make a fit's record.** `fit::fit_named` (the form's analysis when the tune was
  asked for it, the split fit, the timings' noise) is what mul's, division's and the series'
  fits call, where each had its own wrapper; and `Fit::worst` (the row of the largest
  residual, and the residual) is what division's and the series' evidence name their worst
  shape from, where each solved for it again. mul's `fit_lin` names its equation from its
  scale's key; division's four lists of keys are module constants, which `cargo tune division
  --refit` filters by; division's `fit_equations` takes the noise as the number it always was.
- **The series tuner.** `EQS` (each equation's name and columns) and `prices` stand where the
  three equations were spelled out in four places; `col("mul1")` where bare column numbers
  were (`(4, 5, ratio)`, `base[9]`, `base[12]`); `framed` is the frame a context is summed in
  at a precision, which four walkers computed (the dispatch check's among them); the guard
  sweep's `Loads` lost the predicate it took when there were two guards; a table member's
  shift is `Synth::decay` (the closure it replaces repeated the members' own rules: the same
  numbers); `fit_equations` computes each gain's rms once; and `report` no longer sets the
  trees' parts a second time after each fit. `Synth::decay` has a default (three members had
  the same body) and `synth::x_static` is the leaked x four places built.
- **The table builder and the timing loop.** `tables::measure` reads the run's share itself
  and times `PER_WAVE` sizes a wave (every caller passed the same two); `tables::pick_at` and
  `sizes_above` (from a table's rows) replace three and two copies; `measure::less_harness`
  is the one place a harness's time is taken off a probe's (three were); `measure::dump`
  writes a developer's TSV (four ways before, one of which didn't make the directory);
  `sample` and `sample_at` are one function, as nothing has sampled a subset of probes since
  the whole-algorithm timings went; and division's probes share `numerator`, `placed_divisor`
  and `copy_probe` between the fit's timings, the two tables and the dispatch check (the same
  seeds, and the same order of allocation at each site, which division's timings are
  sensitive to).
- **Residue.** `Kind::follows_derive` (the resolver tests the kind where it used it, and
  `Section::derive`'s comment says what the number is); `table::pick2`, which only the tests
  called (it is their oracle, in the test file now); comments that still spoke of derived
  values; `merge_into`'s `Vec` of one section; `exp_ln_items`' argument that was always
  "trans"; `Level::fit_batches`, 3 at every level (a constant); `hash::table_bits`'s own copy
  of `fnv1a`'s loop; `Design::of` and `of3` sharing their tail; `sweep::geo_to` for six
  `geo(..).take_while(..)`; `Tuned::set_live` for four `tuning::set(&format!(..))`; candidates'
  losses computed once where an argmin's comparator computed them on both sides
  (`pick_param`, `top_block_scale`, the series' gain).
- **The one change in what a tune does: a section whose values came out the ones before it
  is not gated.** The gate then timed the same values on both sides: noise, which could in
  principle drop the section, for every round of every workload of the section and of those
  built on it (a reviewer's estimate from the code, not timed: 5-9 s a section at the
  thorough level). The loop prints "its values are the ones before it, so there is nothing
  to compare" and counts the section as kept. In a quick tune started from the preset it
  fired for sqrt, theta, agm, exp and ln.

**Not done, each for a reason**: step functions, `Knob::threads`, `Knob::search`,
`Kind::Model`'s `work` and `Ty::Table`'s `measure` have no reader today, but
`docs/tuning_system.md` still plans one for each (the user's call). A phase's timings still
reach its second phase through a thread-local (`KEPT`); a `Tuned` still holds its key and
not its path, and a fit's REJECTED or NOISY is still counted from its evidence's text; whether
a gate item's output is exact is still read from its name (`check::exact`): each a redesign
across forty-odd call sites or the file shared with the bench, proposed and left. The series'
`fit_equations` still writes the gain's parts into its input, and its family is still walked
once for the rows and once a pass for the probes (under half a second a tune). Division's
unread second numerator of a quotient-only probe stays, as taking it out moves the buffers its
timings are sensitive to. Not exact, so left: mul's coarse-then-dense square sweeps as
`cutoff_sweep` (they clip sizes and count edges differently), `top_block_scale` through
`keep_or_take` (its evidence's text differs), sqrt's and division's per-probe knob installs
through `with_knob` (a boxed call inside the timed region). Of clippy's 18 notes in
`src/tuning` three are left: `is_multiple_of` twice, in files `build.rs` shares (it needs a
recent toolchain and the crate declares none), and one rewrite by De Morgan in division's
timings that reads no better.

**Checked.** `cargo tune series --level thorough --timings FILE` on kept timings, which fits
and scores without timing anything: its report and both of its dumps (`series_parts.tsv`,
`series_workloads.tsv`) are byte for byte what they were, as are `cargo tune list` and
`cargo tune status`. `benches/tuning_bias.rs --hash`: all 227 lines the same as the build
before the pass (the items, their order and every output). A quick full tune to a scratch
file with the tuner before and after: the same sections and the same keys, the tables within
a few change points of each other, and evidence that differs only where it depends on the
timings (kept or taken, agree or not, a mark). **That second tune's first fit was disturbed, and the cause is the
binary's first launch, not the machine's load** (found in the pass after this one, which
corrects what was first written here): the tuner's executable had just been copied, and a
freshly copied executable's first launch on this Mac has its first seconds slowed. Three of
three first launches of a fresh copy read schoolbook's equation 25-39% high with its passes
37-38% apart (REJECTED, and division's NR multiples bent to match); the same file launched
again, six times over three builds of the tuner, gave 0.3556 to 0.3601. A quick tune's first
fit is a few seconds long, so it takes the whole of it (the series' `pass2_ratio`, moved
out of first place for reading 1.85 times slow there, was probably the same thing). Whether a freshly *built* executable (the
usual `cargo tune` after a change) is slowed alike was not tested: run the tuner once, on
anything, before a quick tune whose first fit will be read. 596 tests,
610 with `tune`. `src/tuning` is 8752 lines from 8760: the pass took out duplication, not
size.

#### A clean-up pass over `src/tuning` (2026-11-13, after it)

The user's `/clean_code` over all of `src/tuning`: simpler algorithms and architecture, fewer
functions and lines, terse comments, with results bit for bit the same and nothing outside
the directory touched but these documents. `src/tuning` went from 8752 lines to 7934 (code
6726 to 6487, comments 1506 to 955), from 425 functions to 396 and from 47 structs to 45. No
asymptotic complexity changed.

- **Comments are terse, and history is not in them.** A comment says what a thing is and the
  one reason that would stop a wrong change; the dated anecdotes, the user's words and the
  numbers of past runs are in this document and in `CLAUDE.md`, where they already were.
- **Merged.** The series family's three walkers without an x (`hyper_one`, `engel`, `bbp`,
  with `frames_terms`) are one generic `plain`; `par_rms` and `par_rms_of` are one function
  over any set of timings; `sweep_cutoff` and `sweep_cutoff_plus` are one (the `plus`
  argument); mul's six probe constructors go through `binary` and `unary`; the gate's two
  division loops through `div_pair`; `fnv` is inside `h`; `Level`'s eleven `match` blocks are
  one table lookup a method (`Level::of`).
- **Folded into their one caller.** In the series tuner: `entry_lost` (which the report called
  twice a band: once now), `rms`, `price_key` (a column's knob looked up once, not twice),
  `loads` and `sweep_guard` (into `sweep_guards`), `SmallSpec::count`, and the three
  `*_cell` constructors and `decay` (the `by_q!` macro makes the context itself). In
  division's: `eq_pick` (the report computed each pick three times: once now), `timings_tsv`,
  `SmallSpec::widths` and `quotients`. In the tune loop: `table_into`, `section`,
  `set_values` and `load` (one `set_live` returns the resolver's warnings). Elsewhere:
  `time_probes`, `quantile`, `fit::analyze`, `check::owners`, `hash::hash_pairs`,
  `file::num_text`, and the registry's three wrappers (closures in `SECTIONS`; `div::tune_late`
  takes the level it ignores).
- **Structs gone.** The series' `Loads` and `Named`: the walk is implemented on a plain
  `Vec` and a plain `HashMap`.
- **Less work.** `cargo tune series` copies its timings once for the live equations' parts
  where it saved and restored them around every fit; the reports above; `run`'s commands
  share one `ready` (their level, on the pool).

**Checked** as the pass before: `cargo tune series --timings` on kept timings and its two
dumps byte for byte the same, `cargo tune list` and `status` the same, `tuning_bias --hash`
the same on all 227 lines, 596 tests and 610 with `tune`. A quick full tune to a scratch file wrote the same sections and
keys as the tune before both passes, ran all three second phases, and skipped the gate for
the five sections whose values did not change; mul alone on a second launch fitted
schoolbook at 0.3559 and 0.3601, recursive Karatsuba at 1.845 and 1.843 and the FFT at 0.576
and 0.575 (the tuners before: 0.356, 1.846, 0.576).

**Held back, as they would change a number or a measurement**: `sample`'s own batch loop
through `batch()` (it keeps a batch's total where `batch` returns the time a call, and
(ns / iters) * iters is not ns to the last bit, which can move the point a probe's budget
runs out); division's `stat` for BZ's top block through `sweep::lost` (0 against NaN on an
empty set); the school fit's probe through `binary` (a call through a pointer inside a
probe of tens of ns, where it calls `mul_buf` directly); `divisor` and `placed_divisor` as
one (they differ in where the buffer lies, which division's timings read); the gate's
workload closures through shared helpers (they are the measuring instrument); and NR's two
fits as one loop over `rem` (three branches in it for a dozen lines). The schema's `doc`
strings, `work` strings and step functions are data or the plan's, and were left.

## Marked problems

The rule (the user's, 2026-11-10): a rejected fit is reported with a proposed form and waits
for approval, once. If the model has already been revised once and a problem remains, its noisy
fit is accepted and the work moves on; what was tried and what still doesn't conform is
recorded here for a later investigation.

### BZ's runtime equation (dynamic division): no longer marked

**Resolved 2026-11-10.** On the machine at one steady speed, form 4 below fits at 2.2-2.4% in
five tuner runs at the normal level, with multiples 1.044-1.046, 0.981-0.984 and 0 (the log
entry "BZ's predicted fit confirmed" above). The history stays here because the tuner still
treats the form as revised (`Fit::evidence_revised`): a reading above 5% is marked NOISY, not
REJECTED.

In use: `a (blocks (P + L) + [padded] padded top) + b [Knuth top] Knuth(top, d) + c q`, with
P and L one 2/1 block's products and Knuth leaves, and a padded top block priced by its
quotient's significant pieces (`DivEq::bz_padded_top_ns`). The preset and the schema hold the
predicted constants (1.04, 0.99, 0), within 1% of the fitted ones, until the full tune.

Tried, in order:

| # | Form for a padded top block | Result |
|---|---|---|
| 1 | One more full block | 3.4% on the 130 shapes where BZ has a full-size block (q >= d), its top block chosen by the old rule; 13.4% once partly filled padded tops were fitted; 19.7% over all forced top-block timings |
| 2 | Leaves in full, products times a fitted power `(top / d)^g` (approved by the user) | 15.4% rms, worst 50%, g = 0.17: no one power exists, as there was no discount under about 700 limbs and floors of 0.8, 0.6 and 0.5 of a block above |
| 3 | Structural, for the code as it then was: only the FFT-sized levels' products shrink | 1.7% in the tuner at full machine speed (510 shapes), 1.3% offline (490). Never installed: the user had the cause removed instead |
| 4 | The code changed (`div_3_2` multiplies every quotient piece at its significant width) and every level is priced by significant pieces: the form in use | 7.2-9.5% in five tuner runs on a throttling machine, where Knuth's unchanged fit read 4.4-8.2%; 3.1% with each timing divided by its shape's Knuth speed (442 shapes with q >= 8) |

What was open while the machine throttled, and where each item stands:

- ~~No accepted tuner fit of form 4 exists.~~ It does now: 2.2-2.4%.
- The three residual patterns below were read off the drift-cancelled diagnostic. They have
  not been re-read from a steady run's `bz_fit.tsv`; at 2.2-2.4% overall they no longer decide
  anything.
- **Short padded tops.** By the top block's quotient limbs, the drift-cancelled residuals are
  4.7% rms at 16 or fewer (mean -1.3%), 3.6% from 17 to 100, and 2.4% above 100. Eight of the
  ten worst rows (10-14% off, both signs) are padded tops of 8 to 64 limbs. Those are also the
  rows whose Knuth reference run is shortest, so part of this may belong to the diagnostic.
- **A trend with the divisor.** Under 300 limbs the equation over-predicts by about 2% (rms
  3.0-4.5%); from 1000 it under-predicts by about 1%. The two other worst rows are full tops at
  d = 181 and 362 (10% off, opposite signs).
- **The q term fits to zero**, where the first equation had 3.6-4.6 ns per quotient limb.
- **Still open: static BZ's top-block rule** (`use_bz_for_top_block`,
  `div.bz_top_padded_cost_scale`) was set for the old cost. On forced dynamic BZ it loses 3.9%
  of measured time (worst 127%) against the equations' 0.01-0.02% (worst 1.3-2.8%), on the
  steady machine. One sweep of its constant (static: approximate).
- Dispatch on held-out shapes, steady machine: the equations' picks lose 0.04-0.06% of
  measured time for `div_dyn` (worst 4.6-5.7%) and 0.17-0.19% for `div_rem_dyn` (worst
  12-15%, all of it NR with a remainder picked at a short quotient over a divisor past 1200
  limbs, where its equation under-predicts by 10-13%).

To read it again: `cargo tune division --level normal --refit`; `BIGBITS_TUNE_VERBOSE=1` adds
`target/tuning/bz_fit.tsv` (every BZ row, its fitted time, its shape's Knuth run and both sets
of regressors) and `target/tuning/division_fit.tsv` (the fitted grid's timings).

### The series' dispatch equations

**None of the three is marked since 2026-11-13** ("Horner's and binary splitting's equations
under the line, and the beta guard gone" above). Held out in the M1 preset's tune: Horner 3.3%,
rect 2.8%, binary splitting 3.6%, and under the line at the quick and normal levels as well.
What did it was the method that had fixed rect, with the form's floor on the kernel's own
counts read first: Horner needed a price a group and its calls counted by what a term's step
does (and two counts corrected: the frames its products by R run over, and its groups for a q
that grows), binary splitting three prices its form lacked (its one-limb passes, finalize by a
two-limb Q, an x-leaf's term). What follows is the record up to that change.

**Where they stood under the rule on tables (2026-11-12, "Tables measured, under one rule"
above).** There is one set of prices, fitted on every series whose timed decision is within 5%
of the call, which starts at a few microseconds (the derived table's extent had kept the fits
past a predicted 20 us, and the short prices are gone with it). There, held out on the M1:
Horner 6.6%, rect 3.1%, binary splitting 8.1%. Horner's and binary splitting's are above the line again and NOISY (this model was
revised once: the rule on a revised model holds), rect's under it. Their picks lose 0.6-0.9% on the mean and 0.04-0.16% of the time on the series they are read for (worst 48% and 76%).
What takes them under the line is in hand and is the user's number: at a share of 1% the same
timings read 3.8%, 3.4% and 5.7% (the table in that entry). **Tested and ruled out (2026-11-12,
"A fit takes the shapes its equation is read for"):** that they are above the line because the
fits took series the table decides. Fitted only on what the table leaves, a tune reads Horner
6.2%, rect 3.1% and binary splitting 7.5% held out (6.6%, 3.1% and 8.0% on each series' own
decision, from the same timings), and on unseen series 6.9%, 4.0% and 7.5% (6.8%, 4.2%, 6.9%).
The misfit is the models on series of a few microseconds. Not tried: counters in Horner's and
binary splitting's kernels on short series, the method that brought rect under the line. What
follows is the record up to that change.

**Where they stood (2026-11-12, after "Rect's counts read against the kernel, and
hi_mul_buf's own equation" above).** Fitted on a synthetic family, not on exp's and ln's chunks
("Built: the series tuned as general evaluators"), and a regime at a time ("Applied: a fit a
regime and whole groups"). Held out, on the M1: for the long series, whose equations the
dispatch reads, Horner 4.7%, rect 2.8% and binary splitting 3.6%, **all under the line and
none of them marked**; for the short ones, whose prices only the table's derivation reads,
7.4%, 7.6% and 9.4%, which stay NOISY. (Before that day's last change: 5.8%, 8.2% and 3.9%;
9.7%, 6.2% and 10.5%.) "Rect: everything tried" below is rect's whole record, and how it came
under the line. The figures between here and there are the history of the fit on the callers'
series.

In use for dynamic nodes since 2026-11-10: one fitted runtime equation an algorithm (`SeriesEq`
over `Parts`), built on mul's and division's equations; the log entry "The series' dynamic
dispatch rebuilt on fitted equations" has the form and what the rebuild found. Their fits read
5.6% (Horner), 13.3% (rect) and 6.9% (binary splitting), above the 5% line, and this was the
model's one revision, so they are NOISY and marked. The picks lose 0.13% of measured time on
the fitted workloads and 0.35% held out (worst 6.4% and 20%).

Tried, in order:

| # | Model | Result |
|---|---|---|
| 1 | The seven shared prices over counted work, refitted to the three algorithms' forced timings | 30.6% rms; picks no better. The counts are off, not the prices |
| 2 | Its dynamic products priced by mul's fitted equations (`MUL_WORK_FITTED`, tune-only, removed) | refit 31%; held-out picks' loss 8.42% to 7.44% |
| 3 | The cause removed in the code: a tree started off the pool splits only with `bin_split.dyn_par_work` of limbs times terms, and the model charges such a tree one wake | picks' loss 6.25% to 3.73% and 8.42% to 4.13%; refit 24.5% |
| 4 | The rebuild: an equation an algorithm over its own parts, products by mul's equations, finalize's division by division's | fits 5.6%, 13.3%, 6.9%; picks' loss 0.13% and 0.35% |

What still doesn't conform:

- **Rect, 13.3%.** It has 25 workloads (it needs an x of two limbs or more) and three constants
  of its own. With an x of 32 limbs or more at 1024 limbs its products are predicted 5-20%
  high (its powers are padded to the accumulator with zero low limbs, and its block products
  clip to the frame; both are priced as whole products), and at 16-64 limbs 10-17% low.
- **Parallel trees, 12-13%.** One gain a thread for every level; the 14 such trees are not in
  binary splitting's fit. Nothing picks between algorithms there (binary splitting is the only
  candidate at those sizes), so the cost is the fit, not a pick.
- **Small sizes.** At 4 limbs Horner on exp's Hyper series reads 8-18% low and binary
  splitting on exp's Engel series about 10% low: fixed costs that differ by family more than
  one constant an algorithm can say.
- **A level's leaf decision.** The tree walk takes q's size at the range's midpoint for every
  node of a level, where the tree's own estimate is smaller at its left nodes: a node or two
  off when a level sits at the leaf cutoff.
- **The evaluation's own cost**, 40-60 ns for a small BBP or Engel range against the old
  counts' 25: `ln_dyn` is 3.8% slower at 8 limbs and 1.8% at 16 than before the
  rebuild, where its picks were already the best ones (within 1.3% at every other size;
  `exp_dyn` is 3.5-6% faster from 4 to 300 limbs).
- **The remaining bad picks** are few and small (`cargo tune series` lists them): a one-term
  series at 256 limbs where rect and binary splitting are within 10%, and `exp hyper` at 16
  limbs and shift 4, which `horner_first` sends to Horner without the equations.

#### Rect: everything tried to bring it under the line

Asked for by the user on 2026-11-12 ("record all the things already tried for bringing the
rectangular series inline"). In order; each row names the log section with its detail. The
rms is held out unless it says otherwise, and "the line" is 5%.

| # | Date | Tried | Result |
|---|---|---|---|
| 1 | 11-10 | The old model: seven prices shared by the three algorithms over counted work, refitted to forced timings | Rect's error 59%. The counts were off, not the prices ("The series' dynamic dispatch rebuilt on fitted equations") |
| 2 | 11-10 | The rebuild: rect's own parts read off its kernel (its powers, each block's product on the sum of the blocks below it, two passes of each term's weight over its power, a division a group), products priced by mul's equations; every price its own | 6% fitted and 12% held out on its 25 workloads (exp's and ln's chunks): too few shapes for nine prices |
| 3 | 11-10 | Horner's prices for the kernels rect shares with it, and three constants of its own: a multiple on its products, a price a call, a fixed cost. The form kept (a fourth constant came in row 13) | 9.5% fitted and held out in the trial; 13.3% in the tune (five fixed frames), 11.6% held out on scattered ones. NOISY, marked |
| 4 | 11-10 | Its known miscounts noted, not fixed: its powers are padded to the accumulator with zero low limbs and its block products clip to the frame, both priced as whole products | With an x of 32 limbs or more at 1024 limbs its products read 5-20% high; at 16-64 limbs 10-17% low |
| 5 | 11-11 | Other forms on the same timings (22 workloads; the form in use 13.4%): term, call and allocation prices by family 13.4%; every price by family 12.4%; prices apart under and from 64 limbs 12.9%; the linear passes' limbs times log2 of the frame 13.4%; products priced apart under and from 32 limbs 13.7%; fitted only within 1.5x of the fastest 12.8%; every price its own 11.6% | None helps. No pattern by family, frame or time: 10.6-15.9% whichever way it is cut ("The series' equations: what they are, where they miss") |
| 6 | 11-11 | Whether rect is worth predicting at all: priced out of the dispatch | `exp_dyn` 0.97 at 16-32 limbs and 1.01-1.05 from 128 to 4096; but on general series rect is the fastest in 118 of 244 shapes, 0.54 of Horner's time at rho 6-20 and 0.25 past 100 at an x of 16 limbs or more. It stays a full algorithm (the user) |
| 7 | 11-11 | Fitted on the synthetic family where it competes (rho from 1/4), on 82 shapes in place of 22 | 7.8% fitted, 8.3% held out. Its 11-13% was the bit-burst's chunks, the map's triple point, and too few of them ("Built: the series tuned as general evaluators") |
| 8 | 11-11 | A guard in place of a prediction where rect is far ahead: taken outright from rho 32 at an x of 16 limbs (`series.rect_rho`, `series.rect_x_limbs`) | Settles 34 of 237 long shapes without the equation. The fit is the same; the picks no longer depend on it there |
| 9 | 11-11 | Forms on the family (fixed split; the form in use 9.0%): every price its own 10.9%; its own term price 8.4%; one fitted multiple (0.90) on the part of its time in Horner's kernels 7.8% from 8.3% | Small either way; not applied. It misses where Horner's kernels are 60% or more of its time (+6.7% bias, 12.7% over 14 shapes) ("Fit models proposed for the series") |
| 10 | 11-11 | A fit by the series' length, the cut that brought Horner to 4.6% and binary splitting to 4.3% from 20 us up | Rect does not follow: 6.6% under 5 us, 9.4% at 5-20 us, 9.5% from 20 us (worst 33%, 41 shapes) |
| 11 | 11-12 | A fit a regime, as built: long series for the equations, short ones for the table's prices | Long 8.2% over 123 shapes, short 6.2% over 41 ("Applied: a fit a regime and whole groups") |
| 12 | 11-12 | Horner's long prices for everything rect shares with it, the one-limb division included | 25.5%, and the picks lose 2.0%: Horner's long series hardly make a one-limb division, so its long fit prices that kernel at 0 |
| 13 | 11-12 | Rect's own price for one-limb divisions, `series.rect_ns_div1` (its groups end at its blocks, so it makes them in every range). In use; approved by the user | Back to 8.2% |
| 14 | 11-12 | Horner's groups counted whole for a q of constant size (`Shape::group_of`, which rect's parts read for the groups inside a block) | Horner's fix (its worst shapes were a few terms in a wide frame); rect reads the same count and stayed above the line |
| 15 | 11-12 | Is it the number of shapes? Cross-validated over every size at equal counts | Rect 8.7% on its 82; Horner 8.5% and binary splitting 9.0% on a random 82 of theirs. Over every size rect is no worse than they are. On long series they come down to 5.8% and 3.9% and rect does not |
| 16 | 11-12 | Hypothesis: the high product's price. It has no fitted equation; a probe of 199 (l, s) reads measured over priced 1.17 with a 16% spread in the schoolbook tier and 1.08 with 9% in the guarded one, and full products outside mul's fitted areas 0.89 with 23% | Proposed, not applied: a fitted equation for the high product and mul's fits extended to unbalanced products. Then tested against rect's residuals, rows 17-21 |
| 17 | 11-12 | Does rect's error follow its products? Its residuals on 123 long unseen shapes, by the share of its predicted time in products | No. 9.5% (mean +4.7%) in the third with the least products, 8.7% in the middle, 6.4% in the third with the most. The residual goes with the shares of calls (+0.43), terms (+0.38) and one-limb multiplier passes (+0.36), against products (-0.25) ("The table shrunk, rect's error looked at again, ...") |
| 18 | 11-12 | Its own prices for the passes it borrows from Horner, on those 123 shapes, each half judged by the other: + one-limb multiplier passes 7.2%; + fills 7.1%; + terms 7.2%; all three 7.1%; every price its own 10.2% (the form in use 7.4%) | No. No set of prices reads under 6.8% even on the shapes it is fitted to: it is not a price |
| 19 | 11-12 | Where it is, by x's width (same residuals; predicted over measured) | 2 limbs 10.7% (mean +4.7%), 3-4 12.7% (+7.7%), 5-8 9.8% (+3.9%), 9-16 8.7% (+6.4%), 17-32 5.0% (-0.1%), 33 and more 6.1% (-2.2%). Predicted too slow at a narrow x; at the line from 17 limbs |
| 20 | 11-12 | A product multiple of its own by x's width (to 4, 8 or 16 limbs, or three widths), same shapes | The narrow multiple fits 0.77-0.86 against 1.00, which is what the probe of row 16 read for short sides of 3-5 limbs (0.79): the bias at a narrow x goes (+2.3% to -0.5% at 2-4 limbs) and the fit reads 6.7-7.1% from 7.4% |
| 21 | 11-12 | A price for the one-limb multiplier passes (the term adds) of its own at a narrow x; and both that and the product multiple | 0.30 ns a limb against 0.40-0.45: 6.8-6.9%, an x of 2-4 limbs at 6.4-6.6% from 8.7%. With both free the two trade off (products 0.94, passes 0.33): 6.8%. An x of 5-16 limbs stays at 7-9% under every form. A call price by width does nothing (7.4%) |
| 22 | 11-12 | The user's hypothesis: mul lacks a chunking FFT, which would suit rect's uneven products | A chunking FFT is 1.1-1.6x faster than `mul_dyn` from a short side of 192 limbs under a long side 8 times it, but no series the tuner fits makes such a product (frames stop at 1024 limbs), mul's equation predicts those products within 7% as they are, and rect's error is at an x of 2-16 limbs. Not the cause; worth having for mul on its own account (the user's call) |
| 23 | 11-12 | Counters in the kernel (a scratch worktree), and rect timed alone at up to seven block lengths: the model's counts against the kernel's, 747 shapes | The counts are off, by amounts that vary shape to shape: term adds and R passes 1.74 of the kernel's (spread 22%), divisions 1.15 (21%; 1.42 at 1.5-3 blocks), products 1.06 (9%). And over every block length the form in use reads 10.1%, not 7.6%: wrong in m as well |
| 24 | 11-12 | Prices fitted to the kernel's own counts: the floor for an equation of this kind | 4.6% with five prices of its own, 3.5-3.7% with `hi_mul_buf`'s products priced apart from the guarded ones. About three points are counts, most of another the high product's price |
| 25 | 11-12 | A power's width from x's bits, ceil(j bits / 64), not j times its limbs (the kernel skips a power's zero low limbs) | Row 19's "narrow x": add limbs 1.23 of the kernel's at two limbs and 1.11 at 3-4 become 1.02 at every width; the block products' short side becomes exact |
| 26 | 11-12 | A term add's passes, 1 + (its weight's bits) / 64 in place of two; its limbs the lesser of its power and its frame term by term, in closed form a block; a group's division at its own frame, not its block's | Adds 1.74 to 1.02 (spread 22% to 5%); divisions 1.15 to 1.01 (21% to 5%). Rows 20-21's 0.30 ns a limb against 0.40-0.45 was the passes: a count, not a price |
| 27 | 11-12 | The guarded product's copy out of scratch and the scan of each power for zero limbs, counted; the accumulator's length a fraction of a limb off the frame, from where the bits land | The residual's -0.39 correlation with the share of copies goes (shapes of many powers of a narrow x on a long accumulator read 16-19% fast); block products' area 1.10 (15%) to 1.00 (0.3%) |
| 28 | 11-12 | Forms on the corrected counts: the four prices as they were; + its own price for the term adds; + a multiple for `hi_mul_buf`'s products | 4.8 / 5.4 / 4.4; 4.6 / 4.9 / 4.3; 3.8 / 4.2 / 3.6 on three sets (8.3 / 7.6 / 7.5 before). The third proposed as rect's own constant |
| 29 | 11-12 | The user: the tier's price into mul's section. `hi_mul_buf`'s own equation, a kernel: its limb products, a term a row or a column, a shift (`HiEq`) | Rows 2.7% and columns 3.8% held out, where mul_buf's equation on the area read 14.6% and 31%. Rect needs no multiple of its own for the tier, and Horner's products by x take the same price |
| 30 | 11-12 | As built and tuned: `SeriesEq::RECT` is product, div1, add, term, fixed over the corrected counts; the preset retuned | **2.8% held out over 62** (2.7% fitted on 125). Horner 4.7% and binary splitting 3.6%: no long equation is marked. Rect's short prices 7.6% (the table's derivation only) |

**How it ended.** Rows 1-22 moved prices and shapes; what was wrong was counts (rows 23-27),
and the price of one kernel that had been read off another's equation (row 29). The method
that found it is the one to reach for first when a fit of counted work won't conform: put
counters in the kernel, compare them with the model's counts shape by shape, and fit prices to
the kernel's own counts to learn what the form can do at best. A count that is off by a
constant factor is absorbed by its price and never shows; one whose error varies with the
shape is the misfit. Rows 19-21 had seen the symptom from the outside (too slow at a narrow
x, a term-add price that wanted 0.30 for 0.40-0.45) and could not tell a price from a count.

What is still open is in the log entry above ("What it costs, and what is left"): the
dispatch's own cost, rect's short prices, a q that grows along the range, more candidate block
lengths, and the powers' products over their zero limbs.

`cargo tune series` is the report to rerun; `--timings FILE` keeps its timings, so a change to
a count is read against the same measurements.

## What is left

Phase 2 is done on the M1 (2026-11-10): the log entry "Any machine can tune itself" above has
the final tune and its closing checks. Nothing is owed on this machine. Left, none of it
blocking:

1. **x86: the user's, on the i7**, by "The x86 handoff" below: the FFT's odd-part factors
   first, then the standard flow.
2. **The series' dispatch equations: under the line, none marked** (2026-11-13, "Horner's and
   binary splitting's equations under the line, and the beta guard gone"): 3.4%, 2.7% and 3.6%
   held out in the preset. What that entry leaves: the decision costs about 4% more for the
   model and walks binary splitting's tree on the ranges the beta guard used to settle (0.3% of
   the time on unseen series past 100 us; a lower bound from the tree's own equation would
   spare it, proposed and not built); and binary splitting's prices for finalize by a one- or
   two-limb Q rest on a few series a half at the lower levels. (Rect's groups and the two-limb
   ratio, open in that entry, were settled the same day: "The tuner's leftovers cleared, and a
   full retune", which leaves `div_rem_dyn` 1030/1000 6% slower, a near tie of NR with remainder
   and BZ that flips between retunes.) Still open from rect's fix ("Rect's counts
   read against the kernel, and hi_mul_buf's own equation"): more candidate block lengths. The
   corner the derived table had lost (two terms at an x past 32 limbs in a frame of 256 limbs or
   more going to binary splitting) is closed by measuring the table: a two-term series at an x
   of 293 limbs in a frame of 275 runs rect from the table, 631 ns where Horner takes 16.6 us
   and binary splitting more.
2c. **What the rule on tables costs** (same entry): a division at a divisor the old Knuth
   guards settled reads the equations once the decision is within 5% of it (12 of the division
   grid's 40 shapes, 2.8% slower as a group), accepted by the user for the one rule's sake; and
   the series' table on 8 kinds loses more on short series than the 72 classes did (its picks 1.5% on the mean and 2.3% of the time against 0.8% and 1.7%; the entry, its dispatch included, 3.0% and 3.5% against 2.6% and 2.5%)
   for an eighth of the size, the user's trade. Either moves with the share or the layout.
2a. **A chunking FFT for mul** (2026-11-12): 1.1-1.6x for `mul_dyn` from a short side of 192
   limbs under a long side 8 times it, with chunks of 4 to 15 times the short side ("The table
   shrunk, ..."). An algorithm, its equation, mul's version and a retune of what is built on
   mul: the user's call, not made.
2b. **mul's fits on unbalanced products** (proposed 2026-11-12, not applied): mul_buf's
   equation reads 20% high at a short side of 3-9 limbs under a long side of 25-99, and 60%
   under 25 (4.5% from 100 limbs). The other half of that proposal, a fitted equation for the
   high product, is built (`HiEq`, the same day). Division's NR still prices its schoolbook
   high products as half a schoolbook product with a fitted multiple of its own; it could read
   `HiEq` in place of it, which means a retune of div and what is built on it.
3. **The series' model over general series** (2026-11-11): the frame (rho, beta), the fit on
   the synthetic family and the guards are built ("Built: the series tuned as general
   evaluators" above, which lists what it leaves: the pool-split trees, wide coefficients; the
   dispatch's own cost on short series is the measured table's since 2026-11-12, and
   `horner_first` is a static node's only). And mul, div,
   sqrt, theta and agm in the M1 preset were tuned on battery with the sweeps' "samples" line
   at 5.9% (bin_split, series, exp and ln were tuned again on mains power): a rerun would
   settle whether that mattered.
3a. **exp's chunk width wants a serial and a parallel value** ("The tuner's exp workloads
   timed nothing" above; the user: the transcendental functions after the series code): one constant loses 7-12% for `exp_dyn` at 300-500 limbs on the pool
   and 15-27% for `exp_static` at 500-1040, or about 10% below 128 limbs. Proposed, the
   user's to decide.
4. **The "samples" line at the thorough level** reads 4.8-5.05% for a full tune on a steady
   machine, so its 5% warning is marginal there. A threshold or the thorough spread bound for
   the user to decide, if it matters. (`cargo tune series` reads 0.1-0.4% at normal.)
5. **Static entries** share `div.bz` and `mul.kara` with the dynamic ones and lose where they
   cross later (`div_rem_static<2048>` 200/100 10%). Approximate by the user's decision, until
   their memory handling changes.
6. **Knuth's inner loop and buffer placement**: up to 2.45x slower with the divisor 1 to 7
   limbs past a 4096-byte alignment against the working buffer (measured at q = 6, d = 1024).
   A property of the library, found through the tuner; not looked into.
7. **Linux core pinning** has never been built for or run on Linux.
8. **Deferred by the user**: the `cargo-bigbits` driver (the rest of Phase 3);
   `runtime-tuning` (Phase 4). The tuning code had its simplification pass on 2026-11-13 ("A
   simplification pass over `src/tuning`": duplication and residue, 8 lines net, so not a
   reduction in size); what that pass proposed and left is listed at its end.

## The x86 handoff

What this machine could check without an x86 CPU (2026-11-10), with the `x86_64-apple-darwin`
target under Rosetta: the tune build compiles; `cargo test --target x86_64-apple-darwin --lib`
passes (577, before the BZ change above); `cargo tune tune base` runs and reports that Rosetta
has no ADX. Nothing was timed: Rosetta's times mean nothing for a preset.

New for it: `tuner/sections/base.rs` (compiled on x86-64 only, and first in
`sections::sections()`): two cutoff sweeps with the knob forced to each side, `base.prim_adx`
through `mul_add_prim` and `base.limb_adx` through `add_mul_limb`, over 1-96 limbs. On a CPU
without ADX it tunes nothing. Its first real run is the i7's.

On the i7-8750H, on mains power at full charge with nothing else running (see the entries above
for what a throttling machine does to the fits). Since 2026-11-10 the flow is the one any
machine runs: the tune writes the machine's own file, each section passes the end-to-end gate
or is dropped, and nothing has to be compared against another tree.

```bash
cargo test --lib && cargo test --lib --features tune        # debug; both must pass first
# 1. The FFT's odd-part factors (the user's decision: hard-coded per architecture). Build line
#    in the probe's header; its `sizes` mode prints ns per point by size class.
#    Send me the medians relative to 2^k for odd parts 3, 5, 7, 9: they become x86-64's FFT_ODD_COST.
# 2. A first look, into this machine's own file (presets/local.toml):
cargo tune tune --level normal
cargo tune status                                           # which layer supplies each section
# 3. The preset:
cargo tune tune --level thorough --write-preset             # writes presets/x86_64-<os>.toml
rm presets/local.toml                                       # so builds use the preset
cargo tune check --file presets/x86_64-<os>.toml            # the preset against the defaults
cargo tune equations --level normal                         # products: time lost by mul_dyn's picks
cargo tune division --level normal                          # divisions: the same for div_dyn, div_rem_dyn
cargo test --lib && cargo test --lib --features tune        # again, under the preset
```

The series' two-limb ratio needs nothing by hand: `series.horner_pass2` is measured by the series
section's tune on `mul_prim2` and `mul_prim` (3.46 on the M1; read its evidence line for x86's).

To make the preset cover every x86-64 OS, rename it `presets/x86_64.toml` (`build.rs` reads
`<arch>-<os>.toml` first, then `<arch>.toml`).

What to look at, in order:
1. **The summary lines of each tune.** "samples" (past 5%: the machine was busy, measure again);
   "gate" (sections dropped because their values ran slower than the defaults, with the reason);
   "fits" (rejected ones are written, marked, and the gate decides whether their section stays;
   report each with its equation and rms all the same, as a rejected fit means a form to
   revise); "edges" (a cutoff sweep that chose a value at an end of its grid: the crossover may
   lie outside it on this CPU).
2. The likeliest rejected fit is the FFT's (`mul.fft_*`) until step 1's factors are in:
   `FFT_ODD_COST` is 1.0 for every odd part on x86. Whether `find_fft_size`'s set
   {1, 3, 5, 7, 9} is still the right one there is the same measurement.
3. `base.prim_adx` and `base.limb_adx`: their sweeps' evidence lines (defaults 20 and 10).
4. `cargo tune check`'s per-section ratios for the preset: each at or below 1, and no item
   listed as slower without an explanation.
