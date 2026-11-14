# Tuning system

How BigBits' performance constants are declared, read, measured and stored. **Status:** Phases 0–2
are built and tuned on the M1 (`presets/aarch64.toml`); the x86 preset is still to be run on the
i7. Phases 3 and 4 (the user-facing driver, runtime loading) wait until the crate ships.
`tuning_implementation.md` is the record of how it got here and what is left.

Two parts are rules for anyone who adds or changes a knob: [Invariants](#invariants) and
[Adding or changing a knob](#adding-or-changing-a-knob).

## Summary

1. **A schema.** Every knob is declared once in `src/tuning/schema.rs`: its type and default, the
   range in which any value gives correct results, the range a sweep searches, and how its value is
   decided.
2. **Generated accessors with two backings.** Code reads a knob as `k::sqrt::leaf()`. In a normal
   build that is a `const fn` returning a constant; with the `tune` feature it reads a live atomic
   the tuner sets between measurements.
3. **One tuner inside the crate** (`cargo tune`, behind `tune`). It measures a *section* of knobs at
   a time, in dependency order, and writes a tuning file. Each section has a version, so a code
   change invalidates only the sections it touches.
4. **Layered values, resolved at build time.** `build.rs` starts from the schema's defaults, then
   the target's preset, then the machine's own file. A partial file is normal.
5. **Dispatch.** A decision over one number is a measured cutoff. A dispatcher over several inputs
   compares fitted runtime equations per call, and reads a small measured table where that
   comparison would cost more than 5% of the call.

## Goals and non-goals

Tune once per machine; tune one section and leave the rest; quick by default, thorough on request;
free in normal builds; works as a dependency (cross-compilation, docs.rs, CI with no file);
correctness never depends on a tuned value.

Not goals: tuning at first use or inside `build.rs` (an explicit step someone chooses to run);
tuning soundness limits or capacities; bit-identical results across machines for functions with
approximate contracts (different tunings pick different algorithms, which agree within the bound).

## Vocabulary

- **Knob.** A value that affects only speed: a cutoff, an equation's constant, a parallel gate, an
  algorithm parameter, a dispatcher's table. Written `section.name`.
- **Section.** Knobs tuned together that go stale together. The unit of the tuner, the file and
  versioning: `base` (x86 only), `mul`, `div`, `sqrt`, `bin_split`, `series`, `consts`, `theta`,
  `agm`, `exp`, `ln`.
- **Fixed constant.** A soundness limit, guard, capacity or policy: a plain `const`, never tuned.
- **Kind.** How a knob's value is decided: `Sweep` (a one-input decision timed directly), `Model`
  (a constant of a fitted runtime equation), `Table` (a dispatcher's measured table), `Interim` (an
  old `mod.rs` constant not yet measured by the tuner).
- **Dispatcher.** A place that chooses among algorithms. **Predicate:** a pure function from a
  call's shape and explicit knob values to an algorithm. **Forced entry:** a function that runs one
  algorithm regardless of dispatch.
- **Valid range.** Every value in it gives correct results (invariant 1), enforced by `build.rs` and
  the live setter. **Search range:** where a sweep looks, inside the valid range.
- **Preset.** A tuning file checked into `presets/`, chosen by target. **Resolved values:** what a
  build uses after layering.
- **Time lost.** At one shape, how much slower the chosen algorithm is than the fastest. Decisions
  are judged by it, not by where a crossover lands. **Tie band (ε):** a relative difference below
  which two candidates count as tied.
- **Section version.** Bumped when a section's knobs or their meaning change; a file's section with
  another version falls through whole. **`derive` number:** the layout of the section's table; a
  file's table with another number falls through alone.
- **Upstream hash.** A hash of the resolved values of the sections a section depends on, recorded
  when it is tuned; a mismatch later is a warning.

## Invariants

1. **Every value in a knob's valid range gives correct results.** A knob only chooses between
   correct paths or sets a parameter of a correct algorithm. A value that panics or returns a wrong
   result is a bug to fix or a reason to narrow the valid range, in the schema, with a comment. The
   random-knob test enforces this.
2. **Results stay within each function's documented error under every tuning.** Exact operations
   give the same bits under any knob values. Tests compare against oracles, never against output
   produced under another tuning.
3. **Soundness limits and capacities are never knobs.** See [Fixed constants](#fixed-constants).
4. **Normal builds pay nothing.** An accessor is a `const fn` returning a generated constant; a
   table's accessor returns a `&'static` slice.
5. **No knob changes while a call is running.** The tuner sets live values only between
   measurements, with rayon's workers idle. Some knobs are read more than once per call. Live values
   are atomics, so breaking this gives inconsistent dispatch, never undefined behaviour.
6. **Every knob that chooses between algorithms has a pure predicate** that takes the knob values as
   arguments; the public dispatcher calls it with the accessor values. So the choice for any shape
   and any knob values can be computed without running it, and it is always the choice that ships.
7. **Read knobs once per call where the read is hot, and pass them down** (`DivCutoffs`,
   `SplitCutoffs`, `SqrtKnobs`, `MulEq`). A read at each level of a recursion is fine; a read inside
   a limb loop is not.
8. **Tests read accessors, not literal values.** A test that straddles a boundary must still
   straddle it under any preset. A count-style test pins what it depends on (explicit cutoffs
   through `_at` entries, a parallel gain of 0), never the values a preset happens to hold.
9. **Measured numbers live in data.** The schema's `doc` says what a knob decides; each value and
   the tuner's evidence for it are in the tuning file.
10. **Bump the right version.** A section's `version` when a knob is added, removed or renamed, an
    equation's form or a work count's unit changes, or a valid range narrows. Its `derive` number
    when its table's layout changes. Neither for a change that doesn't move timings. A version bump
    makes every file ignore that section until it is remeasured.
11. **The tuner sees nothing a normal build doesn't run.** It times the library's own forced
    entries and public dispatchers, never copies of source or a second implementation of a model.
12. **An equation changes with its algorithm.** A change to what an algorithm calls, at which sizes,
    or how many passes it makes updates its parts (`Parts`, `nr_parts`, …) in the same change. A
    stale count can't make a result wrong, only a pick slow.
13. **Every value in a tuning file is measured.** Nothing is derived from other values or written by
    hand; a table is timed at each index.

## Architecture inside the crate

### Files

```
build.rs, build/codegen.rs     resolve the layers, write $OUT_DIR/tuning.rs
presets/                       tuning files per target; local.toml is this machine's (gitignored)
src/tuning/
  mod.rs                       includes the generated code; get/set/apply/snapshot; cli()
  schema_types.rs, schema.rs   plain const data: the only place knobs are declared
  resolve.rs                   schema checks, layering, Follows, upstream hashes
  hash.rs, file.rs, table.rs   FNV-1a; the tuning-file subset parser and writer; table lookup
  tuner/                       (tune builds only)
    mod.rs                     the command line and the tune loop
    measure.rs                 timing; levels; fast-core selection
    design.rs                  where a fit's shapes go
    fit.rs                     SVD fits on relative error; a form's analysis
    sweep.rs                   one-input sweeps and the tie band
    tables.rs, dispatch.rs     the one builder every table is measured by; the rule's check
    check.rs, check_items.rs   the end-to-end gate and its workloads
    synth.rs                   the synthetic series family
    out.rs                     what a section's tune returns
    sections/                  one tuner per section
examples/tune.rs               fn main() { big_bits::tuning::cli() }
```

`build.rs` and the crate both bring in the shared files with `#[path]`, so there is one source of
truth and no build dependency; those files use only `core`/`std`. Equations and their parts live
next to their algorithms in `src/utils/` (invariant 12).

### The schema

Plain `const` data, so people and agents can read, grep and edit it without a macro. A section has
`name`, `version`, `derive`, `after` (its upstream sections) and `knobs`. A knob declares:

- `name`, `old` (the `mod.rs` constant it replaced, for grep), `doc`;
- `ty`: `Usize`, `U64`, `F64`, `Bool`, or `Table { algs, measure, inputs }` (each algorithm with the
  smallest size at which it is eligible);
- `default`: a value, or `Follows { from, plus }`, which resolves after layering;
- `valid` and `search` ranges; a table is valid when its change points ascend and every pick is
  eligible;
- `kind`, `threads` (`One`, `Pool`, `Cpus`), `arch`.

The defaults are the values a target with no preset runs on: mostly the i7's hand-tuned values of
2026-10, and the M1's fit for the equations (with `mul.fft_scale` set so Karatsuba meets the FFT
where the old x86 ratio put it).

### Accessors and their two backings

`build/codegen.rs` generates, per knob: `resolved::<section>::<KNOB>` (a `pub const` in every
build, for const contexts in tests; `src/utils` never names it) and `k::<section>::<knob>()` (a
`const fn` returning that value in a normal build; a relaxed atomic load in a tune build, F64 as
bits). So no call site may use an accessor in a const context: associated consts became trait
methods (`Node::leaf_limbs()`), `const DivCutoffs` became constructors, and a const-generic knob
goes through `knob_generic!` (a const argument normally, a `match` over the valid values in a tune
build).

The live store (tune builds, doc-hidden): `get`, `set` (checks type and valid range), `apply` and
`spec` (`"path=value;…"`, to replay a random-knob vector), `snapshot`/`restore`, `reset`.

### `build.rs`

1. Load and check the schema (defaults in range, `search` inside `valid`, `after` names sections,
   no `Follows` cycles). A failure is a build error.
2. Layer: the schema's defaults; `presets/<arch>-<os>.toml`, else `presets/<arch>.toml`; then the
   file `BIGBITS_TUNING` names (`presets/local.toml` in a checkout, through `.cargo/config.toml`).
   Only the sections and keys present override the layer below.
3. Warnings, with the part ignored: an unknown section or key; a section of another `version` (it
   falls through whole); a table of another `derive` number; a file whose `[meta] target` isn't the
   build's (skipped: a cross build would otherwise take the host's measurements); a set-but-missing
   path outside `presets/`. Warning, values kept: an upstream hash that no longer matches (only the
   top layer's record of a section is checked). **Error:** a value of the wrong type or outside its
   valid range, since valid ranges guard correctness.
4. Resolve `Follows`, write `$OUT_DIR/tuning.rs`.
5. Rerun only when `build.rs`, `build/`, the shared `src/tuning` files, `presets/` or
   `BIGBITS_TUNING` change.

No `[build-dependencies]`: the file format is a small TOML subset parsed by hand.

### The tuning file format

`#` comments, `[section]` headers, `key = value` lines; values are integers, floats (`inf`
included), booleans, a double-quoted string, or a flat array of numbers. A one-input table is
`[size, alg, …]`; a two-input table `[s, l, alg, …]`, row by row.

```toml
[meta]
format = 1
crate = "0.1.0"
target = "aarch64-macos"
threads = 8

[mul]
version = 18
derive = 10
tuned = "2026-11-13"
level = "thorough"
upstream = "base:e15d4450209898e0"
school_scale = 0.3568   # fit: 239 shapes; relative rms 4.0%, worst 14.3%; held out 3.9% over 119; ...
karatsuba_sqr = 71      # sweep 40-140, 101 sizes: kept, measured time lost mean 0.02%, worst 1.5%, ...
```

Every value carries its evidence as a comment. The upstream hash is FNV-1a 64 over the canonical
text of each upstream section's resolved values.

### Presets

`presets/aarch64.toml` is the M1's thorough tune and serves every aarch64 target for now (unmeasured
on other aarch64 hardware). There is no x86 preset yet. `cargo tune tune` writes the machine's own
`presets/local.toml`, which the next build layers over the preset; delete it to return to the
preset. `--write-preset` writes the preset a build resolves, and should be run with no local file
in force, since the gate's "before" is whatever the build resolves.

## The tuner

### The command line

```
cargo tune tune [SECTION ...] [--level quick|normal|thorough] [--out FILE] [--write-preset]
                [--threads N] [--no-gate] [--analyze] [--tables] [--share X]
cargo tune check [--file FILE] [SECTION ...] [--level L]
cargo tune dispatch [--level L] [--share X]
cargo tune equations [--level L]
cargo tune division [--level L] [--out FILE] [--refit]
cargo tune series [--level L] [--set 'PATH=VALUE;...'] [--timings FILE] [--share X]
cargo tune list [SECTION]
cargo tune status [--file FILE]
```

- **`tune`** measures the named sections (all, in order, when none are named): each section's
  sweeps and fits, then its tables, then its fits again on the shapes the tables leave, then the
  end-to-end gate. `--tables` measures only the tables (after a layout change). `--analyze` prints
  each fit's form analysis: for whoever writes an equation, never for tuning a machine. `--share X`
  runs under another number than the rule's 5%.
- **`check`** is the gate alone: public calls timed under FILE's values and under the layers below.
- **`dispatch`** checks the table rule: for each dispatcher, on indices where it reads its
  equations, what the decision costs against the call. Exits non-zero past 1.25× the share. Every
  tune ends with the same lines.
- **`equations`**, **`division`**, **`series`** report the measured time lost by the equations'
  picks and by the entries, on shapes no tune fits (`design::UNSEEN`). `division --refit` fits now,
  to the half not held out. `series --timings FILE` keeps the timings between runs;
  `--set` runs under other knob values.
- **`list`** every knob with its kind, range and live value. **`status`** the layers a build
  resolves and which supplies each section; with `--file`, one file against the schema.
- `BIGBITS_TUNE_VERBOSE=1` writes developers' TSVs under `target/tuning/` (`series_parts.tsv`,
  `series_workloads.tsv`, `series_cells.tsv` with `--tables`, `bz_fit.tsv` with
  `division --refit`).

The tuner runs in the release profile, as that is what ships, and asks for the fastest cores
(`measure::prefer_fast_cores`): on an efficiency core fits read 2.5× slow.

### How knobs are tuned

**What kind of measurement a decision gets.**

- *One input: a sweep.* A cutoff, a gate or a balance point is measured directly: both sides timed
  over sizes, the value with the least measured time lost.
- *Several inputs: fitted runtime equations.* Each algorithm gets an equation in its theoretical
  form, with constants fitted to that algorithm's own timings; dispatch compares them per call. An
  equation may include another's fitted time (a Karatsuba level includes schoolbook's). **Never
  derive a runtime purely from other algorithms' models**: of the first 25 decisions tried that
  way, one held.
- *A table where the decision is dear.* Where reading a dispatcher's equations costs more than
  `tables::SHARE` (5%) of the call, the dispatcher reads a table; past that, its equations. One
  constant for every dispatcher. **The table is measured, never derived:** at every index of its
  grid each algorithm that can take the shape is timed, forced, with the decision timed beside them
  at explicit knobs. A cell is the pick that costs least (within the tie band a pick stays; one cell
  between two of another pick is dropped), and a row ends at the first index where the decision is
  within the share. A row may stand for several kinds of call (the series' table): its cell is then
  the pick that costs them least together, and reading the equations is one of the picks.
  `dispatch::TABLES` lists every dispatcher with its table (`mul_dyn` and `mul.kara`, `div_dyn` and
  `div.dyn_small`, `div_rem_dyn` and `div.dyn_rem_small`, the series' entry and `series.small`).
- *Only what a decision reads is measured.* A fit or benchmark no dispatcher's decision depends on
  is purged from the tuner, the schema and the docs. A procedure used in tuning one algorithm is
  fitted as part of that algorithm.
- *Static entries get approximate values only*: swept at one N or left at their defaults, no fitted
  equations. A static decision's best value depends on N as well as the size, and the static
  entries' memory handling is to be reworked.
- *Fixed constants* are never measured by a tune (below).

**How a fit is found: one way, everywhere.**

1. *Shapes* are scattered over the area where the algorithm competes (`tuner/design.rs`: the R2 and
   R3 low-discrepancy sequences in log scale), not along a few ratios, and every other one is held
   out. Effort goes into more shapes, not a tighter time at each: a fit pools its shapes.
2. *Timing* (`measure::time_fit`): three batches a probe and no spread bound, in rounds that each
   visit every probe once in shuffled order, the whole set twice with its buffers moved. The value
   is the least of the two passes; their difference is the noise ("passes differ"). Rounds are
   needed: back-to-back batches lost whole probes to one disturbance of a few milliseconds.
3. *Solve*: `nalgebra`'s SVD on relative error (`fit::solve`, through `fit::fit_named`). Every
   equation is linear in its constants. `fit_split` solves without the held-out shapes for the rms
   that is judged, then on all of them for the constants written.
4. *A term that solves negative isn't needed*: it is left out at 0 and the rest solved again
   (`solve_nonneg`). It rejects nothing.
5. *A fit takes the shapes its equation is read for.* A dispatcher's equations aren't read where
   its table names an algorithm, so a tune fits twice from one set of timings: once to measure the
   tables under, then on what the tables leave (a section's `late` phase). An equation that is also
   a price inside other equations (schoolbook's, `hi_mul_buf`'s, Knuth's) keeps every shape.
6. *Which terms a form needs is the developer's question, never a tune's.* `--analyze` prints every
   constant against its standard error and its share of the time, and how each form short of some
   terms holds out. A tune only ever fits the form the equation has.

**Acceptance.** A fit is accepted at a held-out relative rms ≤ 5% (`fit::ACCEPT_RMS`). The worst
single residual is reported, never judged. **When a fit is rejected, stop and ask, once**: report
the fit, its equation and its rms, propose a new form or a few, and wait for approval. A model
already revised once that still misses is accepted noisy, marked NOISY (`Fit::evidence_revised`),
written, and recorded under "Marked problems" in `tuning_implementation.md`. A rejected fit is still
written, marked; the gate decides whether its section stays.

**When a count-based fit won't conform**, fit the times to the kernel's own counts first (counters
in a scratch copy). That floor says whether the counts are wrong (floor under the line, model above
it) or the form is (floor itself above the line). Then tune a new form at all three levels before
calling it sound: a price that a handful of series exercise passes or fails by the held-out split.

**Sweeps.**

- A sweep replaces a value only when another loses less mean time by more than the level's ε
  (`sweep::keep_or_take`); otherwise its evidence says "kept". Knobs that interact are swept twice.
- *Tune a helper through its callers.* bin_split's gates are swept through whole `exp_dyn` and
  `ln_dyn_at` calls: on isolated trees the split gate read 1024, which made both 3–5% slower.
- *A sweep's workloads must span every caller's shape.* Swept on exp's shape alone, the Hyper leaf
  read 192 limbs where ln's chunks want 48.
- *Whole-algorithm sweeps are judged by stability*: a fixed number of samples twice, and the choice
  stands only when both measurements support it within the tie band; otherwise the value is kept and
  the evidence says UNSTABLE.
- A gate whose loss is flat past the point where it does its job takes the lowest candidate within
  the tie band (`param_sweep_lowest`).
- *A phase's fits go live before its sweeps.* A sweep through an entry that dispatches on equations
  must run under the equations just fitted, or a tune that starts from defaults keeps a wrong value.

**Parallel algorithms.** A tune runs at the thread count the code will be used at. A
thread-dependent value is a serial/parallel pair: a second plain knob read by the call site for its
thread count (`ln.dyn_agm_limbs` and `ln.dyn_agm_par_until`). On a pool a decision may not be
one-directional, so shape the parallel knob as a bound. Measure from the main thread on rayon's
global pool, not from a private pool's worker.

**The gate.** Each section ends with the end-to-end check: `tuner/check.rs` times the workloads of
that section and of every section built on it (`check_items.rs`, shared with
`benches/tuning_bias.rs`) under the values before and after, in paired rounds in one process. The
section is dropped when the mean is slower by more than the level's allowance with a 95% interval
that excludes 1, or when an exact output differs. A section whose values didn't change isn't timed.
A single workload more than 2% slower is listed and never drops anything: **read that list**, as a
regression concentrated in a few workloads passes on the mean. The gate's "before" is a tune of the
same design, so it can't see a loss against the previous design; only an A/B with the previous tree
can, read item by item.

**Machine conditions.** Every measurement runs warm, and no equation has a cache term. A tune needs
the machine at one steady speed. The "samples" line counts sweep probes that never settled under the
level's spread bound (it sits near its 5% warning at the thorough level even on a steady machine);
the "fit probes" line reports the fits' pass difference and warns above 5%. When several unrelated
fits fail at once, suspect the machine before the forms. There is no steadiness preflight and no
kernel thermometer, by decision.

### Levels

| | quick | normal | thorough |
|---|---:|---:|---:|
| Tie band ε (`Level::eps`) | 3% | 1% | 0.5% |
| Sweep sample spread, IQR/median (`Level::spread`) | 6% | 4% | 2% |
| Gate allowance (`Level::gate`) | 3% | 2% | 1% |
| Fit shapes (`Level::fit_shapes`) | 1× | 2× | 4× |

A full thorough tune of the M1 takes about 36 minutes.

### The tuner's code

- A section's tuner is a `sections::SectionTuner`: its first phase, its tables as (knob key,
  measurement), its second phase. The tune loop names no section, so a section that gains a table
  or a second phase is a line of `sections::SECTIONS`.
- Every fit is made by `fit::fit_named`; its worst shape is `Fit::worst`.
- `tables::measure(level, spec, eps)` builds any table from a `Spec`; `tables::pick_at` is a row's
  pick at a size.
- A harness's time comes off a probe's through `measure::less_harness`; a developer's TSV is written
  by `measure::dump`.
- Division's probes are built by `numerator`, `placed_divisor` and `copy_probe`: keep each site's
  order of allocation, as division's timings follow where its buffers lie.
- The gate's item sections are `check::ITEM_SECTIONS`.
- **Comments in `src/tuning` are terse on purpose**: what a thing is and the one reason that would
  stop a wrong change. History, quotes and past runs' numbers belong in the docs.
- Left on purpose: the schema's unread fields (`Knob::threads`, `search`, `Kind::Model`'s `work`,
  step functions), the `KEPT` thread-locals between a section's two phases, `check::exact` by item
  name, a fit's REJECTED/NOISY counted from its evidence's text.

### Section dependencies and order

```
base ─► mul ─► div ─► sqrt ─────────────┐
         │      │                        ▼
         │      └──► bin_split ─► series ─► consts ─► exp, ln
         └──────────────────────────────► theta ─► ln
                            sqrt, div, mul ─► agm ─► ln
```

The tuner runs base (x86 only), mul, div, sqrt, bin_split, series, theta, agm, exp, ln. `consts`
has no tuner (its gates stay at their defaults). A section's fits read the equations of the
sections above it (division's on mul's, the series' on both), so tune upstream first and retune
downstream after an upstream change; the upstream hash records which values a section saw.

### Validation

1. **Random-knob correctness** (`cargo test --lib --features tune`). `random_knobs` re-runs the
   test binary on `random_knobs_child` in its own process (invariant 5). The child tries the
   defaults, everything low, everything high, everything at its maximum, each knob alone at each
   extreme, and random vectors, and runs every section's `random_knob_check` against the suite's
   oracles. A failure prints the vector; `BIGBITS_KNOB_VECTOR='path=value;…'` replays it
   (`BIGBITS_KNOB_SEED`, `BIGBITS_KNOB_VECTORS`). This is invariant 1 as a test: run it after any
   change to a dispatcher. It has found real bugs (a cancelling parabola in `is_school_at`, a block
   length of 0, an overflowing gate, debug stacks too small) and set the minimums: every sqrt cutoff
   ≥ 3, `mul.karatsuba` ≥ 3, `mul.karatsuba_sqr` ≥ 5, `div.bz` ≥ 3.
2. **Bias and identity checks** (`benches/tuning_bias.rs`): public entries timed in a normal build
   and a tune build (a knob read somewhere hot shows as bias; hoist it, invariant 7), and output
   hashes against a baseline build. `<exe> --hash [--section S] | --compare A B [--procs 6]`.
   Sections: `mul div sqrt series consts trans divgrid`. To compare with an older tree, copy the
   bench and `check_items.rs` into a worktree of it.
3. **Defaults.** On this Mac every test resolves the M1 preset, which hides a test that depends on
   tuned values. `cargo test --target x86_64-apple-darwin --lib` (and with `--features tune`) runs
   under the schema's defaults through Rosetta; it also compiles the x86-only code. Timings there
   mean nothing.

## Fixed constants

Never knobs (invariant 3). A tuning file can't reach them.

| Constant | Why it is fixed |
|---|---|
| `FFT_16BIT_CUTOFF`, `FFT_SQR_16BIT_CUTOFF`, `FFT_MID_16BIT_CUTOFF`, `FFT_BNM1_16BIT_CUTOFF` | Soundness: balanced coefficients stay under 2^48 (a compile-time assert). Past them dispatch takes the NTT |
| `FFT_FOLD_TRIGGER`, `FFT_FOLD_PLAIN`, `FFT_FOLD_WRAPPED` | Soundness of the fold check (`fft_precision.md`) |
| `FFT_RESIDUE_CHECK` | A policy of safety against speed |
| `THETA_STATIC_SLOTS`, `RECT_MAX_M`, `GROUP_MAX`, `THETA_MAX_TERMS`, `MAX_VALUES` | Capacities: array lengths and compile-time asserts |
| `HI_MUL_GUARD_LIMBS`, `EXP_RED_GUARD_BITS` and the guard limbs throughout; `AGM_TAIL` | Accuracy |

Fixed per architecture, checked by validation runs:

- **`FFT_ODD_COST`** (beside `FFT_ODD_PARTS` in `mul.rs`): the FFT's per-point cost for each odd
  part, relative to a power of two. AArch64: the NEON medians (1.00, 1.00, 1.21, 1.18, 1.21 for
  o = 1, 3, 5, 7, 9). x86-64: unmeasured, 1.0; measure it with `benches/probes/fft_sizes.rs`
  (`sizes` mode) before the x86 preset. Changing it changes what `mul.fft_scale` means, so it bumps
  mul's version.
- **`FFT_ODD_PARTS`** feeds const-evaluated sizes and the fold primes: it changes only with a
  soundness review.
- **`static_split_stack(n)`**: a static split thread's stack. Codegen sets the real need.
- The FFT limits and fold trigger were validated on NEON only; each target needs its own run.

## Adding or changing a knob

1. **Declare** it in `schema.rs` with `old`, a default, valid and search ranges, kind, threads,
   arch and a `doc` that says what it decides. Work out the valid range from the code; where unsure,
   keep it wide and let the random-knob test narrow it.
2. **Read it** through its accessor, once per call where hot (invariant 7), behind a pure predicate
   if it chooses an algorithm (invariant 6).
3. **Tune it** in `tuner/sections/<name>.rs`: a sweep for one input; for a dispatcher over several,
   an equation next to the algorithm, a fit, and, if the decision is dear at small sizes, a table
   registered in `sections::SECTIONS` and `dispatch::TABLES`.
4. **Bump** the section's version (invariant 10).
5. **Test**: `cargo test --lib`, `cargo test --lib --features tune`, and the Rosetta run under
   defaults. Add the knob's shapes to the section's `random_knob_check`.
6. **Retune** the section and everything downstream of it, on the user's word, and compare end to
   end with the tree before (`tuning_bias --compare`).
7. **Update every reference**: CLAUDE.md, these docs, comments.

## Phases

| Phase | Contents | Status |
|---|---|---|
| 0 | Dependency hygiene: criterion and rand to dev-dependencies, no empty `main`, hidden constants surfaced | landed 2026-11-08 |
| 1 | Schema, accessors, the `tune` feature, the random-knob test; every section migrated, values unchanged, bit-identical to `faa9971` | landed 2026-11-09 |
| 2 | The tuner, fitted-equation dispatch, measured tables, presets | done on the M1; x86 to run |
| 3 | `cargo-bigbits` driver, `init`/`status` for dependents, user documentation | when the crate ships |
| 4 | Optional runtime loading; sections for the number types | as needed |

### Phase 3: the user-facing system (planned)

```
cargo add big_bits                 # the target's preset (or the defaults) apply
cargo install big_bits             # once per machine: the `cargo-bigbits` binary
cargo bigbits tune                 # on the machine that will do the real run
cargo bigbits init                 # adds BIGBITS_TUNING to .cargo/config.toml
cargo build --release              # big_bits rebuilds with the tuned values
```

- **One package, two entry points.** The library, and `[[bin]] cargo-bigbits`. Cargo never builds a
  dependency's binaries, so library users never compile the driver. The binary uses only `std` (a
  package's dependencies are shared by its library and binaries), has no `required-features`, and
  measures nothing.
- **The driver builds a harness against the project's own `big_bits`**: `cargo pkgid big_bits`
  finds its source and version; a harness crate under `<target dir>/bigbits-tune/` depends on
  exactly that with `features = ["tune"]`, an empty `[workspace]`, the project's
  `[profile.release]` and `Cargo.lock`, and `fn main() { big_bits::tuning::cli() }`; it runs from
  the project's directory so the project's `rustflags` apply. The only contract between driver and
  library is "`tuning::cli()` exists and takes argv".
- It refuses to tune for another target unless `--allow-cross`. `build.rs` warns when `tune` is on
  outside the harness (`BIGBITS_TUNE_HARNESS=1`), since features merge across the graph.
- `init` writes `BIGBITS_TUNING = { value = "bigbits-tuning.toml", relative = true }` under
  `[env]`. An environment variable is the only way a project can pass a value to a dependency's
  build script. Builds without it get the preset or the defaults, with no warning.
- **Publishing checklist:** `tune` not a default feature; no `[build-dependencies]`; `presets/` and
  `src/tuning/` in the package; the driver builds with only `std`; docs.rs builds with no file.

### Phase 4

- **`runtime-tuning`**, for one binary shipped to many machines: accessors read a
  `OnceLock<Params>` filled at startup, falling back to the resolved values (FFTW's wisdom).
- **Sections for the number types**: radix conversion cutoffs, a float type's precision policy.

## Decided

0. **What Phase 1 added (2026-11-09):** `resolve.rs`; the `resolved::` constants; `apply`/`spec`;
   the random-knob test in its own process; `benches/tuning_bias.rs`; `knob_generic!`; knob bundles
   only where a predicate needed explicit values.
1. **FFT odd-part factors (2026-11-08):** hard-coded per architecture, not measured knobs.
2. **Thread counts (2026-11-08, revised 2026-11-11):** nothing is stored per thread count. A
   thread-dependent decision is a serial/parallel pair of plain knobs.
3. **Unequal cores:** not modelled. 4. **Cache levels:** not modelled.
5. **Two-dimensional boundaries (2026-11-08, revised 2026-11-12):** large inputs compare equations
   at run time; tables only where that costs more than 5% of the call, and measured.
6. **Phase timing (2026-11-08):** Phases 0–2 in one pass; code outside `src/utils` may break, and
   comparisons with the old harnesses run in a worktree at `faa9971`.
7. **Fitted forms for mid-range composites (2026-11-09):** superseded by 8.
8. **Sweeps for one input, fitted equations for several (2026-11-10).** One-input decisions are
   sweeps; multi-input dispatchers compare runtime equations whose constants are fitted to each
   algorithm's own measurements; nothing is derived purely from other algorithms' models; the fits
   use a library (`nalgebra`) under the `tune` feature. This replaced the plan's first method
   (measure base units and leaf kernels, derive everything else), which failed its end-to-end
   checks.
9. **Any checkout can tune itself (2026-11-10):** the machine's file is a third layer in `build.rs`
   now; every tune ends each section with the gate; a rejected fit is still written, marked.
10. **Tables are measured, under one rule (2026-11-12);** a fit takes the shapes its equation is
    read for.

## Open questions

- **How finely presets are keyed.** `build.rs` sees the architecture and OS, not the CPU model.
  Decide once a third machine has measured.
- **`FFT_RESIDUE_CHECK` as a cargo feature.**
- **The x86 preset**, and `FFT_ODD_COST` for x86-64.

## Precedents

- **GMP:** `tune/tuneup` measures crossovers into `gmp-mparam.h`, one file per CPU family chosen at
  configure time. Presets and build-time resolution follow that model.
- **FFTW:** wisdom saved to a file and loaded at run time. Phase 4 follows it.
- **ATLAS:** tuned during installation, and builds took very long. That is why tuning here is an
  explicit, modular step, never part of the build.
