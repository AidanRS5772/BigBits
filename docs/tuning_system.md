# Tuning system

**Status: plan, written 2026-11-07; tuning method added 2026-11-08. Phases 0 and 1 landed
2026-11-08/09: every performance constant is a schema knob read through generated accessors, and
`src/utils/mod.rs` keeps only the fixed constants. Phase 2 is next.
[`tuning_implementation.md`](tuning_implementation.md) tracks the order of work and each stage's
status.** When a phase lands, mark its heading (for example "Phase 1 (landed 2026-12-01)") and correct
anything the implementation changed. Keep the [knob inventory](#knob-inventory) in step with
`src/utils/mod.rs` until every section has migrated.

This is the long-lived reference for how BigBits' performance constants are declared, read,
measured, derived and stored. That means the cutoffs, cost-model coefficients and parallel gates in
`src/utils/mod.rs` and anything added later. The document covers two stages: now, while the crate
is developed in this repository, and later, when it is a dependency in other people's projects.
Phases 0 to 2 are internal and are being built now, for continued development. Phases 3 and 4 are
the system users see, and get built once the `bit_nums` API (`UBitInt`, `BitInt`, `BitFloat`, …) is
stable enough to publish.

Two parts are rules for anyone who adds or changes a knob, including agents:
[Invariants](#invariants) and the [per-section migration checklist](#per-section-migration-checklist).
The rest describes the design and explains the reasons behind it.

## Contents

1. [Summary](#summary)
2. [Goals and non-goals](#goals-and-non-goals)
3. [How tuning works today](#how-tuning-works-today)
4. [Vocabulary](#vocabulary)
5. [Invariants](#invariants)
6. [Architecture inside the crate](#architecture-inside-the-crate)
7. [The tuner](#the-tuner)
8. [Knob inventory](#knob-inventory)
9. [Phases](#phases)
10. [Per-section migration checklist](#per-section-migration-checklist)
11. [Worked example: adding a knob](#worked-example-adding-a-knob)
12. [Phase 3 in detail: the user-facing system](#phase-3-in-detail-the-user-facing-system)
13. [Retiring the current harnesses](#retiring-the-current-harnesses)
14. [Open questions](#open-questions)
15. [Precedents](#precedents)

## Summary

Today every performance constant is a `pub const` in `src/utils/mod.rs`. Each was tuned by hand,
on one machine, with a harness written for its own area. Retuning means rebuilding that harness's
setup, reading its output, and pasting numbers back into `mod.rs` together with prose describing
the measurement. Values tuned on one machine are wrong on another. For example, a 2026-10-27 run
on an Apple M1 found that it wants the Karatsuba→FFT ratios near 3.0 and 3.4, where the
x86-tuned values are 1.96 and 1.88, and `FFT_SQR_CUTOFF` near 500 rather than 340. Those values
were never applied.

The plan has five parts:

1. **A schema.** Every knob is declared once, in `src/tuning/schema.rs`. The declaration gives its
   type and default, the range in which any value gives correct results, the range the tuner
   searches, and how its value is decided: measured, or derived from other knobs.
2. **Generated accessors with two backings.** Code reads a knob through a generated function such
   as `k::sqrt::leaf()`. In a normal build the function returns a compile-time constant, so it
   costs nothing and compiles to the same code as today. With the `tune` feature, it reads a live
   value that the tuner sets between measurements.
3. **One tuner, inside the crate.** It sits behind `tune`. For any *section* of knobs (mul, div,
   sqrt, …) it measures the few values that can't be computed on the machine it runs on, derives
   every other knob from them, and writes a tuning file. Sections are independent units, tuned in
   dependency order. Each has a version number, so a code change invalidates only the sections it
   touches.
4. **Layered values, resolved at build time.** `build.rs` starts from the schema defaults,
   overrides them with the preset checked into the repository for the build target, and (Phase 3)
   overrides those with the user's tuning file, named by the `BIGBITS_TUNING` environment
   variable. A partial file is normal: whatever it leaves out falls through to the layer below.
5. **For users (Phase 3), one crate.** `cargo add big_bits` provides the library and
   `cargo install big_bits` provides a small `cargo-bigbits` binary. Inside a project,
   `cargo bigbits tune` builds a tuning harness against the project's own copy of `big_bits` (same
   version, same build profile) and runs it.

The tuning method keeps measurement small: **measure only what can't be computed**. A *leaf
kernel*, code whose cost is its own inner loop such as `mul_buf` or the FFT transform, gets a cost
model with a measured scale and shift over a work function the developer writes. A *composite*
algorithm, one built from other operations such as Karatsuba, NR division or Zimmermann's root, has
no measured values: its cost is predicted from its parts. For large inputs, dispatch compares
those predictions on every call, at the call's thread count. For small inputs, where that
comparison would cost too much of the call, dispatch reads a small table *derived* from the same
predictions. Parallel gates and algorithm parameters are derived the same way. Today's 99
hand-measured performance values become 39 measured values and 66 derived ones, and tuning one
section re-derives everything downstream of it without timing anything. See
[How knobs are tuned](#how-knobs-are-tuned).

| Phase | Contents | Visible to | When |
|---|---|---|---|
| 0 | Dependency hygiene; complete the inventory | nobody | now |
| 1 | Schema, generated accessors, the `tune` feature, a correctness test over random knob values; sections migrate one at a time | developers | now, one section at a time |
| 2 | The tuner (`cargo tune`): base costs and leaf-kernel models measured, everything else derived and checked; presets per target | developers | after Phase 1, in the same pass |
| 3 | `cargo-bigbits` driver, `BIGBITS_TUNING`, `init`/`status`, user documentation | crate users | before publishing |
| 4 | Optional runtime loading; sections for the number types | crate users | as needed |

## Goals and non-goals

Goals:

- **Tune once per machine.** After one tune, every build on that machine uses the values with no
  further work.
- **Modular.** You can tune one section and leave the rest at their preset or default. Partial
  tuning files are the normal case.
- **Quick by default, thorough on request.** The person running it decides when a tune happens and
  how long it may take (`--level`, `--budget`). An interrupted tune keeps every section it finished.
- **Free in normal builds.** An accessor compiles to the same constant that a `pub const` does now.
  The predictions dispatch evaluates on large inputs belong to the algorithms, not to the tuning
  system, and each dispatcher's extent holds them under ε of the call.
- **Few measured values.** Time only what can't be computed: base costs, leaf kernels' cost
  models, and the rare crossover between two code paths of one kernel. Derive everything else.
- **Cheap to extend.** Adding an algorithm means writing its work function and declaring its
  decision. A composite algorithm needs no new measurement at all; a new leaf kernel adds two
  measured values. Nobody writes a new benchmark.
- **Works as a dependency.** That includes cross-compilation, docs.rs, and CI builds that have no
  tuning file.
- **Correctness never depends on a tuned value.** See invariant 1.

Non-goals:

- **Tuning at first use or inside `build.rs`.** Planning at runtime the way rustfft does, or
  tuning during the build the way ATLAS did, makes first calls or builds slow and unpredictable.
  Tuning is an explicit step that someone chooses to run.
- **Tuning soundness limits or capacities.** These include the FFT's precision bounds, guard
  limbs, and stack slot counts. See [Fixed constants](#fixed-constants-never-knobs).
- **Bit-identical results across machines for functions with approximate contracts.** Series
  sums, exp, ln, the AGM, theta and reciprocals guarantee only an error bound. Different tunings
  can pick different algorithms, and those agree within the bound but not bit for bit (see
  invariant 2). Exact operations stay exact under every tuning.

## How tuning works today

This section describes the tree at commit `faa9971`, before Phase 1. Phase 1 deleted the harnesses
below as it migrated their constants; they still run in a worktree at that commit, where Phase 2
compares against them.

There are 106 `pub const`s in `src/utils/mod.rs`. 99 of them are performance knobs. The other 7
are fixed for soundness or capacity reasons and must never be tuned. Two of the knobs,
`BZ_2_1_SCHOOL_TO_KARATSUBA` and `BZ_2_1_KARATSUBA_TO_FFT`, were constants inside `div.rs`'s
`bz_2_1_cost` until Phase 0 moved them. Each constant in `mod.rs` carries a comment recording how and where it was
measured. mul, div and sqrt were mostly measured on an i7-8750H. The series model, ln, the AGM and
ln_agm were measured on an Apple M1. The exp parallel gate was measured on 12 threads.

Some code already passes knob values explicitly. These are the seams the new system builds on:

- `is_school_at(l, s, chunk, c)` and `is_karatsuba(l, s, chunk, reg)` in `mul.rs`
- `div_alg_dispatch(r, c: DivCutoffs)` with its four `DivCutoffs` constants in `div.rs`
- `SplitCutoffs` and `bin_split_tree_at` in `series.rs`
- the `knobs` tuple that `ln_dyn` and `ln_static` pass to `ln_takes_agm` in `trans.rs`
- the doc-hidden `_at` entries that take one knob as an argument: `sqrt_*_dyn_at` (wrapped
  cutoff), `exp_dyn_at` (square count), `ln_dyn_at` (square-root scale), `agm_dyn_at` (tail cap),
  and `ln_agm_dyn_at::<T>` (theta terms)

The current harnesses:

| Area | Harness | How it varies a knob | Analysis | Write-up |
|---|---|---|---|---|
| mul | `benches/mul_cutoffs.rs`, driven by `benches/mul_cutoffs.py` | Forces both sides of each crossover family (the table at the top of `mul_cutoffs.rs`). `--set NAME=VALUE` rewrites a `mod.rs` constant for one build and restores it afterwards, for example to give Karatsuba schoolbook leaves. `build` also compiles three x86 code-alignment variants (`plain`, `jcc`, `jcc_align`) to measure how much code layout moves a crossover. | Python: an exact Wilcoxon test per shape over 24 pinned processes for nanosecond-scale families, process means (`--simple`) for larger ones | `docs/mul_cutoff_tuning.md` |
| div | `benches/cutoffs_bench.rs` (`DIV_TUNE_REGIME`, `DIV_TUNE_FAMILY`, `DIV_TUNE_VALUES`, …) | Forced BZ/NR wrappers; candidate ratios come from environment variables; `DIV_TUNE_STATIC_SIZED` sizes `N` to the operands | `scripts/python/analyze_div_cutoffs.py` fits cost models to per-shape costs normalized to NR | `docs/division_cutoff_retuning.md`, `docs/division_dispatch_split.md` |
| sqrt | `scripts/python/tune_sqrt_cutoffs.py` building `benches/probes/sqrt_cutoffs.rs` | Pastes `src/utils/sqrt.rs` into `target/` and rewrites its cutoffs as const generics. The script's stated rule is "no tuning API enters src/". This is why CLAUDE.md lists text in `sqrt.rs` that must not change. | `scripts/python/analyze_sqrt_cutoffs.py` | `docs/sqrt_cutoff_tuning.md` |
| bin_split, exp, ln, agm, ln_agm | `benches/bin_split_cutoffs.rs` (`cargo bench-bin-split <family>`) | `bin_split_tree_at` and the `_at` entries | In the bench: candidates alternate within rounds; each reports its median relative to the workload's best; summarized by the geometric mean over workloads | `docs/bin_split_cutoff_tuning.md` |
| series model | `benches/series_cutoffs.rs` (`cargo bench-series`) with `scripts/python/tune_series_cutoffs.py` | Forces each algorithm through `series_eval_at`; the bench prints each row's work vector from the library's own model | Python: non-negative least squares on relative error; dispatch regret checked on held-out workloads | `docs/series_cutoff_tuning.md` |

They share no code and no file format, and the only way to put a result back is to edit `mod.rs`
by hand.

## Vocabulary

- **Knob.** A value that affects only speed: a cutoff, a cost-model coefficient, a parallel gate,
  or an algorithm parameter such as exp's square count. Its path is written `section.name`, for
  example `sqrt.leaf`. Every knob is either measured or derived.
- **Section.** A group of knobs that are tuned together and go stale together, such as `mul`,
  `div` or `sqrt`. A section is the unit of the tuner, of the tuning file and of versioning.
- **Fixed constant.** A soundness limit, guard, capacity or policy. Not a knob: it is a plain
  `const`, set by proof and checked by validation, never tuned (invariant 3).
- **Measured knob.** A knob the tuner times: a *model knob* (a coefficient of a cost model or a
  base unit) or a *sweep*.
- **Derived knob** (or derived value). A knob computed from measured knobs by a *derivation*,
  never timed and never set by hand, unless it is pinned: a dispatcher's table or extent, a parallel
  gate, a balance point, a structural value or a predictor coefficient. Most of today's constants
  become part of a derived knob.
- **Kind.** How a knob's value is decided: `Interim` (one of today's constants, migrated as it is
  in Phase 1), `Model`, `Sweep`, or `Derived` with its derivation type (see
  [Derivations](#derivations)).
- **Dispatcher.** A place that chooses among algorithms: `mul_dyn`, the division entries,
  Karatsuba's choice for its sub-products, and so on. At or above its *extent* it compares the
  algorithms' predictions on every call. Below the extent it reads its *table*: change points,
  derived from the same predictions, that say which algorithm has the least prediction at each
  small shape. See [Dispatch](#dispatch).
- **Leaf kernel.** Code whose cost is its own inner loop: `mul_buf`, `sqr_buf`, `hi_mul_buf`,
  `hi_sqr_buf`, `mid_mul_buf`, the FFT and NTT transforms, Knuth division, the binomial root, and
  the base primitives. Each has a cost model.
- **Composite.** An algorithm built from other operations: Karatsuba, the guarded high products,
  the transform entries, BZ, NR, Zimmermann, binary splitting, Horner, rect, theta, the AGM, exp
  and ln. It has no model knobs; its cost is predicted from its parts.
- **Work function.** A function the developer writes. For a leaf kernel it gives the work W, a
  Big-O count in the call's actual sizes, such as l·s for `mul_buf`, or N·log2 N at the actual
  transform size. For a composite it lists its parts: which operations it calls at which sizes,
  and its own passes, kernel calls, scratch acquisitions and array fills.
- **Cost model.** A leaf kernel's time as t = scale·W + shift, fitted to measurements. A kernel
  that counts more than one kind of work has one scale for each kind (the NTT counts butterflies
  by radix).
- **Base units.** Costs shared by every work function, measured in section `base`: a linear pass,
  a multiply-add pass and a `div_prim` pass per limb, a kernel call, a scratch acquisition, a static
  array limb, a rayon wake, a rayon task, a scoped thread, and one prediction term.
- **Prediction.** An operation's predicted time at a shape. For a leaf kernel it is the kernel's
  cost model. For a composite it is the sum of its parts' predictions and its base units. For a
  dispatcher it is the least prediction among the algorithms it can choose.
- **Sweep.** A crossover measured directly, by timing both sides over sizes. Used only between two
  code paths of one kernel (see [Which values are measured](#which-values-are-measured)).
- **Pinned.** A derived knob that is measured directly (by a sweep) because its derivation failed
  its check. Only a gate, a balance point or a one-input table can be pinned. Derivation leaves it
  alone until it is unpinned.
- **Time lost.** At one shape, how much slower the chosen algorithm is than the fastest, relative
  to the fastest. Decisions are judged by it, not by where a crossover lands.
- **Tie band (ε).** A relative difference below which two algorithms count as tied. A wrong pick
  inside it costs less than ε, so the tuner never spends samples resolving it.
- **Valid range.** Every value in it gives correct results. It is a contract (invariant 1),
  enforced by `build.rs` and by the live setter.
- **Search range.** Where a sweep looks. It is always inside the valid range. For a model knob it
  is a sanity bound: a fit outside it is reported.
- **Default.** The value declared in the schema. While a section migrates, its defaults are
  today's `mod.rs` values. A default may *follow* another knob (`knob + integer`), and then it
  tracks that knob until it is tuned or derived itself.
- **Preset.** A tuning file checked into `presets/`, chosen by target architecture and OS.
- **Tuning file.** A file in the [format](#the-tuning-file-format) that the tuner writes. Presets
  are tuning files.
- **Resolved values.** What a build uses: the defaults, then the preset, then (Phase 3) the user's
  file, after resolving defaults that follow other knobs.
- **Normal build.** Any build without the `tune` feature. Accessors return resolved values as
  constants.
- **Tune build.** A build with the `tune` feature. Accessors read live values that the tuner can
  change between measurements.
- **Forced entry.** A function that runs one specific algorithm regardless of dispatch, such as
  `karatsuba_entry_dyn`, `fft_entry`, `bz_div_dyn` or `series_eval_at`.
- **Predicate.** A pure function from a call's shape and explicit knob values to an algorithm, such
  as `is_school_at` or `div_alg_dispatch`.
- **Upstream hash.** A hash of the resolved *measured* values of the sections a section depends on,
  recorded when its derived knobs are computed. A mismatch later means they were derived from
  different values; `derive` fixes it without timing anything.
- **Section version.** An integer in the schema, bumped when a section's measured knobs or their
  meaning change. A tuning file's section with a different version falls through entirely.
- **Derive version.** A second integer per section, bumped when its derived knobs, its derivations
  or the work functions behind them change. A file's derived values with a different derive
  version fall through; its measured values are kept.

## Invariants

These are rules. Apply them to every knob, in every phase from 1 on.

1. **Every value in a knob's valid range gives correct results.** A knob only chooses between
   correct paths or sets a parameter of a correct algorithm. A value that panics, writes outside a
   buffer or returns a wrong result is either a bug to fix or a reason to narrow the valid range,
   in the schema, with a comment explaining why. For example, Karatsuba squares need at least 6
   limbs, so the interim `KARATSUBA_SQR_CUTOFF` can't go below 5, and in Phase 2 `mul.kara_sqr`'s
   `algs` declare a Karatsuba level eligible only from 6. Phase 1's random-knob test enforces this rule. It
   is the general form of the static `bin_split` leaf panics (CLAUDE.md, "Previously-known source
   bugs"), where every cutoff was mathematically valid but some crashed.
2. **Results stay within each function's documented error under every tuning.** Exact operations
   (`mul`, `sqr`, `div`, `div_rem`, the exact sqrt modes) give the same bits under any knob values.
   Operations with approximate contracts may differ in their low bits between tunings, within their
   documented error. Tests compare against oracles, never against output produced under another
   tuning.
3. **Soundness limits and capacities are never knobs.** If a constant bounds accuracy (the FFT's
   16-bit limits and fold trigger, guard limbs and bits) or sizes stack storage
   (`THETA_STATIC_SLOTS`, `RECT_MAX_M`), it stays a plain `const`. A tuning file can never reach
   it. Such a constant may differ between targets (by `cfg`), and each target's value is checked
   by a validation run, never tuned. When a soundness limit also caps a speed decision, as the
   16-bit FFT limits cap the switch from the FFT to the NTT, the capped algorithm's prediction is
   infinite past the limit, so dispatch respects the limit and can still switch earlier. See
   [Fixed constants](#fixed-constants-never-knobs).
4. **Normal builds pay nothing.** In a normal build an accessor is a `const fn` returning a
   constant from generated code. The compiler folds it exactly as it folds today's `pub const`.
   Moving a section to accessors must not measurably change its timings (the bias check in
   Phase 1). A table's accessor returns a `&'static` slice, folded the same way. The predictions a
   dispatcher evaluates above its extent are the algorithm's own work, held under ε of the call by
   that extent.
5. **No knob changes while a call is running.** The tuner sets live values only between
   measurements, while no library call is running on any thread and rayon's workers are idle.
   This matters because some knobs are read more than once per call: the NTT radix-3 and radix-5
   parallel gates both size scratch and choose the parallel path. They must also read the same
   thread count each time. Live values are atomics, so breaking this rule gives inconsistent
   dispatch but never undefined behavior.
6. **Every knob that chooses between algorithms has a pure predicate.** The predicate takes the
   knob values as arguments (as `is_school_at` and `div_alg_dispatch` already do), and the public
   dispatcher calls it with the accessor values. Then the choice the dispatcher makes for any
   shape and any knob values can be computed without running it, by tests and by the tuner's
   derivations, and it is always the choice that ships.
7. **Read knobs once per call where the read is hot, and pass them down.** Do this the way
   `DivCutoffs` and `SplitCutoffs` are passed today. A read at each level of a recursion is fine;
   a read inside a limb loop is not. The same goes for the thread count that predictions and gates
   read: read it once per call and pass it down. Phase 1's bias check is what catches violations.
8. **Tests read accessors, not literal values.** A test that straddles a boundary (for example
   sizes either side of `k::sqrt::dyn_rcp_carry()`) must still straddle it under any preset or
   tuning file.
9. **Measured numbers live in data; the schema's prose describes the decision.** After a section
   migrates, its schema `doc` says what is compared, over which shapes, and why the predicate has
   its shape. Each value and the tuner's evidence for it go in the tuning file as comments: the
   fit's quality for a model knob, the check's time lost for a derived one (see
   [Evidence](#evidence)). The measurement prose currently next to each constant in `mod.rs`
   moves to a dated section of that area's tuning doc, so the history is kept (the update-all-refs
   rule: dated reports keep their history).
10. **Bump the right version.** Bump a section's `version` when its measured knobs or their
    meaning change: a model knob or sweep added, removed or renamed, a work function W of a leaf
    kernel changing its unit, a narrowed valid range, or a change to a leaf kernel or sweep large
    enough that you would remeasure it. Bump its `derive` version when a derived knob is added,
    removed or renamed, when a predicate's form changes, or when a derivation or a work function a
    derivation reads changes. Don't bump either for a change that doesn't move timings. A version
    bump makes every preset and user file ignore that section until it is remeasured; a derive
    bump costs only a `derive` run, which times nothing.
11. **The tuner sees nothing that a normal build doesn't run.** The tuner times the library's own
    forced entries and public dispatchers. It never uses copies of source, pasted source, or a
    separate implementation of the model. Its predictions come from the work functions next to
    each algorithm, and its derivations call the crate's own predicates. (`series_cutoffs.rs`
    already follows this rule by printing the library's own work vectors.)
12. **A work function changes with its algorithm.** A change to an algorithm's structure (what it
    calls, at which sizes, how many passes, scratch acquisitions or array fills) updates its work
    function in the same change, and bumps the derive version (invariant 10). A stale work function
    can't make a result wrong, only a derived knob slow, so only the derived-knob
    [check](#checking-derived-values) catches it.
13. **Derived knobs are computed, never written by hand.** A tuning file holds a derived value only
    as the output of `derive` (with its upstream hash) or as a pinned measurement with its sweep's
    evidence. To change a derived value, change a measured knob or a work function, or pin it.

## Architecture inside the crate

### Files

```
build.rs                       reads the schema (and from Phase 2 the presets), writes $OUT_DIR/tuning.rs
presets/                       (Phase 2) tuning files checked into the repository, one per target
  x86_64.toml
  aarch64-macos.toml
src/tuning/
  mod.rs                       include!(concat!(env!("OUT_DIR"), "/tuning.rs")); the public tune API
  schema_types.rs              Section, Knob, Ty, Range, DefaultValue, Kind, Derivation, Threads (plain data)
  schema.rs                    pub const SCHEMA: &[Section] = &[...]  (the only place knobs are declared)
  hash.rs                      FNV-1a over a section's canonical values (shared with build.rs)
  file.rs                      the tuning-file subset parser and writer (shared with build.rs)
  # Phase 2, only with the `tune` feature:
  cli.rs                       the `tune` command line (shared by `cargo tune` and later `cargo bigbits`)
  measure.rs                   batched, interleaved timing of forced entries; a median per shape
  fit.rs                       weighted non-negative least squares on relative error; lack-of-fit report
  sweep.rs                     crossovers between two code paths of one kernel
  derive.rs                    the derivation types (table, extent, parallel gate, balance point, …)
  sections/{base,mul,div,sqrt,bin_split,series,consts}.rs   each section's kernels, shapes and derivations
src/utils/*.rs                 work functions and runtime predictors, next to their algorithms
src/tests/test_tuning.rs       schema checks; the random-knob correctness test (`tune` feature)
examples/tune.rs               (Phase 2) fn main() { big_bits::tuning::cli() }
```

`build.rs` and the crate both bring in `schema_types.rs`, `schema.rs`, `hash.rs` and `file.rs`,
with `#[path = "src/tuning/schema.rs"] mod schema;` and the same for the others. So there is one
source of truth and no build dependency. These four files may use only `core`/`std`, and nothing
else from the crate.

Work functions live next to their algorithms in `src/utils/`, so a change to an algorithm and to
its work function land in the same diff (invariant 12). Those that only derivations read are
`#[cfg(feature = "tune")]`. Runtime predictors, which dispatch reads on every call (`mul_ns`, the
series model's costs, `agm_wins`), are in every build and read the same knobs. Derivation runs in
the tune build, never in `build.rs`: it uses the crate's own code (invariant 11), and `build.rs`
stays a parser.

### The schema

The schema is plain `const` data, chosen so that people and agents can read, grep and edit it
without learning a macro. One entry, with illustrative values (the real valid range is set when
sqrt migrates):

```rust
// src/tuning/schema.rs
Section {
    name: "sqrt",
    version: 1,                      // measured knobs and their meaning (invariant 10)
    derive: 1,                       // derived knobs, derivations and the work functions they read
    after: &["base", "mul", "div"],  // upstream sections: their measured values are hashed into upstream
    doc: "Square roots: binomial or Zimmermann per output mode, and the wrapped remainder.",
    knobs: &[
        Knob {
            name: "binomial_scale",
            old: "",                              // new in Phase 2; no mod.rs constant
            ty: Ty::F64,
            default: DefaultValue::Float(0.4),    // a first estimate, in ns per limb step
            valid: Range::Float { min: 0.0, max: f64::MAX },
            search: Range::Float { min: 0.01, max: 10.0 },   // a sanity bound on the fit
            kind: Kind::Model { work: "binom_sqrt_core: s.len() * x.len() limb steps" },
            threads: Threads::One,
            arch: None,
            doc: "The binomial root's cost per limb step.",
        },
        Knob {
            name: "stage",
            old: "ZIMMERMAN_SQRT_LEAF_CUTOFF",   // the mod.rs name it replaced (for grep and history)
            ty: Ty::Table {
                algs: &[Alg { name: "binomial", min: 1 }, Alg { name: "zimmermann", min: 2 }],
                measure: "root limbs",
            },
            default: DefaultValue::Table(&[1, 0, 11, 1]),   // binomial from 1, Zimmermann from 11
            valid: Range::Table,                  // any table whose picks are eligible
            search: Range::Int { min: 6, max: 40 },   // where a sweep looks if it is pinned
            kind: Kind::Derived(Derivation::Table),
            threads: Threads::One,
            arch: None,                           // Some("x86_64") for the ADX cutoffs
            doc: "Zimmermann's recursion: a stage, or the binomial root as its leaf.",
        },
        Knob {
            name: "stage_extent",
            ty: Ty::F64,                          // in the dispatcher's measure; inf: the table covers every size
            kind: Kind::Derived(Derivation::Extent),
            // ...
        },
        // ...
    ],
}
```

In Phase 1 the same section holds today's constants instead, as `Interim` knobs under their own
names (`leaf`, `dyn_rem`, …, `dyn_wrapped_rem`), with today's values. Phase 2 replaces them with the
entries above and bumps the version.

Each knob declares:
- **`name`, `old`, `doc`.** The path, the `mod.rs` constant it replaced (empty for a knob the
  method adds), and the decision it makes or the cost it models.
- **`ty`.** `Usize`, `U64`, `F64`, `Bool`, or `Table { algs, measure }` for a dispatcher's table:
  its algorithms, each with the smallest size at which it is eligible, and its size measure. A
  `Bool` knob is a choice that can be timed, never a policy. A balance point declared `Pool` or
  `Cpus` holds a step function of the thread count (see [Dispatch](#dispatch)).
- **`default`.** An `Int`, `Float`, `Bool` or `Table`, or `Follows { from, plus }`, which resolves
  after layering. In Phase 1, `sqrt.dyn_wrapped_rem` follows `mul.sqr_bnm1_fft`, keeping the
  relationship that `mod.rs` currently writes as `DYN_SQR_BNM1_FFT_CUTOFF + 1`.
  - An `Interim` knob's default is today's value, so Phase 1 dispatches exactly as today.
  - A Phase 2 knob, model or derived, takes its default from the x86 preset, matching today's
    mostly-x86 defaults. A build with no tuning file and no preset for its target dispatches as
    the x86 machine would. Before that preset exists, a model knob starts as a first estimate where
    today's model puts it (for example `mul.school_scale` at `SERIES_NS_PRODUCT`), and its derived
    knobs are derived from those estimates.
- **`valid` and `search`.** The correctness domain, and where a sweep looks (`search` ⊆ `valid`).
  - Every non-negative value is valid for a model knob, as predictions only choose among correct
    paths. Its `search` is a sanity bound, and a fit outside it is reported.
  - A table is valid when its change points ascend and every pick is eligible at its sizes.
  - Derived values are clamped into `valid`.
- **`kind`.** `Interim` (Phase 1 only); `Model { work }`, with the work function W described in
  prose; `Sweep`; or `Derived(derivation)`, with one of the [derivation types](#derivations). The
  derivation itself is code in the section's tuner module, keyed by the knob's path.
- **`threads`.** Which thread count the knob's predictions or value read: `One` (single-threaded
  work), `Pool` (the size of rayon's pool), or `Cpus` (static scoped threads, which follow
  `available_parallelism`; macOS can't limit that).
- **`arch`.** The targets it applies to, if fewer than all.

### Accessors and their two backings

`build.rs` generates one module per section and one function per knob. The call sites are the
same in both builds:

```rust
use crate::tuning::k;
if s.len() < k::sqrt::leaf() { binom_sqrt_core(x, s); return; }
```

What it generates, sketched:

```rust
// Normal build. RESOLVED holds the defaults (Phase 1), or defaults + preset (+ user file), resolved.
pub(crate) mod k { pub(crate) mod sqrt {
    #[inline(always)] pub(crate) const fn leaf() -> usize { super::super::RESOLVED.sqrt.leaf }
} }

// Tune build. Live values are atomics; a relaxed load is a plain load on x86-64 and AArch64.
// F64 knobs are stored as their bits in an AtomicU64.
pub(crate) mod k { pub(crate) mod sqrt {
    #[inline(always)] pub(crate) fn leaf() -> usize {
        super::super::LIVE.sqrt.leaf.load(Ordering::Relaxed) as usize
    }
} }
```

The live store starts at the resolved values. Its public API (`tune` feature only; doc-hidden
until Phase 3):

```rust
pub fn set(path: &str, v: Value) -> Result<(), KnobError>;   // checks type and valid range
pub fn get(path: &str) -> Value;
pub fn snapshot() -> Snapshot;  pub fn restore(s: &Snapshot);
pub fn derive() -> DeriveReport;   // recompute every unpinned derived knob from the live measured ones, set live
pub fn schema() -> &'static [Section];
pub fn cli();                                                 // the tuner's entry point
```

Derived knobs have accessors like any other knob, and call sites can't tell them apart. Two
shapes differ from a plain value:
- A table's accessor returns its change points: `&'static [u32]` in a normal build.
- A balance point stored by thread count takes the count:
  `k::ln::dyn_agm_terms(threads)`.

In a tune build both live in fixed-capacity atomic arrays, sized by the schema. The `DivCutoffs`
and `SplitCutoffs` rows below describe Phase 1. In Phase 2 those forms give way to tables and
runtime comparisons.

The accessor is `const fn` in a normal build and an ordinary function in a tune build, so no call
site may use a knob in a const context. A few consumers do so today and change when their section
migrates:

| Consumer today | Where | Change |
|---|---|---|
| `Node::LEAF_LIMBS`, `Node::LEAF_TERMS` (associated consts on the six nodes) | `series.rs` | Make them trait methods `leaf_limbs()`/`leaf_terms()` that return accessors. `SplitCutoffs::of::<N>()` stops being `const fn`. |
| `DYN_DIV`, `DYN_DIV_REM`, `STATIC_DIV`, `STATIC_DIV_REM` (`const DivCutoffs`) | `div.rs` | Replace each with a function that builds `DivCutoffs` from its three accessors. `div_alg_dispatch` is unchanged. |
| `ln_agm_dyn_at::<DYN_LN_AGM_TERMS>`, `ln_agm_static_at::<N, STATIC_LN_AGM_TERMS>` (const generic) | `trans.rs` | Normal build: `::<{ k::ln::dyn_agm_terms() }>` (a const-fn call is a valid concrete const argument). Tune build: a `const_dispatch!` `match` over `0..=THETA_MAX_TERMS`, which compiles 17 monomorphizations, in tune builds only. |
| `DYN_SQR_BNM1_FFT_CUTOFF = FFT_SQR_CUTOFF`, `DYN_SQRT_WRAPPED_REM_CUTOFF = DYN_SQR_BNM1_FFT_CUTOFF + 1` | `mod.rs` | `Follows` defaults in the schema, until Phase 2 derives both. |
| `mul_work`, which scales each tier at the crossover that selects it | `mul.rs` | It calls `is_school` and reads `KARATSUBA_CUTOFF`, `CHUNKING_KARATSUBA_CUTOFF` and the FFT and NTT ratios directly. When mul migrates, those reads become accessors like any other, and the series model then follows tuned mul values automatically. That dependency is why `series` lists `mul` as upstream. In Phase 2, `mul_ns` replaces it: a runtime predictor that evaluates mul's cost models along mul's dispatch (with derived closed-form coefficients for the Karatsuba tiers), so the series model reads products in nanoseconds instead of multiplying limb products by `SERIES_NS_PRODUCT`. |

Fixed constants that feed const contexts, such as `FFT_*16BIT_CUTOFF` into `find_fft_size` and
the 2^48 assert, or `THETA_STATIC_SLOTS` into array lengths, stay `const` and aren't affected.

### `build.rs`

Phase 1's `build.rs` reads only the schema and writes defaults. Phase 2 adds the presets, and
Phase 3 adds `BIGBITS_TUNING`. In Phase 3 it does the following:

1. Load the schema and check it. Defaults must lie in their valid ranges, `search` must lie inside
   `valid`, every `after` must name a section, `Follows` defaults must have no cycles, and knob
   paths must be unique. A failure is a build error, because the schema is the crate's own data.
   (That every derived knob has a derivation is checked by `test_tuning.rs` in tune builds, since
   derivations are crate code.)
2. Layer the values:
   - schema defaults;
   - `presets/<arch>-<os>.toml`, otherwise `presets/<arch>.toml`, otherwise nothing, using
     `CARGO_CFG_TARGET_ARCH` and `CARGO_CFG_TARGET_OS`;
   - `$BIGBITS_TUNING` if it is set and non-empty.

   Within each file, only the sections present (and the keys present in them) override the layer
   below.
3. Handle problems in a file:
   - **Warning** (`cargo:warning=`), with the offending part ignored: an unknown section or key; a
     section whose `version` differs from the schema's (the whole section falls through); a
     section whose `derive` differs from the schema's (its derived values fall through, its
     measured values are kept: "sqrt's derived knobs are from an older crate; run `derive`").
   - **Warning**, values still used: a section whose recorded upstream hash differs from the
     resolved upstream measured values ("sqrt was derived from different mul values; run
     `derive`").
   - **Error**: a value of the wrong type, or outside its valid range. Valid ranges guard
     correctness, so a typo must not be silently clamped. For a table, that means its change
     points must ascend and each pick must be eligible at its sizes (invariant 1).
   - **Error**: the `BIGBITS_TUNING` file is missing.
4. Resolve `Follows` defaults, then write `$OUT_DIR/tuning.rs`: the section structs, `RESOLVED`,
   both backings of the accessors (each behind `#[cfg(feature = "tune")]` or its negation), the
   live store, and the schema re-exported for runtime use.
5. Emit `cargo:rerun-if-changed` for `src/tuning/` and `presets/`, and for the `BIGBITS_TUNING`
   path when set, plus `cargo:rerun-if-env-changed=BIGBITS_TUNING`. Use the single-colon `cargo:`
   form, which every supported toolchain reads.

`build.rs` has no `[build-dependencies]`. The file format is a small subset of TOML that
`file.rs` parses by hand, so library users don't compile a TOML parser.

### The tuning file format

Presets and user files share one format. It is a subset of TOML, written by the tuner and readable
by any TOML parser:

- `#` comments; `[section]` headers; `key = value` lines;
- values are integers (decimal, `_` allowed), floats (`inf` included), `true`/`false`, a double-quoted string
  without escapes, or a flat array of numbers, which may span lines;
- no nested arrays, inline tables or other multi-line values.

Arrays hold two things:
- **Tables.** A one-input table is `[size, alg, size, alg, …]`: from each size up to the next,
  that algorithm (an index into the schema's `algs`). A two-input table is
  `[s, l, alg, s, l, alg, …]`: the same lists, row by row, for each value s of the shorter input.
- **Step functions** of the thread count: `[threads, value, threads, value, …]`, each value from
  its thread count up.

```toml
# BigBits tuning file. Written by `cargo tune` (big_bits 0.1.0). Edit with care: values outside a
# knob's valid range fail the build, and a section whose version differs from the crate's is ignored.
[meta]
format = 1
crate = "0.1.0"
target = "aarch64-macos"
cpu = "Apple M1"
cores = 8
threads = 8                  # the most threads checks and step functions went up to

[sqrt]
version = 1
derive = 1
tuned = "2026-11-20"
level = "normal"
upstream = "base:5be0c2a7e9d14f30 mul:9f2c41d0a1b2c3d4 div:03ab77e1deadbeef"
pinned = "dyn_rem"
binomial_scale = 0.41        # fit: 9 shapes, 2–24 root limbs; relative rms 2.1%
binomial_shift = 18.5
stage = [1, 0, 12, 1]        # derived (table): binomial from 1, a Zimmermann stage from 12; check: worst 0.4% lost
stage_extent = inf           # derived (extent): Zimmermann wins every larger root, so no predictions
dyn_rem = [1, 0, 18, 1]      # pinned: sweep 12–28; tied within 1% from 16 to 19
dyn_only = [1, 0, 13, 1]     # derived (table); check: worst 0.7% lost
# ... dyn_approx, the static modes and their extents likewise
wrapped = [1, 0]             # derived (table): classic below the extent
wrapped_extent = 300         # derived (extent); check above it: worst 1.6% lost at 340 (flagged)
dyn_rcp_carry = 680          # derived (structural): the first root to reach a second NR stage
```

`pinned` lists the section's derived knobs whose value is a measurement, as a space-separated
string. Every value with no "derived" or "pinned" note in its
evidence is measured. The upstream hash is FNV-1a 64 over the canonical text of each upstream
section's resolved measured values (`key=value` lines, sorted, with integers in decimal and floats
in shortest round-trip form). The tuner records the hashes of the measured values that derivation
read. `build.rs` recomputes them from what it resolved.

### Presets

`presets/` holds one tuning file per target the developers can measure. At first that means
`x86_64.toml` from the i7-8750H and `aarch64-macos.toml` from the M1. A build on any other target
uses the schema defaults, which stay at today's mostly-x86 values. A preset is written only by
`cargo tune <section> --write-preset`, which merges the measured sections into the file for the
host target and keeps the others. Edit a preset by hand only to record a measurement made some
other way, and say so in a comment on the value.

Derivation only computes, so a developer rederives every preset on one machine after a derive
bump: `cargo tune derive --file presets/x86_64.toml` reads that preset's measured values and
rewrites its derived ones. For that to hold, work functions and derivations must not read the
host's `cfg`. Anything target-specific, such as the x86-only `base.prim_adx`, is read from the
file, never from the host.

## The tuner

### The command line

`tuning::cli()` is shared by `cargo tune` (Phase 2) and `cargo bigbits tune` (Phase 3), so both
take the same arguments:

```
tune [SECTION ...]           measure these sections' model knobs and sweeps (all, in order, when none
                             are named), then derive every derived knob and check it
    --level quick|normal|thorough
    --budget 30m             stop starting sections after this; report what was skipped
    --threads N              pool size for parallel predictions and Pool knobs (default: available_parallelism)
    --pin KNOB               measure this derived knob by a sweep and pin it (repeatable)
    --unpin KNOB             return a pinned knob to its derivation
    --out FILE               tuning file to merge into (default target/tuning/latest.toml)
    --write-preset           (developer) merge into presets/<host target>.toml instead
    --resume                 skip sections already in FILE whose version matches
    --csv DIR                also write raw timings (default target/tuning/<run>/)
    --seed S
derive [--file FILE]         recompute derived knobs from FILE's measured values; times nothing
status [--file FILE]         per section: tuned, stale (version, derive or upstream), pinned, flagged,
                             or falling through
list [SECTION]               knobs, kinds, valid and search ranges, resolved and live values
check [SECTION ...]          random-knob correctness smoke test; no timing
```

`--resume` makes long tunes safe to interrupt. Each section is written to `--out` as soon as it
finishes, so a stopped run loses at most the section that was in progress. Every run ends with a
derive, even a stopped one, so the file's derived knobs always match its measured ones.

### How knobs are tuned

> **Revised 2026-11-10 (the user's decision; it supersedes this section where they differ).**
> Phase 2's first sections showed that deriving a decision purely from other algorithms' models
> doesn't hold: of mul's and sqrt's 25 decisions tuned so far, one could be derived, and seven
> derivations failed their end-to-end checks (the models carry 2-9% error, more than most
> crossings tolerate). The method is now:
>
> 1. **One input: a sweep.** A decision whose shape is one number (a cutoff) is measured directly:
>    both sides timed over sizes, and the value with the least measured time lost. A coarse pass,
>    then a dense one around the crossing.
> 2. **More than one input: fitted runtime equations.** For a dispatcher over two or more inputs
>    (products, divisions), each algorithm gets a runtime equation in its theoretical form, with
>    constants fitted to that algorithm's own measured times over a grid of shapes. Dispatch
>    compares the fitted equations (per call for large inputs; as a table computed from them for
>    small ones).
> 3. **No purely derived runtimes.** An algorithm's equation may depend on another's (a Karatsuba
>    level's includes its sub-products' fitted schoolbook time), but its constants are always
>    fitted to its own measurements, never assembled from base units alone.
> 4. **A fitting library.** Fits use a linear-algebra dependency rather than a hand-written
>    solver: `nalgebra`, an optional dependency of the `tune` feature only, so normal builds of
>    the library never compile it. Since 2026-11-11 every fit is one unconstrained SVD solve
>    (`fit::solve`): every equation is linear in its constants (recursive Karatsuba's fitted
>    exponent became a schoolbook term with its own weight), so `levenberg-marquardt` is gone; a
>    constant solved negative rejects the fit instead of being dropped silently; and a fit is
>    judged on held-out shapes scattered over the area its algorithm competes on
>    (`tuner/design.rs`, `measure::time_fit`). The user's decisions; the rules are in CLAUDE.md
>    ("How a fit is found").
>
> 5. **Acceptance.** A fit is accepted at relative rms <= 5%; its worst residual is reported and
>    not judged. A level's thoroughness is the spread its samples must come under (6%, 4%, 2%).
>    Measurements run warm; no equation has a cache term.
> 6. **A rejected fit is the user's call.** The developer reports the fit, its equation and its
>    rms, proposes one or more new forms, and waits for approval.
>
> 7. **Only what a decision reads is measured (2026-11-10).** A fit or benchmark that no
>    dispatcher's decision depends on is removed from the tuner, the schema and these documents.
>    A procedure used in tuning only one algorithm is fitted as part of that algorithm, not
>    measured separately: its end-to-end fit captures it. Removed under this rule: all of
>    `base`'s cost units (`pass`, `call`, `mul_pass`, `div_pass`, `div_setup`, `alloc`, `fill`,
>    `rayon_wake`, `rayon_task`, `thread`, `eval`), the schoolbook-square and Karatsuba-square
>    level fits, `model.rs`, `dispatch.rs`, `threads.rs` and `DeriveCost`/`RuntimeCost`. Every
>    mention of them below is historical. Chunking Karatsuba, which alone read `base`, has its
>    own fit (`mul.chunk_*`).
> 8. **Parallel algorithms (2026-11-10).** A tune runs at the thread count the code will be used
>    at (users will be told to do so). The tuner finds each thread-dependent value twice, serial
>    and fully parallel, and the library chooses between the two sets at run time. Parallel gates
>    are one-input sweeps; equations for pool algorithms are fitted on the pool.
>
> 9. **Static algorithms: approximate values (2026-11-11).** Their memory handling is to be
>    reworked, and their best values depend on N as well as the size, so they are swept at one N
>    or left at their defaults. No fitted equations and no two-input dispatch for static entries.
>
> 10. **The two multi-input dispatchers run on fitted equations (2026-11-11).** `mul_dyn` past
>    its small-size table compares recursive Karatsuba's and the FFT's equations (`MulEq`), and
>    `div_dyn`/`div_rem_dyn` compare Knuth's, NR's and BZ's (`DivEq`, built on `MulEq::mul_ns`).
>    Each was adopted on a measured-loss comparison with the rule it replaced, on shapes neither
>    was tuned on (`cargo tune equations`, `cargo tune division`). The hand-fitted ratios and
>    BZ-vs-NR constants this document inventories for the dynamic entries are gone; the static
>    entries keep theirs. Details and numbers: `docs/tuning_implementation.md`, Steps 4 and 5.
>
> 11. **A model is revised once (2026-11-10).** Item 6 holds for a model's first form. If a
>    model has already been revised once and its fit still misses, the noisy fit is accepted as
>    it is and the work moves on: the tuner marks it NOISY, not REJECTED (`Fit::evidence_revised`),
>    writes its constants, and the problem is recorded with what was tried and what still
>    doesn't conform ("Marked problems" in `docs/tuning_implementation.md`: the series cost
>    model; BZ's equation was marked while the machine throttled and fits at 2.2-2.4% on a
>    steady one), to be investigated later.
>
> 12. **Any machine with a checkout can tune itself (2026-11-10).** The third layer this
>    document gives to Phase 3 is in `build.rs` now: after the target's preset it reads the
>    file `BIGBITS_TUNING` names, and the checkout's `.cargo/config.toml` points that at
>    `presets/local.toml`, which `cargo tune tune` writes by default and git ignores. So a
>    machine that has tuned itself builds with its own values, and any other with the preset or
>    the defaults; deleting the file returns to the preset. The file lives in `presets/` because
>    cargo can only notice a newly created file cheaply inside a directory `build.rs` already
>    watches. A file whose `[meta] target` isn't the build's is skipped with a warning (a cross
>    build would otherwise take the host's measurements), and a section set by several layers is
>    checked against the top layer's upstream record only. `cargo tune status` lists the layers
>    in force and which supplies each section. The `cargo-bigbits` driver, for projects that
>    depend on the crate, still waits for Phase 3.
>
> 13. **Every tune ends each section with the end-to-end check as a gate (2026-11-10).**
>    `tuner/check.rs` times one list of public calls (`tuner/check_items.rs`, shared with
>    `benches/tuning_bias.rs`) under two sets of knob values in one process, in paired rounds.
>    After a section is measured, its values are compared with the values before it over that
>    section's workloads and those of every section built on it. If the mean is slower by more
>    than the gate's allowance (3% quick, 2% normal, 1% thorough: `Level::gate`, the user's
>    numbers, set wider than the sweeps' tie band so that noise from one tune to the next
>    doesn't reject a section), with a 95% interval that excludes 1, or an exact output differs,
>    the section is dropped: its values and the file go back, and the summary says so. A single
>    workload more than 2% slower is listed and never drops anything. A tune therefore can't
>    leave a machine's workloads more than that allowance slower, on average, than it found them. `cargo tune check` runs the same
>    comparison for a file against the layers below it. A rejected fit is still written, marked
>    REJECTED (the user's decision): the gate, not the fit's quality, decides whether its
>    section stays.
>
> 14. **Decisions of the same day, for the record.** The FFT's odd-part factors stay hard-coded
>    per architecture (x86's to be measured on the i7). The tuner gets no steadiness preflight:
>    the "samples" line stays its one check of the machine. NR division's equation prices each
>    of its middle and high products at its own shape (`DivEq::nr_parts`), one fitted multiple a
>    tier. Binary splitting's parallel split, started off the pool, also needs
>    `bin_split.dyn_par_work` of limbs times terms, and the series cost model charges such a
>    tree one measured wake (`bin_split.dyn_wake_ns`). The tuner pins itself to the fastest
>    cores on Linux as well as macOS (compile-checked only; untested on a hybrid Linux machine).
> 15. **The series' dynamic dispatch is a third multi-input dispatcher on fitted equations
>    (2026-11-10).** Its marked cost model (seven prices shared by Horner, rect and binary
>    splitting over counted work) was rebuilt at the user's request on the other fitted
>    equations: each algorithm has its own equation over its own parts (`SeriesEq` over `Parts`
>    in `series.rs`), with multi-limb products priced by mul's equations and finalize's division
>    by division's, and a fitted price for each kind of linear pass, call, term and allocation.
>    Rect, which runs Horner's kernels, takes Horner's prices for them. Section `series` is in
>    `cargo tune tune` (version 2, 23 new knobs, upstream now includes `div`). On exp's and
>    ln's own series the picks lose 0.13% of measured time on the fitted workloads and 0.35%
>    held out, from 3.8% and 4.3%; the fits read 5.6%, 13.3% and 6.9%, above item 6's line
>    and so NOISY under item 11 (this was the model's one revision). What the rebuild found was
>    wrong counts, not wrong prices, and that one shared set of prices cannot fit three
>    algorithms (17%): item 10's rule again. A dispatcher's own evaluation is part of its cost:
>    the equations first made `exp_dyn` 15-23% slower at small sizes, until they stopped
>    predicting what could not change the pick. Static nodes keep the old shared prices
>    (item 9). `mul.fft_karatsuba` and `mul.fft_chunking_karatsuba`, read by no decision since,
>    were purged with mul's version 17 on 2026-11-11.
>
> What stays from the text below: fixed constants, the fit on relative error, fitting only where
> an algorithm competes, levels and budgets, evidence in the tuning file, and the end-to-end check
> of every decision. What goes: "composites have no measured values", the 39 measured / 66 derived
> split, and derivations of tables, extents, gates and balance points from predictions alone.
> Gates and balance points are one-input decisions, so they are sweeps.

The method has one rule: **measure only what can't be computed.** Every value is one of three
things:

| Value | What it is | Decided by | Count (target) |
|---|---|---|---|
| Fixed constant | a soundness limit, guard, capacity or policy, or a per-architecture cost factor | proof, or a developer's measurement per architecture, checked by validation; never tuned (invariant 3) | not knobs |
| Measured knob | a base unit, a leaf kernel's cost-model coefficient, or a sweep | timing, by the tuner | 39 |
| Derived value | a dispatcher's small-input table and extent, a parallel gate, a balance point, a predictor coefficient | computation from the measured knobs | 66 |

Today all 99 performance values are measured by hand, each area by its own harness. Here the
tuner measures 39 of them (37 off x86), and the rest take no timing at all (see the
[parameter count](#parameter-count)). One tune of a section runs five steps: measure its sweeps,
fit its leaf kernels' cost models, derive, check the derived values, and write the file.

The method refines a four-camp framework:
- **Correctness parameters:** universal, plain constants.
- **Unary operations:** one size decides between a stack of algorithms, by an integer cutoff
  swept over sizes, allowing for an algorithm that passes another more than once.
- **Binary operations:** two sizes decide. A developer-specified Big-O form per algorithm, with a
  scale and a shift fitted to samples by maximum likelihood, predicts wall time, and dispatch
  compares the predictions.
- **Everything else.**

Five refinements make up this method:
1. Correctness constants are universal per target rather than per computer, and checked by
   validation.
2. The binary method's models serve one-size decisions too (the one-dimensional case), so a unary
   sweep survives only where a model wouldn't pay for itself.
3. A composite algorithm's prediction is assembled from its parts' models, so the scale-and-shift
   form is needed only for leaf kernels.
4. Dispatch compares predictions at run time wherever doing so costs a negligible share of the
   call, as the binary method intended. Only small inputs read a table, derived from the same
   predictions.
5. Nearly everything in "everything else" is a derivation from those predictions.

The decisions behind these refinements are recorded under [Open questions](#open-questions).

#### Which values are measured

1. **Base units** (section `base`). Primitive costs that every work function counts:
   - a linear pass (add, subtract, copy, clear, negate) per limb;
   - a multiply-add by one limb, per limb;
   - a `div_prim` pass per limb;
   - the call overhead those three passes share;
   - a `ScratchGuard` acquisition;
   - a limb of a `[u64; N]` array zeroed or moved;
   - a rayon wake from outside the pool, a rayon task inside it, and a scoped thread;
   - one prediction term: a leaf model evaluated at a shape. This sets how large a call must be
     before comparing predictions pays (see [Dispatch](#dispatch)).

   Each is the scale or shift of its primitive's model, measured by timing that primitive over
   sizes.
2. **Leaf kernels' cost models.** t = scale·W + shift: two measured knobs per kernel, or one
   scale per kind of work for a kernel that counts more than one kind. The leaf kernels are
   `mul_buf`, `sqr_buf`, `hi_mul_buf`, `hi_sqr_buf`, `mid_mul_buf`, the FFT, the NTT, Knuth division
   and the binomial root. Three more model knobs complete the measured list:
   - `series.term`, Horner's and rect's per-term bookkeeping;
   - `series.eval`, one evaluation of the series dispatch's own model, which is too intricate to
     count in prediction terms;
   - `ln.par_fraction`, an Amdahl fraction measured end to end, which stands in for parallel
     structure too nested to predict (below).
3. **No composite has measured knobs of its own.** Its prediction is the sum of its parts'
   predictions and its own base units. This is the largest saving: Karatsuba, the transform
   entries, BZ, NR, Zimmermann, binary splitting and everything in `trans.rs` need no fitting.
4. **One model per kernel serves both allocation models.** Dynamic and static entries run the same
   inner loops. They differ in scratch: a dynamic entry acquires pooled buffers (`base.alloc` for
   each) and a static one zeroes or moves `[u64; N]` arrays (`base.fill` per limb). Work functions
   count both, so a static decision sees `N` itself instead of assuming `N` close to the operands,
   as today's static values do. Where a static path runs different code (its guarded high product
   is Karatsuba, not the FFT), the dispatch's set of algorithms differs, not the models.
5. **A decision is a sweep only between two code paths of one kernel, when neither path needs a
   model for anything else.** A cutoff is one measured value. A model is two, but serves every
   decision its kernel takes part in. So a decision between kernels that have models anyway is
   derived. A sweep remains only where modeling both paths would add values: today that's the ADX
   loops (`base.prim_adx`, `base.limb_adx`) and the high-product row kernels (`mul.hi_mul_rows`,
   `mul.hi_sqr_rows`). A sweep also stands in for any derived value that fails its check
   ([pinning](#checking-derived-values)).
6. **Everything else is derived.**

#### Work functions and predictions

A leaf kernel's work W is a Big-O count in the call's **actual** sizes:
- l·s for `mul_buf`;
- n(n+1)/2 for `sqr_buf`;
- about s·x.len() limb steps for the binomial root, one pass over x per root limb;
- ρ(o)·N·log2 N for the FFT at the transform size N the entry really uses (`find_fft_size`), where
  ρ(o) is a fixed factor for N's odd part o, hard-coded per architecture (`FFT_ODD_COST`, see
  [Fixed constants](#fixed-constants-never-knobs));
- N·log2 N per radix stage for the NTT, at its actual size.

Using actual sizes means the steps in transform cost, and the crossovers they cause more than once,
come out of W instead of needing more knobs. For example:
- `STATIC_NTT_SQR_CUTOFF`'s comment records the NTT winning from about 3070, losing at 3320 and
  3884, and winning again from 4040.
- `DYN_PARTIAL_MUL_CUTOFF`'s comment records the guarded product winning from 320 but losing at
  304–312.

The transform models are shared across entries:
- **FFT.** One real transform, shared by the product, square, middle-product and cyclic-square
  entries. Each entry's work function counts its transforms and its per-point passes
  (`mul.fft_point`).
- **NTT.** It counts butterflies per radix stage (`mul.ntt_radix{2,3,5}`), plus pointwise products
  and CRT per limb (`mul.ntt_point`), for every NTT entry, dynamic and static.

A composite's work function lists its parts:
- **One Karatsuba level over l × s:** three sub-products, each predicted as the least prediction
  among the algorithms Karatsuba's recursion could choose at its shape, plus its additions (linear
  passes), calls and scratch acquisitions.
- **BZ:** Knuth leaves and predicted products.
- **NR:** the predicted high and middle products of its chain, plus correction passes.
- **A Zimmermann stage:** a predicted division, a predicted square, and passes.

Derivation memoizes predictions by shape, so predicting a whole recursion stays cheap. At run time a
recursive composite uses a closed form instead (a [predictor coefficient](#derivations)), fitted to
that full prediction.

Parallel work is predicted from its structure. A fork-join of known pieces costs its most expensive
piece plus its overhead:
- `base.rayon_task` inside the pool;
- `base.rayon_wake` from outside it;
- `base.thread` for each scoped thread.

The pieces are shared out over the thread count of the call (see [Dispatch](#dispatch)). Every
thread counts as equal. A fitted Amdahl fraction stands in only where the parallelism is too nested
to follow: the series logarithm, with chunks on rayon, trees splitting inside them, and a tail
beside them (`ln.par_fraction`).

Dispatchers evaluate predictions on every large call, so work functions and their runtime
predictors, such as `mul_ns(l, s, stat)` for products, are code in every build, and read the same
knobs as derivation. The series model's `costs` and `pick`, and `agm_wins`, are runtime predictors
of this kind already.

#### Dispatch

Every choice among algorithms is made by a *dispatcher*:
- the product dispatchers `mul_dyn` and `mul_static`;
- the division dispatchers;
- Karatsuba's choice for its sub-products;
- the series dispatch;
- and so on.

A dispatcher has two regions, split at its *extent*:

- **At or above the extent, it compares predictions.** It evaluates each eligible algorithm's
  runtime prediction at the call's shape and thread count, and runs the least. This is the binary
  method's dispatch by predicted wall time, used directly. It follows transform steps exactly and
  needs no boundary formula, so today's hand-made forms (`is_school_at`'s parabola,
  `is_karatsuba`'s ratio, `DivCutoffs`) go away.
- **Below the extent, it reads a table.** Here evaluating predictions would cost too much of the
  call. The table is derived from the same predictions: the algorithm with the least prediction
  at every shape below the extent, stored as change points.
  - A one-input table is a short list of sizes, usually one, which is what today's cutoff is.
  - A two-input table has one short list per value of the shorter input, and the extent bounds its
    rows. Tables cover only small inputs. No dispatcher tabulates large ones.

The extent is in the dispatcher's own size measure (n, l·s, q·d, …). It is derived as the smallest
size from which evaluating the dispatcher's predictions costs at most ε of the call's predicted
time: its count of prediction terms times `base.eval`, or `series.eval` for the series dispatch.
If one algorithm wins at every shape past the table, as Zimmermann does for large roots, the table
ends with it, the extent is `inf`, and the dispatcher never evaluates a prediction.

The rules that apply in both regions:

- **Eligibility and soundness.** A prediction is infinite where its algorithm can't run:
  - Karatsuba squares below 6 limbs;
  - NR where `nr_split` refuses;
  - the 16-bit FFT past its limits.

  So soundness limits cap the choice without being knobs, and on a machine with many cores the
  comparison can switch from the FFT to the pool's NTT below the limit. The schema declares each
  algorithm's eligible sizes, and `build.rs` rejects a table that picks an algorithm outside them
  (invariant 1).
- **Recursions.** A recursion's own choice is a dispatcher over the sub-shapes it visits and the
  algorithms it may call. That covers Karatsuba's sub-products, BZ's blocks, Zimmermann's stages
  and binary splitting's nodes. A table entry there compares one level over leaves against the
  leaf.
- **Thread count.** A runtime prediction reads the call's thread count once (invariant 7): rayon's
  pool size for a dynamic entry, or `available_parallelism`, cached once per process, for a static
  one. That covers every thread count from 1 to the machine's exactly, and nothing is stored per
  count.
  - **Tables never depend on the thread count.** A parallel path pays only on work far larger than
    a thread's start-up, which lies far past any extent. If derivation ever found a parallel
    algorithm competing below an extent, it would lower the extent below that shape.
- **Parallel gates.** A gate chooses between serial and parallel for one piece of code. Its
  comparison, serial against a fork-join of p pieces, reduces to a closed form: work ≥
  W* / (1 − 1/min(k, p)), with k the call's thread count and W* a derived break-even work. That is
  one division per call, so gates have no table and no extent. Any scratch sized from a gate
  (the NTT's radix gates, invariant 5) reads it with the same k.
- **Balance points that depend on threads.** A balance point is an algorithm parameter, not a
  choice, so it can't be compared per call. When its predicted optimum moves with the thread
  count, as theta's term count does (5 on one thread, 10 to 12 on eight), it is derived for every
  thread count up to the machine's (computation only) and stored as a step function: the value
  from each count at which it changes.

#### Fitting a cost model

1. **Shapes.** Fit only where the kernel competes: from about half its smallest crossing with
   another algorithm to about twice its largest. Crossings come from predictions at the current
   values (the defaults, on a first run).
   - A model linear in its coefficients learns most from the two ends of that range, so a fit
     takes a few shapes at each end and a few in between for the lack-of-fit check.
   - A two-input kernel spreads its shapes over both inputs: square, very unbalanced, and in
     between.
2. **Timing.** Each shape runs its forced entry in batches long enough that the clock's
   resolution doesn't matter.
   - Rounds interleave every algorithm competing at that shape, so drift cancels between them.
   - The shape's value is the median of its rounds.
3. **Fit.** Weighted non-negative least squares on relative error. Timing noise is multiplicative
   and right-skewed, so this is the maximum-likelihood fit under log-normal error. It also keeps
   the largest shapes from deciding everything, as plain least squares on absolute time would. The
   series model's fit (`tune_series_cutoffs.py`) already works this way.
4. **Lack of fit.** Structure in the residuals means the work function is wrong, and more samples
   won't help. Examples: a run of same-signed residuals, or a pattern by transform size. The tuner
   flags the knob with its residuals and carries on with the fitted values; fixing W is a
   developer task.
5. **Settle the decisions.** Derive, then time both sides of every decision the section's kernels
   take part in. Below an extent that means each table change point. Above it, it means each
   crossing of the runtime predictions, at one thread and at the machine's full thread count.
   - A shape is *settled* when the confidence interval of the measured time ratio lies outside
     1 ± ε (decided) or inside it (tied). An unsettled shape gets more rounds, up to the level's
     cap.
   - Where a decided shape contradicts the prediction, the leaf kernels' own times at that shape
     join their fits, and the section is refit and rederived. That loop runs at most a few times.
   - Where the leaf kernels predict their own times but a composite's prediction is off, the
     composite's work function is flagged (invariant 12).

   This step is also the [check](#checking-derived-values) of the derived values.

The stopping rule is about decisions, not coefficients: the coefficients are good enough once
every decision they drive is settled. The tie band is what keeps that cheap. Near a crossover the
two sides are within noise by definition, and picking either costs less than ε, so the tuner never
spends samples there. Most samples go to the shapes where a decision is actually made.

#### Sweeps

A sweep times both code paths over sizes:

1. **Grid.** A geometric grid over the knob's search range, a few points per octave, refined
   wherever the faster side changes between neighbors.
2. **Settling.** Each size settles like a decision's shape: decided, or tied within ε.
   - Nanosecond-scale kernels need many short processes rather than one long one, because code
     layout and process state move them. The old mul harness ran 24 pinned processes per shape
     with an exact Wilcoxon test.
   - `HI_MUL_ROW_CUTOFF`'s comment records layout alone moving that crossover by up to 20 limbs,
     all inside 1%.
3. **Choice.** The value is the threshold with the least total time lost over the range, with
   sizes weighted uniformly in log size. With one crossing, that is the crossing. With several, it
   is the best single integer. Inside a tie the choice doesn't matter, and the evidence records the
   tie band.

#### Derivations

A derivation is computation on predictions. It reads only measured knobs, work functions and the
crate's predicates.

| Type | Examples | Computation |
|---|---|---|
| Table | `mul.kara`, `div.div_dyn`, `sqrt.stage`, bin_split's leaves | The algorithm with the least prediction at every shape below the dispatcher's extent, stored as change points: a list of sizes for one input, a list per value of the shorter input for two. A recursion's entry compares one level over leaves against the leaf, not whole recursions, which is what makes a leaf a crossover |
| Extent | every dispatcher's `_extent` | The smallest size, in the dispatcher's measure, from which its prediction terms times `base.eval` cost at most ε of the call's least prediction |
| Parallel gate | `mul.ntt_par*`, `bin_split.dyn_par`, ln's chunk gates | The break-even work W* at which serial and parallel predictions tie on unlimited threads; dispatch applies it with the call's thread count |
| Balance point | `exp.*_sqr_bits`, `ln.*_sqrt_a`, `ln.*_agm_terms`, `agm.*_tail_terms`, the chunk widths | The value with the least predicted end-to-end time, as a geometric mean over the section's sizes: the old scans' statistic, computed instead of measured. Const generics range over their compiled values. A balance point declared `Pool` or `Cpus` is a step function of the thread count, with one step when its optimum doesn't move |
| Structural | `sqrt.dyn_rcp_carry` | Read off the dispatch, with no cost comparison: the smallest root whose recursion reaches a second NR stage |
| Predictor coefficient | `mul.kara_pred`, `mul.chunk_pred`, `mul.kara_sqr_pred`, `ln.*_agm_limbs` | Closed forms fitted to a recursive composite's full prediction, for its runtime predictor; `agm_wins`' anchor κ |

Every derived value is clamped into its valid range, so a derivation can make a call slow, never
wrong (invariant 1). A decision that also depends on input values, such as ln's on δ's lead, keeps
those values as run-time inputs to its predictor.

#### Checking derived values

Step 5 of the fit times both sides of every decision. That is the check:
- Table change points are checked at their shapes.
- Runtime comparisons and gates are checked at their crossings, at one thread and at the full
  thread count.
- A balance point is checked against its neighboring values at a few sizes (and at both thread
  counts if it is a step function).
- Structural values need no check.

Each derived value's evidence records its worst time lost over its check shapes. A value that loses
more than the level's threshold at a decided shape is *flagged*, and `status` lists it. Then:

- **Usually, fix the work function.** A flag usually means a work function is wrong or missing a
  part (invariant 12). Fix it, bump the derive version, and rederive. This is the only fix for a
  runtime comparison or a two-input table, which have no single value to measure.
- **Otherwise, pin it** (`tune --pin KNOB`). A gate, a balance point or a one-input table can be
  measured directly by a sweep over its search range (a gate at the full thread count), and
  derivation leaves it alone until `--unpin`. A pinned value counts as measured, and the target
  counts above assume none.

#### Levels and budgets

| Level | Tie band ε | Rounds per shape | Check | For |
|---|---|---|---|---|
| `quick` | 3% | a fixed few | none; derived values are written unchecked | development, after a code change |
| `normal` | 1% | until settled, capped | every decision, on a coarse grid | the default; users |
| `thorough` | 0.5% | until settled; nanosecond sweeps over many processes | a dense grid | presets |

These numbers are starting values, to adjust with experience. ε also sets every extent, so a
tighter level gives larger tables and fewer runtime comparisons.
- `--budget` stops starting new sections when time runs out.
- A section that wasn't measured keeps the layer below's measured values. The final derive still
  uses them, so a partial tune leaves a consistent file.
- The tuner reports what it skipped: no time left, or not applicable on the target (the ADX sweeps
  off x86).

#### Evidence

Each value gets one evidence comment (invariant 9):

```toml
school_scale = 0.31          # fit: 14 shapes, 3×3 to 96×96; relative rms 1.8%
fft_scale = 0.0021           # fit: 11 shapes; lack of fit: +6% at 245760 (flagged)
hi_mul_rows = 26             # sweep 16–64: tied within 1% from 20 to 31
sqr_static = [1, 0, 72, 1]   # derived (table): sqr_buf from 1, Karatsuba from 72; check: worst 0.4% lost
sqr_static_extent = 640      # derived (extent): 3 prediction terms within 1% from 640 limbs
ntt_par = 2900               # derived (gate W*); check at 1 and 8 threads: worst 1.1% lost
bz_leaf = [1, 0, 150, 1]     # pinned: sweep 100–300; tied within 1% from 140 to 170
```

Each section's metadata records its conditions: the level, the date and the upstream hash. The
machine's thread count is recorded in `[meta]`.

#### Parameter count

Here a dispatcher (its table and its extent) counts as one derived value.

| Section | Today (hand-measured) | Measured | Derived |
|---|---|---|---|
| `base` (new) | — | 10, plus 2 sweeps on x86 | — |
| `mul` | 24 | 18, plus 2 sweeps | 20: 13 dispatchers, 4 gates, 3 predictor coefficients |
| `div` | 19 (2 of them `bz_2_1_cost`'s scales) | 2 | 8 dispatchers |
| `sqrt` | 9 | 2 | 9: 8 dispatchers, 1 structural |
| `bin_split` | 10 | — | 8: 6 dispatchers, 2 gates |
| `series` | 9 | 2 | 1 dispatcher |
| `consts` | 6 | — | none: runtime comparisons |
| `theta`, `agm`, `exp`, `ln` | 22 | 1 | 20: 4 dispatchers, 2 gates, 12 balance points, 2 predictor coefficients |
| **Total** | **99** | **39 (37 off x86)** | **66** |

Where the savings come from:
- **Composites have no coefficients.** div goes from 19 values to 2, sqrt from 9 to 2, and
  bin_split, consts and the `trans.rs` sections to almost nothing.
- **Dynamic and static entries share models.** The 54 of today's knobs that come in
  `DYN_`/`STATIC_` pairs need no static-only measurements.
- **The series model's per-unit costs become base units** that every section shares.
- **Large inputs store nothing.** Above each extent, dispatch computes its answer per call, at the
  call's thread count. Transform-step crossovers, crossovers against parallel algorithms, and the
  FFT-to-NTT switches need no stored value.
- **Parallel gates are predicted** from three overheads.

mul measures about as many values as today, because it holds the leaf kernels that everything else
is built from.

### Section dependencies and order

Dependencies come from the code: who calls whose dispatch, and whose prediction reads whose knobs.
Measured and derived knobs depend on other sections differently.

- **Measuring** a leaf kernel times only that kernel, so its fit depends on no other section
  except `base`, whose sweeps can change the kernel's inner loop (`mul_buf`'s multiply-add rows
  switch at `base.prim_adx`). So `base` is measured first, and the other leaf kernels in any order
  after it. The exception is `ln.par_fraction`, which is measured end to end through every
  section, so it comes last.
- **Deriving** follows the call graph, because a composite's prediction calls its parts'
  predictions:

```
base ──► mul ──► div ──► sqrt ──────────────┐
          │       │                          ▼
          │       └────► bin_split ──► series ──► consts ──► exp, ln
          └──────────────────────────────────────► theta ──► ln
                                  sqrt, div, mul ──► agm ──► ln
```

- Every section's work functions count base units.
- `div` uses mul's products: NR's high and middle products, and `bz_2_1_cost` calls `dyn_dispatch`.
- `sqrt` divides at each stage and squares, including the square mod B^m − 1.
- `bin_split` merges with products and finalizes with `ratio_to_fraction_*` (a division or a
  reciprocal).
- `series`' model reads mul's predictions (`mul_ns`, which replaces `mul_work`) and bin_split's
  cutoffs.
- `consts` builds through bin_split and the series, and π takes a root.
- exp and ln sit on everything.

The tuner measures in the order base, then the rest, and derives in topological order: base, mul,
div, sqrt, bin_split, series, consts, theta, agm, exp, ln. Derivation only computes, so every
tune ends by rederiving every section. Measuring mul alone therefore updates div's, sqrt's and
every other section's derived knobs, without timing them. The upstream hash records which measured
values each section's derived knobs came from. It goes stale only when a file is edited or layered
by hand, and `derive` fixes it.

### Validation

There are three checks, run once per section when it migrates; each section passes all three
before its old harness is retired. They are separate from the [check of derived
knobs](#checking-derived-values) that every tune runs:

1. **Random-knob correctness** (`src/tests/test_tuning.rs`, `cargo test --features tune`). This
   samples knob vectors: uniform draws within each knob's valid range (capped to keep sizes
   practical), each range's extremes, and the "everything low" and "everything high" vectors.
   Everything low forces the upper algorithm at tiny sizes: Karatsuba at a few limbs, FFT and NR
   division early, Zimmermann at the smallest roots. For each vector it runs a fixed set of
   operations against the suite's oracles: schoolbook `mul_buf`, Knuth division, the sqrt and
   series oracles, `exp_full_oracle` and `ln_full_oracle`. This is invariant 1 as a test. Run it
   after any change to a dispatcher.
2. **Bias check.** For each section, once, at migration: time its public entries in a normal build
   and in a tune build at the same values, on the same inputs. Build both binaries once, alternate
   them over several processes, and compare paired ratios. Accept the section if the two agree within noise at the
   knob's own sizes. If they don't, a knob is being read somewhere hot; hoist the read
   (invariant 7). This is the answer to the sqrt script's reason for keeping tuning out of `src/`:
   live values are accepted once they're shown not to move timings.
3. **Comparison with the old harness.** On the machine where a section was last tuned, compare
   the new tuner's values with the old harness's (for example the "0.2% plateau" ranges in
   `docs/sqrt_cutoff_tuning.md`). Explain any difference before retiring the old harness. A
   derived value outside the old harness's tie band is a finding about a work function until shown
   otherwise. A two-dimensional boundary may differ where the old form was a compromise that no
   single parameter fitted, and then the comparison is by time lost, not by value.

## Knob inventory

This covers all 106 constants in `src/utils/mod.rs` after Phase 0 (`faa9971`'s 104, plus the two
fitted scales Phase 0 moved out of `div.rs`'s `bz_2_1_cost`), and the measured knobs that the
[method](#how-knobs-are-tuned) adds.

The knob names are the proposal for Phase 2, following the rule `<section>.<dyn|static>_<what>`:
- drop `dyn_`/`static_` when one value serves both;
- drop `_CUTOFF`;
- name a model's pair `<kernel>_scale` and `<kernel>_shift`;
- give a dispatcher two values, its table `<dispatcher>` and its `<dispatcher>_extent`.

Phase 1 migrates today's constants under their own names first (see [Phase 1](#phase-1-the-seam)).

Each section has two tables:
- **Measured knobs.** "Work" gives the work function W, or what a base unit costs. "Today"
  names the constant it replaces, or "—" for a knob the method adds.
- **Derived values.** "Absorbs" names today's constants that the value replaces. A dispatcher's
  crossings below its extent become change points in its table; crossings above the extent become
  runtime comparisons, with nothing stored (see [Dispatch](#dispatch)).

Other conventions:
- "Threads" in the notes: `pool` means the dispatcher's predictions read rayon's pool size, `cpus`
  means they read `available_parallelism`. Tables are single-threaded throughout.
- "GUESS" marks a value that `mod.rs` says was never measured.

### Section `base` (new: 10 model knobs, 2 sweeps; upstream: none)

Phase 1 migrated 2026-11-08: `base.prim_adx` and `base.limb_adx` (x86_64 only; their bias check is pending on the i7-8750H, as they compile out on AArch64).

| Knob | Today | Kind | Work | Notes |
|---|---|---|---|---|
| `base.prim_adx` | `MUL_PRIM_ADX_CUTOFF` | Sweep | `mul_add_prim`'s plain loop vs its mulx/adcx loop | x86_64 only; nanosecond-scale |
| `base.limb_adx` | `MUL_LIMB_ADX_CUTOFF` | Sweep | `add_mul_limb`/`sub_mul_limb`'s one-limb loops vs the ADX rows | x86_64 only; used in `utils.rs` |

### Section `mul` (18 model knobs, 2 sweeps, 20 derived; upstream: base)

Phase 2 so far (2026-11-10): measured `mul.school_{scale,row,shift,err}`, `mul.school_sqr_{scale,row,shift,err}`, `mul.kara_{scale,pass,shift,err}` and `mul.kara_sqr_{scale,pass,shift,err}` (the Karatsuba levels are fitted forms, decision 7; the `_err` knobs are each fit's relative rms, used as decision margins); derived `mul.kara` (a two-input table, read by `is_school`) and `mul.karatsuba_sqr`. The names differ from the proposal below where the implementation settled them. The other interim knobs remain.

Phase 1 migrated 2026-11-08: the 22 interim knobs `mul.<old name, lowercased, without _CUTOFF>` (`mul.ntt_par_radix3` and `_radix5` for the radix gates, `mul.sqr_bnm1_fft` a `Follows` of `mul.fft_sqr`). The random-knob test found `is_school_at`'s parabola cancelling catastrophically for large c; it is computed as `1.5 l − l²/(2c)` now (same boundaries at practical c), and `mul.karatsuba` ≥ 3.

| Knob | Today | Kind | Work | Notes |
|---|---|---|---|---|
| `mul.school_{scale,shift}` | — | Model | `mul_buf`: l·s limb products | scale defaults to `SERIES_NS_PRODUCT` |
| `mul.school_sqr_{scale,shift}` | — | Model | `sqr_buf`: n(n+1)/2 | |
| `mul.hi_school_{scale,shift}` | — | Model | `hi_mul_buf`: the window's column products plus guard columns | columns or rows per `mul.hi_mul_rows` |
| `mul.hi_school_sqr_{scale,shift}` | — | Model | `hi_sqr_buf` | |
| `mul.mid_school_{scale,shift}` | — | Model | `mid_mul_buf`: n² | |
| `mul.fft_{scale,shift}`, `mul.fft_point` | — | Model | one real transform, ρ(o)·N·log2 N at the actual N; per-point passes (decompose, pointwise product, rounding, accumulate) | shared by the product, square, middle-product and cyclic-square entries; ρ(o) is the fixed per-architecture `FFT_ODD_COST` |
| `mul.ntt_radix{2,3,5}`, `mul.ntt_shift`, `mul.ntt_point` | — | Model | one NTT's butterflies per radix stage at the actual size; pointwise products and CRT per limb | shared by every NTT entry, dynamic and static |
| `mul.hi_mul_rows` | `HI_MUL_ROW_CUTOFF` | Sweep | `hi_mul_buf`'s columns vs rows | within about 1% over a wide band; layout moves it up to 20 limbs |
| `mul.hi_sqr_rows` | `HI_SQR_ROW_CUTOFF` | Sweep | `hi_sqr_buf`'s columns vs rows | |

| Value | Kind | Algorithms | Absorbs | Notes |
|---|---|---|---|---|
| `mul.mul_dyn` | dispatcher (l, s; extent in l·s) | schoolbook, chunking Karatsuba, Karatsuba, FFT, NTT | `CHUNKING_KARATSUBA_CUTOFF`, `KARATSUBA_CUTOFF`, `FFT_CHUNKING_KARATSUBA_CUTOFF`, `FFT_KARATSUBA_CUTOFF`; the switch to the NTT that today sits at `FFT_16BIT_CUTOFF` | pool. The FFT's prediction is infinite past its limit. The M1 wanted ratios of about 3.4 and 3.0 |
| `mul.mul_static` | dispatcher (l, s) | schoolbook, chunking Karatsuba, Karatsuba, NTT | `NTT_CHUNKING_KARATSUBA_CUTOFF`, `NTT_KARATSUBA_CUTOFF` | pool; counts its fills |
| `mul.kara` | dispatcher (l, s) | schoolbook, chunking, a Karatsuba level | `is_school`'s use in `kdispatch` | Karatsuba's sub-products |
| `mul.sqr_dyn` | dispatcher (n) | `sqr_buf`, Karatsuba square, FFT, NTT | `FFT_SQR_CUTOFF`, `KARATSUBA_SQR_CUTOFF` at the top level; the switch at `FFT_SQR_16BIT_CUTOFF` | pool; the M1 wanted about 500 |
| `mul.sqr_static` | dispatcher (n) | `sqr_buf`, Karatsuba square, NTT | `STATIC_NTT_SQR_CUTOFF`, `KARATSUBA_SQR_CUTOFF` at the top level | pool |
| `mul.kara_sqr` | dispatcher (n) | `sqr_buf`, a Karatsuba square level | `KARATSUBA_SQR_CUTOFF` as the recursion's leaf | a level needs at least 6 limbs |
| `mul.hi_mul_dyn`, `mul.hi_mul_static`, `mul.hi_sqr_dyn`, `mul.hi_sqr_static` | dispatcher (window) | `hi_*_buf`, the guarded product (the top limbs' predicted product plus guard passes) | `{DYN,STATIC}_PARTIAL_{MUL,SQR}_CUTOFF` | 4 dispatchers; the static ones count fills |
| `mul.mid_dyn` | dispatcher (n) | `mid_mul_buf`, FFT, NTT | `FFT_MID_CUTOFF`; the switch at `FFT_MID_16BIT_CUTOFF` | pool |
| `mul.mid_static` | dispatcher (n) | `mid_mul_buf`, NTT | `NTT_MID_CUTOFF` | pool; 997 on one core, 672 on the pool |
| `mul.bnm1` | dispatcher (m) | a square and fold, the cyclic FFT | `DYN_SQR_BNM1_FFT_CUTOFF`; the limit at `FFT_BNM1_16BIT_CUTOFF` | GUESS today |
| `mul.ntt_par_conv` | parallel gate | the three primes' convolutions | `NTT_PAR_CUTOFF_NTT_CONV` | pool |
| `mul.ntt_par` | parallel gate | a convolution's two forward transforms | `NTT_PAR_CUTOFF_NTT` | pool |
| `mul.ntt_par_radix3`, `mul.ntt_par_radix5` | parallel gate | a radix-3 or radix-5 step's sub-transforms | `NTT_PAR_CUTOFF_NTT_3`, `NTT_PAR_CUTOFF_NTT_5` | pool; scratch sizing reads them with the call's thread count (invariant 5) |
| `mul.kara_pred`, `mul.chunk_pred`, `mul.kara_sqr_pred` | predictor coefficient | | `mul_work`'s scaling at the crossovers | closed forms of the Karatsuba predictions, for `mul_ns` |

### Section `div` (2 model knobs, 8 derived; upstream: base, mul)

Phase 1 migrated 2026-11-08: 19 interim knobs, `div.<old name, lowercased, without _CUTOFF>` (`div.knuth`, `div.bz`, `div.bz_top_padded_cost_scale`, `div.bz_2_1_school_to_karatsuba`, `div.bz_2_1_karatsuba_to_fft`, the twelve BZ-vs-NR values, the two reciprocal cutoffs). The random-knob test set `div.bz`'s valid minimum to 3.

| Knob | Today | Kind | Work | Notes |
|---|---|---|---|---|
| `div.knuth_{scale,shift}` | — | Model | Knuth (`div_buf_of`): q·d, one `sub_mul_limb` row and one quotient estimate per quotient limb | it could be a composite of base passes, but it is every division's leaf, so it gets its own model |

| Value | Kind | Algorithms | Absorbs | Notes |
|---|---|---|---|---|
| `div.div_dyn`, `div.div_rem_dyn`, `div.div_static`, `div.div_rem_static` | dispatcher (q, d; extent in q·d) | Knuth, BZ, NR | `DIV_KNUTH_CUTOFF` and the twelve `{DYN,STATIC}_{DIV,DIV_REM}_*_NR_BZ_CUTOFF` | `hi_div` uses div's. NR's prediction is infinite where `nr_split` refuses. `div_rem`'s finish and static's fills are in the work functions |
| `div.bz_leaf` | dispatcher (d) | Knuth, a BZ level | `BZ_CUTOFF` | BZ's recursion |
| `div.bz_top` | dispatcher (d, q) | the top block by Knuth, BZ's padded 2/1 block | `BZ_TOP_PADDED_COST_SCALE`; `SCHOOL_TO_KARATSUBA` and `KARATSUBA_TO_FFT` in `bz_2_1_cost` | the hidden constants go away with `bz_2_1_cost` (Phase 0 moves them to `mod.rs` until then) |
| `div.rcp_dyn`, `div.rcp_static` | dispatcher (R) | Knuth, NR | `DYN_RCP_KNUTH_NR_CUTOFF`, `STATIC_RCP_KNUTH_NR_CUTOFF` | |

### Section `sqrt` (2 model knobs, 9 derived; upstream: base, mul, div)

Phase 1 migrated 2026-11-08: the nine constants are `Interim` knobs `sqrt.leaf`, `sqrt.dyn_rem`,
`sqrt.dyn_only`, `sqrt.dyn_approx`, `sqrt.static_rem`, `sqrt.static_only`, `sqrt.static_approx`,
`sqrt.dyn_wrapped_rem` and `sqrt.dyn_rcp_carry`. Bias check: 1.0014× over 42 sqrt items (normal
against tune build, M1). The random-knob test set every cutoff's valid minimum to 3 (see
`docs/sqrt_cutoff_tuning.md`).

| Knob | Today | Kind | Work | Notes |
|---|---|---|---|---|
| `sqrt.binomial_{scale,shift}` | — | Model | `binom_sqrt_core`: one pass over x per root limb, about s·x.len() limb steps | |

| Value | Kind | Algorithms | Absorbs | Notes |
|---|---|---|---|---|
| `sqrt.{dyn,static}_{rem,only,approx}` | dispatcher (x limbs) | binomial, Zimmermann | the six `{DYN,STATIC}_SQRT_{REM,ONLY,APPROX}_ZIMMERMAN_CUTOFF` | 6 dispatchers. Each output mode's work function differs (the root-only top usually skips its square); the static ones count fills |
| `sqrt.stage` | dispatcher (root limbs) | binomial, a Zimmermann stage (a predicted division and square) | `ZIMMERMAN_SQRT_LEAF_CUTOFF` | the recursion's leaf; shared by dyn and static |
| `sqrt.wrapped` | dispatcher (root limbs, per stage) | the classic remainder, the wrapped remainder | `DYN_SQRT_WRAPPED_REM_CUTOFF` | dynamic only; GUESS today |
| `sqrt.dyn_rcp_carry` | structural | | `DYN_SQRT_RCP_CARRY_CUTOFF` | the smallest root whose recursion reaches a second NR stage: about 680 root limbs at today's cutoffs (its comment) |

### Section `bin_split` (8 derived; upstream: base, mul, div)

Phase 1 migrated 2026-11-08: `bin_split.<dyn|static>_<bbp|engel|hyper>_leaf_limbs`, `bin_split.<dyn|static>_hyper_leaf_terms`, `bin_split.dyn_par` and `bin_split.dyn_merge_par`; every value valid.

| Value | Kind | Algorithms | Absorbs | Notes |
|---|---|---|---|---|
| `bin_split.{dyn,static}_{bbp,engel,hyper}_leaf` | dispatcher (the node's estimated limbs; and terms, for Hyper) | one leaf, a split into two leaves and a merge | the six `{DYN,STATIC}_{BBP,ENGEL,HYPER}_LEAF_LIMBS` and the two `{DYN,STATIC}_HYPER_LEAF_TERMS` | 6 dispatchers. Leaf terms cost multiply-add passes, merges are predicted products, and the static side counts its fills. Today these are associated consts (`Node::LEAF_LIMBS`) |
| `bin_split.dyn_par` | parallel gate | a split's two halves | `DYN_BIN_SPLIT_PAR_CUTOFF` | pool; also read by the series model |
| `bin_split.dyn_merge_par` | parallel gate | a merge's independent products | `DYN_BIN_SPLIT_MERGE_PAR_CUTOFF` | pool |

### Section `series` (2 model knobs, 1 derived; upstream: base, mul, div, bin_split)

Phase 1 migrated 2026-11-08: `series.ns_{div,scalar,product,term,call,alloc,fill}`, `series.horner_limbs`, `series.horner_terms`. The random-knob test found `rect_best` returning block length 0 when every prediction is infinite; fixed.

**As built (2026-11-10, revision note 15), not the table below:** dynamic nodes dispatch on three
fitted equations, `series.horner_ns_{product,div1,div2,mul1,mul2,light,term,call,one}`,
`series.rect_ns_{product,call,one}` (rect takes Horner's prices for the linear kernels) and
`series.bin_split_ns_{product,division,div1,mul1,mul2,add,light,term,call,alloc}`, each a
Model knob fitted by `tuner/sections/series.rs` to that algorithm's forced timings on exp's and
ln's own series with the entry's own work taken off, plus `series.bin_split_par_gain`, what a
tree's parallel split gains a thread. The `product` and `division` prices are multiples of
mul's and division's predictions (1.01 and 1.00 for binary splitting on the M1). Static nodes
keep `series.ns_{div,scalar,product,term,call,fill}` (`ns_alloc` is gone: no static algorithm
allocates). `series.horner_limbs` and `series.horner_terms` stay unswept at 28 and 12.

**Since 2026-11-11 (series version 4; the user's direction: the evaluators are general, and
their fitting takes no caller's series):** the equations are fitted on a synthetic family
(`tuner/synth.rs`: coefficients of a set size with and without convergence of their own, a
factorial's, and BBP's two) scattered over frame, terms and x's width, each algorithm where it
competes; and the dispatch has two guards ahead of the equations, on the shape's ratios rho
(x's bits over the bits a term fixes) and beta (the exact fraction's width over the frame):
`series.bin_split_beta`, `series.rect_rho` and `series.rect_x_limbs`, Sweep knobs measured
through the entry itself on series of 50 us and more. The constants that went on 2026-11-11
(`horner_ns_one`, `bin_split_ns_mul1`) are not in the lists above any more. The log is in
`docs/tuning_implementation.md`, "Built: the series tuned as general evaluators". And a short
series reads no equation: `series.small` (Derived, a table like `mul.kara`) holds what to run
by (class x 16 + the frame's bucket, terms), derived from the equations at each class's
representative shape out to where they put a series past 20 us ("The small-series table" in the
same file; its layout was cut to 2602 change points on 2026-11-12: octave frames, four q
buckets, ten kinds); `cargo tune derive` rederives it (`cargo tune derive --file FILE series`
for it alone). Since 2026-11-12 the prices are two sets: `series.{horner,rect,bin_split}_ns_*`,
fitted on the series past the table's extent (where the dispatch reads the equations), and
`series.short_{horner,rect,bin_split}_ns_*`, fitted on those under it and read only by the
table's derivation; rect has `rect_ns_div1`, its own price for one-limb divisions.

| Knob | Today | Kind | Work | Notes |
|---|---|---|---|---|
| `series.term` | `SERIES_NS_TERM` | Model | a Horner or rect term's group and weight bookkeeping | |
| `series.eval` | — | Model | one evaluation of the dispatch's cost model (`costs` and `pick`) | sets the dispatcher's extent; 200–330 ns at 1–4 limbs (`SERIES_HORNER_LIMBS`' comment) |

`SERIES_NS_DIV`, `SERIES_NS_SCALAR`, `SERIES_NS_CALL`, `SERIES_NS_ALLOC` and `SERIES_NS_FILL` move to
`base`. `SERIES_NS_PRODUCT` goes away, as products are predicted by mul's models through `mul_ns`.

| Value | Kind | Algorithms | Absorbs | Notes |
|---|---|---|---|---|
| `series.dispatch` | dispatcher (output limbs and terms, per series kind) | Horner, rect, bin_split | `SERIES_HORNER_LIMBS` (its extent), `SERIES_HORNER_TERMS` (its table) | Above the extent it runs the existing runtime model, so the series dispatch already has the shape every dispatcher takes. `SERIES_HORNER_TERMS` exists because the model wrongly favors bin_split on short BBP and Engel ranges. If the refit fixes that, the table needs no terms entry |

### Section `consts` (no derived values; upstream: base, bin_split, series, sqrt, div)

Phase 1 migrated 2026-11-08: `consts.dyn_ln2_par`, `consts.dyn_ln2_par_half`, `consts.dyn_pi_par`, `consts.dyn_pi_par_half`, `consts.static_ln2_par`, `consts.static_pi_par`. `cache_par` saturates now, so every value is valid.

`DYN_LN2_PAR_CUTOFF`, `DYN_LN2_PAR_HALF`, `DYN_PI_PAR_CUTOFF`, `DYN_PI_PAR_HALF`,
`STATIC_LN2_PAR_CUTOFF` and `STATIC_PI_PAR_CUTOFF` become runtime comparisons. Each compares a cache
extension's predicted serial time with its parallel time plus a rayon wake (or the scoped threads'
start), at the call's thread count. Extensions happen only past the 1024-limb tables, where
evaluating the predictions costs a negligible share of the call, so `cache_par`'s two-parameter form
and its four values go away. They were measured on an older base-16 ln 2 series and have low
priority.

### Sections `theta`, `agm`, `exp`, `ln` (1 model knob, 20 derived; their Phase 2 waits until those algorithms settle)

Phase 1 migrated 2026-11-09: 22 interim knobs (`theta.<dyn|static>_par_limbs`, `agm.<dyn|static>_tail_terms`, `exp.<dyn|static>_{chunk_par_work,chunk_min_bits,sqr_bits}`, `ln.<dyn|static>_{sqrt_a,chunk_min_bits,chunk_par_work,agm_limbs,series_par_fraction,agm_terms}`). The theta term counts, const generics, go through `knob_generic!`.

| Knob | Today | Kind | Work | Notes |
|---|---|---|---|---|
| `ln.par_fraction` | `DYN_LN_SERIES_PAR_FRACTION`, `STATIC_LN_SERIES_PAR_FRACTION` | Model | Amdahl's parallel fraction of the series logarithm, fitted over thread counts | one value for both, as their chunk structure is the same; static's is a GUESS today |

| Value | Kind | Algorithms | Absorbs | Notes |
|---|---|---|---|---|
| `theta.dyn`, `theta.static` | dispatcher (frame limbs) | the `SERIAL` plan, the `PARALLEL` plan (threaded on two or more threads) | `DYN_THETA_PAR_LIMBS`, `STATIC_THETA_PAR_LIMBS` | pool, cpus. `PARALLEL`'s prediction is its products along the critical path plus overhead. GUESS today |
| `agm.dyn_tail_terms`, `agm.static_tail_terms` | balance point | | `{DYN,STATIC}_AGM_TAIL_TERMS` | the tail at t terms vs one more mean step and a tail of about t/2; at most `AGM_TAIL`'s 19 |
| `exp.dyn`, `exp.static` | dispatcher (chunks × working limbs) | serial with squares, parallel without | `DYN_EXP_CHUNK_PAR_WORK`, `STATIC_EXP_CHUNK_PAR_WORK` | pool, cpus |
| `exp.{dyn,static}_chunk_min_bits` | balance point | | `{DYN,STATIC}_EXP_CHUNK_MIN_BITS` | the first chunk's width |
| `exp.dyn_sqr_bits`, `exp.static_sqr_bits` | balance point | | `{DYN,STATIC}_EXP_SQR_BITS` | squares vs the series terms they save; static is a GUESS today |
| `ln.dyn_sqrt_a`, `ln.static_sqrt_a` | balance point, by threads | | `{DYN,STATIC}_LN_SQRT_A` | pool, cpus. Square roots vs the series terms they save; static is a GUESS today |
| `ln.{dyn,static}_chunk_min_bits` | balance point | | `{DYN,STATIC}_LN_CHUNK_MIN_BITS` | the first chunk's width; GUESS today |
| `ln.dyn_chunk_par`, `ln.static_chunk_par` | parallel gate | ln's chunk series | `{DYN,STATIC}_LN_CHUNK_PAR_WORK` | pool, cpus; GUESS today |
| `ln.{dyn,static}_agm_limbs` | predictor coefficient | | `{DYN,STATIC}_LN_AGM_LIMBS` | `agm_wins`' anchor κ, from where the predicted AGM and series logarithms tie for a generic X on one thread. `agm_wins` already compares them at run time with the series' thread speedup. GUESS today |
| `ln.dyn_agm_terms` | balance point, by threads | | `DYN_LN_AGM_TERMS` | pool. Theta's terms vs the AGM steps they save, over the compiled T: 5 on one thread, 10 to 12 on eight. Const generic (`const_dispatch!` in tune builds); `docs/agm-ln-theta.md` predicts about 2/(R ln 2), for R a term's cost over a step's |
| `ln.static_agm_terms` | balance point, by threads | | `STATIC_LN_AGM_TERMS` | cpus; over the T whose plans fit `THETA_STATIC_SLOTS`; const generic |

### Fixed constants (never knobs)

| Constant | Why it is fixed |
|---|---|
| `FFT_16BIT_CUTOFF`, `FFT_SQR_16BIT_CUTOFF`, `FFT_MID_16BIT_CUTOFF` | Soundness: the largest sizes that keep balanced coefficients under 2^48, enforced by the assert in `mod.rs` and feeding `find_fft_size` in const contexts. Past them the FFT's prediction is infinite, so the dispatchers switch to the NTT there at the latest, and earlier when the NTT's prediction wins |
| `FFT_BNM1_16BIT_CUTOFF` | The same assert. Its value is still an old measured trip rate (see its comment), so it caps `mul.bnm1`'s cyclic FFT; raising it needs a soundness argument, not a measurement |
| `FFT_FOLD_TRIGGER` | Soundness of the fold check (`docs/fft_precision.md`) |
| `FFT_RESIDUE_CHECK` | A policy of safety against speed. If users ever choose it, that is a cargo feature (see [Open questions](#open-questions)), never a tuned value |
| `THETA_STATIC_SLOTS` | Capacity: array lengths, and compile-time asserts on theta's plans |

The same applies to accuracy and capacity constants outside `mod.rs`:
- guard limbs and bits: `HI_MUL_GUARD_LIMBS`, `NR_RCP_GUARD_LIMBS`, div's `GUARD`,
  `EXP_RED_GUARD_BITS`, trans's `GUARD_LIMBS`, `AGM_TAIL_GUARD`, `HORNER_GUARD`, and the guard limbs
  throughout;
- bounds: `NR_DIV_MAX_CORRECTIONS`, `MAX_TERMS`;
- capacities: `RECT_MAX_M`, `GROUP_MAX`, `MAX_PAR_CHUNKS`, `THETA_MAX_TERMS`, `MAX_VALUES`;
- the soundness tables `FFT_FOLD_PLAIN`, `FFT_FOLD_WRAPPED` and `AGM_TAIL`.

Some fixed constants depend on the machine. They are fixed per architecture (by `cfg` where
needed) and checked by validation runs, never tuned:

- **`FFT_ODD_COST`** (new, beside `FFT_ODD_PARTS` in `mul.rs`): the FFT's per-point cost factor
  ρ(o) for each odd part, relative to a power of two. It is hard-coded for each architecture the
  library has assembly for:
  - **AArch64:** the NEON medians in `docs/fft_precision.md`: products 1.00, 1.00, 1.21, 1.18,
    1.21 for o = 1, 3, 5, 7, 9, and squares within 0.02 of those, so one factor serves both.
  - **x86-64:** not yet measured. rustfft's AVX planner chains its radix steps differently, so the
    NEON tiers may not hold. Measure it with `benches/probes/fft_sizes.rs` (`sizes` mode) before
    the x86 preset ships.
  - **Other targets:** 1.0 for every part.

  Changing the factors changes what `mul.fft_scale` means, so it bumps mul's version.
- **`FFT_ODD_PARTS`** ({1, 3, 5, 7, 9}, `mul.rs`): the transform sizes, chosen from rustfft's NEON
  costs (its comment says to remeasure on x86). It feeds the const-evaluated FFT maximum sizes and
  the `FFT_FOLD_WRAPPED` primes, so it changes only with a soundness review.
- **`LANES`, `MIN_SEGMENT`** (`verify.rs`): the residue fold splits into four chains per prime from
  128 limbs. That is a code-path crossover, but it matters only with `FFT_RESIDUE_CHECK` or in
  debug builds. It becomes a `base` sweep if the residue check ever ships on.
- **`SQR`, `MUL`** (`theta_plan.rs`): the square and product weights of theta's compile-time plans.
  The plans are built in const contexts, so these stay constants.
- **`static_split_stack(n)`** (`series.rs`): a static split thread's stack, 1024 bytes per limb of
  `N` plus 1 MiB. Codegen sets the real need, so it can differ by target and compiler. The open
  stack overflow at `N = 16600` (CLAUDE.md) runs in threads sized by it.
- **The FFT limits and fold trigger** are proved, but were validated on NEON only. Under Rosetta
  (SSE) a few worst-case middle products inside the limits went wrong, and AVX2 is unmeasured
  (CLAUDE.md). Each target needs its own validation run before its preset ships.

## Phases

### Phase 0: hygiene and inventory (no behavior change) (landed 2026-11-08)

1. Move `criterion` and `rand` from `[dependencies]` to `[dev-dependencies]` only. The library
   never uses them: only `src/main.rs`, the tests and the benches do. As it stands, every
   downstream build would compile criterion.
2. Delete `src/main.rs` (an empty `main` that only imports criterion and rand) and the empty
   `src/bin/`. Otherwise `cargo install big_bits` would install a binary that does nothing.
3. Move the fitted constants hidden in `div.rs`'s `bz_2_1_cost` into `mod.rs`. Check for others
   with `grep -rnE '^\s+const [A-Z_]+: f64' src/utils` and add any you find to the inventory.
   (Done: they are `BZ_2_1_SCHOOL_TO_KARATSUBA` and `BZ_2_1_KARATSUBA_TO_FFT`; no others exist.
   `itertools`, which nothing used, was also removed from `[dependencies]`.)
4. Until each section migrates, any constant added to or removed from `mod.rs` is added to or
   removed from the inventory above, in the same change.

Done when: `cargo build`, `cargo test` and `cargo bench --features _bench_internals --no-run`
pass, and outputs and timings are unchanged.

### Phase 1: the seam (landed 2026-11-09)

1. Add `src/tuning/` (types, schema, `mod.rs`, `hash.rs`, `file.rs`) and a `build.rs` that writes
   defaults only. Add `tune = []` to `[features]`, off by default.
2. Migrate sqrt first, following the [checklist](#per-section-migration-checklist). It is small (9
   knobs), stable, and its old harness gives Phase 2 something to compare against. The migration
   breaks that harness (`scripts/python/tune_sqrt_cutoffs.py` matches the exact text of the
   cutoff checks in `sqrt.rs`), so it is deleted; Phase 2 runs it in the `faa9971` worktree.
3. Add `src/tests/test_tuning.rs`: schema checks in every build; the random-knob test with
   `tune`. Run the bias check for sqrt.
4. Migrate the other sections one per change: series and bin_split, then mul, div, consts, and
   theta, agm, exp and ln. Phase 1 moves values without changing them, so migrating code that
   still changes costs nothing (version bumps handle churn). Only those sections' Phase 2 waits
   until their algorithms settle, because a tuner for code that changes weekly is wasted work.
   (Decided 2026-11-08; until then the trans sections stayed in `mod.rs`.)

Phase 1 migrates today's constants as they are: `Interim` knobs at today's values, read by
today's dispatch forms. Phase 2 replaces each section's interim knobs with the knobs the
[inventory](#knob-inventory) lists (model knobs, sweeps and derived values), and rewrites its
dispatchers to read a table below the extent and compare predictions above it. The infrastructure
Phase 1 builds (schema, accessors, `build.rs`, the `tune` feature, the random-knob test, the bias
check) carries over unchanged.

Done when (per section): a normal build gives bit-identical outputs and timings within noise
against the commit before; `cargo test` and `cargo test --features tune` pass; and the section's
constants are gone from `mod.rs`, with every reference updated.

### Phase 2: the tuner and the presets (done on the M1, 2026-11-10; the x86 preset is still to be run)

The runtime prediction layer and the tuner's skeleton have landed, and the pilot measures `base` and
mul's schoolbook models; see [`tuning_implementation.md`](tuning_implementation.md) for status and
the pilot's findings (among them `base.div_setup`, a work-function term the first fit showed was
missing, and the tuner running on performance cores).


1. **The tuner.** Add `cli`, `measure`, `fit`, `sweep`, `derive` and `sections/*`. Add an example
   `examples/tune.rs` declared with `[[example]] name = "tune"` and `required-features = ["tune"]`,
   and an alias in `.cargo/config.toml` next to the existing ones:
   `tune = "run --release --features tune --example tune --"`. The release profile (fat LTO,
   `codegen-units = 1`) is what ships, so the tuner measures under it, even though a rebuild after
   a code change is slower.
2. **`build.rs`.** Add the presets layer, validation, derive versions, `pinned`, and the
   upstream-hash warning.
3. **`base` and a pilot.** Measure `base`. Then fit mul's `school` and `school_sqr` models, write
   the work function of one Karatsuba level, and derive the tables of `mul.kara` and
   `mul.kara_sqr` (today's schoolbook and Karatsuba cutoffs) with their extents. Check them against
   `benches/mul_cutoffs.py`'s results on both preset machines. This is the smallest closed loop
   through the whole method: measure, fit, compose, derive, dispatch, check. Settle the levels'
   starting numbers here.
4. **The rest of mul.**
   - The transforms: FFT with `FFT_ODD_COST`, and the NTT with its parallel predictions.
   - The high and middle products.
   - `mul_ns`, replacing `mul_work`.
   - The product, square, high-product, middle-product and cyclic dispatchers rewritten as table
     plus comparison. Today's boundary forms and their interim knobs retire with them.
5. **div, then sqrt.** Each has one leaf model, and every dispatcher's table and extent is derived.
   Compare with `cutoffs_bench` and the sqrt probe, then retire the probe and its scripts (see
   [Retiring](#retiring-the-current-harnesses)).
6. **series and bin_split.**
   - Base units replace five of the series coefficients, and `mul_ns` replaces a sixth.
   - Add `series.term` and `series.eval`.
   - Compare with the Python fit and `cargo bench-bin-split`.
7. **consts.**
8. **Presets.** For each preset machine run `cargo tune --level thorough --write-preset`, writing
   `presets/x86_64.toml` (i7-8750H) and `presets/aarch64-macos.toml` (M1).

Done when (per section):
- every derived knob passes its check at `normal`, or is pinned with a note saying why;
- the comparison with the old harness is explained;
- presets exist for both machines.

### Phase 3: the user-facing system

This waits until the `bit_nums` API is stable enough to publish a 0.x. It is purely additive: it
changes where `RESOLVED` gets its values, adds a driver binary, and adds documentation. Nothing in
`src/utils/` changes. See [Phase 3 in detail](#phase-3-in-detail-the-user-facing-system).

### Phase 4: optional extensions

- **`runtime-tuning`.** For applications that ship one binary to many machines. With this feature,
  accessors read from a `OnceLock<Params>`, which `big_bits::tuning::load(path)` or
  `install(params)` fills at startup, and which falls back to `RESOLVED` when unset. It costs one
  load per accessor read, the same as a tune build, and the values can't change after
  initialization. This follows FFTW's wisdom.
- **Sections for the number types.** Once the API exists: radix conversion (divide-and-conquer
  `to_string`/`from_str` cutoffs) and `BitFloat`'s precision and guard policy. The 19-digit
  decimal chunks in `ubitint_static.rs` are structural, not knobs.

## Per-section migration checklist

When you move a section from `mod.rs` constants to the schema, do all of these in one change:

1. **Declare.** Add the section to `schema.rs` with its `version` and `derive` (each starting at
   1) and its `after` list. Give every knob `old`, a default equal to today's value, its valid and
   search ranges, threads, arch and a `doc` that describes the decision (invariant 9). In Phase 1
   every knob is `Interim`. In Phase 2, declare the knobs the [inventory](#knob-inventory) lists,
   with their kinds, and bump the version. Work out the valid range from the code. Where you are
   unsure, keep it wide and let the random-knob test narrow it.
2. **Switch call sites.** Replace every use of the old constants in `src/` with accessors.
   Convert any const-context consumers (see the table under
   [Accessors](#accessors-and-their-two-backings)). Make sure each choice knob has a pure predicate
   (invariant 6), and read knobs once per call where it is hot (invariant 7).
3. **Switch tests and benches.** Tests, benches, probes and scripts that named the constants now
   read accessors, or use `tuning::get` in tune builds (invariant 8).
4. **Remove the constants** from `mod.rs`. Move their measurement prose into a dated
   "values before the tuner" section of the area's tuning doc.
5. **Update references** in CLAUDE.md, docs, scripts, probes and comments (the update-all-refs
   rule). Search the whole repository for every old constant name. Check that the search can find
   a known name before trusting an empty result.
6. **Check.** `cargo test`, `cargo test --features tune` (the random-knob test), and the bias check
   against the commit before.
7. **Update this doc.** Mark the section migrated in the inventory and record the bias-check
   result.

Phase 2 adds these steps for each section:
- write the work functions and runtime predictors next to their algorithms (leaf kernels' W,
  composites' parts, closed forms for recursions);
- rewrite each dispatcher as a table below its extent and a comparison of predictions above it,
  with eligibility declared in the schema;
- write `sections/<name>.rs` (each kernel's forced entry and shapes, and each derived knob's
  derivation);
- check the derived knobs, then pin or fix any that are flagged;
- compare with the old harness;
- write the presets;
- retire the old harness.

## Worked example: adding a knob

Say a new algorithm X joins `sqrt` for roots above some size, after Phase 2:

1. **Composite or leaf?**
   - **Composite.** If X is built from divisions, products and passes, as most new algorithms
     are, it is a composite. Write its work function next to it in `sqrt.rs`: the operations it
     calls at which sizes, plus its own passes, calls, scratch acquisitions and fills.
   - **Leaf kernel.** If X has its own inner loop, it is a leaf kernel. Write its W, add two model
     knobs (`sqrt.x_scale`, `sqrt.x_shift`, `kind: Kind::Model { work: … }`), and give the tuner a
     `pub(crate)` forced entry for X.
2. **Add X to its dispatcher.** In the schema, append X to the `algs` of the sqrt dispatchers it
   competes in, with the smallest size at which it is eligible. In `sqrt.rs`, give X's runtime
   predictor to the dispatchers' comparison, infinite where X can't run. No new knob is needed:
   derivation rebuilds those dispatchers' tables and extents with X among the choices.
3. **Keep the predicate pure.** The dispatchers' predicates already take their tables, extents and
   thread count explicitly (invariant 6). X only adds an arm.
4. **Bump the versions.** Bump sqrt's `derive`, because the dispatchers' algorithm sets changed
   and their tables may move. Bump `version` too only if you added model knobs.
5. **Test and retune.** Run `cargo test --features tune`. Then:
   - for a composite, `cargo tune derive --write-preset`, which times nothing and can run on any
     machine (follow it with a check run on each preset machine);
   - for a leaf, `cargo tune sqrt --write-preset` on each preset machine.

There is no new bench, no new script, no `mod.rs` edit and no hand-copied number, and a composite
needs no measurement at all. Before Phase 2, there is no dispatcher to join. Measure X with
whatever harness exists, add an `Interim` cutoff whose default is your estimate, and mark it
`// GUESS` until Phase 2 replaces it.

## Phase 3 in detail: the user-facing system

### What a user does

```
cargo add big_bits                     # develop: the target's preset (or the defaults) apply
...
cargo install big_bits                 # once per machine; installs the `cargo-bigbits` binary
cargo bigbits tune --budget 45m        # or: tune mul div / tune --level thorough
cargo bigbits init                     # adds BIGBITS_TUNING to .cargo/config.toml
cargo build --release                  # big_bits rebuilds with the tuned values
cargo bigbits status                   # later: which sections are tuned, stale or defaulted
```

The tune runs on the machine that will do the real run. The user documentation says so plainly,
and the driver refuses to tune for another target (below).

### Packaging: one package, two entry points

`cargo install` installs only binaries. A library is never installed: each project fetches and
compiles it. So `big_bits` stays one package with:

- the library, as now;
- `[[bin]] name = "cargo-bigbits"`, path `src/bin/cargo-bigbits.rs`. Cargo runs any `cargo-<name>`
  on `PATH` as `cargo <name>`.

Cargo never builds a dependency's binaries, so library users never compile the driver. The
binary has three constraints:

- **Only `std`.** A package's dependencies are shared by its library and its binaries, so anything
  the driver imports would be compiled by every library user. bindgen moved its command line into
  a separate `bindgen-cli` crate partly for this reason.
- **No `required-features`.** That would make `cargo install big_bits` fail without
  `--features tune`.
- **It measures nothing.** The real tuner is the library's own `tuning::cli()`, built with `tune`
  against the user's project (next section). `cargo install` still compiles the library once,
  because a package's binaries always link its library. That is a one-time cost.

### The driver: build a harness against the project's own big_bits

Run inside a project, `cargo bigbits <args>` does the following:

1. **Find the project's `big_bits`.** `cargo pkgid big_bits` prints its source and version, for
   example `registry+https://github.com/rust-lang/crates.io-index#big_bits@0.2.0`, or in this
   repository `path+file:///Users/aidansgarlato/Programing/Rust/BigBits#big_bits@0.1.0`, or a git
   URL and commit. If the graph has more than one version, `pkgid` reports the ambiguity; the
   driver lists the versions and takes `--package <spec>`. If there is none, it says to
   `cargo add big_bits`.
2. **Write the harness** to `<target dir>/bigbits-tune/`:
   - `Cargo.toml` depends on exactly that source (`version = "=0.2.0"`, `path = …` or
     `git = …, rev = …`) with `features = ["tune"]`;
   - it declares an empty `[workspace]`, so cargo doesn't treat it as part of the user's
     workspace;
   - it copies the project's `[profile.release]` and `Cargo.lock`, so the measured code is built
     the way the project builds it and with the same versions of other crates;
   - `src/main.rs` is `fn main() { big_bits::tuning::cli() }`.
3. **Build and run it** with `cargo run --release --manifest-path <harness>/Cargo.toml -- <args>
   --out <project>/bigbits-tuning.toml`, from the project's directory. Cargo reads
   `.cargo/config.toml` from the working directory upwards, so the project's `rustflags` (such as
   `-C target-cpu=native`) apply to the tuning build too. `--offline` and `--locked` pass through.
4. **Report.** The harness prints its progress and writes the file. The driver adds only the
   hint to run `cargo bigbits init` if `BIGBITS_TUNING` isn't set.

So the driver needs almost no knowledge of the library: it passes the arguments through and
`tuning::cli()` does the rest. Any installed driver works with any library version that has
`tuning::cli()`. It checks for that version and says clearly when the project's version is too old.

Other cases:

- **Cross-compiling.** If the project builds for another target (`--target`, or `build.target`
  in config), the driver refuses unless `--allow-cross` is given, because tuning must run on the
  machine that runs the code. The usual answer is to install and run the driver on that machine.
- **Workspaces.** In a workspace, `pkgid` resolves through the workspace's lock file, and the
  tuning file and `init` go at the workspace root.
- **Accidentally enabling `tune`.** Features merge across the dependency graph, so a crate in a
  user's tree that turns on `tune` would make their whole build read live values. That is slower,
  but still correct. The driver's harness sets `BIGBITS_TUNE_HARNESS=1`, and `build.rs` warns when
  `tune` is enabled without it.

### `init`, the environment variable, and the layers

`init` writes this to the project's `.cargo/config.toml`, creating the file if needed:

```toml
[env]
BIGBITS_TUNING = { value = "bigbits-tuning.toml", relative = true }
```

It refuses to overwrite a different existing value. `[env]` sets variables for build scripts as
well as for `rustc`. `relative = true` resolves the path against the directory that contains
`.cargo/`, so the setting can be checked in and works from any subdirectory. An environment
variable is the only way a project can pass a value to a dependency's build script: features carry
no values, and they merge across the graph. From here on, `build.rs` applies the third layer as
described [above](#buildrs). Builds without the variable (docs.rs, CI, other machines) get the
preset or the defaults, with no warning.

### Compatibility across versions

- **Upgrading `big_bits`.** Sections whose version is unchanged keep their measured values.
  Sections whose version was bumped fall through to the new preset, with a warning naming them,
  and `cargo bigbits tune --resume` remeasures only those. A section whose derive version was
  bumped keeps its measured values and loses only its derived ones; `cargo bigbits derive`
  restores them without timing anything. Most upgrades that change an algorithm need only that.
- **The file's `format`.** It changes only if the format itself does. `build.rs` reads every
  format it has ever written.
- **The driver and the library.** The only contract is "`tuning::cli()` exists and takes argv".
  Command-line changes live in the library and follow its version.

### Publishing checklist

- Phase 0 is done (no criterion or rand in `[dependencies]`; no empty `main`).
- `tune` is not a default feature, and its documentation says it is for the tuner only.
- `build.rs` has no `[build-dependencies]`, and `presets/` and `src/tuning/` are in the package
  (check with `cargo package --list`).
- `cargo-bigbits` builds with only `std`.
- The README has a "Tuning" section with the five-command workflow above.
- docs.rs builds with no file (presets only), and its documentation shows normal-build accessors.
- `_bench_internals` and the doc-hidden `_at` entries are still documented as unstable.

## Retiring the current harnesses

Phase 1 deleted every harness below when it migrated their constants (code outside `src/utils` was
allowed to break, decided 2026-11-08). Phase 2's comparisons run them in a worktree at commit
`faa9971`, to which Phase 1's outputs are bit-identical, rather than in this tree.

A harness is retired once its section's comparison with it is done (Phase 2). Retiring means deleting it,
or keeping it as a developer tool with a note in its header, and updating every reference.

| Today | Replaced by | Notes |
|---|---|---|
| `benches/probes/sqrt_cutoffs.rs`, `scripts/python/tune_sqrt_cutoffs.py`, `scripts/python/analyze_sqrt_cutoffs.py` | `cargo tune sqrt` | Also remove CLAUDE.md's list of `sqrt.rs` text that must not change (the leaf check, the marker comment, `dyn_div`, `static_div`, …), unless `benches/probes/zimmermann_stages.rs` still pastes it; that probe is a profile, not a tuner, so decide it separately |
| `benches/series_cutoffs.rs`, `scripts/python/tune_series_cutoffs.py` | `cargo tune series` | Base units and `series.term` replace the seven coefficients; the relative-error NNLS fit lives on in `fit.rs` |
| `benches/mul_cutoffs.rs`, `benches/mul_cutoffs.py` | `cargo tune mul` | Live values replace `--set`, and leaf models replace forcing Karatsuba's leaves. The x86 code-alignment builds have no successor: layout moves only crossovers that are tied within ε, and sweeps report those as tie bands |
| `benches/cutoffs_bench.rs` (division regimes, `DIV_TUNE_*`), `scripts/python/analyze_div_cutoffs.py` | `cargo tune div` | `cutoffs_bench.rs` also has shared helpers that other benches may use; split them out before deleting |
| `benches/bin_split_cutoffs.rs`, families `bbp engel hyper sbbp sengel shyper par e2e` | `cargo tune bin_split` | Their scans become derivations and the tuner's check. Families `ln exp agm lnagm` stay until the trans sections migrate |
| The doc-hidden `_at` entries | kept | Tests use them in normal builds to exercise particular values (for example `test_exp_dyn_sqr_counts` runs square counts 0–130); the tuner uses accessors instead |

The per-area docs (`mul_cutoff_tuning.md`, `sqrt_cutoff_tuning.md`, `series_cutoff_tuning.md`,
`bin_split_cutoff_tuning.md`, the division docs) stay as the history of their decisions. New
results go in the tuning files' evidence comments, and only notable findings get a new dated
section in the doc.

## Open questions

Decide each of these where it first matters, and record the answer here.

1. **Knob names.** The inventory's names are a proposal. Decide them when Phase 1 migrates each
   section; `old` keeps every old constant name searchable.
2. **How finely presets are keyed.** `build.rs` sees the target's architecture, OS and
   `target_feature`s, not the CPU model, so a preset can't tell a Zen 4 from an i7-8750H. Is
   per-architecture good enough, given that users can tune themselves? Probably, but decide this
   once a third machine has measured.
3. **`FFT_RESIDUE_CHECK` as a feature.** Is a user-chosen `fft-residue-check` cargo feature
   worth offering? The project leans toward accuracy over speed (`docs/fft_precision.md`).
4. **Bias that can't be hoisted.** If some section's bias check fails and no read can be hoisted,
   the fallback is a tune build that generates const-generic copies of only that section's hot
   code (what the sqrt probe does now), behind the same accessors.
### Decided

0. **What Phase 1's implementation added (2026-11-09).** These are part of the design now:
   - A fifth shared file, `src/tuning/resolve.rs`: schema checks, layering with every fall-through
     rule, `Follows` and the upstream hash, compiled by `build.rs` and the crate alike.
   - `tuning::resolved::<section>::<KNOB>` constants beside the accessors, for const contexts in
     tests; `src/utils` never names them (a test checks).
   - `tuning::apply("path=value;…")` and `tuning::spec()`, used to replay random-knob failures.
   - The random-knob test runs in its own process (`random_knobs` re-runs the test binary on
     `random_knobs_child`, on a 1 GiB-stack thread), so no other test runs while knobs change.
   - `benches/tuning_bias.rs`: the bias check (normal against tune build) and the identity check
     (against `faa9971`), from public entries only.
   - `knob_generic!` in `trans.rs` for knobs that are const generics.
   - Knob bundles are introduced only where a predicate needed explicit values (sqrt's
     `SqrtKnobs`, `MulKnobs`, `DivCutoffs`, `SeriesModel`); elsewhere accessors are read where they
     are used, once per call or recursion level (invariant 7), and invariant 5 keeps two reads in
     one call equal. Passing gates down with the call's thread count waits for Phase 2.
   - The random-knob test found and fixed: `rect_best` returning block length 0 on infinite
     predictions; `is_school_at` cancelling catastrophically for large c; `cache_par` overflowing;
     debug builds' static split stacks overflowing on deep trees. It narrowed: every sqrt cutoff
     ≥ 3, `mul.karatsuba` ≥ 3, `mul.karatsuba_sqr` ≥ 5, `div.bz` ≥ 3.

1. **FFT odd-part factors (2026-11-08).** Hard-coded per architecture the library has assembly
   for, AArch64 and x86-64, as the fixed constant `FFT_ODD_COST`; other targets use 1.0. They are
   not measured knobs (see [Fixed constants](#fixed-constants-never-knobs)).
2. **Thread counts (2026-11-08).** No crossovers are stored per thread count. Every decision where
   a parallel algorithm competes is made by comparing runtime predictions at the call's own thread
   count, so it is exact for any count from 1 to the machine's, at no storage cost.
   - Gates use a closed form in that count.
   - Tables never depend on it, because they cover only inputs too small for parallelism to pay.
   - Only a balance point whose optimum moves with the thread count is stored per count, as a
     short step function, derived (not measured) for every count up to the machine's.
   - The check times parallel decisions at one thread and at all of them.

   See [Dispatch](#dispatch).
3. **Unequal cores (2026-11-08).** Not modeled. Predictions treat every thread as equal, and the
   check at the full thread count is where a large effect would show.
4. **Cache levels (2026-11-08).** Not modeled. One cost model per kernel, fitted only where the
   kernel competes.
5. **Two-dimensional boundaries (2026-11-08).** Large inputs compare cost models at run time.
   Lookup tables are kept to small inputs, where comparing models wouldn't amortize, and an extent
   per dispatcher bounds them. The same rule applies to one-input decisions, whose tables are a
   few change points each. Today's boundary forms (`is_school_at`, `is_karatsuba`, `DivCutoffs`)
   retire in Phase 2. See [Dispatch](#dispatch).
8. **Sweeps for one input, fitted equations for several (2026-11-10).** See the note at the top
   of [How knobs are tuned](#how-knobs-are-tuned): one-input decisions are sweeps; multi-input
   dispatchers compare runtime equations whose constants are fitted to each algorithm's own
   measurements; nothing is derived purely from other algorithms' models; and the fits use a
   library (`nalgebra`; `levenberg-marquardt` too until 2026-11-11) under the `tune` feature. This widens decision 7
   from "where derivation misses" to every multi-input decision, and retires the pure derivations
   the method section describes.
7. **Fitted forms for mid-range composites (2026-11-09).** The method's rule was that a composite
   has no measured values. Where assembling one from its parts misses the decision it drives, it
   keeps its theoretical form but takes constants fitted to its own measured times. This is meant
   for algorithms that are neither large-scale (where predictions from parts hold) nor tiny
   (where a table is read): Karatsuba's level first (`mul.kara_scale`, `kara_pass`, `kara_shift`:
   scale · its sub-products' predictions + pass · (l + s) + shift), and BZ, NR's chain and a
   Zimmermann stage if their derivations miss too. Such knobs are `Model` knobs of the composite,
   fitted only where it competes, and add to the measured count.
6. **Phase timing (2026-11-08).** Phases 0, 1 and 2 are built in one pass, for continued
   development; Phases 3 and 4 wait until the crate ships. Phase 1 migrates every section,
   theta, agm, exp and ln included. Code outside `src/utils` may break: an old harness that a
   migration breaks is deleted, and Phase 2's comparisons with it run in a worktree at `faa9971`,
   to which Phase 1 is bit-identical. See [`tuning_implementation.md`](tuning_implementation.md).

## Precedents

- **GMP.** `tune/tuneup` measures crossovers and prints parameters for `gmp-mparam.h`, which is
  compiled in. GMP ships an `mparam` file per CPU family and chooses one at configure time.
  BigBits' presets and build-time resolution follow the same model. tuneup measures every
  threshold directly; here only sweeps do, and everything else is derived.
- **FFTW.** Its "wisdom" is planning results saved to a file (`fftw_export_wisdom_to_filename`),
  loaded by the application at runtime, with a system-wide file as a fallback. Phase 4's
  `runtime-tuning` follows this model. FFTW plans either from a heuristic cost model
  (`FFTW_ESTIMATE`) or by timing candidate plans (`FFTW_MEASURE`). This plan's derivations are an
  estimate mode whose model is measured once per machine.
- **ATLAS.** It tuned empirically during installation, and builds took a very long time. This is
  the reason tuning here is an explicit, modular step with budgets, never part of the build.

> **2026-11-11:** `ln_series_speedup` and the `ln.*_series_par_fraction` knobs named here were replaced by a serial/parallel pair of measured bounds (`ln.<dyn|static>_agm_limbs` on one thread, `ln.<dyn|static>_agm_par_until` on a pool); see `docs/tuning_implementation.md`.
