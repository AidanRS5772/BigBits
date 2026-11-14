# Tuning system: record and status

How `tuning_system.md`'s design was built into the crate (started 2026-11-08), what is left, and
the record of what was tried, so it isn't tried again. The design and its rules are in
`tuning_system.md`. This file was a running log of about 3700 lines until 2026-11-13; that log is in
git history at `fee4f90`.

## Status

| Stage | Contents | Status |
|---|---|---|
| 0 | Phase 0: dependencies, the empty `main`, `bz_2_1_cost`'s hidden constants | landed 2026-11-08 |
| 1 | Phase 1 infrastructure: schema types, the shared resolver, `build.rs` codegen, the `tune` feature, tests, the bias harness | landed 2026-11-08 |
| 2–7 | Phase 1 migrations: sqrt; series and bin_split; base and mul; div; consts; theta, agm, exp, ln | landed 2026-11-08/09, outputs identical to `faa9971`, bias within noise |
| P2 | Phase 2: the tuner, fitted-equation dispatch, measured tables, the M1 preset | done on the M1; full thorough retune 2026-11-13 |

The preset, `presets/aarch64.toml` (named `aarch64-macos.toml` until 2026-11-13), is a full
thorough tune of 2026-11-13: 36 minutes on mains power, all nine sections kept by the gate, 0 fits
rejected, 0 noisy. Held out: schoolbook 3.9%, `hi_mul_buf` 2.7% (rows) and 3.8% (columns),
recursive Karatsuba 2.7% and chunking 1.6%, FFT 3.5%, Knuth 2.2%, NR 3.0% and 3.8% with remainder,
BZ 2.6%, the series' Horner 3.4%, rect 2.7%, binary splitting 3.6%. Tables: `mul.kara` 94 change
points, `div.dyn_small` 34, `div.dyn_rem_small` 48, `series.small` 232. The sweeps' "samples" line
read 5.9%, past its 5% warning, as every full thorough tune of this machine has (5.9–6.8%). **Don't
retune the M1 unprompted.**

## What is left

None of it blocking.

1. **x86**, the user's, on the i7: [The x86 handoff](#the-x86-handoff).
2. **A lower bound from binary splitting's own equation** (its finalize and its root's products)
   would spare the decision the tree's walk where it can't win: 0.3% of the time on unseen series
   past 100 µs. A shortcut computed from the equations, so proposed and not built.
3. **Rect's block length** is the best of three candidates; the best of the lengths timed is 8%
   faster on average.
4. **Binary splitting's `div1` and `div2` prices** rest on a few series a half at the quick and
   normal levels (it passes as judged at every level; with the halves swapped it reads about 8%).
5. **exp's chunk width** wants a serial and a parallel value (`trans.md`). Proposed; the user put
   the transcendental functions after the series code.
6. **A chunking FFT for mul** (`mul.md`): the user's call.
7. **mul's fits on unbalanced products.** Schoolbook's equation reads 20% high at a short side of
   3–9 limbs under a long side of 25–99, and 60% under 25. Division's NR could read `HiEq` for its
   schoolbook high products in place of its own multiple; that is a retune of div and what is built
   on it.
8. **Near ties that flip between retunes:** `div_rem_dyn` 1030/1000 (6% apart on the clock).
9. **Static entries** share `div.bz` and `mul.kara` with the dynamic ones and lose where they cross
   later (`div_rem_static<2048>` 200/100: 10%). Approximate by decision, until their memory handling
   changes.
10. **Knuth's inner loop and buffer placement** (`div.md`): up to 2.45× slower at an unlucky
    alignment. A property of the library, not looked into.
11. **The "samples" line at the thorough level** sits on its own 5% warning. A threshold for the
    user to move, if it matters.
12. **Linux core pinning** is compile-checked only.
13. **A warm-up before the tuner's first measurement.** The first launch of a freshly copied tuner
    binary reads its first fit 25–39% high on this Mac. It would change what a tune does: proposed.
14. **Not covered by the series' tune:** coefficients past 64 bits, frames past 1024 limbs, trees
    that split across the pool (one fitted gain a thread).
15. **Deferred by the user:** the `cargo-bigbits` driver (Phase 3), `runtime-tuning` (Phase 4).
    Cosmetic and left: clippy's style notes in `src/tuning`, the preset's repeated evidence comments.

## The x86 handoff

Checked on this machine under Rosetta (`x86_64-apple-darwin`, which resolves the schema's defaults
as an x86 machine with no preset does): the tune build compiles with no warning; `cargo tune
status`, `list`, `tune base` (no ADX under Rosetta: it tunes nothing) and `tune sqrt --level quick`
run; 596 tests pass, 610 with `tune`. Nothing was timed there. `tuner/sections/base.rs` (x86-64
only, first in `sections::sections()`) has never run for real: two cutoff sweeps,
`base.prim_adx` through `mul_add_prim` and `base.limb_adx` through `add_mul_limb`, over 1–96 limbs.

Before it is tuned, an x86 machine prints two expected things: a tune of some sections alone ends
with "dispatch: series_eval_dyn ... PAST the rule's 5%", and `cargo tune dispatch` exits non-zero,
because the default `series.small` names nothing until the series section is tuned.

On the i7-8750H, on mains power at full charge with nothing else running:

```bash
cargo test --lib && cargo test --lib --features tune        # debug; both must pass first
# 1. The FFT's odd-part factors (hard-coded per architecture). Build line in the header of
#    benches/probes/fft_sizes.rs; its `sizes` mode prints ns per point by size class. The medians
#    relative to 2^k for odd parts 3, 5, 7, 9 become x86-64's FFT_ODD_COST.
# 2. A first look, into this machine's own file (presets/local.toml):
cargo tune tune --level normal
cargo tune status                                           # which layer supplies each section
# 3. The preset:
cargo tune tune --level thorough --write-preset             # writes presets/x86_64-<os>.toml
rm presets/local.toml                                       # so builds use the preset
cargo tune check --file presets/x86_64-<os>.toml            # the preset against the defaults
cargo tune equations --level normal                         # products: time lost by mul_dyn's picks
cargo tune division --level normal                          # the same for div_dyn, div_rem_dyn
cargo test --lib && cargo test --lib --features tune        # again, under the preset
```

Rename it `presets/x86_64.toml` to cover every x86-64 OS. `series.horner_pass2` needs nothing by
hand: the series tune measures it.

What to look at, in order:

1. **Each tune's summary lines.** "samples" (past 5%: the machine was busy, measure again); "gate"
   (sections dropped, with the reason); "fits" (a rejected one is written, marked, and means a form
   to revise: report it with its equation and rms); "edges" (a sweep that chose a value at an end of
   its grid: the crossover may lie outside it on this CPU).
2. The likeliest rejected fit is the FFT's until step 1's factors are in: `FFT_ODD_COST` is 1.0 for
   every odd part on x86. Whether `find_fft_size`'s set {1, 3, 5, 7, 9} is right there is the same
   measurement.
3. `base.prim_adx` and `base.limb_adx`: their evidence lines (defaults 20 and 10).
4. `cargo tune check`'s per-section ratios: each at or below 1, no item slower without a reason.

## History in brief

### Phase 0 and Phase 1 (2026-11-08/09)

Phase 0 moved `criterion` and `rand` to dev-dependencies, deleted an empty `src/main.rs`, and
surfaced `bz_2_1_cost`'s two fitted scales.

**Stage 1** built the seam: the shared files (`schema_types.rs`, `schema.rs`, `resolve.rs`,
`file.rs`, `hash.rs`) compiled by `build.rs` and the crate alike; the generated accessors with
their two backings; the random-knob test in its own process; and `benches/tuning_bias.rs`, which
times public entries only so it also builds against the `faa9971` baseline.

Stages 2–7 migrated every section with values unchanged: 99 knobs in 11 sections, `mod.rs` left
with only the fixed constants. Each stage was checked by output hashes against `faa9971`
(identical, 161 items) and by normal-against-tune bias (0.9986–1.0034×). Every old harness a
migration broke was deleted, not ported: they still run in a worktree at `faa9971`. The random-knob
test found four real bugs on the way (`tuning_system.md`, "Validation").

### The pilot, and the first method abandoned (2026-11-09/10)

The plan's first method was "measure only what can't be computed": base units and leaf kernels'
cost models measured, every composite's cost and every decision derived from them. The pilot built
that for schoolbook, one Karatsuba level and `mul.kara`. What it found:

- Residuals with structure mean a missing term, not too few samples (`div_prim`'s per-call
  reciprocal; a per-row term in schoolbook).
- The tuner must ask for performance cores: on efficiency cores fits read 2.5× slow.
- A table on a hot path needs a row index (`Table2`): a linear scan of 88 change points cost 3–21%.
- `find_karatsuba_scratch` assumes a monotone boundary, so `karatsuba_mul` falls back to schoolbook
  where scratch is short and any valid table is correct.
- A model must not decide differences below its own error: a fit-error margin fixed the leaf
  boundary, where the two sides tie over a wide range, and then made the steep Karatsuba-to-FFT
  crossing worse.
- Measure from the main thread on the global pool: on a private pool's worker the parallel NTT read
  9% faster and put a cutoff at 2000 for 2700.

Score when the user stopped it: of 25 decisions, 1 derived, 7 derivations failed their end-to-end
checks, 17 swept directly. **The method was replaced** (decision 8): sweeps for one input, fitted
runtime equations for several, nothing derived purely from other algorithms' models. `base`'s cost
units, `model.rs`, `dispatch.rs` (the old one), `threads.rs` and every fit no decision read were
purged.

### Completing Phase 2 (2026-11-10/11)

- **Products.** `mul_dyn` past its table compares recursive Karatsuba's and the FFT's fitted
  equations per call. Because the FFT's equation follows its size steps, it loses 0.0–0.2% of
  measured time against the old ratios' 0.2–0.7%, and 0–7% at worst against 8–19%. On the M1 the
  i7's ratios had sent mid-size products to the FFT too early (312 × 104: 0.69×).
- **Divisions.** `div_dyn`/`div_rem_dyn` compare Knuth's, NR's and BZ's equations. The hand-fitted
  cutoff rule had lost 0.8% and 3.4% of measured time (worst 37% and 59%); the equations 0.03% and
  0.15%. The 40-shape division grid is 4% faster, up to 0.65×, nothing slower. NR is priced product
  by product, tier by tier: one multiple of M(h, h) read 4.2–4.7%, since NR runs no Karatsuba.
- **Every one-input decision a sweep**, with a tie band, a per-probe budget, and stability in place
  of spread for whole-algorithm sweeps. ln's AGM was found ahead from 32 limbs on one thread (the
  hand value was 1024), which needed the serial/parallel pair to keep the series on a pool.
- **The power throttle.** On a 15 W supply at 3–7% battery the M1 throttled 1.5–1.9× under load
  and every division fit failed together, Knuth's unchanged one included. About an hour went to
  separating that from a new BZ form. It is why every timing command ends with a "samples" line.
- **Any checkout can tune itself**: `presets/local.toml` as a third layer; the end-to-end gate after
  every section; rejected fits written, marked. "Make this the final tune pass" (the user): Phase 2
  closed on the M1.

### The series' dispatch rebuilt (2026-11-10 to 11-13)

- **On fitted equations (11-10).** Seven prices shared by three algorithms over counted work lost
  3.8–4.3% of measured time (worst 65%). One equation an algorithm over its own parts, with products
  priced by mul's equations and finalize by division's: 0.13% and 0.35%. What the rebuild found was
  wrong counts, not wrong prices (a uniform tree depth; products with a one-limb side priced by
  schoolbook's equation; `div_prim2` priced as `div_prim`; parallel trees at a full speedup; forced
  timings carrying the entry's own work). The largest single cause was in the code, not the model:
  a tree of a few wide terms started off the pool woke the pool for nothing, hence
  `bin_split.dyn_par_work`.
- **A dispatcher's own cost is part of the result, and a tune build hides it** (every knob is an
  atomic there). First wired, the equations made small `exp_dyn` calls 15–23% slower; they had to
  stop predicting what can't change the pick. Compare a normal build against the tree before.
- **One way to fit (11-11):** scattered shapes, held-out halves, rounds, one SVD solve; a negative
  term dropped. The form analysis then found 15 constants no timing needed, and they were removed
  from equations, schema and tuner (schoolbook's term a row; Karatsuba's terms a limb and shifts;
  the FFT's term a point and shift; NR's term in d and shift; BZ's term in q; two of the series').
- **The exp workloads timed nothing (11-11):** `trans.md`.
- **General evaluators (11-11).** The user: the series evaluators are for general series, not
  exp's and ln's helpers, and "the series fitting should stand on its own with no knowladge of what
  uses it". Hence the frame of rho and beta (`series.md`) and the synthetic family. Measured on 244
  synthetic shapes: rect fastest in 118, Horner in 74, binary splitting in 52. On exp's and ln's
  chunks rect had looked marginal (fastest in 9 of 192): they sit at the map's triple point.
- **A table for short series (11-11):** reading the equations was 22% of a series under 1 µs. The
  entry's mean loss went from 9.4% to about 2%.
- **Rect's counts and `HiEq` (11-12):** the record below.
- **Tables measured, under one rule (11-12):** the next section.
- **Horner's and binary splitting's forms (11-13).** Horner's floor on the kernel's own counts was
  above the line, so its form was wrong: a price a kernel call, every call alike, became a price a
  group and the calls of a term's own step (5.8% → 3.3% held out on the kernel's counts), with two
  counts corrected. Binary splitting's counts were sound and its form lacked three prices (its
  one-limb passes: 7.5% → 5.1% alone; finalize by a two-limb Q; an x-leaf's term). First reported as
  passing on a thorough tune only; at the normal level Horner read 5.0–5.1%, because its two-limb
  price sat on six series in the fitted half. Tied to the one-limb price by the kernels' measured
  ratio: 3.5%, 4.0%, 3.5% at the three levels.
- **The beta guard removed (11-13).** It kept binary splitting from being predicted past 8 frames
  of tree, and once the equation fitted it was barring it where the equation was right (67% and 75%
  slow on two unseen series). Without it the worst of the 138 series it settled goes from 75% slow
  to 7%, and the decision walks the tree on them (47 ns → 200–570).
- **End to end**, each step against the tree before it, outputs identical: the rebuild 0.977
  (series) and 0.984 (exp/ln); rect's counts 0.988 and 0.998; the measured tables 1.005 and 1.004;
  the new forms 1.003 and 0.999. The later passes bought soundness (fits under the line, tables that
  are timings, one rule) and not speed: the picks were already within a tenth of a percent.

### Tables measured, under one rule (2026-11-12)

The user: "The table should be reworked and derived from actual measurments at each index size
instead of the table being derived from the fits. This should be the case across the board", and
"if dispatch is more then 5% of runtime at that size then use a table. This should be enforced
system wide." What it replaced: tables derived from fits (`tuner/derive.rs`, `cargo tune derive`),
the Karatsuba level's and chunking's fits, division's Knuth guards, the series' short-series prices
and the fit a regime, `Kind::Derived`, the `pinned` key, the ten `*_err` knobs.

- What prompted it: the derived series table put binary splitting where rect was 30× faster (two
  terms at a wide x), because two equations a factor of three off made the cell a near tie.
- `mul.kara` measured reads Karatsuba from 44 limbs for squares and 49 at l = 4s; derived it read
  46 and 66. Measured through `karatsuba_entry_dyn`, the indices being timed must be Karatsuba's in
  the live table, or both probes run schoolbook and tie ("Karatsuba from 211").
- **The series' table layout is the user's.** The first measured table had 72 classes (kind × q's
  bits × an r flag): 1873 change points. The user: "that complex of a table is causing an explosion
  in entries", then, of a layout of about 200: "That is roughly the size I want cahces to be at and
  the mean is still less then 2% and the weighted less then 3%." As built, 8 kinds: on 459 unseen
  short series its picks lose 1.5% on the mean and 2.3% of the time (the 72 classes: 0.8% and 1.7%).
- **My numbers for that choice were wrong** (1.9% and 2.7% quoted; 2.9% and 4.1% built, with a
  hundred longer series made worse, one 426% slow). The simulation scored only the series the table
  is for and fitted cells to data the builder doesn't pick from. Two changes brought it to what was
  asked: reading the equations became one of a cell's picks, and an x past 32 limbs a kind of its
  own.
- Sizes hardly follow the share (the 72 classes: 2636 change points at 1%, 1873 at 5%, 1310 at
  20%). What the share moves is which series the equations are read for.
- **What it costs:** division's hand-over (`div.md`), accepted: "just accept it then it is within
  the noise threshold roughly anyway. (For the sake of consistency)". End to end against the tree
  before: series 1.005, exp/ln 1.004, mul 1.001, div 1.002, the division grid 1.011.
- **A fit takes the shapes its equation is read for** ("it is just good practice to not fit over
  samples that the fits arnt used for"): the two-phase tune. It moved no fit by more than half a
  point.
- **A tuner bug the comparison found:** mul's second phase swept before its own refitted equations
  were live, so a tune starting from schema defaults kept `mul.dyn_partial_mul` at 312 where 242 is
  right (13% on a 300-limb high product). The gate couldn't see it; the A/B with the tree before
  did, item by item.

### The tuner's code (2026-11-13)

A simplification pass (`SectionTuner` registry, `fit::fit_named`; one behaviour change: the gate
skips a section whose values didn't change) and a clean-up pass (8752 → 7934 lines, comments 1506 →
955, results bit for bit the same). Both were checked by replay: the series report on kept timings
byte for byte the same, `list`/`status` identical, bench hashes, both suites. The preset then became
`presets/aarch64.toml` for every aarch64 target ("for now").

## Lessons

Each is from something that went wrong.

- **Suspect the machine before the forms** when several unrelated fits fail at once. Check
  `pmset -g batt`, `pmset -g adapter` and `uptime`, and time one fixed workload for 40 s, before a
  tune whose values are kept.
- **Launch a freshly copied binary once before timing with it.** Its first seconds are slowed: a
  quick tune's first fit read 25–39% high, REJECTED.
- **One bench item can move 10–15% from binary layout alone** (`mul_dyn` 100 × 100 after a schema
  change, mul's source untouched). Check the tune's in-process gate, build the new code with the old
  values, and compare each executable with itself, before blaming a tune.
- **Replay before quoting.** A simulated loss for a table layout or a dispatch rule must come from a
  replay of the real builder and lookup, checked against the live system's picks on every workload,
  and scored on every call the candidate touches.
- **Test an explanation against the fit's own residuals before presenting it as the cause.** "It is
  the products' price", from one night's probe, did not survive the next day's residuals.
- **Counts before prices.** A count off by a constant factor is absorbed by its price; one whose
  error varies with the shape is the misfit, and no change of prices or shapes finds it.
- **Tune a new form at all three levels** before calling it sound.
- **Take the entry's common work off forced timings** (the term count's bisection, the eligibility
  pass: 100–200 ns of a 350 ns call), as division's numerator copy is taken off.
- **Run the suite under the schema's defaults** (Rosetta) before a handoff. The M1 preset hid a test
  that depended on the parallel gates.
- **Exact-match edits for multi-file changes.** Scripted edits that sliced by markers clobbered code
  twice; an unquoted heredoc ate backticks once.

## Marked problems

The rule: a fit above the 5% line is reported with a proposed form and waits for approval, once. A
model already revised once that still misses is accepted noisy, marked NOISY, and recorded here.

**Nothing is marked.** The records below are kept so that what was tried isn't tried again, and
because the tuner still treats BZ's, NR's and the series' forms as revised: a future miss reads
NOISY and is written, not a stop (confirmed by the user, 2026-11-13).

### BZ's runtime equation

In use: `block`·(whole blocks' products and Knuth leaves) + `top`·(the top block: Knuth's equation,
or a padded block priced by its quotient's significant pieces).

| # | Form for a padded top block | Result |
|---|---|---|
| 1 | One more full block | 3.4% where BZ has a full-size block (q ≥ d); 13.4% once partly filled padded tops were fitted |
| 2 | Leaves in full, products times a fitted power (top/d)^g | 15.4%, g = 0.17: no one power exists (no discount under about 700 limbs, floors of 0.5–0.8 above) |
| 3 | Structural, for the code as it then was: only FFT-sized levels' products shrink | 1.7%. Never installed: the user had the cause removed instead |
| 4 | The code changed (`div_3_2` multiplies each quotient piece at its significant width), every level priced by significant pieces | 7.2–9.5% on a throttling machine; 2.2–2.4% on a steady one. In use |

The cause behind forms 2 and 3: only the FFT entry trimmed its operands' zeros, so only FFT-level
products shrank under a short quotient. The term in q fitted to zero and is out of the form. To read
it again: `BIGBITS_TUNE_VERBOSE=1 cargo tune division --level normal --refit`.

### The series' dispatch equations

| # | Model | Result |
|---|---|---|
| 1 | Seven shared prices over counted work, refitted to forced timings | 30.6% rms; picks no better. The counts are off, not the prices |
| 2 | Its products priced by mul's fitted equations | 31%; held-out picks' loss 8.4% → 7.4% |
| 3 | The cause removed in the code: an off-pool tree splits only with `dyn_par_work` of limbs × terms, and the model charges one wake | picks' loss 6.3% → 3.7% and 8.4% → 4.1%; refit 24.5% |
| 4 | An equation an algorithm over its own parts | 5.6%, 13.3%, 6.9%; picks' loss 0.13% and 0.35% |
| 5 | One set of prices shared by the three, on the new parts | 17%; loses 2–2.6%. Each algorithm's constants are fitted to its own timings |
| 6 | Fitted on the synthetic family, not exp's and ln's chunks | 8.2%, 8.3%, 9.7% held out |
| 7 | A fit a regime: long series for the equations, short ones for the derived table | long 5.8%, 8.2%, 3.9% |
| 8 | Rect's counts corrected, `hi_mul_buf` its own equation | long 4.7%, 2.8%, 3.6% |
| 9 | One fit on every series the 5% rule gives the equations (from a few µs) | 6.6%, 3.1%, 8.1%. At a 1% share the same timings read 3.8%, 3.4%, 5.7% |
| 10 | Leaving out the series the table decides | 6.2%, 3.1%, 7.5%: not why they were above the line |
| 11 | Horner's form and two counts; binary splitting's three missing prices | **3.3%, 2.8%, 3.6%**. In use |

Alternatives on exp's and ln's chunks (2026-11-11, held out; the form then in use read 6.3% Horner,
7.5% binary splitting, 13.4% rect). None applied; prices by family were withdrawn when the
evaluators became general.

| Alternative | Horner | Binary splitting | Rect |
|---|---|---|---|
| A fixed cost a call | 6.2% | 7.5% | (has one) |
| The per-term price by family | 5.2% | 7.6% | – |
| Term, call and allocation prices by family | 5.2% | 6.6% | 13.4% |
| Every price by family | 7.7% | 6.2% | 12.4% |
| Prices apart under and from 64 limbs | 6.0% | 7.8% | 12.9% |
| The linear passes' limbs × log2(frame) | 6.3% | 7.3% | 13.4% |
| Products priced apart under and from 32 limbs | 7.2% | 8.1% | 13.7% |
| Fitted only within 1.5× of the fastest | 9.1% | 6.9% | 12.8% |

Ruled out in the pass of 2026-11-13: a price of their own for the family's 16- to 32-bit
denominators (`div_prim`'s data dependence: 5.8% → 5.7%); a price for terms that land within a limb
of the last; a fixed cost for Horner, a price a product, a price a leaf or a merge for binary
splitting (each solved 0 or negative).

Tried for the table's index: the theory's two ratios and a size (rho, beta, and the frame, the terms
or their product) lost 10% at every grain from half an octave to eight. rho is 0 for the half of
short series with no x, and a short series follows its fixed costs, which follow the raw sizes.
In the derived table's layouts, both the 8-bit and 16-bit q buckets had to stay.

Tried for the guards: swept by mean relative loss over all sizes, a guard gets used to dodge the
dispatcher's cost on sub-microsecond calls and overrides correct equations on long ones (a 100 µs
series 35% slower). Their sweeps take series of 50 µs and more. A blunt "small series to Horner"
rule can't replace the table: binary splitting is fastest in 22 of 99 shapes under a microsecond.

#### Rect: everything tried to bring it under the line

Asked for by the user on 2026-11-12 ("record all the things already tried for bringing the
rectangular series inline"). The rms is held out unless it says otherwise. Add a row for anything
new tried.

| # | Tried | Result |
|---|---|---|
| 1 | The old model: seven shared prices over counted work | 59%. The counts were off |
| 2 | The rebuild: rect's own parts read off its kernel, every price its own | 6% fitted, 12% held out on 25 workloads: too few shapes for nine prices |
| 3 | Horner's prices for the kernels it shares, three constants of its own | 9.5% in the trial; 13.3% in the tune; 11.6% on scattered frames |
| 4 | Its known miscounts noted, not fixed (powers padded to the accumulator, block products clipped to the frame) | products 5–20% high at a wide x at 1024 limbs, 10–17% low at 16–64 |
| 5 | Other forms on the same timings: prices by family, by frame, by time; log2 terms; products priced apart; every price its own | 11.6–13.7% whichever way it is cut |
| 6 | Whether rect is worth predicting at all | On general series it is fastest in 118 of 244 shapes: 0.54 of Horner's time at rho 6–20, 0.25 past 100. It stays a full algorithm (the user) |
| 7 | Fitted on the synthetic family where it competes, 82 shapes for 22 | 8.3%. Its 11–13% was the bit-burst's chunks, and too few of them |
| 8 | A guard where rect is far ahead (rho ≥ 32, x ≥ 16 limbs) | Settles 34 of 237 long shapes without the equation; the fit is the same |
| 9 | Forms on the family: every price its own 10.9%; its own term price 8.4%; a multiple on its share in Horner's kernels 7.8% | Small either way; not applied |
| 10 | A fit by the series' length (what brought Horner to 4.6%) | Rect does not follow: 6.6% under 5 µs, 9.4–9.5% above |
| 11 | A fit a regime | long 8.2% over 123 shapes, short 6.2% over 41 |
| 12 | Horner's long prices for everything shared, one-limb divisions included | 25.5%: Horner's long series hardly make a one-limb division, so that price fits 0 |
| 13 | Rect's own price for one-limb divisions, `series.rect_ns_div1` (approved; in use) | back to 8.2% |
| 14 | Horner's groups counted whole for a q of one size | Horner's fix; rect stayed above the line |
| 15 | Is it the number of shapes? Cross-validated at equal counts | No: 8.7% on its 82 against 8.5% and 9.0% for the others' random 82 |
| 16 | Hypothesis: the high product's price (no fitted equation; a probe read measured over priced 1.17 with a 16% spread) | Proposed, then tested against rect's residuals, rows 17–21 |
| 17 | Does rect's error follow its products? | No: 9.5% in the third of shapes with the least products, 6.4% in the third with the most |
| 18 | Its own prices for the passes it borrows from Horner | No set reads under 6.8% even in sample: it is not a price |
| 19 | Where it is, by x's width | 10.7% at 2 limbs, 12.7% at 3–4, 9.8% at 5–8, 8.7% at 9–16, 5.0% at 17–32. Predicted too slow at a narrow x |
| 20 | A product multiple of its own by x's width | The narrow multiple fits 0.77–0.86; 6.7–7.1% from 7.4% |
| 21 | A price for the term adds at a narrow x | 0.30 ns a limb against 0.40–0.45; 6.8–6.9%. An x of 5–16 limbs stays at 7–9% |
| 22 | The user's hypothesis: mul lacks a chunking FFT | Not rect's cause (no tuned series makes such a product), but a real gain for mul |
| 23 | **Counters in the kernel**, the model's counts against the kernel's on 747 shapes | The counts are off by amounts that vary shape to shape: term adds 1.74× (spread 22%), divisions 1.15× (21%), products 1.06× (9%) |
| 24 | Prices fitted to the kernel's own counts: the floor of the form | 4.6% with five prices; 3.5–3.7% with `hi_mul_buf`'s products priced apart |
| 25 | A power's width from x's bits, ceil(j·bits/64), not j limbs of x | Row 19's "narrow x": add limbs 1.23× and 1.11× become 1.02× at every width |
| 26 | A term add's passes 1 + bits(weight)/64, not 2; over the lesser of power and frame term by term; a group divides at its own frame | adds 1.74 → 1.02; divisions 1.15 → 1.01. Row 21's 0.30 was a count, not a price |
| 27 | The guarded product's copy and the scan for zero limbs counted; the accumulator's length from where the bits land | block products' area 1.10 (15%) → 1.00 (0.3%) |
| 28 | Forms on the corrected counts: four prices; + a term-add price; + a multiple for `hi_mul_buf`'s products | 4.8–5.4%; 4.3–4.9%; 3.6–4.2% |
| 29 | The user: the tier's price into mul's section. `HiEq`, `hi_mul_buf`'s own equation | rows 2.7%, columns 3.8% held out, where schoolbook's equation on the area read 14.6% and 31% |
| 30 | As built: product, div1, add, term, fixed over the corrected counts | **2.8%** held out. Horner improved for free (5.8% → 4.7%) |

**How it ended.** Rows 1–22 moved prices and shapes; what was wrong was counts (23–27) and the
price of one kernel read off another's equation (29). Rows 19–21 had seen the symptom from outside
and could not tell a price from a count. The method to reach for first, with the step added on
2026-11-13: counters in the kernel in a scratch copy (wrappers of the same names over the
primitives the module imports, global atomics, composite calls priced at the dump by the model's own
environment), **then fit the times to the kernel's own counts under the form as it is.** That floor
says whether the counts are wrong or the form is. The user's rule for the price: a kernel's price
belongs to the kernel's own section ("Move the tier price in to the mul section").

## The user's decisions, in their words

- Method (2026-11-10): one-input decisions are sweeps, "that is the right way"; never derive a
  runtime purely from other algorithms; use a fitting library.
- Thresholds (2026-11-10/11): a fit at relative rms ≤ 5%, read on held-out shapes; the worst
  residual is not judged; spread bounds 6% / 4% / 2%; gate 3% / 2% / 1%.
- A revised model (2026-11-10): "if we have revised a fit model once already and a problem still
  remains then we should except the noisy fit and move on."
- Form analysis (2026-11-11): "This kind of analysis shouldn't be done for when someone uses the
  tuner, this is the kind of stuff we do to find the proper tuning approach for every algorithm."
- Static entries (2026-11-11): approximate values only; their memory handling is to change.
- Series (2026-11-11): general evaluators; "Definitely Keep 'rect'"; guards "as a tunable
  paramter"; no trimming of the accumulator ("the series code works at an exact level"); "for small
  series use a table as is done for other algorithms."
- The beta guard (2026-11-13): "if the beta gaurd is causing a slow down remove it and rely purely
  on the table and the fired equations."
- Failed fits of revised forms (2026-11-13): "leave failed fits as marked noisy."
- The tuner has no kernel thermometer and no steadiness preflight: built once, removed at the
  user's request (it made thorough tunes 45 minutes).
