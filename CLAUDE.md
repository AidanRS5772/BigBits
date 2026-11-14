# CLAUDE.md

BigBits is a big-integer library (floats and rationals planned) built on inline assembly (x86-64,
AArch64) and adaptive algorithm selection. This file is the map and the rules. The detail is in
`docs/`: read the doc for an area before changing it.

## Commands

```bash
cargo build
cargo test --lib                              # debug; never judge by --release (see Testing)
cargo test --lib --features tune              # adds the tuner's tests and the random-knob test
cargo test test_mul                           # one module; test_mul::<name> for one test
cargo test --target x86_64-apple-darwin --lib # under the schema's defaults (Rosetta)

cargo tune tune [SECTION...] [--level quick|normal|thorough] [--write-preset] [--tables] [--analyze]
cargo tune check | dispatch | equations | division | series | list | status
#   the tuner: release profile, `tune` feature. Full reference: docs/tuning_system.md

cargo bench-utils                             # Criterion (needs _bench_internals)
cargo bench --bench tuning_bias --no-run [--features tune]
#   <exe> --hash [--section S] | --compare A B [--procs 6]   end-to-end A/B and output hashes

cargo build --profile prof                    # symbols for samply (release and bench are stripped)
cargo run --release --example gen_ln2 [-- <limbs>]   # regenerate consts/ln2.bin; gen_pi likewise
scripts/coverage.sh                           # needs cargo-llvm-cov
```

## Layout

```
src/utils/       raw limb-buffer arithmetic: mul, div, sqrt, series, trans, consts, verify
src/tuning/      performance knobs: schema, resolver, file format; tuner/ (tune builds only)
build.rs         generates knob accessors from the schema, the target's preset and the machine's file
presets/         aarch64.toml (the M1's tune, for every aarch64 target); local.toml (gitignored)
src/bit_nums/    number types over utils
src/tests/       the suite (cfg(test))
benches/         utils_bench (Criterion), tuning_bias, profiles, probes/
docs/            the reference for each area
```

| Doc | Read it for |
|---|---|
| `docs/mul.md` | products, squares, high/middle products, Karatsuba, FFT, NTT, the residue verifier |
| `docs/fft_precision.md` | the 16-bit FFT's soundness: balanced digits, fold check, limits, how to remeasure |
| `docs/div.md` | division and reciprocal contracts, Knuth/BZ/NR, the reciprocal carry, dispatch |
| `docs/sqrt.md` | binomial and Zimmermann roots, the wrapped remainder |
| `docs/series.md` | series kinds, binary splitting, Horner, rect, the series dispatcher and its equations |
| `docs/trans.md` | exp, ln, theta, the AGM, the constant caches |
| `docs/ln_agm.md` | the theta–AGM logarithm's mathematics |
| `docs/tuning_system.md` | knobs, accessors, `build.rs`, the tuner, the method, invariants, validation |
| `docs/tuning_implementation.md` | status, what is left, the x86 handoff, everything already tried |
| `docs/tests.md` | what each test file covers and its oracle |
| `docs/fixed_bugs.md` | root causes of past wrong-result bugs and their regression tests |
| `docs/holonomic_series.md` | a design sketch for joint sin/cos (nothing built) |

Reports removed in the 2026-11-13 condensation are in git history at `fee4f90`; the pre-tuner
harnesses still run in a worktree at `faa9971`.

## Conventions that hold everywhere

- Buffers are little-endian `u64` limbs. Operands are trimmed; divisors nonzero.
- **Dynamic** entries (`*_dyn`) take pooled scratch from `ScratchGuard` (`utils/mod.rs`, a
  thread-local RAII pool). **Static** entries (`*_static<N>`) use `[u64; N]` and assume every input
  and output has at most N limbs: behaviour past that is out of contract, not a bug. A static entry
  whose N is too small returns `CapacityError { min_n }` before touching its output.
- A product or quotient into an exactly sized output returns its overflow limb; a wider output
  stores it and returns 0.
- Exact operations give the same bits under every tuning. Approximate ones (series, exp, ln, AGM,
  theta, reciprocals, high and middle products) guarantee an error bound only: compare them by
  value against an oracle.
- `exp`/`ln` take X = 2^(64m)·x with x's **top** limb the integer part: m = −1 is a full-precision X
  in [0, 1).
- Doc-hidden `*_at` and `*_raw` entries force an algorithm or a knob's value, for tests and the
  tuner.
- Avoid `u128` arithmetic in `const fn`: it lengthens compile times badly.
- **Exactness comes before speed.** A dispatcher that silently returns wrong limbs is worse than a
  slower tier.

## The utils layer

- **`mul.rs`** — schoolbook (asm rows) → Karatsuba → FFT (`rustfft`, 16-bit balanced digits,
  dynamic only) → NTT (three primes, CRT). Also squares, high products (`hi_*`, within an ulp or
  a few), middle products (`long[2n−1] × short[n] → out[n]`) and squares mod B^m − 1. FFT calls are
  checked (fold check, residues in debug) and redone on the NTT when a check fails. The four
  `FFT_*16BIT_CUTOFF` limits in `utils/mod.rs` are soundness limits under a compile-time assert,
  never knobs.
- **`verify.rs`** — linear-time product checks by residues mod the NTT primes.
- **`div.rs`** — quotient, quotient with remainder, high quotient and reciprocal over Knuth,
  Burnikel–Ziegler and Newton–Raphson. A call is preflight, dispatch, then a prepared driver; NR
  falls back to BZ and the reciprocal to Knuth for correctness. `RcpCarry` shares a reciprocal
  across divisions by growing prefixes of one divisor.
- **`sqrt.rs`** — binomial below each output mode's cutoff, Zimmermann's recursion above; exact
  root and remainder, exact root only, and approximate. Large dynamic stages form their remainder
  from one square mod B^m − 1 and carry a reciprocal up the recursion.
- **`series.rs`** — general evaluators for BBP, Engel and Hyper series: reverse Horner, rectangular
  splitting, binary splitting. Public entries `series_eval_dyn`/`series_eval_static` take a typed
  context (`Bbp`, `Engel`, `Hyper`); `bin_split*` builds exact (dynamic) or truncated (static) nodes.
- **`trans.rs`**, **`theta_plan.rs`** — `exp` (reduction by ln 2, bit-burst chunks, squares) and
  `ln` (square roots, atanh chunks; or theta and the AGM), `theta_*`, `agm_*`, compile-time plans
  for theta's powers.
- **`consts/`** — ln 2 and π: compile-time tables to 1024 limbs, a growable cache past them.
- **`utils.rs`** — `trim_lz`, `add_buf`, `sub_buf`, `cmp_buf`, shifts, residues mod B^m − 1.

`src/bit_nums/` compiles `UBitInt`, `BitInt`, `UBitIntStatic<N>`, `BitIntStatic<N>` and
`traits.rs` (`Sqr`, `DivRem`, `PowI`, `LogI`, `Abs`, the `impl_commutative*` macros).
`bitfloat.rs`, `bitfloat_static.rs` and `bitfrac.rs` are on disk but **not compiled** (absent from
`bit_nums/mod.rs`, broken since a 2024 refactor): reviving them is a rewrite against the current
utils API. Whether to delete them is the user's open decision.

## Dispatch

- **A decision over one number** (a cutoff, a gate, an algorithm parameter) is a swept knob.
- **A dynamic dispatcher over several inputs** (`mul_dyn`, `div_dyn`, `div_rem_dyn`, the series'
  entry) reads a small measured table at small sizes (`mul.kara`, `div.dyn_small`,
  `div.dyn_rem_small`, `series.small`) and past it compares fitted runtime equations per call
  (`MulEq`, `HiEq`, `DivEq`, `SeriesEq`), each algorithm's own equation next to its code.
- **Static dispatchers** keep cutoff rules and the old shared cost model, with approximate values.
- Every dispatcher has a pure predicate at explicit knob values (`*_at`), which tests and the tuner
  call. Knobs are read once per call and passed down.
- Tests must find a boundary with the dispatcher's predicate (`is_school`, `dyn_div_alg`), never
  from a default value: the M1 preset moves every boundary.

## Tuning rules (the user's decisions)

Full statement and reasons: `docs/tuning_system.md`. History and what was tried:
`docs/tuning_implementation.md`.

1. **One input: a sweep. Several inputs: fitted runtime equations**, each in its algorithm's
   theoretical form with constants fitted to that algorithm's own timings. Never derive a runtime
   purely from other algorithms' models.
2. **A dispatcher's table is measured at each index, never derived from fits**, and exists only
   where reading the equations costs more than 5% of the call (`tables::SHARE`, the user's number).
   Tables stay small (about 200 change points; mean loss under 2%, time-weighted under 3%). Don't
   add a guard, table or shortcut computed from the equations: propose it.
3. **A fit is accepted at a held-out relative rms ≤ 5%**, on the shapes its equation is read for.
   Effort goes into more shapes scattered over the area, not tighter samples. A term that solves
   negative is dropped, not a rejection. Which terms a form needs is developer analysis
   (`--analyze`), never part of a tune.
4. **A rejected fit: stop and ask, once.** Report the fit, its equation and its rms, propose a form,
   wait. Changing an equation's terms is the user's call. A model already revised once that still
   misses is accepted, marked NOISY and recorded under "Marked problems".
5. **When a count-based fit won't conform**, fit the times to the kernel's own counts first
   (counters in a scratch copy), then compare counts before touching prices. Tune a new form at all
   three levels before calling it sound. A kernel's price belongs in the kernel's own section.
6. **Only what a decision reads is measured.** Purge a fit or benchmark nothing reads, from the
   tuner, the schema and the docs.
7. **Tune a helper through its callers**, and give a sweep workloads that span every caller's shape.
8. **Static algorithms get approximate values only**: no fitted equations or two-input work.
9. **The series evaluators are general evaluators.** Their section is tuned on the synthetic family
   (`tuner/synth.rs`) and on no caller's series: never put `Exp` or `ATanh` into its fits, sweeps or
   gate items, and don't special-case an evaluator on exp's or ln's evidence.
10. **Every section ends with the end-to-end gate.** Read its list of slower workloads. The gate
    can't see a loss against the previous design: for that, A/B a normal build against the tree
    before with `tuning_bias`, item by item. A dispatcher's own cost is part of the result and a
    tune build hides it.
11. **Before quoting a simulated loss** for a table layout or dispatch rule, replay the real builder
    and lookup, check the replay against the live picks, and score every call the candidate touches.
12. **A tune needs a steady machine**: mains power, nothing else running, a fixed workload timed
    first. When several unrelated fits fail at once, suspect the machine. Don't retune the M1 or
    write the preset unprompted.
13. **Every exp workload takes `check_items::EXP_FRACTION`**; m = −(n) times a no-op.
14. **Knob changes** follow `docs/tuning_system.md`'s invariants: every value in a valid range is
    correct; soundness limits and capacities are never knobs; bump a section's `version` when its
    knobs or an equation's form change and its `derive` number when its table's layout changes.
15. **Comments in `src/tuning` are terse**: what a thing is and the one reason that would stop a
    wrong change. History and numbers go in the docs.

## Testing

- Verify with the debug suite. `cargo test --release` fails 7 `should_panic` tests by design
  (they expect `debug_assert!` messages).
- `test_ntt_sqr_entry_dyn_parallel_branch` is a known flake in full-suite runs: rerun it alone
  before suspecting a diff.
- Oracles for big-number tests are schoolbook `mul_buf` and forced Knuth, never `mul_dyn`.
- **Random operands hide the worst bugs here.** Every wrong-result bug in `docs/fixed_bugs.md`
  passed the random suite. Near Karatsuba's buffer arithmetic, NR's corrections, high-product
  truncation or the FFT, test structured operands (zero halves, powers of two, all-ones runs,
  sparse divisors, worst-case digit patterns).
- After any change to a dispatcher or a knob, run `cargo test --lib --features tune`.
- After adding a test that reads live knobs, and before handing the tree to another machine, run
  the suite under the schema's defaults (the Rosetta line above). A count-style test pins what it
  depends on.

## Known issues

- `ln_static_at::<16600>` and `ln_agm_static_at::<16600, 6>` at 16384 output limbs overflow a
  spawned worker's stack (fine at N = 4200). Not investigated.
- Static Hyper binary splitting can panic in `hi_mul_shifted` when an R product is zero.
- `ln_dyn_at` (the series path) trips `ratio_to_fraction_dyn`'s "finalize expects S < 1" debug
  assertion for X = 1 + δ with a multi-bit δ far below 1, from about 150 output limbs.
- Products and squares haven't been swept with worst-case operands on the 2^19-digit FFT transform;
  AVX2 is unmeasured for FFT soundness.
- No x86 preset yet: x86 builds run on the schema's defaults until the i7 is tuned.

## Features and build

- `tune` — the tuner, and knob accessors that read live atomics (constants in normal builds). For
  the tuner and the random-knob test only. Never change a knob while a library call runs.
- `_bench_internals` — exposes utils internals to Criterion. Not for non-bench code.
- `_fft_cutoffs` — enables the long FFT soundness test.
- `build.rs` layers the schema's defaults, then `presets/<arch>-<os>.toml` (else `<arch>.toml`),
  then the file `BIGBITS_TUNING` names (`presets/local.toml` through `.cargo/config.toml`). Delete
  `presets/local.toml` to build with the preset again.

## Working here

- **With any change, update every reference to what changed**: this file, `docs/`, comments,
  scripts, probes. Search the whole repo for each removed or renamed name.
- Keep this file a map. New detail goes in the area's doc; a dated record of what was tried goes in
  `docs/tuning_implementation.md`.
- Benchmark-only wrappers and forced-algorithm harnesses live in `benches/` and `src/tests/`, never
  in the algorithm modules.
