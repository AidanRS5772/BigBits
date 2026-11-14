# The test suite (`src/tests/`)

What each file covers and which oracle it trusts. Tests are included from `lib.rs` under
`#[cfg(test)]`; `test_div_nr.rs` is included from `div.rs` for private access (check that a
`#[path]`-included file still compiles after a large refactor: it was once orphaned for a week).

## Running

```bash
cargo test --lib                                   # debug; about 2 minutes
cargo test --lib --features tune                   # adds the tuner's tests and the random-knob test
cargo test test_mul                                # one module
cargo test test_utils::test_trim_lz_empty          # one test
cargo test --target x86_64-apple-darwin --lib      # under the schema's defaults (Rosetta)
```

- **Verify in debug, never `--release`**: 7 `should_panic` tests in `test_mul` expect
  `debug_assert!` messages that release compiles out, so a release run looks like a regression.
- `test_ntt_sqr_entry_dyn_parallel_branch` is a known flake under full-suite parallelism. Rerun it
  alone before suspecting a diff.
- Tests read accessors, never literal knob values: a preset moves every boundary. Find a boundary
  with the dispatcher's own predicate (`is_school`, `dyn_div_alg`).
- The M1 preset is in force for every native run, so a test that depends on tuned values passes
  here and fails on a machine with no preset. Run the Rosetta line after adding a test that reads
  live knobs, and before a handoff.
- `--features _fft_cutoffs` adds the long FFT soundness test (128 worst-case operands per shape).

## Oracles

Big-number oracles use schoolbook `mul_buf` (`mul_ref`), never `mul_dyn`, and forced Knuth for
division. Shared helpers in `tests/mod.rs`: `rand_vec`, `rand_nonzero_vec`, `to_u128`,
`structured_vec` (limbs biased to zeros, ones and powers of two), `fft_worst_operand` (balanced
worst-case kinds), `mul_ref`, `verify_divmod`, `exact_fraction` (a dynamic node's exact P/Q).

Each section has a `random_knob_check` that the random-knob test runs under every knob vector
(`SECTION_CHECKS`); see `tuning_system.md`, "Validation".

## Files

**`test_utils.rs`** — every buffer helper (`trim_lz`, `add_buf`, `sub_buf`, `cmp_buf`, shifts), and
the residues mod B^m − 1 against division by the all-ones divisor.

**`test_verify.rs`** — `Residues::of`/`of_with_top` against u128 Horner (every length to 40, and
around the four-lane path's 128-limb threshold with every leftover count); that
`verify_mul`/`verify_sqr` accept correct products and reject every single-bit flip of a small
product, ±2^(16k) coefficient errors alone, in pairs (including pairs one limb apart, which cancel
mod B − 1) and in clusters.

**`test_mul.rs`** — products, squares, high and middle products, dyn and static, against schoolbook;
boundary finders for algorithm transitions.
- *FFT machinery:* `find_fft_size` against a direct search; `fft_twidles` exact; `scale_and_round`
  bit for bit against its scalar path; `recombine`/`sqr_recombine` within a few ulps of their scalar
  loops; `decompose`/`sqr_decompose` against their operands' values and digit ranges.
- *FFT soundness:* the raw entries exact at every odd part of the size set (a wrong FFT in a checked
  entry just gets redone, so only raw entries test the arithmetic). The fold check on shapes whose
  raw transform is wrong for worst-case kinds, which on the M1 exist only past the limits (the
  98304-limb square, 98304 × 98304, the middle product at n = 98303, the cyclic square at
  m = 114688): the result must be exact, every wrong raw result must fail the fold check, and
  `fft_reroutes()` must grow. Also: a checked entry past its limit goes straight to the NTT; right
  results reading high keep the FFT's answer; the fold primes are prime, divide 2^(16N) − 1 and fit
  the kernel's bound; `fft_fold_agrees`'s two sparse blind spots are pinned. The residue test turns
  the fold check off and requires `fft_residue_reroutes()` to grow (it returns early in a release
  build without `FFT_RESIDUE_CHECK`).
- *Dispatch limits:* each dynamic entry runs a worst-case and a random operand at each limit and
  one past it.
- *`sqr_mod_bnm1_dyn`* against the folded schoolbook square at every length up to m, and against the
  folded NTT square at and past its limit.
- *Equations:* `test_hi_window_products_counts_the_kernels` counts the limb products directly;
  `test_hi_equation_polynomial_is_the_equation` checks `HiEq::poly` against `HiEq::ns`.

**`test_div.rs`, `test_div_nr.rs`** — `div_prim`/`div_buf_of`, the forced dynamic/static entry
matrix, top-level dispatch, BZ blocks, NR's fallbacks, the reciprocal's < B bound.
- `test_div_2_1_short_quotients`: a 2/1 block at every count of significant quotient limbs.
- `test_dyn_div_small_table`: the table's lookup at a table made by hand (a pick, a pick that can't
  take the shape handed on, past a row's extent, below the first row). `test_dyn_div_table_and_entries`:
  live dispatch returns Knuth where nothing else can run and the table's algorithm where it names
  one.
- `test_dyn_div_alg_compares_equations`: the equations by hand at explicit constants, NR's product
  tiers included (`nr_parts` at q = 8 and q = 700), and that the tier boundaries are the
  dispatchers' own.
- `test_div_nr.rs`: static NR completes for every R ≤ N through both middle products; NR's outcomes
  and transactional remainder; the reciprocal carry (a chain resumed from one, several or no steps
  back lands as close as one from its one-limb seed; foreign, flipped and zero seeds cost a restart,
  not a wrong result).

**`test_sqrt.rs`** — binomial and Zimmermann roots in every output mode against a schoolbook oracle.
The wrapped-remainder tests run each mode at cutoffs from the leaf to the default and require
bit-identical roots (and `sqrt_dyn`'s remainder) against the classic stages at root lengths 24–1500
and every top-level shape. Edge inputs: perfect squares, Y² − 1, (Y + 1)² − 1, all-ones X (every
stage saturates), padded tops. The carried-reciprocal tests run across `sqrt.dyn_rcp_carry` and past
the first stage that reads the carry, on roots whose low limbs are zero (the final −1 borrows into a
lower stage's divisor).

**`test_bin_split.rs`** — any tree shape matches a single leaf (dynamic, exact); static nodes match
dynamic within one ulp at every cutoff and N up to finalize's N − 2 bound and keep all but a few
bits of 64N; wide twins (a narrow series scaled by a 70-bit K from some term on) sum what their
narrow series do, by cross-multiplication; a Chudnovsky-shaped wide series against a naive sum; the
constant caches; `Node::limbs` within 2× of real node sizes; the constants' series paths match the
tables. `test_split_work_gate_keeps_trees_exact` builds one tree under every setting of the off-pool
gate, from off the pool and from a worker.

**`test_series.rs`** — every algorithm `series_eval_at` can force, and the dispatcher's own pick,
within one ulp of the exact sum, over BBP, Engel and Hyper with x = 1, a one-limb x and a multi-limb
x, zeros > 0, signed series, every rect block length, static N = 4–64 up to both capacity bounds,
u128 coefficient boundaries, exact cancellation, zero multipliers, and the eligibility edges.
- *Contexts:* each in-crate tail bound (`Exp`, `ATanh`, `Ln2S`, `Pi`) against the exact tail,
  `zeros()`, `Hyper::new`'s errors, the 2^40-term cap.
- *The model's counts by hand:* `test_bin_split_parts_count_the_tree`;
  `test_horner_products_closed_form`; `test_horner_groups_are_whole` and
  `test_horner_groups_of_a_growing_q` (the kernel's grouping rule walked term by term, the model
  within one group of it); `test_horner_products_by_r_run_over_what_the_group_below_left`;
  `test_rect_groups_divide_at_their_own_frames`, `test_rect_powers_are_as_wide_as_x_bits`,
  `test_rect_adds_closed_form`; `test_bin_split_finalize_and_x_leaf_terms` (counts serial, with the
  parallel gain pinned to 0); `test_series_equations_share_horners_kernel_prices`.
- *Dispatch:* every prediction finite and positive with an eligible pick over edge shapes;
  `test_small_series_table_layout` (buckets, kinds, a lookup at a table made by hand);
  `test_guards_settle_ranges_by_their_ratios` (rect's guard from both sides of each threshold, none
  for a static node, and that live dispatch is `decide_at` under the live knobs).

**`test_trans.rs`** — `exp_dyn_entry` against a naive fixed-point Taylor oracle; `exp_dyn` against
`exp_full_oracle` (squares e^(X/2^j) j times; uses neither ln 2 nor division); `ln_*` against
`ln_oracle` and `ln_full_oracle` (e·ln 2 + ln u with ln 2 = 2 atanh(1/3), precision grown until
cancellation leaves 3 guard limbs). The cases are helpers taking the function as an `ExpFn`/`LnFn`,
so the static entries and the theta–AGM logarithm run all of them.
- exp: top limbs at 2^-192 through 2^64, X near multiples of ln 2 (either neighbouring exponent
  accepted at the 2^(64k) boundary), the overflow bound (n = 2^69 − 1 is the case that catches too
  few reduction guard bits), square counts 0–130, both sides of the parallel gate, the capacity
  errors leaving `out` untouched. `test_exp_scale_is_the_top_limbs` pins what m scales.
- ln: X = 2^e, X = 1, X within 2^-64t of 1 from either side, X in [1/2, 1), m = ±2^60, X = 0,
  every square-root scale, sparse and tiny δ, parallel against serial.
- `test_agm_wins_shape` and `test_ln_takes_agm_reads_the_bound_for_its_threads` check the dispatch
  rule at explicit cutoffs; `test_ln_dyn_dispatch_by_threads` on 1, 2 and 8 threads.

**`test_theta.rs`, `test_theta_plan.rs`** — theta within one unit of `theta_oracle` (every power by
binary powering with schoolbook products, not theta's plans), in each `ThetaMode`, with the two
`PARALLEL` runs bit-identical; l at and around limb boundaries; very unequal output lengths;
`test_theta_agm_identity` checks π/(4·AGM(θ2², θ3²)) = ln(1/q) in f64. The plan tests check both
plans for every T from 0 to 16 (values are their operands' sums, every q^(n²) appears once, no slot
is overwritten while it has a reader, `slots` is the live peak); T = 3's sequences are golden.

**`test_agm.rs`** — `agm_dyn_at`/`agm_static_at` at tail caps 0–40 against `agm_oracle` (exact floor
means two limbs past the frame, roots checked by schoolbook), with a first-order tolerance carried
back through each step's Jacobian. `test_agm_tail_every_term_count` starts pairs inside the tail at
every gap needing 0–19 terms; `test_agm_tail_coefficients` checks `AGM_TAIL` against K's series
exactly; `test_agm_theta_ln_identity` checks 4s·M·ln 2 = π against `pi_dyn` and `ln2_dyn`.

**`test_tuning.rs`** — the schema's checks; `test_accessors_match_resolved` (layering the machine's
own file as `build.rs` does); the file format; the resolver on a schema of its own (layers,
`Follows`, a version's fall-through, upstream hashes, a machine layer over a preset, a file tuned on
another target refused); that `src/utils` never names `resolved::`. With `tune`: the gate's verdicts
on made-up rounds (`test_check_gate_verdicts`); fit marks and recovered coefficients
(`test_fit_evidence_marks`, `test_fits_recover_coefficients`); fit designs
(`test_fit_designs_cover_their_area`); the synthetic family
(`test_synthetic_series_converge_and_agree`); the table rule on made-up timings
(`test_table_rule_on_made_up_timings`); `test_every_table_is_a_dispatchers`;
`test_fits_leave_out_what_tables_decide`; `test_series_table_cells_are_of_their_class`; and
`random_knobs`.
