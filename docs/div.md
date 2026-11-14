# Division (`src/utils/div.rs`)

Quotient, quotient with remainder, high quotient and reciprocal, in dynamic and static allocation
models, over Knuth, Burnikel–Ziegler (BZ) and Newton–Raphson (NR). How the knobs are measured is in
`tuning_system.md`. Numbers are the M1's unless noted.

## Contracts

Operands must be trimmed and `d` nonzero. Let M = `div_quotient_len(n.len(), d.len())` =
`n.len().saturating_sub(d.len())`.

- **`div_*`, `div_rem_*`** require `q.len() ≥ M`, write the low M-limb quotient body and return the
  possible structural limb at index M. When `q.len() > M` that limb is stored in `q[M]`, the tail is
  zero-filled and the call returns 0 (multiplication's overflow convention). Equal-width operands
  allow an empty body and return their single-limb quotient. Quotient-only entries preserve both
  operands; `div_rem` leaves the exact remainder in `n`.
- **`hi_div_dyn`, `hi_div_static<N>`** (no remainder variant): for `q.len() = K < M` they return
  the same overflow and write body digits M−K..M−1; for K ≥ M they delegate to standard division.
  Preflight drops irrelevant low numerator limbs and exact whole-limb divisor factors.
- **`rcp_*`** preserve `d`; `rcp.len()` = R is the requested precision. With D = `d.len()` and
  X = min(⌊B^(D+R−1) / d⌋, B^R − 1), the result's error is strictly below B: the lowest output limb
  is not guaranteed exact. Drivers remove exact low-zero limb factors and keep only the top R + 1
  divisor limbs, so work doesn't scale with irrelevant low limbs.
- **Static entries** assert that operands fit N once exact low-zero factors are removed
  (`rcp_static` asserts `d` and `rcp` each fit N). Behaviour reachable only with longer inputs is out
  of contract.
- Number-type callers allocate the output and call an entry directly; there are no allocating
  adapters.

## Structure

Each standard call validates and prepares operands (`division_preflight`), dispatches on the logical
quotient width q = M + 1 and the effective divisor width d (after its exact low-zero limb factor),
then runs a prepared driver (`div_prepared_*`, `div_rem_prepared_*`; reciprocals:
`reciprocal_preflight`, `rcp_prepared_*`). Four division drivers apply operand windows, attempt NR
and retry through BZ on failure before preparing classical storage; two reciprocal drivers own the
NR/Knuth choice and retry. Dynamic drivers provide `ScratchGuard` buffers, static ones `[u64; N]`.
Tests and benches force `DivAlg`/`RcpAlg` through the same preflight and drivers; the forced-entry
harnesses live in `src/tests/test_div.rs` and `benches/`, never in the algorithm module.

Building blocks: `div_prim`/`div_buf_of` (normalized Knuth), `div_2_1`/`div_3_2` (BZ recursion),
and NR's `nr_err_band`/`nr_refine_rcp`/`nr_rcp_chain`/`nr_refine_quo`. Only shared arithmetic
kernels take multiplication callbacks.

- **Knuth** returns its initial top estimate directly. Its upward-biased seed for NR reuses
  `knuth_rcp_normalized` with seed 1 and an explicit increment.
- **BZ** uses Knuth for its recursive base cases (divisors up to `div.bz`, at least 3: the 2/1 step
  halves the divisor and Knuth needs two limbs). The top digit comes from its initial block.
  `div_3_2` multiplies each quotient piece by the divisor's low half at the piece's significant
  width (`buf_len`) and skips a zero piece.
- **NR** keeps a contiguous K + 1-limb internal quotient and bridges exact-width and high-window
  callers through temporary storage; an output with an overflow slot is a direct fast path. Attempts
  report success or failure (`NrEstimate`: proven, needs exact correction, failed) and preserve the
  numerator on failure; remainder correction commits only after success. The reciprocal attempt uses
  two internal guard limbs and omits the full `d·rcp` verification product. Its work buffer and last
  divisor window run up to 3 limbs past R; the static attempt keeps those in `[[u64; N]; 2]`, caps
  its error bands at N, and relies on the window's limbs past N being zero padding, so static NR
  completes for every R ≤ N (only N ≤ 3 leaves no room and retries Knuth).
- **Correctness fallbacks** (not performance dispatch): NR division → BZ when estimation or bounded
  correction fails; the reciprocal driver → Knuth when refinement or normalization can't produce a
  usable approximation.

`nr_refine_quo` forms its error band from the normalized windows of d·2^sh and n·2^sh, and the
correction from the reciprocal before its shift, which bounds the middle product's dropped carry at
about 2(h + 1) ulps of the second guard limb. It reads the correction's sign after the subtraction
and decrements after a positive one (floor of y − c is y − ⌈c⌉). Both were wrong-result bugs that
random operands never reach (`fixed_bugs.md`): test NR changes with structured operands (all-ones
quotient runs, sparse divisors, small top limbs) against forced Knuth.

### The reciprocal carry

`RcpCarry` lets successive dynamic divisions by growing prefixes of one divisor share a reciprocal
(sqrt's stages). `div_carry_dyn`/`div_rem_carry_dyn` are the plain entries plus a carry that
`nr_quo_est_dyn` reads and leaves; BZ and Knuth ignore it. `nr_rcp_resume` restarts the chain
(`nr_rcp_steps`) from the widest step the carried precision reaches, truncating the seed to that
step's p + 1 limbs, so usually only the last refinement runs. A seed without a limb to spare is
rejected and the chain starts from its one-limb seed: a bad seed costs a restart, not a wrong
result. A zero seed is the one its own band can't see; the quotient's band fails it and the driver
retries through BZ. `keep` stores the normalized reciprocal only when the quotient didn't fail.
Static division has no carry.

## Dispatch

**Dynamic quotient entries** (`div_dyn`, `div_rem_dyn`, `hi_div_dyn`; `dyn_div_alg` →
`dyn_div_alg_at`, with a `DivEq` and mul's `MulEq`, each read once):

1. No equation is read where nothing else can run. NR needs `nr_split`'s room (q ≥ 8); BZ needs a
   divisor past `div.bz` and a 2/1 block of its own (`bz_plan`: a whole block, q > d, or a top block
   predicted cheaper padded than Knuth); Knuth always can.
2. The measured tables `div.dyn_small` and `div.dyn_rem_small` (rows by the divisor's limbs, change
   points over the quotient's; a cell is Knuth, BZ, NR or "the equations"; `dyn_div_small`). A pick
   that can't take the shape is handed on. On the M1 they hold only Knuth cells, with rows at
   divisors to 38 limbs (quotient only) and 108 (with remainder).
3. The cheapest fitted prediction among the algorithms that can take the shape.

`dyn_div_best` is the same choice with its predicted ns (a series' finalize is priced by it). It is
`#[inline(always)]`: out of line it cost `div_dyn` 18 ns at 76/64.

| Equation | Form | Knobs |
|---|---|---|
| Knuth | `a·q·d + b·q + c·d + e` | `div.knuth_ns_*` |
| NR | its products by tier, one fitted multiple a tier, `+ b·q` | `div.nr_ns_*` |
| NR with remainder | the same, `+ r·Mul(d, min(q − 1, d + 1)) + b·q + c·d` | `div.nr_rem_ns_*` |
| BZ | `block`·(whole 2/1 blocks) + `top`·(top block) | `div.bz_ns_*` |

- **NR's products** (`DivEq::nr_parts`): the reciprocal chain to h + 2 limbs (h = q/2 + 1; from each
  precision p a middle product with a (p + 1)-limb short side and a high product into p limbs) and
  `nr_refine_quo`'s three (the numerator's top by the reciprocal into h + 2 limbs, the error band's
  middle product, the correction into l + 2 limbs). Each is put in the tier its own dispatcher picks
  (`dyn_mid_dispatch_at`, `hi_alg`): school middle, FFT middle, school high, guarded high. The
  school tiers are closed forms, so a small division's dispatch reads no product equation. NR's time
  has no term in d for a quotient: it reads the divisor's top window. A single multiple of M(h, h)
  fitted at 4.2–4.7%, because NR runs no Karatsuba.
- **BZ** as its recursion: (q − 1)/d whole 2/1 blocks, each Σ 2^k M(d/2^k) down to Knuth leaves
  (`bz_block_parts`), and a top block of (q − 1) % d + 1 quotient limbs that is either Knuth's
  equation or, on a padded numerator, a block priced by its quotient's significant pieces
  (`bz_padded_top_ns`). The fitted multiples are near 1: the recurrence over mul's and Knuth's
  equations is nearly exact. Dynamic BZ's own top-block choice compares the same two predictions
  (`dyn_bz_top_padded`; the tune-only `BZ_TOP_FORCE` overrides it for forced timings).
- **Fits.** Shapes are (q, d) scattered over 4 to 4096 limbs each, every other one held out; each
  forced probe times the division alone (a remainder probe's numerator copy is timed separately and
  taken off). NR's and BZ's fits leave out the shapes the tables decide; Knuth's keeps every shape,
  as it is also a price inside BZ's equation and a series' finalize. Held out in the preset: Knuth
  2.2%, NR 3.0%, NR with remainder 3.8%, BZ 2.6%. BZ's and NR's forms count as revised: a miss is
  marked NOISY, not REJECTED.
- **What the table rule costs.** An index just past a table pays up to 5% for its decision. The
  tables replaced guards computed from the equations (`div.dyn_knuth_d`), and 12 of the division
  grid's 40 shapes run 2.8% slower as a group for it (0.9–7.3%), accepted for one rule everywhere.
- `cargo tune division` times every algorithm and the entries on shapes no tune takes. Under the
  2026-11-11 preset the picks lost 0.07% of measured time for `div_dyn` (worst 4.9%) and 0.30% for
  `div_rem_dyn` (worst 13.2%: NR with remainder at a short quotient over a divisor past 1200 limbs).

**Static quotient entries** keep the cutoff rule (approximate, like every static decision):
`div.knuth` is the mandatory Knuth width; a two-regime NR-vs-BZ cost model per family
(`div.static_div_*`, `div.static_div_rem_*`) picks; when it picks BZ, or the shape can't enter NR,
they call Knuth up to `div.bz` and BZ above. Each reads its knobs once into a `DivCutoffs`
(`DivCutoffs::static_div()`, `static_div_rem()`), so `div_alg_dispatch` is a pure predicate. Static
BZ's top block uses `use_bz_for_top_block` with `div.bz_top_padded_cost_scale`.

**Reciprocals** depend only on R: Knuth up to `div.dyn_rcp_knuth_nr` / `div.static_rcp_knuth_nr`,
NR above. There is no BZ reciprocal.

## Facts to know when measuring

- **Buffer placement.** Knuth by a 1024-limb divisor runs up to 2.45× slower with the divisor 1 to 7
  limbs past a 4096-byte alignment against its working buffer (its loads alias the numerator's
  stores; not looked into). Division's probes are therefore timed at two placements, and the
  tuner's probe builders (`numerator`, `placed_divisor`, `copy_probe`) must keep their order of
  allocation.
- **Static N.** Static drivers zero-fill `[u64; N]` buffers on every call (about eight for NR, three
  for BZ). At an N far above the operands a small division mostly times memsets: static NR read
  3–10× slower than BZ below 400 limbs at N = 16384. Static values are measured with N sized to the
  operands, and one leaf knob is shared with the dynamic path: `div_rem_static<2048>` at 200/100 is
  10% slower at the M1's `div.bz` of 99 than at the default 140.
- `div_prim2`/`div_prim` are data-dependent: 11–17% slower on fresh data for some divisors, and a
  repeated buffer hides it (the predictor memorises the correction branch).
- Knuth's equation is fitted from 4-limb divisors and reads 27% high at 2 limbs, 6% at 3.
- A near tie flips between retunes: `div_rem_dyn` 1030/1000 (NR with remainder against BZ at a
  quotient of 31 limbs) is 6% apart on the clock.

## Findings not acted on

- **Where NR's time goes** (profiles, 2026-07-25 and 2026-11-04): 92–99.9% multiplication at every
  size past the smallest; wrappers, closures, scratch and copies are all under 1%. Don't chase
  wrapper overhead. rustfft's cost per point at mixed sizes was the real lever, since taken by
  `find_fft_size`'s stricter set.
- **`nr_rem_finish`** computes the full d × q_low product and reads only its low `d.len() + 1`
  limbs. A wrapped (mod B^m − 1) or low product measured about 0.90× on `div_rem`.
- **A row-kernel middle product** measured 0.82–0.84× at 128–240 limbs.
- **Sharing FFT transforms inside a Newton step** (analysis, 2026-11-06, no prototype): x appears in
  two products of each step, so a reciprocal step could go from 6 half-length transforms to 5 and a
  quotient refinement from 9 to 8: about 0.87× a step, 5% of an AGM, 7% of a root. It needs a new FFT
  interface that keeps x's spectrum. Nothing carries across steps (operands change and lengths
  double), and merging two products into one triple product doesn't fit 16-bit digits in a double.
- Knuth-seeding the NR chain gains nothing. Forcing NR below where dispatch takes it doesn't pay:
  NR with a remainder costs 1.2–1.6× BZ/Knuth below about 1536 limbs.

Harnesses that still run: `benches/div_profile.rs` with `scripts/profile_division.sh` and
`scripts/python/analyze_div_profiles.py` (samply attribution of the six public dispatchers), and
the `division_flow` group of `cargo bench-utils` (`benches/probes/division_flow.rs`). The reports
and CSVs behind the old cutoff rule (`division_*.md`, `division_*.csv`, `nr_transform_sharing.md`)
are in git history at `fee4f90`; their harness (`cargo bench-cutoffs`) is at `faa9971`.
