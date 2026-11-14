# Square roots (`src/utils/sqrt.rs`)

Binomial roots below each output mode's top-level cutoff, Zimmermann's recursion above. How the
knobs are measured is in `tuning_system.md`. Numbers are the M1's unless noted.

## Entries

| Output | Dynamic | Static |
|---|---|---|
| Exact root and remainder | `sqrt_dyn` | `sqrt_static` |
| Exact root only | `sqrt_only_dyn` | `sqrt_only_static` |
| Floor or one above | `sqrt_approx_dyn` | `sqrt_approx_static` |

The `zimmerman_sqrt_*` names are `pub use` aliases of these. Inputs shorter than `2·s.len()` are
read as padded, x·B^(2·s.len() − x.len()). The approximate root satisfies
⌊√X⌋ ≤ s ≤ ⌊√X⌋ + 1: a numeric bound, not "only the last limb is off", since the +1 can carry. The
doc-hidden `sqrt_dyn_at`, `sqrt_only_dyn_at` and `sqrt_approx_dyn_at` take the wrapped-remainder
cutoff explicitly (`usize::MAX`: classic stages everywhere).

## Structure

- `binom_sqrt_core` is the binomial root (one `knuth_est` per root limb, quadratic).
- `zimmerman_sqrt_entry` runs one Zimmermann stage: the top one in each output mode, and, as
  `SqrtOutput::Stage`, every recursive one, which leaves x above the remainder unspecified.
  `zimmermann_sqrt_core` is the leaf check in front of it. Both are shared by the two allocation
  models through division and square callbacks. `normalized` is every top level's normalization,
  one-limb roots included; `sqrt_denormalization` returns at once when the shift is 0.
- A stage: one recursive root of the high half, a division by it, a square of the low root, and a
  linear correction. Only the top stage can skip work: root-only output returns before the low
  square when the remainder's high limbs already decide the correction (usual on random inputs), and
  the approximate output divides quotient-only and has no square. Every recursive stage needs its
  exact remainder.
- Knobs (section `sqrt`): `leaf` (the recursive leaf, shared by both models), six top-level
  cutoffs (`<dyn|static>_<rem|only|approx>`), `dyn_wrapped_rem`, `dyn_rcp_carry`. They are read
  once per call into a `SqrtKnobs` that the recursion passes down, and `sqrt_alg(root_len, cutoff)`
  is the pure predicate for the top level and for a stage. **Every cutoff is at least 3**:
  Zimmermann's split takes (s.len() − 1)/2 low root limbs, which is none for a root of 1 or 2.

### The wrapped remainder (dynamic only)

A stage's remainder R = X − s² lies in [−(2s − 1), 2s]. Dynamic stages of at least
`sqrt.dyn_wrapped_rem` root limbs whose m = `bnm1_size(k + 1)` is at most `FFT_BNM1_16BIT_CUTOFF`
form it from residues mod B^m − 1: they fold X before the child overwrites its top half
(`fold_mod_bnm1`, then a limb rotation for a padded top), divide quotient-only, square s once with
`sqr_mod_bnm1_dyn`, and read R's sign from the residue's top bit. That drops NR division's whole
q·s′ remainder product. Roots and remainders are bit-identical to the classic stages'. Recursive
stages take it in every output mode; at the top only the root-and-remainder output does. Static
passes `None` and keeps classic stages (a static version needs a cyclic NTT square).

Wrapped ÷ classic: 0.93 at 2084 limbs, 0.88 at 4164, 0.85 at 16644; a stage of a few hundred
limbs loses, since its wrapped square is an FFT where the classic half-size square is Karatsuba.
`sqrt.dyn_wrapped_rem` is its own sweep (438 on the M1, default 341) and does not follow
`mul.sqr_bnm1_fft`: following it to 81 made `sqrt_dyn` 40% slower at 200 limbs.

### The reciprocal carry (dynamic only)

The recursion is linear and each stage divides by the whole root of the stage below, so roots of at
least `sqrt.dyn_rcp_carry` limbs carry one reciprocal up it (`RcpCarry`, `s.len()/8 + 8` limbs; see
`div.md`). The carry lives in `sqrt_dyn_output`'s division closure and the stages never see it. A
stage's NR division resumes from what the last NR stage left: the child's precision passes the
parent's second-to-last chain step by 1–2 limbs (modelled at every width to 40000), so only the last
refinement runs. Every divisor's top bit is set, so normalization never shifts; a stage's final −1
moves the divisor a unit, far below the reciprocal's precision. Roots are bit-identical.
`dyn_rcp_carry` is not a crossover: smaller roots skip a scratch acquisition they couldn't use
(the first read is at about 680 root limbs).

New ÷ old with the carry: `sqrt_approx_dyn` 0.958 at 1044 limbs to 0.875 at 16644; `agm_dyn` 0.974
to 0.922.

## Facts to know when measuring

- **The crossover depends on the input's shape**, which dispatch doesn't see: full (2n) and
  minimally padded (n + 1) inputs favour Zimmermann several limbs earlier than 2n − 1 and 3n/2 + 1.
  Each top-level cutoff is the best single threshold on root width.
- **Root and remainder switches later** than the other two modes, because its Zimmermann entry
  always pays for the final low square and the denormalization.
- **Static N.** Static division and squaring zero `[u64; N]` scratch on every call, so cost grows
  with N: on the i7 the best `rem` cutoff went from 20 at N = next_pow2(2n) to past 49 at N = 4096.
  Static cutoffs assume N close to `x.len()`.
- **A/B traps.** Two runs of one binary share heap and code layout: 128-limb roots once read 16%
  faster by luck, and relinking identical code moved `sqrt_dyn` at 16384 limbs by 6%. Randomize the
  heap with a padding global allocator and give the control copy other seeds; past about 1024 limbs
  build everything with `-C llvm-args=-align-all-functions=6 -C
  llvm-args=-align-all-nofallthru-blocks=6`.
- `benches/probes/zimmermann_stages.rs` pastes this file's text (with a
  `pub use big_bits::{tuning, utils}` shim), so it depends on `zimmerman_sqrt_entry`'s argument
  list, `SqrtKnobs::dyn_read`, `dyn_div`, `static_div` and `sqrt_denormalization`.

## Where the time goes

Zimmermann, root and remainder (i7, 2026-10): division callbacks are 51% of a 32-limb root, 60% at
64–128 and 80% from 512; squares 12–30%; direct control under 1% above 64 limbs. In the AGM's
shape at 4096 limbs a top division is about 55% of a root and the wrapped square 8%. Binomial
(i7, 2026-09): the seed is 85% of a 1–2 limb root, quotient estimation a third at 8–16 limbs, and
the multiply-subtract and add/subtract rows 94% at 1024.

## Findings not acted on

- **Forcing NR lower than dispatch does not pay**, carry or no carry: NR with a remainder costs
  1.2–1.6× BZ/Knuth below about 1536 limbs, and a lower wrapped cutoff pays for a whole-root square.
  The one gain (the top of a small `sqrt_approx_dyn`, 110–250 limbs, up to 9%) was `div_dyn`'s
  crossover, which the fitted equations decide now.
- **A saturated low root** (all-ones inputs saturate at every level) has the known square
  (B^k − 1)² = B^(2k) − 2B^k + 1. Building it directly took 47–90% off such roots in a probe.
- **Large roots** follow the FFT limits: past the 2^19-digit transform a root's products are NTT.
  Letting the FFT run further read 0.26× at 1M limbs in a scratch copy, but that needs a check that
  holds past 2^48 (`fft_precision.md`).
- **A cached reciprocal of the leading root limb** for binomial roots below 64 limbs: 3–4× on the
  estimate chain in `benches/probes/binom_sqrt_reciprocal.rs`. The Knuth rewrite's table-based 2/1
  reciprocal has since taken most of this.

Harnesses that still run: `benches/binom_sqrt_profile.rs` (with `binom_sqrt_sampler.c` and
`scripts/python/profile_binom_sqrt.py`, `compare_binom_sqrt.py`, `verify_binom_sqrt.py`), the
probes `benches/probes/binom_sqrt_*.rs`, `binom_submul.rs` and `zimmermann_stages.rs` (with
`scripts/python/profile_zimmermann_sqrt.py`). The reports these findings came from
(`sqrt_cutoff_tuning.md`, `sqrt_optimization_analysis.md`, `binom_sqrt_profile.md`,
`binom_sqrt_validation.md`, `zimmermann_sqrt_profile.md`) are in git history at `fee4f90`; the
old cutoff probe is at `faa9971`.
