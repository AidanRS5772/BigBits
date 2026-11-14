# Fixed bugs: root causes and regression tests

Kept because most of these returned **silently wrong results**, not panics, passed the whole random
suite, and are easy to reintroduce with a plausible-looking change. The common lesson: random
operands almost never reach a carry, zero or boundary case (probability about 2^-64 per
sub-product). Test near any of this code with structured operands (zero halves, powers of two,
all-ones runs, sparse divisors) against an independent oracle: schoolbook `mul_buf` or forced Knuth,
never `mul_dyn`.

## Multiplication

**`karatsuba_core`'s cross-term offset (wrong products).** `karatsuba_mul(a1, b1, z2)` returns the
product limb at index `z2.len()`, but the cross-term correction subtracted it at
`cross[buf_len(z2)..]`, which is lower whenever z2's top limb is zero. Triggers: both operands with
an all-zero low half from 19 limbs, and structured operands whose sub-products land a zero top limb
deep in the recursion. Tests: `test_karatsuba_core_overflow_offset_zero_top`,
`test_karatsuba_structured_operands`, `test_mul_structured_operand_fuzz` (limbs biased to zeros and
powers of two: 571 bad products in 6000 trials with the old offset).

**`karatsuba_entry_dyn`'s chunking panic, and what fixing it exposed.** `chunking_karatsuba`'s last
chunk can be up to 2s − 2 limbs, on the exact Chunking/Recurse boundary, where `karatsuba_core`
needed an output of 3·half + 1 limbs for its cross term. Now it adds the part of the cross term that
fits and folds the rest into the overflow limb. The fix exposed a latent second bug:
`karatsuba_mul`'s `Prim`/`Prim2` arms multiplied the *full* `out` slice, so a wider `out` fed stale
scratch into the product (silent corruption). They truncate to `long.len()` now. Tests:
`test_karatsuba_entry_unbalanced`, `test_karatsuba_core_recurse_boundary_sizing`,
`test_karatsuba_entry_unbalanced_random_sweep`,
`test_karatsuba_mul_prim_dispatch_ignores_stale_out_tail`. Verified first with an independent Python
differential-fuzz model: worth doing again for any change near this buffer arithmetic.

**High products truncating without guard limbs (`rcp_*` breaking its < B contract from R = 727).**
Above the partial cutoff `hi_mul_*`/`hi_sqr_*` truncated each operand to exactly `out.len()` limbs;
the dropped partial products are worth up to about 2B ulps, the whole low limb. A Newton step
squares its input error, so one untruncated step injected about B ulps that grew 1 → 2 → 4 → 8 limbs
and outran the two guard limbs. Fixed with `HI_MUL_GUARD_LIMBS` = 2, the guard terms added by linear
passes. The old test tolerated a fully wrong low limb, which is how it slipped through. Tests:
`test_nr_reciprocal_high_precision_regressions`,
`test_nr_reciprocal_static_high_precision_regressions`,
`test_hi_mul_truncation_error_is_a_few_ulps`, `test_hi_mul_guard_terms_match_exact_window`.

**`hi_mul_static` panicking at `out.len() == N` with two N-limb operands, N even.** The split
path's l1·s0 cross term needed N + 1 limbs. Its top limb is now added into `out` separately, and the
guard clamp lost a term that had left `out.len() = N − 1` with no guard at all. Test:
`test_hi_mul_static_near_capacity` (exact windows at N − 1 and N).

**NTT middle products with limb 1 off by about 2^60.** `ntt_mid_accumulate_scaled` reads two
coefficients below the band, and each prime's transform was sized for 2n − 1 coefficients, which
wraps the product's top columns onto exactly those two. Where 2n − 1 was an NTT size for some primes
and not others (81, 125, 243, 375, 405, 625, 1125, 1215) the CRT combined residues that disagreed.
Transforms are sized for 2n + 1 now (`mid_ntt_size`). Test: `test_ntt_mid_mul_exact_size_transform`.

**The 16-bit FFT wrong inside its old bounds.** Precision, not arithmetic: `fft_precision.md`. Tests:
`test_fft_fold_check_catches_wrong_results`, `test_fft_checked_entries_stop_past_their_limits`,
`test_fft_residue_check_catches_wrong_results`, `test_dyn_dispatch_exact_around_fft_limits`,
`test_sqr_mod_bnm1_large_against_full_square`.

**`powi_dyn_entry`.** Exponent 0 indexed an empty output (`powi_sz` returned (0, 0); now (1, 1)).
Exponent 1 and others were wrong because bit 0 of `reverse_pow(pow)` was always 1; it is rebuilt
from the exponent's rest-bits under a sentinel, which flipped the parity check that picks the
starting buffer. The two changes only work together. Test:
`test_powi_dyn_entry_small_exponents_exhaustive`.

**`hi_sqr_buf(&[], ..)`** underflowed; it has `hi_mul_buf`'s empty-input guard now.

## Division

**NR quotient one too small on an exact division.** `nr_refine_quo`'s error band used the raw top
window of d, so the middle product's dropped carry was scaled by B/(d's top limb); at an odd
quotient width the band's base lines up with the guard limb the `Exact` verdict trusts. Fixed by
forming the band from the normalized windows of d·2^sh and n·2^sh. `div_rem` was unaffected. Test:
`test_nr_exact_division_small_top_divisor` (divisor top limbs 1 to 2^20 at odd quotient widths).

**NR quotient one too large, and `div_rem` returning q·d + r ≠ n.** Both in how `nr_refine_quo`
applies its correction c. For err > 0 the floor of y − c is y − ⌈c⌉, but it subtracted ⌊c⌋: a true
quotient with an all-ones run just below the cut was wrong 73–88% of the time from 4 limbs. For
err < 0, subtracting x_norm's top limbs could leave a tiny positive c whose negation was nearly all
ones; `nr_rem_finish` reads only the low limbs, so it could accept that when d has zero limbs.
Fixed by reading c's sign after the subtraction and `dec_buf` on the positive branch. A structured
differential fuzz against forced Knuth gave 7646 wrong results in 40000 shapes before, none in
400000 after. Tests: `test_nr_quotient_floor_when_estimate_rounds_up`,
`test_nr_rem_scaled_multiple_of_sparse_divisor`.

## Series

**Static `bin_split` nodes panicking at large leaf cutoffs.** Static merges drop low limbs, but
leaves build some products exactly into `[u64; N]`, so a leaf whose range was too large for N
overflowed: whether a cutoff worked depended on N, the leaf's position and the series, though every
cutoff is mathematically valid. Each static leaf now checks a conservative bit-length bound
(`prod_fits`) and splits at its midpoint when it fails. This is invariant 1 of `tuning_system.md` in
its original form. Tests: `test_static_large_leaf_regressions`, `test_static_nodes_any_cutoff`,
`test_static_leaf_size_sweep`, `test_static_bbp_shift_past_n`.

## Found by the random-knob test

`is_school_at`'s parabola cancelling for a large cutoff (a 3 × 3 product then recursed forever);
`rect_best` returning block length 0 when every prediction was infinite; `cache_par` overflowing;
debug builds' static split stacks overflowing on one-term leaves. Each is fixed, and each narrowed a
knob's valid range or a computation (`tuning_system.md`, "Validation").
