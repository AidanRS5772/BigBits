# Multiplication (`src/utils/mul.rs`, `verify.rs`)

Entries, contracts, dispatch and the transform machinery. The 16-bit FFT's soundness (balanced
digits, the fold check, the limits) is in `fft_precision.md`; how the knobs are measured is in
`tuning_system.md`. Numbers are the M1's unless noted.

## Entries and contracts

Buffers are little-endian `u64` limbs. Dynamic entries take pooled scratch (`ScratchGuard`); static
entries take `[u64; N]` and assume every input and output has at most N limbs.

| Operation | Dynamic | Static |
|---|---|---|
| Product | `mul_dyn` (`mul_vec`) | `mul_static<N>` (`mul_arr<N>`) |
| Square | `sqr_dyn` (`sqr_vec`) | `sqr_static<N>` (`sqr_arr<N>`) |
| High product | `hi_mul_dyn`, `hi_sqr_dyn` | `hi_mul_static`, `hi_sqr_static` |
| Middle product | `mid_mul_dyn` | `mid_mul_static` |
| Square mod B^m − 1 | `sqr_mod_bnm1_dyn` | none |

- **Overflow.** A product into `out` of l + s − 1 limbs returns the top limb. A wider `out` stores
  it and zero-fills the tail, and the call returns 0. Division follows the same convention.
- **High products** compute a limb window of the product. `hi_mul_buf`/`hi_sqr_buf` are within a few
  ulps. Above `mul.<dyn|static>_partial_<mul|sqr>` the guarded truncated product takes the operands'
  top `out.len()` limbs plus `HI_MUL_GUARD_LIMBS` = 2 below them (error at most 1 ulp; with no guard
  the whole low limb is wrong, which broke NR's reciprocal from R = 727). The guard limbs' terms are
  linear `add_mul` passes (`hi_guarded_window`), so the core product keeps its unguarded size: one
  limb past a transform size can cost 2×. The static guard is clamped so the split paths' low
  buffers fit N; windows from `out.len() = N − 1` are exact, which static `series` merges rely on.
- **Middle products** take exactly `long[2n−1] × short[n] → out[n]`; callers materialize any low zero
  padding. `ntt_mid_mul_static` also accepts a `long` that passes N by low zero limbs
  (`long = T·B^z`): it convolves T at the unchanged transform size and reads the band z coefficients
  lower, bit-identical to the dynamic product. Both NTT middle products size each prime's transform
  for 2n + 1 coefficients (`mid_ntt_size`), because the accumulation reads two coefficients below
  the band. The FFT band holds the value to about 2^34 units of its lowest limb, which can flip a
  run of zeros or ones: compare middle products by value, never limb by limb.
- **`sqr_mod_bnm1_dyn(a, out)`** squares mod B^m − 1 (m = `out.len()`, `a.len() ≤ m`) into the
  canonical residue in [0, B^m − 1). With m from `bnm1_size(min)` (the smallest m ≥ min whose 4m
  digits are an FFT length), a longer than `mul.sqr_bnm1_fft` and m at most `FFT_BNM1_16BIT_CUTOFF`,
  it is one cyclic FFT (`fft_sqr_mod_bnm1`), half the transform of a full square; otherwise
  `sqr_dyn` and a fold.

## Dispatch

Every dispatcher has a pure predicate at explicit knobs (invariant 6 of `tuning_system.md`):
`dyn_dispatch_at`, `static_dispatch_at` (a `MulKnobs`), `dyn_sqr_dispatch_at`,
`static_sqr_dispatch_at`, `hi_alg`, `dyn_mid_dispatch_at`, `static_mid_ntt`, `bnm1_fft_at`.

**Dynamic products** (`mul_dyn`):

1. s = 1 → `mul_prim`; s = 2 → `mul_prim2`.
2. `is_school` reads the measured table `mul.kara` (change points over (s, l) with a row index,
   `tuning::table::Table2`, `school_at`): schoolbook `mul_buf` or Karatsuba.
3. Past the table, fitted runtime equations are compared per call (`MulEq`, read once;
   `dyn_dispatch` → `dyn_large_dispatch`): recursive Karatsuba against the FFT at the transform size
   it really uses. Past `FFT_16BIT_CUTOFF` (131066 output limbs) the NTT takes the FFT's place in the
   same comparison.

| Equation | Form | Knobs |
|---|---|---|
| Schoolbook | `scale·l·s + shift` | `mul.school_*` |
| Karatsuba, near-square | `a(l^e − (l − s)^e) + q·l·s`, e = log2 3 | `mul.kara_rec_*` |
| Karatsuba, chunking (s ≤ l/2) | `a·l·s^(e−1) + q·l·s` | `mul.chunk_rec_*` |
| FFT | `a·ρ(o)·N·log2 N` at `fft_mul_size` | `mul.fft_scale` |
| `hi_mul_buf`, rows kernel | `scale·W + row·s + shift` | `mul.hi_rows_*` |
| `hi_mul_buf`, columns kernel | `scale·W + col·l + shift` | `mul.hi_cols_*` |

ρ(o) is `FFT_ODD_COST`, the per-point cost of the transform's odd part, hard-coded per architecture
(x86's is unmeasured). W = `hi_window_products(l, s)` = l·s − (s − 2)(s − 3)/2, the limb products of
the window's columns and its two guard columns. `HiEq::poly` gives the high product's equation as a
polynomial in the window for closed-form sums; its readers are the series' equations. Schoolbook's
and `hi_mul_buf`'s equations are prices inside other equations, so their fits keep every shape;
Karatsuba's and the FFT's are fitted on the shapes the table doesn't give to schoolbook.

`mul.kara` on the M1: Karatsuba from 44 limbs for squares and 49 at l = 4s; every row from s = 61
is Karatsuba. The schema default is the old `is_school_at(l, s, 63, 52.0)` boundary. A preset
replaces it, so **tests must find boundaries with `is_school`, never from default values**.
`karatsuba_mul` falls back to schoolbook where its scratch is too short, so any valid table is
correct. `mul.chunking_karatsuba` and `mul.karatsuba` only scale `mul_work` now (the static series
model's product count); `mul.karatsuba` ≥ 3 keeps 3 × 3 on schoolbook, as a Karatsuba level's cross
product there is 3 × 3 again.

**Static products** (`mul_static`): schoolbook → Karatsuba → NTT, no FFT. Karatsuba against the NTT
is `is_karatsuba`'s two ratios (`mul.ntt_karatsuba`, `mul.ntt_chunking_karatsuba`), swept at one N:
approximate, like every static decision.

**Squares.** `sqr_dyn`: `sqr_buf` (≤ `mul.karatsuba_sqr`) → `karatsuba_sqr_entry_dyn`
(≤ `mul.fft_sqr`) → FFT (≤ `FFT_SQR_16BIT_CUTOFF`, 65533 operand limbs) → NTT. `sqr_static`: school →
Karatsuba (≤ `mul.static_ntt_sqr`) → NTT. A square has one input, so its crossovers are sweeps. Both
Karatsuba square entries take one level at any length from 6: at 3 and 5 `karatsuba_sqr_core`'s
cross term doesn't fit `out[half..]`.

**High products.** `hi_mul_buf` runs its columns kernel up to `mul.hi_mul_row` limbs of window and
its rows kernel above (`mul.hi_sqr_row` for squares); the guarded product takes over above the
partial cutoffs.

**Middle products.** Dynamic: `mid_mul_buf` below `mul.fft_mid`, the FFT to `FFT_MID_16BIT_CUTOFF`
(65533 short limbs), then the NTT. Static: `mid_mul_buf` below `mul.ntt_mid`, then the NTT.

## Karatsuba

`karatsuba_core`'s Recurse path adds the part of the cross term that fits `out[half..]` and folds
its carry and the first omitted limb into the one overflow limb (higher omitted limbs are provably
zero), so `find_karatsuba_scratch` reserves nothing for it. `chunking_karatsuba`'s last chunk can be
up to 2s − 2 limbs, which lands on the Chunking/Recurse boundary. Three silent wrong-result bugs
lived here (`fixed_bugs.md`); random operands reach none of them. Any change near this buffer
arithmetic needs structured operands (zero halves, powers of two) against schoolbook `mul_buf`.

## The 16-bit FFT

Real FFT through `rustfft` on 16-bit digits. Details and evidence: `fft_precision.md`.

- **Sizes.** `find_fft_size(n)` is the smallest 4·o·2^a ≥ n with o in {1, 3, 5, 7, 9}. rustfft's
  NEON planner runs only 2^k and 3·2^k through radix-4; other odd parts cost 1.2–1.9× per point, and
  sizes outside this set lose to a slightly longer cheap one.
- **Per-thread cache.** `FFT_CACHE` holds the plans and `fft_twidles(n)`, one exact `cis` table per
  transform size.
- **Stages.** `decompose`/`sqr_decompose` write balanced digits in [−2^15, 2^15) (`balance_limb`; an
  operand's top digit stays unsigned, `fft_absorb_top_carry`, so no transform grows). `recombine`
  and `sqr_recombine` take two conjugate pairs per NEON step (x86 stays scalar). `scale_and_round`
  multiplies by 1/n at power-of-two lengths, bit-identical to the division other lengths keep, and
  returns the call's `FftRounding`. `fft_accumulate_signed` carries the signed coefficients.
- **Checks.** A call whose rounding distance passes `FFT_FOLD_TRIGGER` (0.25) runs the fold check
  (`fft_fold_agrees`) and is redone on the NTT only if that fails, outside the `FFT_CACHE` borrow.
  Products and squares also check residues (`verify.rs`) in debug builds, and in release when
  `FFT_RESIDUE_CHECK` is set (off: 2.55 ns per limb, 5.6–20% of the FFT). Counters:
  `fft_reroutes()`, `fft_fold_checks()`, `fft_fold_reroutes()`, `fft_residue_reroutes()` (should
  stay 0).
- **Entries.** `fft_entry`, `fft_sqr_entry`, `fft_mid_mul`, `fft_sqr_mod_bnm1` are checked. The
  crate-internal `fft_entry_checked`/`fft_sqr_entry_checked` take the fold check as a flag. The
  doc-hidden `*_raw` entries redo nothing and return the `FftRounding` beside the result. A checked
  entry just redoes a wrong FFT, so tests of FFT arithmetic must go through the raw entries.
- **Limits** (`mod.rs`): `FFT_16BIT_CUTOFF` 131066 output limbs, `FFT_SQR_16BIT_CUTOFF` and
  `FFT_MID_16BIT_CUTOFF` 65533, all on the 2^19-digit transform and the largest a compile-time
  assert allows (a balanced coefficient stays under 2^48); `FFT_BNM1_16BIT_CUTOFF` 57344 modulus
  limbs. They bound soundness, not only speed, and are never knobs.

## The NTT

Montgomery arithmetic over three NTT-friendly primes (P1, P2, P3), recombined by CRT. Radix-2, -3
and -5 butterflies, so transform sizes are 5-smooth. Twiddles: `StaticNTTTwidles<N, P>` at compile
time; `DynNTTTwidles<P>` computed at run time and cached per thread by size. Static entries handle
transforms larger than N (N not 5-smooth) through `ntt_chunked_convolution`, which emulates the
declared size with rescaled sub-transforms clamped to N. The three primes' convolutions and the
transforms' internal splits run on rayon behind `mul.ntt_par_conv`, `mul.ntt_par`,
`mul.ntt_par_radix3` and `mul.ntt_par_radix5`, swept through the NTT entries. In debug builds
`mul_dyn` and `sqr_dyn` assert every NTT result against the residue verifier.

## The residue verifier (`verify.rs`)

Linear-time checks of products mod the NTT primes (together about 2^188). `Residues::of(x)` folds x
from its low limb with one Montgomery reduction per limb per prime; from 128 limbs it splits x into
four segments per prime (12 independent chains) recombined by Horner. `of_with_top`, `of_limb`,
`add`, `sub`, `mul` and `shl_limbs` compose identities such as n = q·d + r; `verify_mul` and
`verify_sqr` check c = a·b. Any error 2^s·E with 0 < |E| < P fails for certain; errors don't cancel
periodically as they do mod B − 1. 2.55 ns per limb. It can't check a middle product's band or a
residue mod B^m − 1; tests compare those with the NTT.

## Other

- `powi_dyn_entry`/`powi_static_entry`: `powi_sz` returns (1, 1) for exponent 0; `reverse_pow` holds
  the exponent's rest-bits reversed under a sentinel bit, and the entries' parity check that picks
  the starting buffer depends on it. Change both together.
- Assembly: `mul_asm_x86`/`mul_asm_aarch` are the 64 × 64 → 128 primitives. On x86-64 the row
  kernels have ADX copies behind `rows_dispatch!` (`utils/mod.rs`) and the `base.prim_adx` and
  `base.limb_adx` crossovers.

## Findings not acted on

- **`s == 1` and `s == 2` tiers (i7-8750H, 2026-10-27).** `mul_buf`'s ADX rows beat `mul_prim` from
  l = 15 and `mul_prim2` from l = 7, by up to 37% at 256 limbs. Moving those fixed tiers is a
  dispatch change, not made.
- **A chunking FFT (2026-11-12, emulated).** Cutting the long operand into chunks of 4 to 15 times
  the short side is 1.1–1.6× faster than `mul_dyn` from a short side of 192 limbs under a long side
  8 times it. Nothing at 96 limbs and under; a loss under 4 times. The user's call.
- **Thread-aware FFT/NTT dispatch.** The FFT runs on one thread and the NTT's primes on the pool.
  Past 2^19 digits the 8-thread NTT ties or wins at some sizes, so raising a limit needs this as
  well as a new answer to the 2^48 bound.
- **Middle and high products against a full product (2026-08-09).** A middle product costs
  0.57–0.65 of the full product of its own operands in the schoolbook and NTT ranges, but 1.2–2.3×
  an n × n product in the transform range. A high product saves only in the schoolbook range (0.66–0.69
  of a full product at 16–32 limbs); the guarded path costs one full product.
- The schema's defaults for mul's sweeps are the i7's 2026-10-27 values, measured at
  `find_fft_size`'s old sizes. Any target without a preset runs on them until it is tuned.

The reports these came from (`mul_cutoff_tuning.md`, `multiplication_band_benchmark_report.md`) are
in git history at `fee4f90`, with their harnesses at `faa9971`.
