# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a single test module
cargo test test_utils
cargo test test_mul
cargo test test_div

# Run a single test by name
cargo test test_utils::trim_lz_basic

# Run benchmarks (aliases defined in .cargo/config.toml)
cargo bench-utils     # runs utils_bench (requires _bench_internals)
cargo bench-cutoffs   # runs cutoffs_bench (requires _bench_internals)
# Or directly:
cargo bench --features _bench_internals --bench utils_bench
cargo bench --features _bench_internals --bench cutoffs_bench

# Build with profiling info (debug symbols, thin LTO, unwind)
cargo build --profile prof

# Test coverage (requires cargo-llvm-cov)
cargo llvm-cov --html --include-pattern 'src/utils' --open
```

## Architecture

BigBits is a big-integer/big-float library optimized with inline assembly (x86-64 and ARM64) and adaptive algorithm selection.

### Layer structure

```
src/utils/       ← raw buffer arithmetic (Vec<u64> limb arrays)
src/bit_nums/    ← ergonomic number types wrapping utils
src/tests/       ← test suite (gated by #[cfg(test)] in lib.rs)
benches/         ← Criterion benchmarks
```

### Utils layer (`src/utils/`)

All algorithms operate on `Vec<u64>` limb arrays (little-endian: index 0 is the least significant limb).

- **`mul.rs`** — Multiplication with dynamic dispatch: schoolbook → Karatsuba → FFT (via `rustfft`) → NTT. Assembly primitives (`mul_prim_asm`, `mul_asm_x86`, `mul_asm_aarch`) handle 64×64→128-bit multiply. Also provides "short"/"middle-product" variants (`short_mul_buf/dyn/static`, `mid_mul_buf/dyn`) that compute only a limb-range of a full product — these back the Newton-Raphson reciprocal refinement in `div.rs`.
- **`div.rs`** — Three division strategies, layered:
  1. `div_buf_of` — Knuth normalized schoolbook division (with `div_prim` for single-limb divisors).
  2. `div_2_1`/`div_3_2`/`bz_div_dyn`/`bz_div_static` — Burnikel-Ziegler divide-and-conquer, falling back to Knuth below `BZ_CUTOFF`.
  3. `nr_div_dyn`/`nr_rcp_dyn` — Newton-Raphson reciprocal division: seeds a precision-1 reciprocal with a tiny Knuth division (`div_buf_of` on the top 3 limbs of `d`), then follows the backward ceiling-halving schedule from `nr_rcp_schedule` — each `nr_rcp_refine_step_dyn` either doubles precision or doubles-minus-one-limb (`trunc` flag), so the chain lands exactly on the requested precision with no oversized final iteration. Refinement uses the middle-product primitives from `mul.rs` (`bz_rcp_seed_dyn` remains only as the reseed fallback when a refine step bails), finishing with a guard-limb quotient correction (`nr_rem_correction`).

  This is under active rework (see recent commits on the `nr_opt` branch). The top-level dispatch entry points `div_vec`/`div_arr` are currently dummy stubs (`Vec::new()`/`[0; N]`) — `UBitInt::div_rem` calls `div_vec` and will not produce correct results until dispatch is wired up. Exercise the algorithms directly (`div_buf_of`, `bz_div_dyn`, `nr_div_dyn`, etc.) as `src/tests/test_div.rs` does, not through the bit_nums layer.
- **`utils.rs`** — Buffer helpers: `trim_lz`, `add_buf`, `sub_buf`, `cmp_buf`, `eq_buf`, `combine_u64`, etc.
- **`mod.rs`** — Algorithm cutoff constants and `ScratchGuard`, a thread-local RAII scratch-buffer pool that reuses allocations across recursive calls (`get_splits` carves one acquired buffer into several disjoint mutable slices). Key constants: `KARATSUBA_CUTOFF: f64 = 17.0`, `FFT_KARATSUBA_CUTOFF: f64 = 1.92`, `FFT_16BIT_CUTOFF: usize = 1<<16`, `BZ_CUTOFF: usize = 88`. NTT parallelization has four separate cutoffs (`NTT_PAR_CUTOFF_NTT_CONV/_NTT/_NTT_3/_NTT_5`) for its different radix paths. Many NTT/squaring cutoffs are marked `// GUESS` and are candidates for tuning via `cutoffs_bench`.

### Algorithm dispatch in multiplication

There are two dispatch paths, each with different algorithm sets:

**Dynamic path** (`mul_dyn` → `mul_vec`): used by heap-allocated `UBitInt`/`BitInt`.
1. `s == 1` → `mul_prim` (single-limb, asm)
2. `s == 2` → `mul_prim2` (two-limb 128-bit, asm)
3. Small → `mul_buf` (schoolbook, asm inner loop)
4. Medium → Karatsuba (recursive with chunking for unbalanced)
5. `output_limbs ≤ 2^16` → FFT (real FFT via `rustfft`, 16-bit decomposition to stay within f64 precision)
6. Large → NTT

**Static path** (`mul_static<N>` → `mul_arr<N>`): used by `UBitIntStatic<N>`/`BitIntStatic<N>`.
Same tiers except FFT is skipped; jumps directly from Karatsuba to NTT.

The `is_school` and `is_karatsuba` boundary functions use the float cutoffs (`KARATSUBA_CUTOFF`, `FFT_KARATSUBA_CUTOFF`, etc.) to compute 2D boundaries over `(long_len, short_len)` space — not simple size thresholds.

### NTT implementation

NTT uses Montgomery modular arithmetic over three NTT-friendly primes (P1, P2, P3 in `mul.rs`), then reconstructs the true product via CRT. This avoids precision loss for large multiplications.

Twiddle factor tables come in two variants:
- `StaticNTTTwidles<N, P>` — compile-time fixed size, used by the static path
- `DynNTTTwidles<P>` — runtime-computed, cached in thread-local `HashMap<usize, DynNTTTwidles<P>>`

NTT supports radix-2, radix-3, and radix-5 butterflies to handle transform sizes that are smooth 5-smooth numbers.

### BitNums layer (`src/bit_nums/`)

| Type | Description |
|---|---|
| `UBitInt` | Unsigned arbitrary-precision (`Vec<u64>`) |
| `BitInt` | Signed arbitrary-precision (magnitude + sign bit) |
| `UBitIntStatic<N>` | Fixed-size unsigned (stack, no heap) |
| `BitIntStatic<N>` | Fixed-size signed |
| `BitFloat` | Arbitrary-precision float (mantissa + exponent) |
| `BitFloatStatic<N>` | Fixed-precision float |
| `BitFrac` | Rational arithmetic |

`traits.rs` defines shared traits (`U`, `I`, `Sqr`, `DivRem`, `LogI`, `PowI`) and macros (`impl_commutative`, `impl_commutative_div_rem`) for operator boilerplate.

### Test suite (`src/tests/`)

- `mod.rs` — Shared helpers: `rand_vec`, `rand_nonzero_vec`, `to_u128`
- `test_mul.rs` — Tests `mul_prim`, `mul_buf`, `mul_vec`, squaring, short/middle-product multiply; includes boundary-finder utilities for locating algorithm transition points
- `test_div.rs` — Tests `div_prim`, `div_buf_of`, BZ division (including forced top-block path and cost-model monotonicity checks), and the Newton-Raphson reciprocal/division path
- `test_utils.rs` — Tests all buffer utility functions

**Known source bugs** (do not fix without explicit instruction):
- `karatsuba_entry_dyn`'s chunking path panics (`lhs must be longer then rhs` from `add_buf`) for certain unbalanced long/short input sizes — see `test_karatsuba_entry_unbalanced` in `test_mul.rs` (currently failing).
- `powi_vec(&[3], 1)` returns `[9]`; expected `[3]`. Suspected cause: `reverse_pow` encodes the exponent one step too high, so the powi core squares for exponent 1. Regression documented via `#[should_panic]` in `test_powi_non_identity_regression`.
- `powi_vec(&[3], 0)` panics with an index-out-of-bounds write; expected `[1]`. Suspected cause: `powi_sz` returns max size `0`, so `powi_vec` allocates an empty output before `powi_dyn_entry` writes `out[0]`. Regression documented via `#[should_panic]` in `test_powi_vec_zero_regression`.

Run `cargo test --lib` to see current pass/fail status — as of this writing only `test_karatsuba_entry_unbalanced` fails unexpectedly; the powi issues are wrapped in `#[should_panic]` so they show as passing.

### Compile-time constraints

Avoid `u128` arithmetic in `const fn` contexts — it significantly increases compile times. Use `u64` or split operations instead.

### Feature flags

- `_bench_internals` — exposes internal `utils` functions for Criterion benchmarks; do not use in non-bench code
- `_fft_cutoffs` — gates FFT-specific cutoff paths (used during cutoff-tuning benchmarks)
