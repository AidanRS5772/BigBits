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

# Run benchmarks (requires nightly or feature flag)
cargo bench
cargo bench --features _bench_internals

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

- **`mul.rs`** — Multiplication with dynamic dispatch: schoolbook → Karatsuba → FFT (via `rustfft`) → NTT. Assembly primitives (`mul_prim_asm`, `mul_asm_x86`, `mul_asm_aarch`) handle 64×64→128-bit multiply.
- **`div.rs`** — Knuth normalized division (`div_buf_of`), primitive single-limb divide (`div_prim`), Burnikel-Ziegler divide-and-conquer.
- **`utils.rs`** — Buffer helpers: `trim_lz`, `add_buf`, `sub_buf`, `cmp_buf`, `eq_buf`, `combine_u64`, etc.
- **`mod.rs`** — Algorithm cutoff constants (e.g., `KARATSUBA_CUTOFF: 21`, `NTT_CUTOFF: 20`, `BZ_CUTOFF: 64`) and `ScratchGuard`, a thread-local RAII scratch-buffer pool that reuses allocations across recursive calls.

### Algorithm dispatch in `mul_vec`

Selection is driven by the cutoff constants in `src/utils/mod.rs`:
1. Small inputs → schoolbook
2. Balanced inputs → Karatsuba (with chunking for unbalanced sizes)
3. Large inputs → FFT-based (real FFT via `rustfft`)
4. Modular-suitable → NTT (recently implemented, currently being integrated)

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
- `test_mul.rs` — Tests `mul_prim`, `mul_buf`, `mul_vec`, squaring; includes boundary-finder utilities for locating algorithm transition points
- `test_div.rs` — Tests `div_prim`, `div_buf_of`, BZ division
- `test_utils.rs` — Tests all buffer utility functions

**Known source bugs** (do not fix without explicit instruction): `sqr_arr` OOB, `sqr_buf` loop bound, `div_buf_of` OOB, `bz_div_init` formula, recursive BZ odd-length divisor. Tests that hit these are expected to fail at runtime; this documents real source bugs.

### Feature flag

`_bench_internals` exposes internal `utils` functions for Criterion benchmarks. Do not use in non-bench code.
