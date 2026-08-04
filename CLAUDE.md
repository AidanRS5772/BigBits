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

All algorithms operate on little-endian `u64` limb buffers (index 0 is the least significant limb). Dynamic entries use pooled `Vec<u64>` scratch storage; static entries use caller-selected `[u64; N]` stack storage.

- **`mul.rs`** — Multiplication with dynamic dispatch: schoolbook → Karatsuba → FFT (via `rustfft`) → NTT. Assembly primitives (`mul_prim_asm`, `mul_asm_x86`, `mul_asm_aarch`) handle 64×64→128-bit multiply. Also provides "short"/"middle-product" variants (`short_mul_buf/dyn/static`, `mid_mul_buf/dyn`) that compute only a limb-range of a full product — these back the Newton-Raphson reciprocal refinement in `div.rs`. The dyn and static middle products accept a `long` shorter than `2*short-1`: it is treated as top-aligned with implicit low zero limbs (the band stays at the virtual positions), which lets division pass divisor top-slices without materializing pad copies. Static NTT entries handle transforms larger than `N` (non-5-smooth `N`) via `ntt_chunked_convolution`, which emulates the declared transform size with rescaled sub-transforms clamped to `N`.
- **`div.rs`** — Division exposes standard quotient-only, quotient-with-remainder, reciprocal, and short quotient-only operations in dynamic and static allocation models. Quotient operations dispatch among Knuth, Burnikel-Ziegler, and Newton-Raphson; reciprocal operations dispatch directly between Knuth and Newton-Raphson and have no standalone Burnikel-Ziegler path. Forced algorithm entry harnesses live in `src/tests/test_div.rs` and `benches/`, where they compose the public preflight and wrapper stages; they do not live in the algorithm module.

  Division operands must be trimmed and `d` must be nonzero. Let `M = div_quotient_len(n.len(), d.len()) = n.len().saturating_sub(d.len())`. Standard `div`/`div_rem` require `q.len() >= M`, write the low `M`-limb quotient body, and return the possible structural limb at index `M`. When `q.len() > M`, that limb is stored in `q[M]`, the remaining tail is zero-filled, and the function returns zero, matching multiplication's overflow convention. Equal-width operands therefore permit an empty quotient body while returning their complete single-limb quotient. Quotient-only entries preserve both operands, and standard `div_rem` always leaves the exact remainder in `n`.

  Short quotient entries `short_div_dyn` and `short_div_static<N>` preserve the former high-window capability without a remainder variant. For `q.len() = K < M`, they return the same structural overflow and write body digits `M-K..M-1` into `q`; when `K >= M`, they delegate to standard division. Short preflight drops irrelevant low numerator limbs and exact whole-limb divisor factors before the rigid backend. Newton-Raphson keeps its contiguous `K+1` internal quotient and bridges exact-width/short callers through pooled dynamic or `[u64; N]` static temporary storage; an output with an overflow slot is a direct fast path.

  Each standard call validates and prepares operands, dispatches using the logical quotient width `M+1`, and then runs a backend with a quotient body of width `M`. Knuth returns its initial top estimate directly. Burnikel-Ziegler returns the top digit from its initial block while later blocks fill the body. Newton-Raphson retains its contiguous full-width core and BZ correctness fallback. Tests and benchmarks force an algorithm by locally composing the same preflight and wrapper stages.

  Reciprocal entries preserve `d`, and `rcp.len()` is the requested precision. For `D = d.len()`, `R = rcp.len()`, `B = 2^64`, and `X = min(floor(B^(D + R - 1) / d), B^R - 1)`, the returned approximation has numeric error strictly below `B`; the least-significant output limb is intentionally not guaranteed exact. Reciprocal wrappers remove exact low-zero limb factors and retain only the high divisor window relevant to `R`, so work does not scale with irrelevant low divisor limbs. The NR reciprocal wrapper uses two internal low guard limbs and discards them, but omits the full `d * rcp` verification product and bounded final correction pass. Division's private upward-biased Knuth reciprocal seed and quotient/remainder corrections remain exact.

  Burnikel-Ziegler still uses Knuth for its recursive base cases. Newton-Raphson division falls back to Burnikel-Ziegler when an error bound, bounded exact-correction step, or static-capacity check fails; the NR reciprocal wrapper falls back to Knuth when refinement, normalization, or static capacity cannot produce a usable approximation. These are correctness fallbacks, not top-level performance dispatch.

  The public dispatchers are `div_dyn`, `div_rem_dyn`, `short_div_dyn`, `rcp_dyn`, and their static counterparts. Standard and short quotient operations use the shared `BZ_CUTOFF` for the Knuth-to-Burnikel-Ziegler boundary. Division selection measures the effective rigid divisor width after its exact low-zero limb factor and chooses BZ directly when the requested rigid shape cannot enter NR; its dynamic and static BZ-vs-NR cost models remain separately tunable. Reciprocal selection depends only on `R = rcp.len()` and switches directly from Knuth to NR after `DYN_RCP_KNUTH_NR_CUTOFF` or `STATIC_RCP_KNUTH_NR_CUTOFF`. Number-type callers allocate the quotient or reciprocal buffer and call one of these entries directly; there are no allocating `div_vec`/`div_arr` adapters.

  The low-level building blocks remain `div_prim`/`div_buf_of` for normalized Knuth division, `div_2_1`/`div_3_2` for Burnikel-Ziegler recursion, and the shared `nr_err_band`/`nr_refine_rcp`/`nr_rcp_chain`/`nr_refine_quo` correction pipeline for Newton-Raphson. Dynamic drivers provide `ScratchGuard` buffers and dynamic multiplication closures; static drivers provide `[u64; N]` buffers and `*_static::<N>` multiplication closures.
- **`utils.rs`** — Buffer helpers: `trim_lz`, `add_buf`, `sub_buf`, `cmp_buf`, `eq_buf`, `combine_u64`, etc.
- **`mod.rs`** — Algorithm cutoff constants and `ScratchGuard`, a thread-local RAII scratch-buffer pool that reuses allocations across recursive calls (`get_splits` carves one acquired buffer into several disjoint mutable slices). `BZ_CUTOFF` is the single shared Knuth/BZ boundary and also controls BZ's recursive Knuth base case; `BZ_TOP_PADDED_COST_SCALE` tunes BZ's top block. Division and division-with-remainder retain independent dynamic/static BZ-vs-NR cost-model families. Reciprocal dispatch instead has one integer BZ/NR precision cutoff per allocation model. Multiplication and NTT parallelization retain their separate cutoff families. Constants marked `// GUESS` need measurement via `cutoffs_bench`.

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
- `test_div.rs` — Tests `div_prim`/`div_buf_of`, the forced dynamic/static entry matrix, top-level dispatch, Burnikel-Ziegler block behavior, Newton-Raphson fallbacks, and the bounded reciprocal-precision invariant
- `test_utils.rs` — Tests all buffer utility functions

**Previously-known source bugs, now fixed**:
- `karatsuba_entry_dyn`'s chunking path panic (`lhs must be longer then rhs` from `add_buf`) — **fixed**. Root cause: `chunking_karatsuba`'s last-chunk remainder can be up to `2*short.len()-2` limbs, landing at the exact Chunking/Recurse dispatch ratio boundary. The resulting `karatsuba_core` call needs its output buffer to be `>= 3*half+1` limbs (`half = ⌈a.len()/2⌉`) for its cross-term reconstruction (`add_buf(&mut out[half..], &cross)`), which exceeds the standard `a.len()+b.len()-1` sizing convention by 1-2 limbs whenever `b.len()` is only just above `half` — a generic gap in `karatsuba_core`'s Recurse path, not specific to chunking. Fixed by having `karatsuba_core` redirect through a scratch-backed buffer sized exactly `3*half+1` when `out` is too small, copying back only the caller's limbs and folding the one legitimate overflow limb (provably ≤ 1, by the basic a.len()+b.len() product-size bound) into the return; `find_karatsuba_scratch`'s Recurse branch reserves the extra space unconditionally since it can't see the caller's actual `out.len()`. Fixing this exposed a second, independent, previously-latent bug: `karatsuba_mul`'s `Prim`/`Prim2` dispatch arms called `mul_prim`/`mul_prim2` on the *full* `out` slice rather than truncating to `long.len()`, so any caller passing a wider `out` (which the sizing fix above newly does, for nested `a1×b1` sub-calls) fed whatever stale scratch-pool garbage sat beyond `long.len()` into the multiply as if it were part of the multiplicand — silent corruption, not a panic, and only reachable when the scratch pool happened to hold nonzero leftover data from an earlier call. Fixed by truncating to `long.len()` before the multiply and explicitly zero-filling/folding any excess. Regression tests: `test_karatsuba_entry_unbalanced`, `test_karatsuba_core_recurse_boundary_sizing` (engineers the exact last-chunk boundary ratio), `test_karatsuba_entry_unbalanced_random_sweep`, `test_karatsuba_mul_prim_dispatch_ignores_stale_out_tail` (poisons `out`'s tail explicitly). Verified via an independent Python differential-fuzz model (tens of thousands of trials) before porting to Rust — worth doing again for any future change near `karatsuba_core`'s buffer arithmetic, since two of these three bugs produced *silently wrong results*, not panics, and are easy to reintroduce with a plausible-looking size-only fix.
- `powi_vec(&[3], 0)` — **fixed**. Root cause confirmed: `powi_sz` returned `(0, 0)` for `pow == 0`, so `powi_vec` allocated an empty `Vec` and `powi_dyn_entry`'s `out[0] = 1` indexed out of bounds. Fix: `powi_sz` special-cases `pow == 0` to return `(1, 1)`.
- `powi_vec(&[3], 1)` — **fixed**. Root cause was more precise than "one step too high": bit 0 of `reverse_pow(pow)` was *always* 1 for every `pow` (an artifact of `pow`'s own leading bit leaking into the construction), forcing an unwanted multiply on the first step of every call, with the real exponent bits all landing one position higher than intended. Fixed by rebuilding `reverse_pow` directly from `pow`'s rest-bits (`pow - (1 << pow.ilog2())`) reversed into a `k`-bit field with a sentinel at position `k`, instead of the old shift-and-reverse-all-64-bits trick. This changes how many src/dst swaps `powi_dyn_core`/`powi_static_core` perform, so the parity check in `powi_dyn_entry`/`powi_static_entry` that decides which buffer starts as the answer flipped too (`% 2 == 0` → `% 2 == 1`) — the two changes only work together. Verified exhaustively (`test_powi_vec_small_exponents_exhaustive`) against a repeated-multiplication oracle across every exponent bit-length/popcount combination up to 20.

Run `cargo test --lib` to see current pass/fail status. A `test_ntt_sqr_entry_dyn_parallel_branch` failure under full-suite parallelism is a known flake; rerun it alone before suspecting a regression.

### Compile-time constraints

Avoid `u128` arithmetic in `const fn` contexts — it significantly increases compile times. Use `u64` or split operations instead.

### Feature flags

- `_bench_internals` — exposes internal `utils` functions for Criterion benchmarks; do not use in non-bench code
- `_fft_cutoffs` — gates FFT-specific cutoff paths (used during cutoff-tuning benchmarks)
