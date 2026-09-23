# BigBits

> [!WARNING]
> **BigBits is under active development and is not production-ready.**
> The crate is pre-1.0 (`v0.1.0`) and is not published on crates.io. Its public API is unstable and may change without notice. Some number types are still being built. Please don't depend on it for production work yet.

BigBits is a high-performance arbitrary-precision arithmetic library for Rust. It aims to match or beat established big-number libraries by combining two things. The first is hand-written inline assembly for **x86-64** and **AArch64**. The second is adaptive algorithm selection, where every cutoff comes from benchmarks. You can keep numbers on the heap (dynamically sized) or on the stack (fixed size, no allocation).

## Goals

- **Speed.** Use the fastest known algorithm for each operand size and shape, with architecture-specific assembly in the hot inner loops.
- **Choice of memory model.** Every core operation has a *dynamic* path (heap `Vec<u64>` with pooled scratch space) and a *static* path (`[u64; N]` on the stack, no heap allocation).
- **Ergonomics.** Number types that work with Rust's standard operators and conversions, while still giving access to the low-level buffer routines.
- **Correctness.** A broad test suite, differential testing against reference models, and exhaustive checks at tricky algorithm boundaries.
- **Cutoffs from data.** Algorithm switch points come from Criterion benchmarks and profiling. The reports are in [`docs/`](docs/).

## Features

### Number types

| Type | Description | Status |
| --- | --- | --- |
| `UBitInt` | Unsigned arbitrary-precision integer (heap) | Available |
| `BitInt` | Signed arbitrary-precision integer (heap) | Available |
| `UBitIntStatic<N>` | Fixed-width unsigned integer (stack, `N` limbs) | Available |
| `BitIntStatic<N>` | Fixed-width signed integer (stack, `N` limbs) | Available |
| `BitFloat` | Arbitrary-precision floating point | In progress |
| `BitFloatStatic<N>` | Fixed-precision floating point | In progress |
| `BitFrac` | Arbitrary-precision rational numbers | In progress |

The shared traits live in `src/bit_nums/traits.rs`. They include `Sqr`, `DivRem`, `PowI`, `LogI`, `Abs` and `Rounding`.

### Multiplication

The dispatcher picks among these tiers using 2D boundaries over the `(long_len, short_len)` shape of the operands, not a single size threshold:

1. **Single- and two-limb products** use 64×64→128-bit assembly primitives.
2. **Schoolbook** multiplication uses an assembly inner loop.
3. **Karatsuba** is recursive, and it splits unbalanced operands into chunks.
4. **FFT** is a real-valued FFT via `rustfft` (dynamic path only).
5. **NTT** uses Montgomery arithmetic over three NTT-friendly primes, recombines the results with CRT, and supports radix-2/3/5 butterflies. Large transforms run in parallel with Rayon.

Squaring, high products and middle products are also available. The middle products drive Newton–Raphson refinement.

### Division

- **Knuth** long division for small operands.
- **Burnikel–Ziegler** recursive division for medium operands.
- **Newton–Raphson** reciprocal-based division for large operands. If it can't produce a result, it falls back to Burnikel–Ziegler.
- The available operations are quotient-only, quotient with remainder, high-quotient windows (only the top limbs of the quotient) and reciprocal approximation. Each has a dynamic and a static entry point.

### Square root

- **Binomial** square root and **Zimmermann** (Karatsuba) square root.
- Three output modes: root with remainder, root only, and approximate root. Each has dynamic and static variants.

### Infrastructure

- `ScratchGuard` is a thread-local, RAII-managed scratch-buffer pool. It reuses allocations across recursive calls.
- Twiddle factors are cached: fixed at compile time for static NTTs and computed at runtime for dynamic ones.

## Project layout

```
src/utils/      Raw limb-buffer arithmetic (mul, div, sqrt, helpers, cutoffs)
src/bit_nums/   Ergonomic number types built on utils
src/tests/      Test suite
benches/        Criterion benchmarks and profiling harnesses
docs/           Cutoff-tuning and profiling reports
```

## Building and testing

```bash
cargo build
cargo test --lib

# Benchmarks (these need the internal `_bench_internals` feature)
cargo bench-utils
cargo bench-cutoffs
```

## Trying it out

BigBits isn't on crates.io yet. To experiment with it, add it as a git dependency:

```toml
[dependencies]
big_bits = { git = "https://github.com/AidanRS5772/BigBits" }
```

Expect breaking changes between commits until the API is stabilized.

## Roadmap

- [ ] Finish `BitFloat`, `BitFloatStatic<N>` and `BitFrac` and include them in the crate
- [ ] Stabilize and document the public API
- [ ] Keep re-tuning algorithm cutoffs across more hardware
- [ ] Add usage examples and API documentation
- [ ] Publish to crates.io

---

**BigBits is still under development.** Feedback, bug reports and issues are welcome.
