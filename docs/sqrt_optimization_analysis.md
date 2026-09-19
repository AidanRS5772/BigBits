# Zimmermann square-root optimization analysis

Analyzed on 2026-09-05. Production source was not changed. Experiments used a temporary snapshot of the working tree, including the user's removal of the reciprocal-square-root experiment.

## Findings

Keep the existing Zimmermann recursion and introduce separate exact-remainder, exact-root-only, and bounded-error-root contracts. The largest demonstrated improvement comes from eliminating work at the outermost level whose only purpose is producing or checking the remainder.

For an `n`-limb root, the current full-width core performs one recursive square root, a division with remainder, a square of the low root block, and linear-time correction/reconstruction. It already reuses the radicand buffer for the low square and normalizes only once. Its cost depends substantially on the division and squaring dispatchers; the square-root control flow alone is not the main cost at large sizes.

## Measurements

Exploratory Criterion measurements on the local aarch64 machine, using dynamic arithmetic, full-width normalized radicands of `2n` limbs, eight deterministic inputs per size, 20 samples, 200 ms warm-up, and 500 ms measurement per case. Input cloning and root allocation used Criterion's batched setup. Times below are fitted estimates in microseconds.

| Root limbs | Existing exact root + remainder | Skip unnecessary denormalization | Exact root only | Approximate root, quotient-only division |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.483 | 1.457 | 1.309 | 1.318 |
| 128 | 13.833 | 13.730 | 9.829 | 9.832 |
| 512 | 109.750 | 109.386 | 98.169 | 83.232 |
| 2048 | 686.578 | 684.522 | 632.976 | 565.646 |

Exact root only reduced measured time by about 8–29%; the approximate quotient-only variant reduced it by about 11–29%. These are square-root kernel results, not AGM speedups. Short runs, one architecture, and full-width random inputs do not establish performance for static arithmetic, virtual padding, or adversarial square-boundary inputs.

A copied baseline agreed closely with the production baseline at every size. The approximate variant that retained division-with-remainder performed almost identically to exact root only on these random inputs. Using quotient-only division provided the additional improvement at 512 and 2048 limbs; at 32 limbs its copying/setup cost slightly outweighed its savings.

The existing benchmark alias, run on the same snapshot for the dynamic 128-limb root, measured 13.744 microseconds, consistent with the custom baseline.

## Exact root without an exact remainder

At `src/utils/sqrt.rs:236`, the core always squares `s_lo`, subtracts it, and restores the remainder if correction is required. Only the sign of that subtraction is needed to produce an exact root.

Let `B = 2^64`, `l = s_lo.len()`, and `q = value(s_lo)`. Immediately before the square, call the pre-subtraction remainder `Rpre`. Since `q < B^l`, we know `q^2 < B^(2l)`. Therefore:

```rust
// Outermost full-width stage only, when the caller does not need a remainder.
if x[2 * lo..s_len + 1].iter().any(|&limb| limb != 0) {
    // Rpre >= B^(2*lo) > q*q, so the root requires no correction.
    return;
}
// Otherwise, compute q*q and test whether the candidate needs decrementing.
```

This is a sufficient condition with an exact fallback, not a heuristic. Root denormalization still follows the return from this stage.

The current split, `lo = (s_len - 1) / 2`, is particularly favorable: the high root block is one limb longer for odd root lengths and two limbs longer for even lengths. The relevant high part of `Rpre` spans only two or three limbs. Random inputs commonly pass this test, avoiding the entire outer low-block square.

The virtually padded entry uses a different split. Near its minimum input length, the high and low root blocks can be equal in length, so its fallback frequency must be measured independently. The buffer positions in the example are specifically for the full-width core.

Do not apply this early return at recursive levels: their parent needs the exact remainder. Likewise, an exact-root-only entry can omit final remainder zero-filling, remainder denormalization, and remainder repair after a correction; it still must decrement the root when necessary.

GMP's description independently identifies computing only enough of the final remainder to decide correction as a root-only optimization. [GMP square-root algorithm](https://gmplib.org/manual/Square-Root-Algorithm)

## Bounded-error root variant

The simplest useful approximation is much tighter than losing an entire limb. Keep all recursive stages exact, but at the outermost stage:

1. Compute the exact high root and its remainder.
2. Preserve remainder reduction, quotient saturation, quotient halving, and high-bit insertion.
3. Use quotient-only division for the extension.
4. Omit quotient-remainder restoration, the low-block square, and the final root correction.
5. Shift the root back after normalization; no remainder is returned.

For the untruncated input, the candidate satisfies:

`floor(sqrt(X)) <= candidate <= floor(sqrt(X)) + 1`.

Here `X` is the integer radicand, including any virtual low zero padding. Zimmermann's normalized construction provides the one-unit correction bound; the final even-power-of-two denormalization preserves the integer bound. [GMP square-root algorithm](https://gmplib.org/manual/Square-Root-Algorithm)

At the low-level API this should be an explicit numeric error guarantee. “Only the last limb is inaccurate” is not a safe bitwise promise: adding one can carry through arbitrarily many limbs. Correct rounding requires an error interval and a fallback when that interval crosses a rounding boundary.

The first prototype kept exact division and changed only the outer correction. The quotient-only prototype then used the existing `div_dyn`; its recursive calls continued to use `div_rem_dyn`. Replacing recursive roots with approximate roots is a different algorithm because their remainders would no longer satisfy the current division invariants.

### Limiting the radicand precision

There are two different opportunities:

* To obtain only the high `p` limbs of a much larger integer root, exact truncation obeys `floor(sqrt(A) / B^k) = floor(sqrt(floor(A / B^(2k))))`. Two discarded radicand limbs correspond to one discarded root limb.
* To approximate an `n`-limb root, one can often retain only about `n` significant radicand limbs, then virtually pad the rest. This changes the exact-root contract and needs an error budget.

For the second claim, normalize `A` so `B^(2n)/4 <= A < B^(2n)`, retain its high `n+g` limbs, and zero the bottom `n-g` limbs to obtain `A0`, where `1 <= g <= n`. Then

`0 <= sqrt(A) - sqrt(A0) = (A-A0)/(sqrt(A)+sqrt(A0)) < B^(-g)`.

This derivation assumes the retained window preserves the normalization bound. It shows why the existing virtually padded interface, with `x.len() = s.len() + 1`, is a natural starting point for floating-point approximation. An exact root of the truncated input can differ from the original integer root by one near a square boundary. Combining this input truncation with an uncorrected outer root gives a conservative error of less than two units of the working root's least-significant position, rather than exact rounding.

For AGM, the product may therefore be rounded to working precision plus guards before square root. However, `hi_mul_dyn` currently falls back to a full product of truncated operands above `PARTIAL_MUL_CUTOFF`; when the operands already have the requested precision, simply requesting their high product does not halve the multiplication work. Its truncation error must also be included in the AGM budget. See `src/utils/mul.rs:3591`.

## Other optimizations within the current exact implementation

1. **Skip zero work in denormalization.** At `src/utils/sqrt.rs:169`, return immediately when `sh == 0`. When the discarded root bits `d` are zero, omit `add_mul` and subtraction of `d^2`, while retaining the shifts if `sh != 0`. The existing shifts already handle zero shifts, but `add_mul` still traverses every root limb with a zero multiplier. This is a modest measured gain and particularly relevant to already-normalized floating-point inputs.
2. **Specialize the normalized division interface.** The recursive root `s_hi` has its top bit set. Its division does not need fresh divisor normalization, and the dividend is expendable scratch. A normalized, destructive quotient-only interface could avoid `div_dyn`'s preserving copy in its Knuth path. Keep backend selection: forcing Knuth at large sizes would defeat the fast arithmetic. Benchmark the actual square-root quotient/divisor shapes before changing dispatch cutoffs.
3. **Inspect squaring transitions.** Dynamic squaring currently goes from schoolbook directly to FFT above 33 limbs, whereas static squaring includes Karatsuba. The outer square is a large part of the 128-limb-root cost. A dynamic Karatsuba square tier is a tuning candidate, not a demonstrated win; removing unnecessary outer squares is already measured.
4. **Optimize the small seed if small sizes matter.** In `sqrt_4x2`, `rem / s_hi` is in `{0,1,2}` because `rem <= 2*s_hi`; comparisons and subtraction can replace that quotient/remainder calculation. Verify generated code and measure before replacing the other division. The `/` and `%` expressions do not by themselves prove that the compiler emits two divisions.
5. **Retune around the actual cutoff.** `ZIMMERMAN_SQRT_CUTOFF` is 17. Several tests named for cutoff boundaries still use 49/50/51. Add 16/17/18 and odd/even cases when implementing changes. Existing cutoff probes compare one descent with binomial arithmetic, so root-only and approximate entries may need separate top-level decisions. Generic callbacks could also be compared with `dyn FnMut`, but this is a lower priority than arithmetic elimination and may already benefit from LTO.

## Suitability for AGM

Yes: a bounded-error Zimmermann root is suitable inside AGM when computed at the final working precision plus guards.

For positive arguments, the arithmetic mean and geometric mean propagate bounded relative input errors without the cancellation of opposite signs. Local arithmetic errors accumulate over iterations. If each iteration contributes at most approximately `C * 2^(-w)` relative error, a useful budgeting model after `K` iterations is `O(C*K*2^(-w))`. Thus `w = p + ceil(log2(C*K)) + margin` is a starting point for a `p`-bit target; the actual bound must include multiplication, square root, rounding, and the final application formula. This budgeting model is an analysis recommendation, not a proved bound for the current floating-point code.

If the approximation sacrifices an entire low limb, budget those 64 bits in addition to the accumulation guards. The outer-only variant above loses at most one integer unit, so it is substantially tighter. A guard limb is generous for many practical iteration counts, but it does not prove correct rounding near a boundary.

The ordinary AGM recurrence is not self-correcting. Increasing precision according to how many bits `a` and `b` currently share can make both converge closely to the wrong value. A common relative perturbation persists because `AGM(lambda*a, lambda*b) = lambda*AGM(a,b)`. Full working precision is needed from the beginning; low precision is appropriate for explicitly small correction terms whose error has been analyzed. [Brent and Zimmermann, Modern Computer Arithmetic, §4.8.4](https://members.loria.fr/PZimmermann/mca/mca-cup-0.5.9.pdf#page=178)

The dormant `ln` implementation at `src/bit_nums/bitfloat_static.rs:540` follows the AGM recurrence and chooses precision from the agreement of the two iterates. That schedule should be reconsidered before reuse. The floating-point files are currently absent from `src/bit_nums/mod.rs`, so they are not wired to the current utils square root.

There is also a useful late-stage AGM optimization. Let `m = (a+b)/2` and `d = (a-b)/2`. Once `|d/m|` is sufficiently small,

`AGM(a,b) = m - d^2/(4*m) + O(d^4/m^3)`.

This can replace the final square-root iterations with a small correction. MPFR documents the equivalent correction `(a-b)^2/(16*m)` and a termination criterion based on approximately one-quarter of the target bits already agreeing; it evaluates the small correction at reduced precision. Port its constants and error analysis if correct rounding is required. This formula approximates the AGM limit, not just the next geometric mean. [MPFR Algorithms, §4.29](https://www.mpfr.org/algorithms.pdf#page=44)

For a Gauss–Legendre pi implementation, the weighted correction sum also needs an error budget. For logarithms, subtraction of the scale correction can lose precision near one. A test based only on `a == b` or a tiny gap does not account for accumulated error.

## Correctness issue found during analysis

The current core can panic when the division numerator is zero or has nonzero limbs only below `lo`:

```rust
let mut x = vec![0_u64; 34];
x[33] = 1 << 62;
let mut root = vec![0_u64; 17];
zimmerman_sqrt_dyn(&mut x, &mut root);
```

This is an exact power-of-two square. The observed panic is “slice index starts at 8 but ends at 0.” At `src/utils/sqrt.rs:225`, `buf_len(&x[..s_len + lo])` can return a length smaller than `lo`, making `x[lo..d_len]` invalid. Slice the numerator window first and trim within it, allowing the division preflight to handle an empty numerator. This fix and a regression test should precede relying on the kernel in AGM.

## Validation and reproduction

The working-tree `cargo test test_sqrt` initially failed to compile because `sqrt.rs` imported the benchmark-gated root re-export `hi_mul_buf`, and multiplication tests referenced the old `HI_MUL_CUTOFF` and `HI_SQR_CUTOFF` names. In the temporary snapshot only, the unused sqrt import was removed and those test identifiers were changed to the existing `PARTIAL_*` names. No arithmetic changes were made to the reference implementation.

* All 11 existing square-root tests passed in the snapshot.
* Five prototype modes passed 1,920 output comparisons over 384 deterministic inputs, including non-normalized radicands and values at or adjacent to perfect squares. These comparisons validate the sampled cases, not the complete proposed API.
* The additional power-of-two-square regression failed as described above.
* The custom Criterion benchmark measured full-width dynamic inputs only. Static and virtually padded candidate variants remain unmeasured.

Temporary snapshot: `/tmp/bigbits-sqrt-analysis-q7o48nv3`. The prototype is `benches/sqrt_analysis.rs` in that snapshot. It intentionally contains measurement-only copies and wrappers. Raw Criterion estimates are under its `target/criterion/sqrt_analysis/` directory. These temporary artifacts may be removed by the operating system.

Commands run in the snapshot:

```sh
cargo test test_sqrt --offline --target-dir /Users/aidansgarlato/Programing/Rust/BigBits/target/sqrt-analysis
cargo test --offline --test sqrt_edge_analysis --target-dir /Users/aidansgarlato/Programing/Rust/BigBits/target/sqrt-analysis
cargo bench --offline --bench sqrt_analysis --target-dir /Users/aidansgarlato/Programing/Rust/BigBits/target/sqrt-analysis
cargo bench-utils --offline --target-dir /Users/aidansgarlato/Programing/Rust/BigBits/target/sqrt-analysis -- 'zimmermann_sqrt/aarch64/zimmermann_dyn/128' --quick
```
