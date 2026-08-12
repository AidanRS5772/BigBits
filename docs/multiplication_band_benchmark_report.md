# Full, high, and middle multiplication benchmark report

Date: 2026-08-09

## Executive summary

This benchmark compares the production dynamic-dispatch implementations of a
full multiplication, a high multiplication, and a middle multiplication at a
target precision of `n` limbs. It now includes two full-product baselines:

- an `n` by `n` product for comparison with the usual `M(n)` notation;
- a same-operands `(2n-1)` by `n` product for comparison with the exact inputs
  consumed by `mid_mul_dyn`.

- `hi_mul_dyn` is meaningfully faster than `mul_dyn` only in the smaller
  schoolbook range. It costs 0.657 times a full multiplication at 16 limbs and
  0.686 times at 32 limbs, but the advantage falls to 0.5% by 89 limbs.
- The original high-product run used `HI_MUL_CUTOFF = 114`. At that cutoff the
  specialized high product was 4.4% slower than the full product. The working
  tree has since changed the cutoff to 90 limbs, so the 114-limb high-product
  row records the old dispatch rather than the current one.
- The current middle-product implementation is not generally cheaper than an
  `n` by `n` full multiplication. Its ratio ranges from 1.03 at 8 limbs to
  2.33 at 16,384 limbs, with a few sizes near 114 limbs approximately equal.
- Against a full multiplication of the same `(2n-1)`- and `n`-limb operands,
  the middle product is substantially cheaper in the schoolbook range: 0.582
  times full at 8 limbs and 0.565 times full at 16 limbs.
- The same-operands transform comparison is irregular. Middle multiplication
  is 11–26% slower from 128 through 16,384 limbs, but becomes 24–39% faster as
  the wide product approaches or crosses its FFT/NTT boundary. At 32,769 limbs
  it costs 0.655 times the complete wide product, close to a `2/3` ratio.

These are measurements of the current implementations, not the theoretical
complexity constants of ideal truncated products. Transform-size selection
causes large discontinuities, so no single ratio describes every precision.

## Same-operands middle-product comparison

This is the direct comparison requested for the middle product. Both calls use
the same `long[2n-1]` and `short[n]` operands:

- full wide: `mul_dyn(long, short, out[3n-2])`;
- middle: `mid_mul_dyn(long, short, out[n])`.

All times are Criterion mean point estimates in microseconds. `Middle / wide`
is the timing ratio, and `Saved` is `1 - Middle / wide`. A negative saving
means the middle product was slower.

| `n` limbs | Full `(2n-1) × n` (µs) | Middle (µs) | Middle / wide | Saved |
|---:|---:|---:|---:|---:|
| 8 | 0.139 | 0.081 | 0.582 | 41.8% |
| 16 | 0.537 | 0.303 | 0.565 | 43.5% |
| 32 | 1.833 | 1.152 | 0.629 | 37.1% |
| 64 | 5.755 | 4.478 | 0.778 | 22.2% |
| 89 | 9.653 | 8.668 | 0.898 | 10.2% |
| 90 | 9.884 | 7.490 | 0.758 | 24.2% |
| 114 | 17.906 | 7.365 | 0.411 | 58.9% |
| 115 | 17.192 | 7.376 | 0.429 | 57.1% |
| 128 | 9.716 | 11.040 | 1.136 | -13.6% |
| 256 | 21.155 | 25.914 | 1.225 | -22.5% |
| 1,024 | 104.481 | 116.036 | 1.111 | -11.1% |
| 4,096 | 485.877 | 577.435 | 1.188 | -18.8% |
| 16,384 | 2,486.852 | 3,145.098 | 1.265 | -26.5% |
| 21,846 | 6,098.870 | 3,730.058 | 0.612 | 38.8% |
| 21,847 | 4,928.555 | 3,746.681 | 0.760 | 24.0% |
| 32,768 | 6,126.950 | 4,527.402 | 0.739 | 26.1% |
| 32,769 | 8,129.615 | 5,321.693 | 0.655 | 34.5% |

### Interpretation of the same-operands ratios

At 8 and 16 limbs, both operations use schoolbook arithmetic. The complete
wide product contains approximately `2n²` limb products, while the requested
middle band contains approximately `n²`. The measured ratios of 0.582 and
0.565 are therefore consistent with a one-half arithmetic target plus band
setup and carry overhead.

At 32–89 limbs, the wide full product has moved to Karatsuba while the middle
product remains schoolbook. That progressively reduces the saving from 37.1%
to 10.2%. When the middle product switches to FFT at 90 limbs, its saving rises
again to 24.2%.

Both operations use FFT at 114–21,846 limbs, but their transform lengths and
factorizations differ. This produces much larger effects than the one-limb
changes in input size: the middle product costs only 0.411–0.429 times full at
114–115, becomes 1.136 times full at 128, and remains slower through 16,384.
These reversals are transform-bucket effects, not changes in asymptotic
complexity.

The wide output has exactly 65,536 body limbs at `n = 21,846`, the largest size
that remains on its FFT path. At `n = 21,847` it switches to NTT and becomes
faster in absolute terms on this machine, narrowing the middle-product saving
from 38.8% to 24.0%. The middle product remains on FFT until `n = 32,768`.

Once both calls use NTT, middle multiplication costs 0.739 times the complete
wide product at 32,768 limbs and 0.655 times at 32,769 limbs. The latter is
close to the proposed `2/3` multiplier, although two points near a transform
boundary are not enough to establish a stable asymptotic constant.

## Original `M(n)`-baseline run

All times are Criterion mean point estimates in microseconds. Ratios divide
the corresponding time by the full-multiplication time at the same `n`; lower
is better, and `1.000` means equal time.

| `n` limbs | Full (µs) | High (µs) | High / full | Middle (µs) | Middle / full |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.078 | 0.061 | 0.783 | 0.081 | 1.034 |
| 16 | 0.286 | 0.188 | 0.657 | 0.302 | 1.058 |
| 32 | 0.936 | 0.642 | 0.686 | 1.156 | 1.235 |
| 64 | 2.827 | 2.500 | 0.885 | 4.510 | 1.596 |
| 89 | 4.783 | 4.761 | 0.995 | 8.723 | 1.824 |
| 90 | 4.883 | 4.882 | 1.000 | 7.493 | 1.535 |
| 114 | 7.387 | 7.711 | 1.044 | 7.354 | 0.996 |
| 115 | 7.387 | 7.397 | 1.001 | 7.380 | 0.999 |
| 128 | 9.069 | 9.086 | 1.002 | 11.036 | 1.217 |
| 256 | 13.634 | 13.648 | 1.001 | 25.905 | 1.900 |
| 1,024 | 66.320 | 66.492 | 1.003 | 115.286 | 1.738 |
| 4,096 | 307.399 | 308.846 | 1.005 | 577.460 | 1.879 |
| 16,384 | 1,349.098 | 1,358.865 | 1.007 | 3,138.912 | 2.327 |
| 32,768 | 2,973.515 | 3,003.637 | 1.010 | 4,436.234 | 1.492 |
| 32,769 | 4,909.431 | 4,966.127 | 1.012 | 4,451.722 | 0.907 |

## Operand and output semantics

The benchmark uses `n` as the target precision and records both full-product
baselines:

- **Full:** `mul_dyn(a[n], b[n], out[2n-1])` computes the complete product and
  returns its possible overflow limb.
- **Full wide:** `mul_dyn(long[2n-1], short[n], out[3n-2])` computes the complete
  product of the exact operands passed to the middle product.
- **High:** `hi_mul_dyn(a[n], b[n], out[n])` computes the high `n`-limb band
  using the library's approximate carry convention for omitted low columns.
- **Middle:** `mid_mul_dyn(long[2n-1], short[n], out[n])` computes the standard
  `n`-limb middle-product band and returns its two carry limbs.

The asymmetric middle-product input is important when interpreting the two
tables. The `n` by `n` baseline compares against the usual `M(n)` operation but
does not use the same two integers. The full-wide baseline uses exactly the
same two integers and measures the work avoided by discarding the low and high
bands.

## Dispatch-boundary interpretation

The cutoff-adjacent sizes in the original run expose three implementation
effects:

1. With the then-current cutoff of 114, `hi_mul_dyn` called the specialized
   `hi_mul_buf` through 114 output limbs. Its benefit was substantial at 8–64
   limbs, nearly gone by 89 limbs, and negative at 114 limbs. The working tree
   now uses `HI_MUL_CUTOFF = 90`, so current 114-limb calls take the full-product
   fallback instead.
2. `mid_mul_dyn` switches from schoolbook middle multiplication to FFT at 90
   limbs. Its ratio improves from 1.824 at 89 limbs to 1.535 at 90 limbs, but
   the transform-backed implementation remains more expensive than the full
   `n` by `n` product at most measured sizes.
3. Middle multiplication switches to NTT when `2n` reaches the 65,536-limb FFT
   limit, so 32,768 limbs is its first NTT point. Full and high multiplication
   still use FFT there because their full output has 65,535 body limbs. They
   switch to NTT at 32,769 limbs, which explains both the full-product timing
   jump and the middle product's temporary 0.907 ratio.

The high-product results showed that `HI_MUL_CUTOFF = 114` was too high on this
machine if the cutoff is intended to minimize latency. The current value of 90
matches the measured break-even region, although a denser cutoff benchmark is
still appropriate before treating it as portable across architectures.

## Implications for reciprocal square root

The measured high-product behavior agrees with the structural limitation that
motivated the Karp-Markstein discussion: the quasi-linear high path does not
save a transform, so the implementation costs approximately one full
multiplication above the schoolbook range.

The same-operands comparison shows that the current generic middle product can
avoid substantial work: about 42–44% in the small schoolbook cases and 24–35%
at the measured NTT sizes. Its 0.655 ratio at 32,769 limbs is close to the
proposed `2/3` cost relative to the complete `(2n-1)` by `n` multiplication.

That does not make it a `1/3 M(n)` residual square. Relative to an `n` by `n`
`M(n)` baseline, the current middle path still usually costs 1.2–2.3 times as
much in the transform range. A Karp-Markstein final iteration still needs a
specialized low/cyclic square or residual primitive to obtain the more
aggressive constant discussed earlier.

## Harness and method

The repeatable harness is `bench_mul_band_ratios` in
[`benches/utils_bench.rs`](../benches/utils_bench.rs). Each case generates its
random operands once, reuses preallocated output buffers, and passes operands
and outputs through Criterion's `black_box`. `full_wide` and `middle` use the
same generated `long` and `short` values. The measured functions perform no
per-iteration input generation or output allocation. Scratch-pool acquisition
and normal production dispatch remain part of the measured operation.

Run the benchmark with:

```bash
cargo bench-utils -- mul_band_ratio --noplot
```

Run only the same-operands comparison with:

```bash
cargo bench-utils -- 'mul_band_ratio.*(full_wide|middle)' --noplot
```

Criterion configuration for this group:

- wall-clock measurement;
- flat sampling with 100 samples;
- 1 second warmup per case;
- 3 seconds measurement per case;
- release benchmark profile with thin LTO and one codegen unit.

The table reports mean point estimates. Criterion's 95% confidence intervals
are retained under `target/criterion/mul_band_ratio_aarch64/`, but that target
directory is intentionally not versioned. Large FFT/NTT cases may use Rayon,
so their wall time includes the production parallel scheduling behavior.

Measured system:

- Apple M1 MacBook Air, 8 cores and 8 GB RAM;
- AArch64, macOS 26.5.2;
- Rust 1.95.0;
- Criterion 0.5.1.

CPU frequency, thermal state, and background processes were not pinned.
Consequently, ratios measured within this run are more portable than the
absolute timings, and cutoff decisions should be confirmed on each target
architecture. The largest parallel cases showed more run-to-run variation
than the small and medium cases, so the 16,384–32,769-limb ratios should be
treated as directional until repeated under controlled thermal conditions.
