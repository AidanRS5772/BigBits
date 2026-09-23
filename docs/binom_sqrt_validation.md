# Binomial sqrt validation and before/after benchmarks

Validated the working-tree halved-numerator `sqrt_4x2`, its four-limb window
refactor, and merged-carry `sub_mul_of` assembly against commit `d655804`.
The fixed-width seed correction and saturated-digit `knuth_est` bypass remain
prototypes; neither is included in these production measurements.

Two assembly issues were corrected before measuring: the AArch64 block lacked
a comma after its final `subs`, and the x86 loop counter required an `inout`
operand because `dec` changes it. The x86 constraint fix is also applied to the
baseline, so the comparison does not rely on an invalid asm operand contract.

## Correctness

- Existing 22 sqrt tests passed before the assembly fixes.
- Added three tests covering seed remainder boundaries, all supported input
  widths for root lengths 1–63, normalization, and exact-square boundaries.
- Debug sqrt: 25 passed. Optimized `prof` sqrt: 21 passed (four debug-only
  capacity checks do not apply). Optimized division: 50 passed.
- An optimized adapter matched Python `math.isqrt` for 9,328 exact root and
  remainder checks, including boundary cases up to 256 root limbs.
- Full debug suite: 332 passed, two unrelated multiplication failures:
  `test_empty_input_public_paths` and
  `test_karatsuba_static_chunking_scratch_fallback`. Both failures were
  reproduced against the baseline source. Neither was changed.
- AArch64 execution was unavailable on this x86 host.

## Timings

Intel i7-8750H, Rust 1.94.0, `prof` optimization, pinned to CPU 2. Each case
uses 256 deterministic operands and seven 200 ms timing rounds per variant,
with a 100 ms warmup per process and rotating variant order. Input buffers are
reused; copying the input back is included. Calls go directly to `binom_sqrt`,
independent of the dispatcher cutoff. Sizes below are **root limbs**; full
inputs contain twice as many limbs. All benchmark output checksums matched.

Baseline and assembly-only snapshots were explicitly rebuilt; executable
hashes and Cargo logs are recorded to rule out reused working-tree artifacts.

| Root limbs | Shape | Pattern | Baseline ns | Assembly only ns | Both changes ns | Time reduction |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | full | random | 111.6 | 113.6 | 113.2 | -1.4% |
| 2 | full | random | 206.0 | 207.2 | 190.8 | 7.4% |
| 3 | full | random | 300.1 | 302.9 | 284.8 | 5.1% |
| 4 | full | random | 402.9 | 406.3 | 374.7 | 7.0% |
| 8 | full | random | 855.2 | 856.2 | 797.8 | 6.7% |
| 16 | full | random | 2046.3 | 1976.1 | 1882.7 | 8.0% |
| 32 | full | random | 5211.1 | 4965.0 | 4734.1 | 9.2% |
| 48 | full | random | 9425.7 | 8687.3 | 8289.3 | 12.1% |
| 63 | full | random | 13864.1 | 12753.6 | 12319.9 | 11.1% |
| 64 | full | random | 14207.3 | 13024.4 | 12587.2 | 11.4% |
| 128 | full | random | 42574.6 | 37821.6 | 37250.4 | 12.5% |
| 256 | full | random | 140632.3 | 121071.0 | 118224.7 | 15.9% |
| 2 | padded | random | 206.6 | 208.9 | 192.1 | 7.0% |
| 4 | padded | random | 407.9 | 413.5 | 390.2 | 4.3% |
| 8 | padded | random | 877.4 | 870.2 | 830.1 | 5.4% |
| 16 | padded | random | 2111.7 | 2039.5 | 1947.4 | 7.8% |
| 32 | padded | random | 5288.2 | 5051.1 | 4828.6 | 8.7% |
| 63 | padded | random | 14167.2 | 12976.9 | 12534.6 | 11.5% |
| 2 | full | shifted | 232.7 | 231.6 | 227.0 | 2.4% |
| 8 | full | shifted | 955.9 | 942.9 | 896.3 | 6.2% |
| 16 | full | shifted | 2264.5 | 2164.2 | 2094.0 | 7.5% |
| 32 | full | shifted | 5654.5 | 5333.6 | 5187.4 | 8.3% |
| 63 | full | shifted | 14608.7 | 13316.8 | 12914.1 | 11.6% |
| 2 | full | ones | 202.3 | 205.6 | 148.6 | 26.6% |
| 8 | full | ones | 790.4 | 791.3 | 687.5 | 13.0% |
| 16 | full | ones | 2042.5 | 2005.4 | 1832.8 | 10.3% |
| 32 | full | ones | 6302.8 | 6048.2 | 5764.0 | 8.5% |
| 63 | full | ones | 22639.8 | 22309.1 | 21731.6 | 4.0% |
| 2 | full | power | 132.7 | 132.8 | 130.2 | 1.9% |
| 8 | full | power | 445.1 | 434.9 | 408.8 | 8.2% |
| 16 | full | power | 990.3 | 920.4 | 858.4 | 13.3% |
| 32 | full | power | 2610.9 | 2210.2 | 2065.6 | 20.9% |
| 63 | full | power | 7490.3 | 6219.3 | 5991.3 | 20.0% |

The random full-input speedups are approximately 5–12% for 2–63 root limbs.
The one-limb control is 1.4% slower, with overlapping run ranges; that path
uses neither modified arithmetic kernel. Padded inputs and structured patterns
also improved in this sweep. These are paired whole-call measurements, not
projections from the earlier seed-only probe.

The existing Criterion binomial cases also completed at 16, 32, and 64 limbs
(approximately 1.580, 3.665, and 11.230 microseconds). Their single-operand
batching differs from the comparison harness; historic Criterion baselines
were not used to establish the before/after improvements above.

## Reproduction and artifacts

- `benches/binom_sqrt_profile.rs`: timing harness.
- `scripts/python/compare_binom_sqrt.py`: compare saved baseline, assembly-only,
  and current harness executables, built with the same toolchain/profile.
- `benches/probes/binom_sqrt_verify.rs` and
  `scripts/python/verify_binom_sqrt.py`: independent Python oracle check.
- `target/binom_sqrt_profiles/comparison/`: build and test logs, binary hashes,
  Criterion output, and `timings/{rounds.csv,summary.json}` with all raw rounds.

Commands used include `cargo test test_sqrt`, `cargo test --lib`,
`cargo test --profile prof test_sqrt`, the optimized test binary with the
`tests::test_div` filter, `cargo bench-utils --no-run`, and the resulting
Criterion executable with `--bench 'zimmermann_sqrt/.*/binom/(16|32|64)$' --quick`.
