# Division control flow refactor

The division module now uses four prepared division drivers and two prepared reciprocal drivers. Public division, remainder, high-quotient, and reciprocal signatures and contracts are unchanged. Forced algorithm tests and benchmarks use `division_preflight` plus `div_prepared_*`/`div_rem_prepared_*`, or `reciprocal_preflight` plus `rcp_prepared_*`, with public `DivAlg`/`RcpAlg` enums. The algorithm-specific wrappers have been removed.

The division drivers apply operand windows, attempt NR, and return on success. Failed attempts select BZ, after which the driver prepares classical storage once and calls Knuth or BZ directly. NR uses caller output when the overflow slot fits; only exact-sized outputs need a temporary full quotient. Failed NR attempts leave the numerator intact, and remainder finishing commits only after successful correction. Static capacity is checked against prepared operands, excluding unused output tails.

`NrEstimate` distinguishes a proven quotient, a quotient needing exact correction, and failure. Quotient-only attempts retain the fast path that avoids a full product; remainder attempts retain their smaller windowed product. BZ cores now each contain normalization, top-block handling, remaining block iteration, and remainder restoration. Allocation, correction, and retry callbacks have been removed; shared arithmetic kernels still accept multiplication callbacks.

Dynamic NR reciprocal repair now uses `knuth_rcp_normalized` with seed 1, followed by the upward bias increment, reusing the reciprocal-error buffer. Before replacing the old power-of-base construction, both implementations were checked against exact division over 120 deterministic cases. Public reciprocals retain their distinct approximation contract, two guard limbs, and explicit Knuth retry. Static reciprocal guard-capacity and BZ padding-capacity checks remain; the unreachable static quotient-estimator capacity branch is a documented assertion.

The module has 69 functions (previously 82). Cutoffs and selection formulas, machine arithmetic and architecture gates, and unrelated square-root changes were preserved. A source comparison confirmed 27 arithmetic kernels and selection functions are unchanged apart from whitespace or location.

Validation:

- `cargo test`: 338 passed; the same two baseline multiplication failures remain: `test_empty_input_public_paths` and `test_karatsuba_static_chunking_scratch_fallback`.
- All 57 division tests pass, including new tests for exact/ambiguous NR guards, failure preservation and retry, distant correction rejection, reciprocal guard capacity, sliding-seed equivalence, and static capacity after exact low-zero divisor factors.
- `cargo check --all-targets --features _bench_internals` passes.
- `cargo fmt` was run on the changed Rust files; unrelated existing formatting was preserved.

## Performance comparison

The `division_flow` group in `benches/probes/division_flow.rs`, registered by `utils_bench`, uses deterministic inputs with both normalized and shifted divisors. It covers public and forced NR paths, exact output and an extra overflow slot, quotient and remainder operations, dynamic and static allocation, reciprocal precision, and a skewed public BZ case with padded-top recursion. All scratch remains in production drivers; measurement helpers remain under `benches/`.

Run the matrix with:

```sh
cargo bench-utils -- division_flow --nresamples 1000 --noplot --save-baseline before
# After rebuilding the refactor:
cargo bench-utils -- division_flow --nresamples 1000 --noplot --baseline before
```

Measurements use an Intel Core i7-8750H, pinned to logical CPU 2, with 30 Criterion samples, 150 ms warmup and 500 ms measurement per case. Baseline executables were saved before editing; the added BZ case is compared with a build of the saved original source using the expanded harness. Sequential sweeps showed transient outliers, so acceptance uses alternating baseline/refactored runs and the planned 2% family geometric mean / 5% individual-case thresholds. No cutoffs were retuned.

Final comparison: all 92 cases were measured in three paired rounds, in baseline/refactor, refactor/baseline, baseline/refactor order. Each case's ratio is the geometric mean of the three ratios of Criterion mean runtimes. The table aggregates those case ratios by operation and allocation model; positive numbers are slower.

| Allocation | Quotient (20 cases) | Remainder (20 cases) | Reciprocal (6 cases) |
|---|---:|---:|---:|
| Dynamic | +0.16% | +0.53% | +1.24% |
| Static | -0.07% | +0.59% | -0.05% |

All six family geometric means are below the 2% regression limit. The largest individual aggregate increase is 4.75% (`division_flow/dyn/div/nr/exact/16x16`), below the 5% limit. No performance-driven cutoff or arithmetic changes were needed. The [per-case CSV](division_control_flow_benchmarks.csv) includes each paired round, including timing variation, rather than only the aggregate.

The source snapshots, saved executables, intermediate stage results, full test logs, and detailed Criterion logs for this run are under `/tmp/bigbits-div-refactor/`; Criterion's per-run estimates are under `target/criterion/division_flow/`. These are local run artifacts, not required to build or test the crate.
