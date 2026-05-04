# Repository Guidelines

## Project Structure & Module Organization

BigBits is a Rust 2021 crate for big integer and big float arithmetic. Core source lives in `src/`. The low-level limb-buffer algorithms are in `src/utils/`, using little-endian `Vec<u64>` buffers where index `0` is the least significant limb. Public number types and shared traits live in `src/bit_nums/`. Tests are organized under `src/tests/` and are included from `src/lib.rs` with `#[cfg(test)]`. Criterion benchmarks live in `benches/`. Helper research/tuning scripts are in `scripts/`, with Python utilities under `scripts/python/`.

## Build, Test, and Development Commands

- `cargo build`: compile the crate in debug mode.
- `cargo test`: run the full Rust test suite.
- `cargo test test_mul`: run one test module; similarly use `test_utils` or `test_div`.
- `cargo test test_utils::trim_lz_basic`: run a single named test.
- `cargo bench-utils`: run `benches/utils_bench.rs` via the `.cargo/config.toml` alias.
- `cargo bench-cutoffs`: run cutoff-tuning benchmarks.
- `cargo build --profile prof`: build with profiling symbols and release-like optimization.
- `scripts/coverage.sh`: generate coverage for multiplication tests; requires `cargo-llvm-cov`.

## Coding Style & Naming Conventions

Use standard Rust formatting: run `cargo fmt` before submitting changes. Prefer `snake_case` for functions, modules, variables, and test names; use `UpperCamelCase` for types and traits; use `SCREAMING_SNAKE_CASE` for constants such as algorithm cutoffs. Keep performance-sensitive changes close to the existing utils-layer patterns, and avoid introducing heap allocation in static-number paths unless it is intentional and measured. Do not use `_bench_internals` outside benchmark code.

## Testing Guidelines

Add focused tests under `src/tests/`, grouped by module (`test_mul.rs`, `test_div.rs`, `test_utils.rs`). Use deterministic edge cases for carry, borrow, zero trimming, and boundary behavior, then add randomized checks where existing helpers such as `rand_vec`, `rand_nonzero_vec`, or `to_u128` fit. Run `cargo test` before opening a PR. For algorithm cutoff or performance work, also run the relevant Criterion benchmark alias.

## Commit & Pull Request Guidelines

Recent commit messages are short, imperative or present-participle summaries, for example `optimizing NTT` or `adding NTT integration and optimization`. Keep commits narrow and describe the algorithm or module changed. Pull requests should include a concise description, the commands run, any benchmark impact for performance changes, and linked issues when applicable. Include screenshots only for generated reports or visual artifacts.

## Agent-Specific Notes

Respect existing documented bug notes in `CLAUDE.md`; do not repair unrelated known failures while working on a narrow task. Preserve inline assembly architecture gates and feature flags when modifying multiplication or division internals.
