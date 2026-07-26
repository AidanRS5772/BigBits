#!/bin/bash
set -e
mkdir -p coverage
cargo llvm-cov clean --workspace
cargo llvm-cov test --lib --html --output-dir coverage -- test_mul
cargo llvm-cov test --lib --lcov --output-path coverage/lcov.info -- test_mul
echo "HTML report: coverage/index.html"
echo "LCOV data:   coverage/lcov.info"
