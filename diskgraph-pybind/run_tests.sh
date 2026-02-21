#!/bin/bash
set -e

PYTHON=${PYTHON:-python3}

# 1. Build via maturin (produces correctly-named .so)
echo "Building diskgraph-pybind via maturin..."
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" uvx maturin develop --release --manifest-path Cargo.toml

# 2. Run Tests
echo "Running Pytest..."
$PYTHON -m pytest tests/
