#!/bin/bash
set -e

# 1. Build
echo "Building diskgraph-pybind..."
cargo build --release

# 2. Setup Python Env
# Copy .so to current dir so pytest can find it
cp ../target/release/libdiskgraph_pybind.so diskgraph_pybind.so

# 3. Run Tests
echo "Running Pytest..."
export PYTHONPATH=.
~/kumo/venv/bin/python -m pytest tests/

# 4. Cleanup
rm diskgraph_pybind.so
