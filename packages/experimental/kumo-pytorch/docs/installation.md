# Installation

kumo-pytorch isn't published on PyPI. It lives in the
[NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine) repository under
`packages/experimental/kumo-pytorch`, and you install it by building wheels from
a checkout of that repository. It needs Python 3.11 or newer, PyTorch 2.7 or
newer (below 3), and the DiskGraph engine, which is a required dependency built
from `packages/diskgraph` in the same repository. kumo-engine's Rust-based
DiskGraph engine indexes data on local SSDs and retrieves time-correct
neighborhoods for training and batch prediction, without keeping the full graph
index in RAM; the [kumo-engine README](https://github.com/NVIDIA/kumo-engine#readme)
explains how it works.

## Build the wheels

You'll need [uv](https://docs.astral.sh/uv/getting-started/installation/), Git,
a Rust toolchain and ordinary C/C++ build tools. If you use
[rustup](https://rustup.rs/), it picks the Rust version from kumo-engine's
`rust-toolchain.toml` for you.

```bash
git clone https://github.com/NVIDIA/kumo-engine.git
cd kumo-engine
uv build --wheel --out-dir dist packages/diskgraph
uv build --package kumo-pytorch --out-dir dist
```

This leaves a `diskgraph` wheel and a `kumo_pytorch` wheel in `dist/`.

## Install with uv

Create a virtual environment and install both wheels into it. The
`--torch-backend` option tells uv which PyTorch build to pick: `cpu` for the
CPU-only wheels, `cu130` for the CUDA 13.0 wheels, or `auto` to let uv choose
from the GPU driver it finds.

```bash
uv venv --python 3.12 .venv-kumo-pytorch
uv pip install --python .venv-kumo-pytorch/bin/python --torch-backend cpu \
    dist/diskgraph-*.whl dist/kumo_pytorch-*.whl
```

The environment can live anywhere; it doesn't have to be inside the checkout.
To run the example notebooks you also need the `notebooks` extra, which you
can request on the wheel itself:

```bash
uv pip install --python .venv-kumo-pytorch/bin/python --torch-backend cpu \
    dist/diskgraph-*.whl "$(ls dist/kumo_pytorch-*.whl)[notebooks]"
```

## Install with pip

pip can't choose a PyTorch index from package metadata, so install PyTorch
first from the [PyTorch index](https://pytorch.org/get-started/locally/) that
matches your hardware, then the two wheels:

```bash
python3.12 -m venv .venv-kumo-pytorch
source .venv-kumo-pytorch/bin/activate
python -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cpu   # or .../whl/cu130
python -m pip install dist/diskgraph-*.whl dist/kumo_pytorch-*.whl
```

A compatible PyTorch that's already installed satisfies the dependency. On
Apple silicon, MPS is a PyTorch device setting, so there's nothing extra to
install for kumo-pytorch.

## Check the installation

The [quick start](../README.md#quick-start) runs as published. Save it to a
file and run it with the environment's Python, and it prints two inputs and a
`(2, 2)` padded history.

The first `prepare()` with `DiskGraphEngine(index=...)` builds the index if it's
missing, and later runs reuse it. With `ingest="never"` it only uses an index
that already exists. It never overwrites an index that doesn't match your data.
See [engines and data](04-engines-and-data.md#building-and-reusing-the-index)
for details.

## Run the tests and notebooks from a checkout

To work on kumo-pytorch itself, use the kumo-engine workspace the same way
kumo-engine's CI does. At the root of the checkout, install the CPU build of
PyTorch, then kumo-pytorch with its `dev` and `notebooks` extras. The second
sync builds DiskGraph from `packages/diskgraph`, so it needs the Rust toolchain
too.

```bash
uv sync --locked --extra cpu
uv sync --locked --inexact --no-install-package torch --package kumo-pytorch --extra dev --extra notebooks
source .venv/bin/activate
cd packages/experimental/kumo-pytorch
python -m pytest -q
```

Tests that need DiskGraph skip if it can't be imported, and tests that need a
GPU skip without one. Pass `--require-imports` to turn those skips into
failures.
