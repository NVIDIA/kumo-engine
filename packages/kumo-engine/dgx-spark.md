# Build and install on DGX Spark

Build GPU Faiss and `kumo-engine` from source, then install them in a Python
virtual environment.

## Supported versions

Use Python **3.11–3.13** and **Faiss GPU 1.14.3 built with CUDA 13** with one
of these PyTorch versions:

| PyTorch | CUDA wheel variants |
| --- | --- |
| 2.7.0 | cu128 |
| 2.9.0 | cu128, cu130 |
| 2.10.0 | cu128, cu130 |
| 2.11.0 | cu128, cu130 |
| 2.12.0 | cu130 |

The commands below use Python 3.12 and PyTorch 2.12.0 with CUDA 13.

## Prerequisites

On your DGX Spark, install:

- A working NVIDIA driver (`nvidia-smi`) and the CUDA 13 toolkit (`nvcc`).
- A C/C++ build toolchain and OpenBLAS development files (`libopenblas-dev`
  on Ubuntu).
- A current stable Rust toolchain, including `cargo`.
- Git and `uv`.

The build downloads source and dependencies from GitHub and package indexes.
The Faiss builder uses CUDA from `/usr/local/cuda`. If CUDA 13 is installed
elsewhere, set `CUDAToolkit_ROOT` to that directory before building.

## Set up the source and Python environment

```bash
git clone https://github.com/NVIDIA/kumo-engine.git
cd kumo-engine
KUMO_CHECKOUT="$PWD"
SPARK_WORK="$HOME/kumo-spark"
uv venv --seed --python 3.12 "$SPARK_WORK/venv"
source "$SPARK_WORK/venv/bin/activate"
python -m pip install 'torch==2.12.0' \
  --index-url https://download.pytorch.org/whl/cu130
```

## Build and install GPU Faiss

Expect the Faiss build to take 3–5 minutes on DGX Spark.

```bash
FAISS_WHEELS="$(CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  FAISS_CUDA_ARCHES=12.1 \
  FAISS_GPU_CACHE_DIR="$SPARK_WORK/faiss" \
  "$KUMO_CHECKOUT/scripts/build_faiss_gpu_arm64_wheel.sh" \
    --python "$SPARK_WORK/venv/bin/python" --cuda cu130)"
python -m pip install "$FAISS_WHEELS"/faiss_gpu-1.14.3-*.whl
```

## Build and install Kumo Engine

```bash
cd "$KUMO_CHECKOUT"
python -m pip install ziglang
KUMO_WHEEL="$(PYTHON="$SPARK_WORK/venv/bin/python" \
  ./packages/kumo-engine/scripts/build_wheel.sh \
    --platform-tag manylinux_2_34_aarch64 \
    --output-dir "$SPARK_WORK/kumo-wheel")"
python -m pip install "$KUMO_WHEEL"
```

Pip installs the remaining dependencies. To include optional dependencies,
append `[snowflake]` or `[text]` to the full wheel filename when running
`pip install`, and quote the resulting path. These extras install Snowflake
dependencies or GloVe word vectors, respectively.
