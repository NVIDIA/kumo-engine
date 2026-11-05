# Install on DGX Spark

Use the Linux ARM64 Kumo Engine wheel with CPython **3.11–3.14**. The host
needs glibc 2.34 or newer and CPU AES and NEON instructions. Check the
[PyPI files for your release](https://pypi.org/project/kumo-engine/#files)
for an `aarch64` wheel before using the installation steps below. If your
release has no ARM64 wheel, use the
[local source-build instructions](../../docs/development/releasing-kumo-engine.md#build-locally),
including the [Linux ARM64 build notes](../../docs/development/releasing-kumo-engine.md#linux-arm64-builds),
then install the resulting wheel in your virtual environment.

The installed-wheel qualification covers CPU ingestion, training, prediction
and model reload on Python 3.11–3.14. GPU execution and Faiss retrieval have
separate dependency requirements below.

## Install the engine wheel

Create a virtual environment; these commands use Python 3.12:

```bash
python3.12 -m venv "$HOME/kumo-spark-venv"
source "$HOME/kumo-spark-venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --only-binary=kumo-engine kumo-engine
python -m pip check
```

Follow the [quickstart](README.md#quickstart) to ingest data, train and predict.
See the [installation and extras table](README.md#installation) for data
connectors and text encoding, including combinations such as
`kumo-engine[snowflake,text]`.

The base ARM64 install does not require Faiss. Install a Faiss backend only
for link-prediction retrieval. Other prediction tasks can run without it.

## Optional: CPU link prediction

```bash
python -m pip install 'kumo-engine[cpu]'
```

The `cpu` extra installs CPU Faiss; it does not choose the PyTorch build or
the training device. With CPU-only Faiss, explicitly select CPU execution for
link prediction, including when a GPU is available:

```python
from kumo.engine import Trainer

trainer = Trainer(model_plan, accelerator="cpu")
```

Use `Trainer.load(bundle_dir, accelerator="cpu")` when reloading such a model.
Choose one Faiss backend per environment.

## Optional: CUDA training

A compatible NVIDIA driver and CUDA-enabled PyTorch build are required for
GPU execution. Check the driver with `nvidia-smi`. These PyTorch versions
provide the following ARM64 CUDA wheel variants:

| PyTorch | CUDA wheel variants |
| --- | --- |
| 2.7.0 | cu128 |
| 2.9.0 | cu128, cu130 |
| 2.10.0 | cu128, cu130 |
| 2.11.0 | cu128, cu130 |
| 2.12.0 | cu130 |

Python wheel availability varies by PyTorch version and CUDA index; the
table does not imply every combination supports every Python version.
For example, PyTorch 2.7.0 ARM64 wheels do not cover Python 3.14. See
[PyTorch installation](https://pytorch.org/get-started/locally/) for available
builds and driver requirements.

The optional GPU Faiss build below uses **Faiss GPU 1.14.3 built with CUDA 13**.

In the activated Python 3.12 environment, install PyTorch 2.12.0 with CUDA 13:

```bash
python -m pip install 'torch==2.12.0' \
  --index-url https://download.pytorch.org/whl/cu130
```

This does not require rebuilding Kumo Engine. Tasks without link-prediction
retrieval do not need GPU Faiss. CUDA link prediction also needs the optional
GPU Faiss build below.

## Optional: build GPU Faiss for CUDA link prediction

For this optional build, use a source checkout and install:

- The CUDA 13 toolkit (`nvcc`) and a compatible NVIDIA driver.
- A C/C++ build toolchain and OpenBLAS development files (`libopenblas-dev`
  on Ubuntu).
- Git and `uv`.

Use the activated Python 3.12 environment and CUDA-enabled PyTorch from above.
If CPU Faiss is already installed, uninstall it before installing GPU Faiss:

```bash
python -m pip uninstall faiss-cpu
```

The build downloads source and dependencies from GitHub and package indexes.
The builder uses CUDA from `/usr/local/cuda`. If CUDA 13 is installed
elsewhere, set `CUDAToolkit_ROOT` to that directory before building.

```bash
SPARK_WORK="$HOME/kumo-spark"
git clone https://github.com/NVIDIA/kumo-engine.git "$SPARK_WORK/kumo-engine"
KUMO_CHECKOUT="$SPARK_WORK/kumo-engine"
FAISS_WHEELS="$(CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  FAISS_CUDA_ARCHES=12.1 \
  FAISS_GPU_CACHE_DIR="$SPARK_WORK/faiss" \
  "$KUMO_CHECKOUT/scripts/build_faiss_gpu_arm64_wheel.sh" \
    --python "$(command -v python)" --cuda cu130)"
python -m pip install "$FAISS_WHEELS"/faiss_gpu-1.14.3-*.whl
python -m pip check
```

The engine wheel remains installed; no Rust toolchain or Kumo Engine source
build is needed for this GPU Faiss setup.
