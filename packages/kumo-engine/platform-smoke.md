# Platform smoke

The manual [Platform smoke workflow](../../.github/workflows/kumo-engine-platform-smoke.yml)
installs the checked-out engine and DiskGraph source, then runs
[smoke_platform.py](test/smoke_platform.py): local Parquet ingestion, a short
training run, and prediction. Linux ARM64 and WSL train on CUDA; native
Windows uses CPU Faiss and CPU training. Failed installation or a missing
requested device fails the job.
This tests source execution; it does not build or publish a release wheel.

From the repository root, with engine dependencies installed:

```bash
python -m pytest -q -s packages/kumo-engine/test/smoke_platform.py
# For CPU training (PowerShell: $env:KUMO_SMOKE_DEVICE = 'cpu'):
KUMO_SMOKE_DEVICE=cpu python -m pytest -q -s packages/kumo-engine/test/smoke_platform.py
```

Acceptance requires real snapshot ingestion, finite gradients observed
at an optimizer step, finite training metrics, and finite predictions for all
100 unlabeled entities (with 300 labeled entities used for training).
`KUMO_SMOKE_DEVICE` selects `cpu` or `cuda` (default), and
the optimizer check requires gradients on that exact device. CUDA mode sets
`KUMO_REQUIRE_GPU=1` and fails instead of skipping without CUDA; CPU mode
explicitly selects CPU training. Its filename excludes it from ordinary test
discovery. Platform, architecture, Python, Torch, requested device, and CUDA
availability are printed for diagnosis.

Local Linux ARM64 CPU validation passed with four optimizer steps and 100
finite predictions, using Python 3.14, Torch 2.12.0+cpu, and public PyG 2.8.1.
That run used the checkout's Python source and freshly built Linux DiskGraph
debug binaries.
It does not establish a Windows build. CUDA mode also correctly fails when
CUDA is unavailable. Windows, WSL, and GPU execution remain unverified; the
first manual runner executions must validate their installation and runtime.

In GitHub Actions, select **Kumo Engine - Platform smoke**, choose **Run
workflow**, and select the branch containing this workflow. All three jobs run
independently. No push or pull-request triggers are enabled. Each dispatch
tests its selected revision. The workflow builds on native portability commit
`e4b423c1aeff1a83564e1a26e58de6939cb01003`.

| Environment | Runner group | Runner label |
| --- | --- | --- |
| Linux ARM64, L4 | `nv-gpu-arm64-l4-1gpu` | `linux-arm64-gpu-l4-latest-1` |
| Windows x86-64 through WSL, L4 | `nv-gpu-general` | `wsl-amd64-gpu-l4-latest-1` |
| Native Windows x86-64, CPU execution on L4 host | `nv-gpu-general` | `windows-amd64-gpu-l4-latest-1` |

The repository must have access to these runner groups. All images need a
C/C++ compiler and network access to package indexes and Rust dependencies.
Linux ARM64 and WSL also need a working NVIDIA driver. Python 3.12 and Rust
are installed by the workflow. PyTorch 2.12.0 uses CUDA 13 on Linux/WSL and
the CPU build on native Windows, with wheel versions recorded in `uv.lock`.
Other Python dependencies resolve from package metadata; this is not a fully
locked environment.

ARM additionally needs CUDA 13 development tools at `/usr/local/cuda` (or
`CUDAToolkit_ROOT`) and OpenBLAS development headers/libraries. The workflow
checks these prerequisites and uses the existing ARM Faiss source builder.
It does not install system packages or alter the runner image.

The WSL job assumes the Actions agent executes inside the Linux guest. It
checks the Linux kernel for WSL and fails if this runner instead executes
Windows host shells. That registration detail and installed image packages
still need confirmation in the first manual run. Git Bash is not a WSL test.

The native Windows job runs PowerShell and installs the MSVC Rust target.
Its image must already contain Visual Studio 2022 or newer Build Tools with
the x64 C++ toolchain, Windows SDK, and CMake. The workflow locates the installed
tools with `vswhere`, activates their developer environment, and checks that
`cl`, `link`, and `cmake` are available. It fails explicitly when prerequisites
are absent; it does not install system toolchains or reboot the runner.

Engine metadata selects `kumo-core[faiss-cpu]` on native Windows and
`kumo-core[faiss-gpu]` elsewhere, including WSL. Only one Faiss distribution
is installed because both provide the same Python import namespace. Windows
CPU Faiss and CPU training do not cover native Windows GPU execution.
DiskGraph and Python portability changes are maintained in this repository.
The native build and runner image still need validation before native
execution can be called operational.

Native Windows coverage is limited to the isolated engine installation in this
workflow. Root `kumo[cu130]`, `kumo[cu132]`, and full-workspace installs remain
unsupported: root GPU extras also request GPU Faiss, conflicting with CPU
Faiss selection. Root extras are unchanged.

Native versus WSL wheel distribution remains undecided. Enable PR checks only
after manual runs are
reliable; enterprise runner checks should follow the repository's trusted
copy-pr-bot workflow.
