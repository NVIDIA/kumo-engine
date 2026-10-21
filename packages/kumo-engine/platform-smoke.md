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
CUDA mode also correctly fails when CUDA is unavailable.

WSL passed twice: [run 39665870696](https://github.com/NVIDIA/kumo-engine/actions/runs/39665870696)
and [run 39667658574](https://github.com/NVIDIA/kumo-engine/actions/runs/39667658574).
Both completed ingestion, four CUDA optimizer steps, and 100 finite predictions.
The latest test passed in 25.98 seconds at revision `29b255e`, the same
revision as the native Windows pass below. Observed source installation took
5m11s–11m19s; the latest DiskGraph build accounted for 11m02s.
It used WSL2 kernel `6.6.114.1-microsoft`, x86-64, Python 3.12.3,
Torch 2.12.0+cu130, and a Tesla T4 with driver 596.36.

Native Windows [run 39667658574](https://github.com/NVIDIA/kumo-engine/actions/runs/39667658574)
passed ingestion, four CPU optimizer steps, and 100 finite predictions
(one test passed in 3.86 seconds; source installation took 4m03s).
It used GitHub-hosted Windows Server 2022, AMD64, Python 3.12.10,
and Torch 2.12.0+cpu, exercising CPU Faiss and the checkout's DiskGraph source.
The smoke prints a bounded subprocess output tail for its generated fixture
on failure, preserving production privacy and failure behavior.
Linux ARM64 CUDA execution remains unverified.
Native Rust dependencies are cached even after smoke failure; workspace
crates still rebuild from checkout source.

In GitHub Actions, select **Kumo Engine - Platform smoke**, choose **Run
workflow**, and select the branch containing this workflow. All three jobs run
independently. Each dispatch tests its selected revision. The workflow builds
on native portability commit
`e4b423c1aeff1a83564e1a26e58de6939cb01003`.

For pre-merge qualification, PR 76 has a temporary push trigger scoped
only to its trusted `pull-request/76` copy. A maintainer admits the exact PR
head with `/ok to test <full-head-SHA>`; copy-pr-bot's push starts all three
jobs without changing `main`. Remove this temporary trigger before merge so
the delivered workflow remains manual-only. Ordinary fork pushes and
`pull_request` events do not start these jobs.

| Environment | Runner group | Runner label |
| --- | --- | --- |
| Linux ARM64, L4 | `nv-gpu-arm64-l4-1gpu` | `linux-arm64-gpu-l4-latest-1` |
| Windows x86-64 through WSL, T4 | `nv-gpu-amd64-t4-1gpu` | `wsl-amd64-gpu-t4-latest-1` |
| Native Windows x86-64, CPU | GitHub-hosted | `windows-2022` |

The workflow selects runners by label only; the table records catalog group
membership. The current runner-access configuration already grants the T4
group used by WSL. ARM64 L4 access is requested in
[runner configuration PR 549](https://github.com/nv-gha-runners/enterprise-runner-configuration/pull/549);
review and application remain pending.
The missing ARM64 grant is a configuration gap; queued jobs alone do not
establish their live permission or provisioning failure cause. Native Windows
uses a GitHub-hosted CPU runner and needs no enterprise GPU group access.
All images need a
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

The WSL job requires the Actions agent to execute inside the Linux guest. It
checks the Linux kernel for WSL and fails if this runner instead executes
Windows host shells. The successful WSL run verified this registration and
its installed prerequisites. Git Bash is not a WSL test.

The native Windows job runs PowerShell and installs the MSVC Rust target.
The [Windows Server 2022 image](https://github.com/actions/runner-images/blob/main/images/windows/Windows2022-Readme.md)
includes Visual Studio 2022, the x64 C++ toolchain, Windows SDK, and CMake.
The workflow locates the installed
tools with `vswhere`, activates their developer environment, and checks that
`cl`, `link`, and `cmake` are available. It fails explicitly when prerequisites
are absent; it does not install system toolchains or reboot the runner.

Engine metadata selects `kumo-core[faiss]`, whose platform markers choose
CPU Faiss on native Windows and GPU Faiss elsewhere, including WSL.
Only one Faiss distribution
is installed because both provide the same Python import namespace. Windows
CPU Faiss and CPU training do not cover native Windows GPU execution.
DiskGraph and Python portability changes are maintained in this repository.
The native pass qualifies this smoke's source-execution path, not the full
runtime or release-wheel distribution.

Native Windows coverage is limited to this isolated engine installation with
zero DataLoader worker processes, selected by the current Windows and test
configuration. The existing Unix worker-pool paths are not qualified by this
smoke. Root `kumo[cu130]`, `kumo[cu132]`, and full-workspace installs remain
unsupported: root GPU extras also request GPU Faiss, conflicting with CPU
Faiss selection. Root extras are unchanged.

Native versus WSL wheel distribution remains undecided. Enable PR checks only
after manual runs are
reliable; enterprise runner checks should follow the repository's trusted
copy-pr-bot workflow.
