# Development setup

Run commands from the relevant checkout. Tool declarations are in
[.mise.toml](../../.mise.toml); Python package inputs are in
[setup.py](../../setup.py) and package-local
`pyproject.toml` files.

## Setup prerequisites and side effects

The local SDK example requires prepared runtime dependencies, including
`torch-geometric>=2.8.1,<3` from the public package index and a
[Faiss selection](#choose-the-core-faiss-dependency). This guide describes
development setup, not a verified public-package quickstart.

Existing targets have these effects; inspect their scripts before running them:

| Target | Implementation and effect |
| --- | --- |
| `make install-tools` | Runs `mise install` for the declared tools; installs toolchains in mise-managed locations |
| `make install-venv` | [install_venv.sh](../../scripts/install_venv.sh) deletes and recreates `.venv` and the `venv` compatibility link, then installs dependencies and development tools |
| `make update-packages` | [update_packages.sh](../../scripts/update_packages.sh) updates packages in the selected environment and builds native dependencies |
| `make install-diskgraph` | [scripts/diskgraph](../../scripts/diskgraph) builds/installs the in-tree Rust/Python engine |
| `make install-all-dependencies` | [install_all_dependencies.sh](../../scripts/install_all_dependencies.sh) is host provisioning, including system/tool installation and possible shell changes; it is not a routine documentation or agent prerequisite |

Review `.mise.toml` before trusting a checkout. Shell activation, toolchain
installation, and environment recreation are explicit setup actions. A coding
agent should report missing prerequisites rather than run host provisioning as
an automatic repair.

## Choose the core Faiss dependency

The local SDK and assembled engine wheel require GPU Faiss on Linux x86_64
and CPU Faiss on macOS, preserving macOS Metal support. Faiss is optional on
ARM64 Linux and native Windows: affected retrieval operations raise an
actionable error without it, while other tasks remain available.

| Package selection | When to use it | Faiss dependency |
| --- | --- | --- |
| `kumo-engine[faiss-cpu]` | Run CPU retrieval | Uses GPU Faiss on Linux x86_64; installs CPU Faiss elsewhere |
| `kumo-engine[faiss-gpu]` | Use a compatible GPU backend | Installs GPU Faiss except on macOS, which retains CPU Faiss |
| `kumo-engine` | Use platform defaults | GPU Faiss on Linux x86_64, CPU Faiss on macOS, neither elsewhere |

The [core package metadata](../../packages/kumo-core/pyproject.toml) specifies
the versions for each extra. These extras select the Faiss dependency; they do
not select the PyTorch CPU/CUDA build or provision a complete GPU environment.
Bare `kumo-core` remains optional so explicit service CPU/GPU selections stay
independent of the SDK default. Its Faiss extras select their named build on
every platform. Select one variant per environment rather
than combining `[faiss-cpu,faiss-gpu]`. CPU Faiss cannot perform CUDA retrieval;
select a CPU device for that workload. Existing macOS retrieval dispatch is
preserved; no fallback is added for missing Faiss on ARM64 Linux or Windows.

For ARM GPU execution, install your compatible source-built `faiss-gpu` wheel
before installing `kumo-engine`, or supply its directory with `--find-links`
when installing `kumo-engine[faiss-gpu]`. The extra requires the version pinned
in core metadata. Service CPU/GPU extras and their source-build workflows keep
their explicit Faiss selection. Core test/dev extras do not select Faiss;
add a compatible backend for retrieval tests.

For an in-tree installation, use `'./packages/kumo-core[faiss-cpu]'`,
`'./packages/kumo-core[faiss-gpu]'`, or `./packages/kumo-core` as the corresponding
package input to your installer, alongside the other in-tree packages and
runtime prerequisites.

The [local SDK example](../../examples/sdk-local/README.md) was tested using
CPU Faiss. You can retain an already installed compatible Faiss variant instead;
run the example to check the local SDK workflow after preparing the environment.

## Working with an existing environment

Use the checkout's `.venv/bin/python` or `.venv/bin/pytest` for Python work.
Use focused [test commands](testing.md) rather than starting every service.
UI and catalog commands are documented in [UI development](ui.md).
See [documentation validation](../maintenance/README.md#validation) for checks
that run without a Kumo application.

`rest.devserver.multi_main` launches the local service stacks. Current
`make devserver` requires an explicit `CGROUP_CONFIG` or `UNFENCED=1` selection.
`make devserver-local` starts local Temporal before invoking the local variant.
These targets start processes and may perform environment preparation; consult
[devserver documentation](../devserver.md) and the actual
[Makefile](../../Makefile) before using them.
