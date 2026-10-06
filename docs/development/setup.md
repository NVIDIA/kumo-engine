# Development setup

Run commands from the relevant checkout. Tool declarations are in
[.mise.toml](../../.mise.toml); Python package inputs are in
[setup.py](../../setup.py), [3rdparty/BUILD](../../3rdparty/BUILD), and package-local
`pyproject.toml` files.

## Setup prerequisites and side effects

The local SDK example requires prepared runtime dependencies, including the
[repository PyG wheel](../../3rdparty/kumo-wheels/README.md#torch_geometric-290-py3-none-anywhl)
and a [Faiss selection](#choose-the-core-faiss-dependency). Public PyG delivery
is tracked in [#31333](https://github.com/kumo-ai/kumo/issues/31333); this guide
describes development setup, not a verified public-package quickstart.

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

The local SDK imports core code that requires Faiss. Bare `kumo-core` does not
install either Faiss variant. Choose one of these installation inputs according
to the workload and environment:

| Package selection | When to use it | Faiss dependency |
| --- | --- | --- |
| `kumo-core[cpu]` | Use CPU Faiss, including CPU workloads on a GPU node | Installs `faiss-cpu` |
| `kumo-core[gpu]` | Use GPU Faiss with a compatible GPU, driver, and CUDA runtime | Installs `faiss-gpu` |
| `kumo-core` | Reuse a compatible Faiss installation already provided by your environment | Installs neither variant |

The [core package metadata](../../packages/kumo-core/pyproject.toml) specifies
the versions for each extra. These extras select the Faiss dependency; they do
not select the PyTorch CPU/CUDA build or provision a complete GPU environment.
A GPU node can still use CPU Faiss. Select one variant per environment rather
than combining `[cpu,gpu]`.

For an in-tree installation, use `'./packages/kumo-core[cpu]'`,
`'./packages/kumo-core[gpu]'`, or `./packages/kumo-core` as the corresponding
package input to your installer, alongside the other in-tree packages and
runtime prerequisites.

The [local SDK example](../../examples/sdk-local/README.md) was tested using
`[cpu]`. You can retain an already installed compatible Faiss variant instead;
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
