# Public SDK and graph test environment

Use [the public test guide](../../docs/development/public-tests.md) for commands and results.
The [runner](../../scripts/test_public.py) selects the tests and builds their
containers. The lightweight reader image remains in `tests/public/`.

The `sdk` image installs only the API and service SDK test packages.
The `full` image installs `kumo-engine`, native DiskGraph, and hosted Python
modules, with CPU PyTorch and GPU Faiss. It does not test GPU execution.

`Dockerfile.diskgraph` builds the checked-in native source with public tools.
Its Rust settings come from `scripts/diskgraph wheel-env`.
The full image accepts the resulting wheel through Docker's `wheel` build
context. Runtime dependencies come from `uv.lock`, including
the PyG wheel checked into `3rdparty/kumo-wheels`. No private packages are fetched.

The full image uses writable, disposable source files because some existing
tests write beside their source. Tests run without root permissions.
It skips the shared `test/conftest.py`, which imports unrelated hosted setup.
Local test folders load the shared cache fixture through `conftest.py`.
They keep their explicit statistics setup. SDK tests keep their own fixtures.
Shared SQL tests retain the core package's pytest policy.

MinIO and its client build from pinned public source. The test runner creates
fake accounts and a temporary TLS certificate, then removes the service and
its private network. Each run uses a separate Compose project.

The full image emits a Snowflake warning because Arrow 24 exceeds the range
for its pandas extra. Snowflake reader tests run in the separate Arrow 23 image.
See the [test matrix](../../docs/connector-test-matrix.md) for coverage and limits.
