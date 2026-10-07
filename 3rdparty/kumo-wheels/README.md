# Vendored Python wheels

This directory holds pre-built Python wheels that uv can't resolve
from the public package index — usually because the upstream package
doesn't ship binary wheels and has a broken or missing PEP-517 build
declaration.

The root `pyproject.toml` lists this directory under `[tool.uv].find-links`;
dropping a wheel here makes it available to `uv lock` and `uv sync`, with the
wheel's SHA256 recorded in `uv.lock`.

## CUDA Faiss artifact

AMD64 GPU image builds download a checksum-pinned custom CUDA Faiss wheel from a versioned Kumo GitHub Release into this directory. The exact release tag, asset name, and SHA-256 are reviewed in the GPU image workflow; the wheel itself is never committed. ARM64 image builds continue using their architecture-specific source build inside the Dockerfile.

## `torch_geometric-2.9.0-py3-none-any.whl`

`kumo-core` uses the unreleased `torch-geometric` revision
`be5e445f73517440a92f26c956753e7f29d45b16`. Its universal wheel is
included for the same reason `pytorch-frame` used to be vendored here:
pytorch-frame's own unreleased revision shipped a real `0.3.0` PyPI
release, so it now resolves straight from the public index instead.

```bash
TORCH_GEOMETRIC_REF=be5e445f73517440a92f26c956753e7f29d45b16
TORCH_GEOMETRIC_URL=git+https://github.com/pyg-team/pytorch_geometric.git
python -m pip wheel --no-deps \
  "${TORCH_GEOMETRIC_URL}@${TORCH_GEOMETRIC_REF}" \
  -w /tmp/torch-geometric-wheel
cp /tmp/torch-geometric-wheel/torch_geometric-2.9.0-py3-none-any.whl \
  3rdparty/kumo-wheels/
```
