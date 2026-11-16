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
