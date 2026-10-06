# CUDA 13 faiss-gpu wheel

This directory builds a GPU-enabled `faiss-gpu` wheel from [facebookresearch/faiss](https://github.com/facebookresearch/faiss) for Kumo's CUDA runtime.

The public CUDA wheel used by Kumo does not include workstation Blackwell's `sm_120` architecture, so GPU calls fail there with CUDA error 209. This builder adds `sm_120` while keeping a package version compatible with Kumo's public dependency pin.

The build pins the upstream Faiss commit, CUDA base-image digest, CPython and SWIG source checksums, and Python build-tool versions. It includes architectures `75;80;86;89;90;100;120`, covering T4 through workstation Blackwell, and uses CUDA 13 runtime libraries supplied by PyTorch/the container while excluding those libraries and the host-provided NVIDIA driver shim from the wheel.

Build locally:

```bash
docker build -t faiss-cu130-builder \
  -f tools/faiss-gpu-amd64/build.Dockerfile tools/faiss-gpu-amd64/
mkdir -p tools/faiss-gpu-amd64/dist
docker run --rm \
  -v "$PWD/tools/faiss-gpu-amd64:/workspace/pkg:ro" \
  -v "$PWD/tools/faiss-gpu-amd64/dist:/out" \
  faiss-cu130-builder /workspace/pkg/build_wheel.sh
```

Pull requests that change the builder compile and smoke-test the wheel through the stable Kumo-core aggregate check without publishing it. After that validation reaches `master`, an administrator can manually dispatch the builder with `publish` enabled. The build must match the reviewed SHA-256 before publication, and publication never overwrites a differing asset: changed output requires investigation and a new release tag.
