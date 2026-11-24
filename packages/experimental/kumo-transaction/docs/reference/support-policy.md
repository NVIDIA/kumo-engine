# Support policy

The core export surface supports the synthetic decoder-fraud recipe on CPU
(required CI path) and on CUDA when `runtime.device` is `auto` or `cuda`. HuggingFace,
RAPIDS and TabFormer paths are optional extras and unsupported in CI.
Do not declare a feature `reference` without recipe smoke and contract tests.
