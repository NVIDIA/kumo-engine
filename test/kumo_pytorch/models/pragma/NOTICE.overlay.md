# Where these files come from

The files in this folder are from **pragmatiq** (https://github.com/dynamiq-ai/pragmatiq),
version 1.1.0, commit `eb6928c`, Copyright Dynamiq, licensed under the
Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). pragmatiq is an
independent implementation inspired by the PRAGMA paper (arXiv 2604.08649) and
is not affiliated with or endorsed by Revolut.

## Historical source overlay

The paths in this section match pragmatiq's own (`models/` =
`pragmatiq/models/`, and so on). They describe the original optimization overlay,
not the files shipped by the current reference implementation.

Modified from 1.1.0 (to speed up the PRAGMA training step):
`data/collate.py`, `models/heads.py`, `models/layers.py`, `models/pragmatiq.py`,
`training/masking.py`, `training/optim.py`, `training/pretrainer.py`,
`tests/_ddp_grad_accum_runner.py`, `tests/test_ddp_grad_accum.py`.

New: `models/fused_ce.py`, `models/short_attention.py`, `models/sorted_gather.py`,
`tests/test_compile_blocks.py`, `tests/test_cuda_graph_parts.py`, `tests/test_fused_ce.py`,
`tests/test_short_attention.py`, `tests/test_sorted_gather.py`.

`models/_liger_ce_nptr.py` is adapted from Liger-Kernel 0.8.3
(https://github.com/linkedin/Liger-Kernel), Copyright 2024 LinkedIn Corporation,
BSD 2-Clause License; its header says what was changed.

## Arena reference adaptation

The standalone Arena package retains the event/profile/history architecture,
shared embeddings, time and calendar features, prediction heads, and optional
frozen text encoder. It omits the original data/training machinery.

The reference adaptation replaces the overlay's attention dispatch with native
PyTorch scaled dot-product attention for each segment, and sorted gathering
with ordinary tensor indexing and autograd. It uses tied vocabulary projection
and native cross-entropy. The custom attention, sorted-gather, fused-loss and
copied Liger modules, their kernel tests, optional CUDA requirements, backend
switches, and block-compilation helper have been removed. Public packed-batch
fields and model parameter names/shapes are retained; optional maximum-length
hints are validated rather than trusted to bypass checks.

The source attribution and licenses are retained, including `LICENSE.liger` as
historical attribution for the removed copied kernel. See `REFERENCE.md` for
source revisions, the current implementation scope, and validation status.

## Arena native input interface

`native.py` and `_kumo.py` are Arena-owned integration code. `NativePragmaModel`
consumes emitted Kumo fields and constructs the model's packed tensors inside
forward, reusing the unchanged `PragmaModel` neural core in `pragmatiq.py`.
The existing source attribution, licenses, parameter names and packed interface
are retained. The native interface does not include Kumo's implementation.
