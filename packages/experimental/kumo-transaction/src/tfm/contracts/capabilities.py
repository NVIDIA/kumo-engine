# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component capability declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    name: str
    causal: bool
    bidirectional: bool
    requires_timestamps: bool
    support_tier: str


@dataclass(frozen=True)
class BackendCapabilities:
    """What a training backend can execute (backend contracts).

    A backend executes the model's semantics; it does not define them. The
    objective, label alignment, masks, and batch contract belong to the model,
    and every backend must reproduce eager training on them. What a backend
    declares here is only how it executes: where, at what scale, and which
    optional features it supports. The runtime refuses a configuration that
    needs something the selected backend does not declare, before any data is
    batched or any step is taken.
    """

    name: str
    support_tier: str
    backbones: frozenset[str]
    """`model.backbone` values it can train: `eager`, `huggingface`."""
    batch_layouts: frozenset[str]
    """Tokenizer layouts whose batches it consumes, e.g. `flat_fields`."""
    requires_cuda: bool
    multi_process: bool
    """Runs data-parallel under a launcher such as `torchrun`."""
    segment_masks: bool
    """Honors packed-row segment boundaries, so `training.packing` is sound."""
    consolidated_hf_export: bool
    """Writes a single-file-set HuggingFace checkpoint for HF backbones.

    Required of every backend that accepts the `huggingface` backbone: it is
    the checkpoint format every downstream consumer reads.
    """
    distributed_checkpoint: bool = False
    """Writes sharded, resumable distributed checkpoints. Optional."""
    tensor_parallel: bool = False
    pipeline_parallel: bool = False
    shards_parameters: bool = False
    """Each rank holds a slice of the weights after training (FSDP), rather
    than a full replica (DDP). A sharded run's only whole checkpoint is its
    consolidated HuggingFace export."""
    precisions: frozenset[str] = frozenset({"fp32"})
    """`training.precision` values it executes as declared: `fp32`, `bf16`."""
