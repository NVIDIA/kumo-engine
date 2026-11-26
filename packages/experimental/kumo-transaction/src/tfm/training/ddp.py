# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel training with PyTorch DDP (experimental).

Every rank holds a full replica of the model and trains on its own share of the
batches; DDP averages gradients during backward. Nothing is sharded, so this is
the backend for models whose weights and optimizer state fit on one device,
which is every model in this kit up to ~100M parameters. The loop, loss,
schedule, and batch contract are the eager backend's own (`train_model`), so
parity with eager is by construction rather than by reimplementation.

Launch under `torchrun`. CUDA ranks use NCCL, CPU ranks use Gloo. Run with one
process, it is the eager loop. `training.batch_size` is per rank.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tfm.config import OptimizerConfig
from tfm.contracts.capabilities import BackendCapabilities
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.training.distributed import (
    distributed_info,
    join_distributed_group,
    launcher_topology,
    leave_distributed_group,
    mean_across_ranks,
    shard_batches,
)
from tfm.training.progress import TrainingProgress
from tfm.training.trainer import BatchSource, TrainableBatch, export_huggingface, train_model


@dataclass
class _Loss:
    loss: torch.Tensor | None


class _LossOnly(nn.Module):
    """The model with its loss as the only output.

    DDP looks for parameters without gradients by walking back from the
    forward outputs. A pretraining step also returns task-head logits, such as
    `heads: fraud`, that no loss reads; walking back from them would make DDP
    wait for head gradients that never come. From the loss alone, those heads
    are unused and skipped.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: TrainableBatch, compute_loss: bool = True) -> _Loss:
        return _Loss(self.model(batch, compute_loss=compute_loss).loss)


class DdpTrainer:
    """Replicate the model on every rank and average gradients (backend contracts)."""

    capabilities = BackendCapabilities(
        name="ddp",
        support_tier="experimental",
        backbones=frozenset({"eager", "huggingface", "gru"}),
        batch_layouts=frozenset({"flat_fields", "flat_key_value", "key_value_time"}),
        requires_cuda=False,
        multi_process=True,
        segment_masks=True,
        consolidated_hf_export=True,
        shards_parameters=False,
        precisions=frozenset({"fp32", "bf16"}),
    )

    def __init__(self, *, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir

    def fit(
        self,
        model: nn.Module,
        batches: BatchSource[TrainableBatch],
        *,
        max_steps: int,
        learning_rate: float,
        seed: int,
        device: torch.device,
        optimizer: OptimizerConfig | None = None,
        shuffle: bool = False,
        precision: str = "fp32",
        progress: TrainingProgress | None = None,
    ) -> dict[str, float | str]:
        launch_world_size, local_rank = launcher_topology()
        group_world_size, _ = distributed_info()
        if not isinstance(batches, Sequence) and launch_world_size == group_world_size == 1:
            stats = train_model(
                model,
                batches,
                max_steps=max_steps,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
                optimizer=optimizer,
                shuffle=shuffle,
                precision=precision,
                progress=progress,
            )
            return {
                **stats,
                "backend": self.capabilities.name,
                "world_size": 1.0,
                "rank": 0.0,
                "batches_per_rank": float(len(batches)),
                "hf_checkpoint": export_huggingface(model, self.checkpoint_dir),
            }
        if not isinstance(batches, Sequence):
            raise UnsupportedCombinationError(
                "training.backend: ddp gives each rank every world_size-th batch by index, but "
                f"received a {type(batches).__name__}, which can only be iterated. Pass a "
                "sequence of batches, or train with training.backend: eager in one process."
            )
        if not batches:
            raise UnsupportedCombinationError("training.backend: ddp received no batches.")
        if launch_world_size > 1 and device.type == "cuda":
            # Every rank resolved runtime.device independently and would otherwise
            # pile onto the same GPU.
            device = torch.device(f"cuda:{local_rank}")
        world_size, rank, local_rank = join_distributed_group(device.type)
        shard = shard_batches(batches, world_size=world_size, rank=rank)

        model.to(device)
        replica: nn.Module = model
        if world_size > 1:
            replica = DistributedDataParallel(
                _LossOnly(model),
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        stats = train_model(
            replica,
            shard,
            max_steps=max_steps,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            optimizer=optimizer,
            shuffle=shuffle,
            precision=precision,
            progress=progress,
            world_size=world_size,
        )
        for key in ("first_loss", "final_loss"):
            stats[key] = mean_across_ranks(float(stats[key]), world_size=world_size, device=device)

        checkpoint = ""
        if rank == 0:
            checkpoint = export_huggingface(model, self.checkpoint_dir)
        leave_distributed_group(world_size)
        return {
            **stats,
            "backend": self.capabilities.name,
            "world_size": float(world_size),
            "rank": float(rank),
            "batches_per_rank": float(len(shard)),
            "hf_checkpoint": checkpoint,
        }
