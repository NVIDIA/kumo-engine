# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-native FSDP2 trainer extra (experimental).

Drives the Blueprint's training stage: the HuggingFace causal LM behind a
`model.backbone: huggingface` composition, sharded with PyTorch's `fully_shard`,
trained on the TFM `TransactionBatch` contract, and saved as a consolidated
HuggingFace-format checkpoint.

The loop, the loss, the schedule, and the sharding are all this SDK's. The loss
in particular is the composed model's own objective (`NextEvent`), so label
alignment is never delegated to the backend. That is the lesson of the
Blueprint's v1 checkpoint, whose dataset and loss each assumed the other would
shift the labels (Blueprint v1 objectives).

This backend never falls back to the eager loop. If CUDA or a compatible
backbone is missing it raises. Use `training.backend: eager` for the
CPU-correct baseline.

Under `torchrun`, `join_distributed_group` creates the process group the
launcher only sets up the environment for, pins each rank to its local GPU,
shards the batch list across ranks, and gathers an unsharded state dict for the
consolidated checkpoint. Without that call every rank reports `world_size` 1,
shards nothing, and trains on identical data. Single-process runs are unchanged
and still shard nothing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from tfm.config import OptimizerConfig
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.capabilities import BackendCapabilities
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.extras.capability import require_extra
from tfm.training.distributed import (
    distributed_info,
    join_distributed_group,
    launcher_topology,
    leave_distributed_group,
    shard_batches,
)
from tfm.training.distributed import mean_across_ranks as _mean_across_ranks
from tfm.training.progress import TrainingProgress
from tfm.training.reproducibility import seed_everything
from tfm.training.trainer import TrainableBatch, epoch_order, lr_at_step, real_tokens

__all__ = [
    "Fsdp2NextEventTrainer",
    "TransactionBatchDataset",
    "distributed_info",
    "join_distributed_group",
    "launcher_topology",
    "shard_batches",
    "shard_with_fsdp2",
]

log = logging.getLogger(__name__)

#: Steps between rank-averaged entries in `loss_curve.json`.
LOSS_CURVE_EVERY = 10


class TransactionBatchDataset(Dataset):
    """Yields token tensors from already-collated TFM batches."""

    def __init__(self, batches: Sequence[TransactionBatch]) -> None:
        self._batches = list(batches)

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = self._batches[index]
        item = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
        }
        if batch.labels is not None:
            item["labels"] = batch.labels
        return item

    def batches(self) -> Iterator[TransactionBatch]:
        yield from self._batches


def shard_with_fsdp2(model: nn.Module, *, world_size: int) -> nn.Module:
    """Shard a HuggingFace causal LM with PyTorch-native FSDP2.

    Each transformer block is its own FSDP2 unit so parameters are all-gathered
    per block rather than all at once.

    FSDP2 unshards parameters in a pre-forward hook on the module it wrapped, so
    a unit whose `forward` is never called keeps its parameters sharded. This
    backbone calls `model.model` and `model.lm_head` directly rather than the
    `...ForCausalLM` root, so wrapping the root would leave the embedding
    sharded and fail with "got mixed torch.Tensor and DTensor". Wrap the modules
    that are actually invoked instead.
    """
    if world_size <= 1:
        return model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if inner is None or layers is None:
        raise UnsupportedCombinationError(
            f"FSDP2 sharding expected a HuggingFace decoder exposing .model.layers, got "
            f"{type(model).__name__}. Run single-process or use training.backend: eager."
        )
    mesh = init_device_mesh("cuda", (world_size,))
    # Master weights stay fp32; compute runs in bf16 and gradients reduce in fp32.
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(inner, mesh=mesh, mp_policy=policy)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        # Called from `logits()`, outside the base model's forward.
        fully_shard(lm_head, mesh=mesh, mp_policy=policy)
    return model


def _full_state_dict(model: nn.Module, world_size: int) -> dict[str, Any] | None:
    """Gather sharded FSDP2 parameters into one unsharded state dict on rank 0.

    Every rank must call this: gathering is itself a collective. Returns `None`
    off rank 0, which has nothing to write.
    """
    if world_size <= 1:
        return None
    from torch.distributed.checkpoint.state_dict import (  # type: ignore[import-not-found]
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def _save_consolidated_hf_checkpoint(
    model: nn.Module,
    ckpt_dir: Path,
    state_dict: dict[str, Any] | None = None,
) -> None:
    """Write a consolidated HuggingFace-format checkpoint, as the Blueprint expects."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if state_dict is not None:
        # Sharded run: write the gathered weights with the HuggingFace writer.
        from transformers import PreTrainedModel  # type: ignore[import-not-found]

        if not isinstance(model, PreTrainedModel):
            raise UnsupportedCombinationError(
                f"Sharded save needs a HuggingFace PreTrainedModel, got {type(model).__name__}."
            )
        PreTrainedModel.save_pretrained(model, ckpt_dir, state_dict=state_dict)
        return
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
        return
    save_pretrained(ckpt_dir)


class Fsdp2NextEventTrainer:
    """Train a composed `TrxFM` over `TransactionBatch` with torch-native FSDP2.

    Requires CUDA and `model.backbone: huggingface`. The backbone's causal LM is
    sharded in place and trained under the composed model's own objective, so
    the `NextEvent` label alignment and the `TransactionBatch` contract are the
    same as the eager loop's.
    """

    capabilities = BackendCapabilities(
        name="fsdp2",
        support_tier="experimental",
        backbones=frozenset({"huggingface"}),
        batch_layouts=frozenset({"flat_fields"}),
        requires_cuda=True,
        multi_process=True,
        # The HF backbone cannot take a block-diagonal segment mask.
        segment_masks=False,
        consolidated_hf_export=True,
        shards_parameters=True,
        precisions=frozenset({"bf16"}),
    )

    def __init__(self, *, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir

    def fit(
        self,
        model: nn.Module,
        batches: Sequence[TrainableBatch],
        *,
        max_steps: int,
        learning_rate: float,
        seed: int,
        device: torch.device,
        optimizer: OptimizerConfig | None = None,
        shuffle: bool = False,
        precision: str = "bf16",
        progress: TrainingProgress | None = None,
    ) -> dict[str, float | str]:
        settings = optimizer if optimizer is not None else OptimizerConfig()
        name = self.capabilities.name
        if settings.name != "adamw" or settings.max_grad_norm is not None:
            raise UnsupportedCombinationError(
                f"training.backend: {name} trains with AdamW and no clipping; "
                "optimizer.name: muon and optimizer.max_grad_norm need eager or ddp."
            )
        if precision not in self.capabilities.precisions:
            raise UnsupportedCombinationError(
                f"training.backend: {name} computes in bf16 with fp32 master weights; "
                f"set training.precision: bf16, not {precision!r}."
            )
        require_extra("huggingface", "transformers")
        if not torch.cuda.is_available():
            raise UnsupportedCombinationError(
                f"training.backend: {name} requires CUDA. Use training.backend: eager on CPU."
            )
        if device.type != "cuda":
            raise UnsupportedCombinationError(
                f"training.backend: {name} requires a CUDA device, got {device!r}. "
                "Set runtime.device to cuda or auto."
            )

        from tfm.extras.huggingface import HuggingFaceCausalBackbone

        backbone = getattr(model, "backbone", None)
        if not isinstance(backbone, HuggingFaceCausalBackbone):
            raise UnsupportedCombinationError(
                f"training.backend: {name} requires model.backbone: huggingface, whose "
                f"causal LM it shards. Got backbone {type(backbone).__name__}."
            )

        transaction_batches = [b for b in batches if isinstance(b, TransactionBatch)]
        if len(transaction_batches) != len(batches):
            raise UnsupportedCombinationError(
                f"training.backend: {name} consumes TransactionBatch only; the "
                "profile/history stack must use training.backend: eager."
            )
        if not transaction_batches:
            raise UnsupportedCombinationError(f"training.backend: {name} received no batches.")

        seed_everything(seed)
        launch_world_size, local_rank = launcher_topology()
        if launch_world_size > 1:
            # Every rank resolved runtime.device independently and would otherwise
            # pile onto the same GPU.
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)

        world_size, rank, local_rank = join_distributed_group("cuda")
        if world_size > 1:
            transaction_batches = shard_batches(
                transaction_batches, world_size=world_size, rank=rank
            )
        model.to(device)
        # Keep master weights fp32 and get bf16 compute from autocast (single
        # process) or the FSDP2 mixed-precision policy (sharded): training bf16
        # parameters directly can produce a checkpoint that overflows on resume.
        model.to(dtype=torch.float32)
        backbone.model = shard_with_fsdp2(backbone.model, world_size=world_size)
        model.train()

        dataset = TransactionBatchDataset(transaction_batches)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=settings.betas,
            weight_decay=settings.weight_decay,
            foreach=False,
        )
        losses: list[float] = []
        curve: list[dict[str, float | int]] = []
        step = 0
        epoch = 0
        if progress is not None:
            progress.start(
                max_steps=max_steps, batches_per_epoch=len(dataset), world_size=world_size
            )
        while step < max_steps:
            order = epoch_order(len(dataset), seed=seed, epoch=epoch, shuffle=shuffle)
            epoch += 1
            for index in order:
                if step >= max_steps:
                    break
                batch = transaction_batches[index]
                rate = lr_at_step(
                    step,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    optimizer=settings,
                )
                for group in opt.param_groups:
                    group["lr"] = rate
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch.to(device), compute_loss=True)
                if output.loss is None:
                    raise UnsupportedCombinationError(
                        f"{name} backend expected a loss; the objective returned none."
                    )
                output.loss.backward()
                opt.step()
                losses.append(float(output.loss.detach().float().cpu()))
                step += 1
                if progress is not None:
                    # FSDP2 does not clip, so no gradient norm is computed.
                    progress.step(
                        step,
                        loss=losses[-1],
                        lr=rate,
                        grad_norm=float("nan"),
                        tokens=real_tokens(batch),
                    )
                if step % LOSS_CURVE_EVERY == 0 or step == max_steps:
                    # Collective: every rank must reach this on the same steps.
                    mean = _mean_across_ranks(losses[-1], world_size=world_size, device=device)
                    curve.append({"step": step, "loss": mean, "lr": rate, "epoch": epoch})
                    if rank == 0 and step % 100 == 0:
                        log.info("step %d/%d  loss %.4f  lr %.3e", step, max_steps, mean, rate)

        if progress is not None:
            progress.finish()
        first_loss = _mean_across_ranks(
            losses[0] if losses else float("nan"), world_size=world_size, device=device
        )
        final_loss = _mean_across_ranks(
            losses[-1] if losses else float("nan"), world_size=world_size, device=device
        )

        ckpt_dir = self.checkpoint_dir
        checkpoint_path = ""
        # Collective: all ranks gather, rank 0 writes.
        gathered = _full_state_dict(backbone.model, world_size) if ckpt_dir is not None else None
        if ckpt_dir is not None and rank == 0:
            _save_consolidated_hf_checkpoint(backbone.model, ckpt_dir, gathered)
            (ckpt_dir / "tfm_backend_meta.json").write_text(
                json.dumps(
                    {
                        "backend": name,
                        "world_size": world_size,
                        "steps": step,
                        "format": "huggingface-consolidated",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (ckpt_dir / "loss_curve.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")
            checkpoint_path = str(ckpt_dir)

        leave_distributed_group(world_size)

        return {
            "backend": name,
            "final_loss": final_loss,
            "first_loss": first_loss,
            "steps": float(step),
            "precision": precision,
            "world_size": float(world_size),
            "rank": float(rank),
            "dataset_len": float(len(dataset)),
            "hf_checkpoint": checkpoint_path,
        }
