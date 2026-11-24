# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-rank adaptation of frozen linear layers.

The frozen-embedding probe answers "what did pretraining learn?". LoRA answers
a different question: "how much better does this get if the encoder may move a
little?". PRAGMA reports both, so this kit needs both, and the probe stays the
reference baseline.

Implemented directly on `nn.Linear` rather than through `peft` because the
adapted models here are this kit's own modules, not HuggingFace
`PreTrainedModel`s, and because it keeps the CPU test path dependency-free.

`B` is initialized to zero, so an adapted model starts numerically identical to
the model it wrapped. That is what makes a LoRA run comparable to its own
frozen baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from tfm.contracts.errors import IncompatibleCompositionError


@dataclass(frozen=True)
class LoRA:
    """Which linear layers to adapt, and how much capacity to give them.

    `targets` are matched against the suffix of each submodule's dotted name,
    so `qkv` adapts every block's attention projection. The default targets the
    attention projections, which is the usual choice and the smallest change
    that still lets attention re-weight what it reads.
    """

    name: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("qkv", "out")
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise IncompatibleCompositionError(f"LoRA rank must be >= 1, got {self.rank}.")
        if not self.targets:
            raise IncompatibleCompositionError("LoRA needs at least one target module name.")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class LoraStats:
    """What `apply_lora` changed, for the run manifest."""

    n_adapted: int
    n_trainable: int
    n_frozen: int
    adapted_modules: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, float | int | list[str]]:
        total = self.n_trainable + self.n_frozen
        return {
            "n_adapted_modules": self.n_adapted,
            "n_trainable_params": self.n_trainable,
            "n_frozen_params": self.n_frozen,
            "trainable_fraction": self.n_trainable / total if total else 0.0,
            "adapted_modules": list(self.adapted_modules),
        }


class LoraLinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable rank-`r` update.

    Computes `base(x) + scaling * dropout(x) @ A.T @ B.T`.
    """

    def __init__(self, base: nn.Linear, spec: LoRA) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = spec.scaling
        self.lora_a = nn.Parameter(torch.empty(spec.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, spec.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(spec.dropout) if spec.dropout > 0.0 else nn.Identity()

    def forward(self, hidden: Tensor) -> Tensor:
        update = self.drop(hidden) @ self.lora_a.t() @ self.lora_b.t()
        return self.base(hidden) + self.scaling * update

    def merged_weight(self) -> Tensor:
        """The equivalent single weight matrix, for exporting a merged checkpoint."""
        return self.base.weight + self.scaling * (self.lora_b @ self.lora_a)


def _matches(name: str, targets: tuple[str, ...]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def apply_lora(model: nn.Module, spec: LoRA) -> LoraStats:
    """Freeze `model` and replace targeted `nn.Linear` layers with `LoraLinear`.

    Mutates `model` in place. Pass a copy to keep the frozen baseline. Raises
    when no target matches, because a LoRA run that silently adapted nothing
    would look like a successful fine-tune of a frozen model.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    adapted: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _matches(full, spec.targets):
                setattr(module, child_name, LoraLinear(child, spec))
                adapted.append(full)
    if not adapted:
        raise IncompatibleCompositionError(
            f"LoRA targets {spec.targets} matched no nn.Linear in "
            f"{type(model).__name__}. Check the module names."
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return LoraStats(
        n_adapted=len(adapted),
        n_trainable=trainable,
        n_frozen=frozen,
        adapted_modules=tuple(adapted),
    )


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Trainable parameters after `apply_lora`, for the optimizer."""
    return [p for p in model.parameters() if p.requires_grad]
