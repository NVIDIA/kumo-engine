# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizers selected by `training.optimizer.name`.

- `adamw`: one AdamW over every parameter at `training.learning_rate`.
- `muon`: Muon (`torch.optim.Muon`) on the hidden matrices, AdamW on the rest,
  as PRAGMA trains. A hidden matrix is the weight of an `nn.Linear` whose two
  dimensions both exceed 1. Embedding tables (which the KVT models also use as
  the tied output layer), norms, biases, learned slots such as `[EVT]`, and
  one-output heads stay on AdamW, since orthogonalizing them is either
  undefined or harmful. Muon runs at `optimizer.muon.learning_rate`, AdamW at
  `training.learning_rate`; the schedule scales both by the same factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from tfm.config import OptimizerConfig
from tfm.contracts.errors import UnsupportedCombinationError


@dataclass(frozen=True)
class ParameterSplit:
    """Trainable parameters by optimizer, with their qualified names."""

    muon: tuple[tuple[str, nn.Parameter], ...]
    adamw: tuple[tuple[str, nn.Parameter], ...]


def muon_parameter_split(model: nn.Module) -> ParameterSplit:
    """Split `model`'s trainable parameters into Muon and AdamW groups."""
    hidden: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear) and min(module.weight.shape) > 1:
            hidden.add(id(module.weight))
    muon: list[tuple[str, nn.Parameter]] = []
    adamw: list[tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (muon if id(param) in hidden else adamw).append((name, param))
    return ParameterSplit(muon=tuple(muon), adamw=tuple(adamw))


def build_optimizers(
    model: nn.Module, *, learning_rate: float, settings: OptimizerConfig
) -> list[Optimizer]:
    """The optimizers for one run. Each param group records its `base_lr`."""
    name = settings.name
    if name == "adamw":
        opt = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=settings.betas,
            weight_decay=settings.weight_decay,
        )
        for group in opt.param_groups:
            group["base_lr"] = learning_rate
        return [opt]
    if name == "muon":
        split = muon_parameter_split(model)
        if not split.muon:
            raise UnsupportedCombinationError(
                "optimizer.name: muon found no hidden matrix to orthogonalize in this model."
            )
        muon = torch.optim.Muon(
            [p for _n, p in split.muon],
            lr=settings.muon.learning_rate,
            weight_decay=settings.weight_decay,
            momentum=settings.muon.momentum,
            adjust_lr_fn=settings.muon.adjust_lr,
        )
        for group in muon.param_groups:
            group["base_lr"] = settings.muon.learning_rate
        optimizers: list[Optimizer] = [muon]
        if split.adamw:
            adamw = AdamW(
                [p for _n, p in split.adamw],
                lr=learning_rate,
                betas=settings.betas,
                weight_decay=settings.weight_decay,
            )
            for group in adamw.param_groups:
                group["base_lr"] = learning_rate
            optimizers.append(adamw)
        return optimizers
    raise UnsupportedCombinationError(f"Unknown optimizer.name {name!r}; use 'adamw' or 'muon'.")


def set_schedule_factor(optimizers: list[Optimizer], factor: float) -> None:
    """Set every group's rate to `base_lr * factor`."""
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * factor


__all__ = [
    "ParameterSplit",
    "build_optimizers",
    "muon_parameter_split",
    "set_schedule_factor",
]
