# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizers selected by `training.optimizer.name`.

- `adamw`: one AdamW over every parameter at `training.learning_rate`.
- `muon`: Muon (`GroupedMuon`, `torch.optim.Muon`'s update) on the hidden matrices,
  AdamW on the rest,
  as PRAGMA trains. A hidden matrix is the weight of an `nn.Linear` whose two
  dimensions both exceed 1. Embedding tables (which the KVT models also use as
  the tied output layer), norms, biases, learned slots such as `[EVT]`, and
  one-output heads stay on AdamW, since orthogonalizing them is either
  undefined or harmful. Muon runs at `optimizer.muon.learning_rate`, AdamW at
  `training.learning_rate`; the schedule scales both by the same factor.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn
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


class GroupedMuon(torch.optim.Muon):
    """`torch.optim.Muon`'s update, with the Newton-Schulz iterations batched over
    matrices of one shape and the momentum and parameter updates as `_foreach` ops.

    Same arguments, state (`momentum_buffer` per parameter) and update as
    `torch.optim.Muon`, up to the batched bf16 matmuls' rounding; fewer launches.
    CPU parameters use the reference implementation to preserve exact parity.
    """

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        # CPU addmm and batched baddbmm can use different bf16 kernels. Keep
        # the eager reference update on CPU; grouping targets CUDA launches.
        if all(p.device.type == "cpu" for group in self.param_groups for p in group["params"]):
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            for p in params:
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad)
            bufs = [self.state[p]["momentum_buffer"] for p in params]
            momentum, lr = group["momentum"], float(group["lr"])
            torch._foreach_lerp_(bufs, grads, 1 - momentum)  # pyright: ignore[reportPrivateImportUsage]
            updates = torch._foreach_lerp(grads, bufs, momentum) if group["nesterov"] else bufs  # pyright: ignore[reportPrivateImportUsage]
            by_shape: defaultdict[tuple[torch.Size, torch.dtype, torch.device], list[int]] = (
                defaultdict(list)
            )
            for i, p in enumerate(params):
                by_shape[(p.shape, p.dtype, p.device)].append(i)
            for (shape, _dtype, _device), index in by_shape.items():
                ortho = _batched_newton_schulz(
                    torch.stack([updates[i] for i in index]),
                    group["ns_coefficients"],
                    group["ns_steps"],
                    group["eps"],
                )
                members = [params[i] for i in index]
                torch._foreach_mul_(members, 1 - lr * group["weight_decay"])  # pyright: ignore[reportPrivateImportUsage]
                torch._foreach_add_(  # pyright: ignore[reportPrivateImportUsage]
                    members,
                    list(ortho.unbind(0)),
                    alpha=-_adjusted_lr(lr, group["adjust_lr_fn"], shape),
                )
        return loss


def _batched_newton_schulz(
    grads: Tensor, coefficients: tuple[float, float, float], steps: int, eps: float
) -> Tensor:
    """`torch.optim.Muon`'s quintic Newton-Schulz orthogonalisation of each `[k, m, n]`
    slice, in bf16."""
    a, b, c = coefficients
    x = grads.bfloat16()
    tall = grads.size(-2) > grads.size(-1)
    if tall:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.mT
        update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, update, x, beta=a)
    return x.mT if tall else x


def _adjusted_lr(lr: float, adjust_lr_fn: str | None, shape: torch.Size) -> float:
    rows, cols = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    return lr


def build_optimizers(
    model: nn.Module, *, learning_rate: float, settings: OptimizerConfig
) -> list[Optimizer]:
    """The optimizers for one run. Each param group records its `base_lr`."""
    name = settings.name
    if name == "adamw":
        opt = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
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
        muon = GroupedMuon(
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
    "GroupedMuon",
    "ParameterSplit",
    "build_optimizers",
    "muon_parameter_split",
    "set_schedule_factor",
]
