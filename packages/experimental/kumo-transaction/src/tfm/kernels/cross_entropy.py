# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-entropy of a linear layer's logits, computed in row chunks so the full
`[rows, vocab]` logits never exist."""

from __future__ import annotations

import torch
from torch import Tensor

CHUNK_BYTES = 256 * 2**20
"""Bytes of one chunk's logits."""

IGNORE_INDEX = -100
"""The target that adds no loss and no gradient: `F.cross_entropy`'s default
`ignore_index`."""


def linear_cross_entropy(
    hidden: Tensor, weight: Tensor, targets: Tensor, *, label_smoothing: float = 0.0
) -> Tensor:
    """`F.cross_entropy(hidden @ weight.T, targets, label_smoothing=label_smoothing)`
    for `hidden [rows, D]`, `weight [vocab, D]` and `targets [rows]`. Each target is a
    class index in `[0, vocab)` or `IGNORE_INDEX` (-100). A row with target
    `IGNORE_INDEX` adds no loss and no gradient; the loss is the mean over the other
    rows. Any other target fails: on CPU at once, on CUDA as a device-side assert
    after the call, as `F.cross_entropy` does. If every target is `IGNORE_INDEX`, the
    loss and gradients are 0, where `F.cross_entropy` gives NaN.

    Runs in `hidden`'s dtype, casting `weight` to it, as autocast would. The forward
    pass also computes both gradients, chunk by chunk; the backward pass scales them.
    """
    loss, _, _ = _fused(hidden, weight.to(hidden.dtype), targets, label_smoothing)
    return loss


@torch.library.custom_op("tfm::linear_cross_entropy", mutates_args=())
def _fused(
    hidden: Tensor, weight: Tensor, targets: Tensor, label_smoothing: float
) -> tuple[Tensor, Tensor, Tensor]:
    """`(mean loss, d loss / d hidden, d loss / d weight)`, over the rows not ignored."""
    rows, vocab = hidden.shape[0], weight.shape[0]
    chunk = max(1, CHUNK_BYTES // (vocab * hidden.element_size()))
    loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
    grad_hidden = torch.empty_like(hidden)
    grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
    bad = torch.zeros((), dtype=torch.int32, device=hidden.device)
    with torch.autocast(hidden.device.type, enabled=False):
        for start in range(0, rows, chunk):
            part = slice(start, start + chunk)
            # The logits become their gradient in place.
            grad = hidden[part] @ weight.t()
            loss += _loss_and_grad(grad, targets[part], label_smoothing, bad).sum()
            torch.mm(grad, weight, out=grad_hidden[part])
            grad_weight += grad.t() @ hidden[part]
    # The check and the count stay on the device, so the step does not wait for them.
    if bad.is_cuda:
        from tfm.kernels._cross_entropy_triton import assert_targets_in_vocab

        assert_targets_in_vocab(bad)
    scale = 1.0 / (targets != IGNORE_INDEX).sum().clamp(min=1)
    return loss * scale, grad_hidden.mul_(scale), (grad_weight * scale).to(weight.dtype)


@_fused.register_fake
def _(
    hidden: Tensor, weight: Tensor, targets: Tensor, label_smoothing: float
) -> tuple[Tensor, Tensor, Tensor]:
    loss = hidden.new_empty((), dtype=torch.float32)
    return loss, torch.empty_like(hidden), torch.empty_like(weight)


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: tuple) -> None:  # type: ignore[type-arg]
    ctx.save_for_backward(output[1], output[2])


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_loss: Tensor, *_: Tensor
) -> tuple[Tensor, Tensor, None, None]:
    grad_hidden, grad_weight = ctx.saved_tensors  # type: ignore[attr-defined]
    return grad_loss * grad_hidden, grad_loss * grad_weight, None, None


_fused.register_autograd(_backward, setup_context=_setup_context)


def _loss_and_grad(logits: Tensor, targets: Tensor, label_smoothing: float, bad: Tensor) -> Tensor:
    """Return each row's loss and write its gradient over `logits` in place. A row with
    target `IGNORE_INDEX` gets loss 0 and gradient 0. On CUDA, any other target outside
    `[0, vocab)` sets `bad` to 1."""
    if logits.is_cuda:
        from tfm.kernels._cross_entropy_triton import cross_entropy_rows

        return cross_entropy_rows(logits, targets, label_smoothing, IGNORE_INDEX, bad)
    valid = targets != IGNORE_INDEX
    # Any other target outside [0, vocab) makes `gather` raise.
    index = targets.masked_fill(~valid, 0)[:, None]
    x = logits.float()
    lse = x.logsumexp(-1)
    target = x.gather(1, index).squeeze(1)
    loss = lse - (1 - label_smoothing) * target - label_smoothing * x.mean(-1)
    grad = (x - lse[:, None]).exp_().sub_(label_smoothing / x.shape[1])
    grad.scatter_add_(1, index, grad.new_full((len(targets), 1), label_smoothing - 1))
    logits.copy_(torch.where(valid[:, None], grad, 0.0))
    return torch.where(valid, loss, 0.0)


__all__ = ["CHUNK_BYTES", "IGNORE_INDEX", "linear_cross_entropy"]
