# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernel: one row's cross-entropy and its gradient, written over the logits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 4096
"""Logits one program reads per step along a row."""


def cross_entropy_rows(
    logits: Tensor, targets: Tensor, label_smoothing: float, ignore_index: int, bad: Tensor
) -> Tensor:
    """Return each row's cross-entropy loss (fp32) and write its gradient, over
    `logits [rows, vocab]` in place. A row with target `ignore_index` gets loss 0
    and gradient 0. Any other target outside `[0, vocab)` sets `bad` (int32, 0-dim) to 1
    and its row also gets loss 0 and gradient 0. One program per row."""
    rows, vocab = logits.shape
    loss = torch.empty(rows, dtype=torch.float32, device=logits.device)
    _kernel[(rows,)](
        logits,
        logits.stride(0),
        targets,
        targets.stride(0),
        loss,
        vocab,
        label_smoothing,
        ignore_index,
        bad,
        BLOCK=min(BLOCK, triton.next_power_of_2(vocab)),
        num_warps=8,  # type: ignore[call-arg]  # a launch option, not a kernel argument
    )
    return loss


def assert_targets_in_vocab(bad: Tensor) -> None:
    """Fail a device-side assert, without a host sync, if `bad` is not 0."""
    _assert_zero[(1,)](bad)


@triton.jit(debug=True)
def _assert_zero(bad):  # type: ignore[no-untyped-def]
    tl.device_assert(
        tl.load(bad) == 0,
        "cross-entropy target is neither in [0, vocab) nor the ignore_index (-100)",
    )


@triton.jit
def _kernel(
    logits, stride, targets, target_stride, loss, vocab, eps, ignore_index, bad, BLOCK: tl.constexpr
):  # type: ignore[no-untyped-def]
    row = tl.program_id(0).to(tl.int64)
    ptr = logits + row * stride
    target = tl.load(targets + row * target_stride)
    outside = (target != ignore_index) & ((target < 0) | (target >= vocab))
    tl.store(bad, 1, mask=outside)
    valid = (target != ignore_index) & ~outside
    peak = tl.full([], float("-inf"), tl.float32)
    total = tl.zeros([], tl.float32)
    total_logits = tl.zeros([], tl.float32)
    # Pass 1: online log-sum-exp and the sum of logits.
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
        new_peak = tl.maximum(peak, tl.max(x, 0))
        total = total * tl.exp(peak - new_peak) + tl.sum(tl.exp(x - new_peak), 0)
        total_logits += tl.sum(tl.where(cols < vocab, x, 0.0), 0)
        peak = new_peak
    lse = peak + tl.log(total)
    x_target = tl.load(ptr + tl.where(valid, target, 0)).to(tl.float32)
    row_loss = lse - (1 - eps) * x_target - eps * total_logits / vocab
    tl.store(loss + row, tl.where(valid, row_loss, 0.0))
    # Pass 2: softmax - (1 - eps) * one_hot - eps / vocab over the logits; 0 if ignored.
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < vocab, other=0.0).to(tl.float32)
        grad = tl.exp(x - lse) - eps / vocab
        grad = tl.where(cols == target, grad - (1 - eps), grad)
        grad = tl.where(valid, grad, 0.0)
        tl.store(ptr + cols, grad.to(logits.dtype.element_ty), mask=cols < vocab)
