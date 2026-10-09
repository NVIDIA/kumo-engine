# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor


def batched_arange(count: Tensor) -> tuple[Tensor, Tensor]:
    r"""Fast implementation of bached version of :meth:`torch.arange`.

    It essentially does the following:

    >>> batch = torch.cat([torch.full((c,), i) for i, c in enumerate(count)])
    >>> arange = torch.cat([torch.arange(c) for c in count])

    Args:
        count: The count vectors.

    Returns:
        batch: ``batch[i]`` indicates the batch index of ``batched_arange[i]``.
        arange: batched output version of :meth:`torch.arange`.
    """
    ptr = count.new_zeros(count.numel() + 1)
    torch.cumsum(count, dim=0, out=ptr[1:])

    batch = torch.arange(count.numel(), device=count.device)
    batch = batch.repeat_interleave(count, output_size=int(ptr[-1]))

    arange = torch.arange(batch.numel(), device=count.device)
    arange -= ptr[batch]

    return batch, arange
