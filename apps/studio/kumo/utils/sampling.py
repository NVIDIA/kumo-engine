# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import islice
from random import randint
from typing import Iterable, Optional

import torch


def reservoir_sample(iterable: Iterable, k: int, n: Optional[int] = None):
    r"""Returns a list of k samples randomly selected from an iterable object
    of n objects, where n is either a very large number of unknown number.

    Args:
        iterable (Iterable): A iterable object.
        k (int): The sample size.
        n (int): The maximal sample size, by default it is None to sample from
        all elements in from iterable.
    """
    iterator = iter(iterable)
    values = list(islice(iterator, k))
    i = k
    for v in iterator:
        select = randint(0, i + 1)
        if select < k:
            values[select] = v
        i += 1
        if n is not None and i > n:
            break
    return values


def random_sample_from_mask(mask: torch.Tensor, n: int):
    r"""Random sample `n` elements from the mask."""
    # TODO(amitabha): Perhaps make faster by grouped WRS to stay with torch
    # multinomial while working around the 2^24 category limit.
    if mask.sum() <= n:
        return mask

    ret = torch.zeros(mask.shape, dtype=mask.dtype)
    available_indices = torch.nonzero(mask).squeeze()
    shuffle = torch.randperm(available_indices.shape[0])
    id = available_indices[shuffle[:n]]
    ret[id] = True
    return ret
