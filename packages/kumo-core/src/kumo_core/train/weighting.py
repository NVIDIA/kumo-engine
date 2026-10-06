# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor


def build_binary_class_weight(
    target: Tensor,
    class_weight_map: dict[bool, float],
    out_dtype: torch.dtype = torch.float,
) -> Tensor:
    """Build per-row class weights for binary targets.

    Raises:
        ValueError: If a target class present in `target` is missing
            from `class_weight_map`.
    """
    target = target.bool().view(-1)
    observed_classes = {bool(v) for v in target.unique().tolist()}
    missing_classes = sorted(class_value for class_value in observed_classes
                             if class_value not in class_weight_map)
    if missing_classes:
        raise ValueError("Missing majority sampling weights for target classes "
                         f"{missing_classes}")

    class_weight = torch.empty_like(target, dtype=out_dtype)
    for key, value in class_weight_map.items():
        class_weight[target == key] = value
    return class_weight
