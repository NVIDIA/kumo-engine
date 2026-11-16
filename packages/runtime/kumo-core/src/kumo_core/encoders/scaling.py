# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from kumoapi.encoder import MaxLogNumerical, MinLogNumerical, Numerical, Scaler
from kumoapi.typing import ColStatType


def get_offset_and_scale(
    encoder: Numerical | MinLogNumerical | MaxLogNumerical,
) -> tuple[float, float]:
    if isinstance(encoder, Numerical):
        if encoder.scaler is None:
            offset, scale = 0.0, 1.0
        elif encoder.scaler == Scaler.STANDARD:
            offset = encoder._stats[ColStatType.MEAN]
            scale = encoder._stats[ColStatType.STD]
        elif encoder.scaler == Scaler.MINMAX:
            offset = encoder._stats[ColStatType.MIN]
            scale = encoder._stats[ColStatType.MAX] - offset
        elif encoder.scaler == Scaler.ROBUST:
            quantiles = encoder._stats[ColStatType.QUANTILES]
            quantile25, offset, quantile75 = quantiles
            scale = quantile75 - quantile25
        else:
            raise NotImplementedError

        # Handle zero-scale gracefully:
        if scale is None or abs(scale) < 1e-5:
            scale = 1.0

    elif isinstance(encoder, MaxLogNumerical):
        offset, scale = encoder._stats[ColStatType.MIN] - 1, 1.0

    elif isinstance(encoder, MinLogNumerical):
        offset, scale = encoder._stats[ColStatType.MAX] + 1, -1.0

    else:
        raise ValueError(f"Unsupported encoder type '{type(encoder)}'.")

    # Prevent overflows
    offset = min(offset, torch.finfo(torch.float).max)
    offset = max(offset, torch.finfo(torch.float).min)
    scale = min(scale, torch.finfo(torch.float).max)

    return offset, scale
