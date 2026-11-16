# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from torch import Tensor
from torch_geometric.utils.mixin import CastMixin


@dataclass
class RHSTime(CastMixin):
    # The unix start date offset for which we are making a prediction.
    unix_start_offset: int
    # The unix end date offset for which we are making a prediction.
    unix_end_offset: int
    # The create time of the RHS nodes.
    create_time: Tensor | None = None
    # The end time of the RHS nodes. NaN is represented by ``-1``.
    end_time: Tensor | None = None

    def __post_init__(self) -> None:
        if self.create_time is not None:
            self.create_time = self.create_time.view(-1)
        if self.end_time is not None:
            self.end_time = self.end_time.view(-1)

        if self.create_time is not None and self.end_time is not None:
            assert self.create_time.numel() == self.end_time.numel()

    def get_valid_nodes(self, time: Tensor) -> dict[int, Tensor] | None:
        if self.create_time is None and self.end_time is None:
            return None

        out = {}
        for t in time.tolist():
            mask = None
            if self.create_time is not None:
                mask = self.create_time <= (t + self.unix_end_offset)
            if self.end_time is not None:
                _mask = self.end_time == -1
                _mask |= self.end_time > (t + self.unix_start_offset)
                mask = _mask if mask is None else mask & _mask
            assert mask is not None
            out[int(t)] = mask.nonzero().view(-1)

        return out
