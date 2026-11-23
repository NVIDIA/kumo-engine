# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch

from tfm.adaptation.batching import concat_rows, select_rows


@dataclass
class Rows:
    values: torch.Tensor
    metadata: dict[str, torch.Tensor]
    events: dict[str, torch.Tensor]

    @property
    def batch_size(self):
        return len(self.values)


def test_row_selection_preserves_non_row_tensor_metadata():
    a = Rows(
        torch.arange(2),
        {"scale": torch.tensor([2, 3, 4])},
        {"clock": torch.arange(6).reshape(2, 3)},
    )
    b = Rows(
        torch.arange(2),
        {"scale": torch.tensor([2, 3, 4])},
        {"clock": torch.arange(6).reshape(2, 3)},
    )
    picked = select_rows(a, [1])
    assert picked.metadata is a.metadata
    torch.testing.assert_close(picked.events["clock"], a.events["clock"][[1]])
    combined = concat_rows([a, b])
    assert combined.batch_size == 4 and combined.metadata is a.metadata
    torch.testing.assert_close(
        combined.events["clock"], torch.cat([a.events["clock"], b.events["clock"]])
    )
    b.metadata["scale"].add_(1)
    with pytest.raises(ValueError, match="metadata"):
        concat_rows([a, b])
