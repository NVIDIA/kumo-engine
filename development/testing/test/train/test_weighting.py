# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from kumo_core.train.weighting import build_binary_class_weight


def test_build_binary_class_weight():
    target = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    class_weight = build_binary_class_weight(
        target=target,
        class_weight_map={
            False: 3.0,
            True: 1.0
        },
        out_dtype=torch.float,
    )
    torch.testing.assert_close(
        class_weight,
        torch.tensor([3.0, 1.0, 3.0, 1.0, 1.0], dtype=torch.float),
    )


def test_build_binary_class_weight_missing_class():
    with pytest.raises(ValueError, match="Missing majority sampling weights"):
        build_binary_class_weight(
            target=torch.tensor([0, 1], dtype=torch.long),
            class_weight_map={True: 1.0},
        )
