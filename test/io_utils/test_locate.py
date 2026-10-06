# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils.locate import locate


def test_locate():
    cls = locate('torch_geometric.nn.SAGEConv')
    from torch_geometric.nn import SAGEConv
    assert issubclass(cls, SAGEConv)
