# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import Tensor
from torch_geometric.testing import withCUDA

from kumo_core.modeling.base import Module


class MyModule(Module):
    LAZY_ATTRS = {'in_channels', 'out_channels'}

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__(in_channels, out_channels)

    def init_modules(self) -> None:
        self.lin = torch.nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x)


@withCUDA
def test_base_module(device: torch.device) -> None:
    module = MyModule()

    with pytest.raises(ValueError, match="not fully-specified yet"):
        module(torch.randn(5, 32))

    with pytest.raises(ValueError, match="not fully-specified yet"):
        module.to(device)

    with pytest.raises(ValueError, match="not fully-specified yet"):
        module.named_parameters()

    with pytest.raises(ValueError, match="not fully-specified yet"):
        module.named_children()

    with pytest.raises(ValueError, match="not fully-specified yet"):
        module.named_modules()

    module.in_channels = 32
    module.out_channels = 7

    module = module.to(device)
    assert module(torch.randn(5, 32, device=device)).size() == (5, 7)
    assert len(list(module.named_parameters())) == 2
    assert list(module.named_parameters())[0][0] == 'lin.weight'
    assert list(module.named_parameters())[1][0] == 'lin.bias'
    assert len(list(module.named_children())) == 1
    assert list(module.named_children())[0][0] == 'lin'
    assert len(list(module.named_modules())) == 2
    assert list(module.named_modules())[0][0] == ''
    assert list(module.named_modules())[1][0] == 'lin'
