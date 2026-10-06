# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch


def withCUDA(func: Callable):
    import pytest

    devices = [pytest.param(torch.device('cpu'), id='cpu')]
    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device('cuda:0'), id='cuda:0'))

    return pytest.mark.parametrize('device', devices)(func)
