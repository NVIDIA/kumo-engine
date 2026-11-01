# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Imports PyTorch Geometric without its unactionable FutureWarning."""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'`torch\.jit\.script` is deprecated',
                            FutureWarning)
    import torch_geometric.nn  # noqa: F401
