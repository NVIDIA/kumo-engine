# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Imports PyTorch Geometric without the FutureWarning torch 2.15+ raises
for its import-time ``torch.jit.script`` calls, which users cannot act on."""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'`torch\.jit\.script` is deprecated',
                            FutureWarning)
    import torch_geometric.nn  # noqa: F401
