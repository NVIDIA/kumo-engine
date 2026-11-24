# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused GPU kernels, each a `torch.library` custom op with its eager oracle."""

from tfm.kernels.cross_entropy import linear_cross_entropy

__all__ = ["linear_cross_entropy"]
