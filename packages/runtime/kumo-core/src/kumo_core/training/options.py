# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable feature decisions for the DiskGraph-native training path."""


def allow_array_targets() -> bool:
    """TT-NG represents list-distinct targets as arrays unconditionally."""

    return True
