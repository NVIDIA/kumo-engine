# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inclusive whole-second graph cutoff for native anchor tables."""
from datetime import timedelta

from .workload import EPOCH


def inclusive_second(anchor):
    return (anchor - EPOCH) // timedelta(seconds=1)
