# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset/model-independent benchmark interfaces, owned by the test suite."""

from .interface import BatchCounts, Workload
from .runner import run

__all__ = ["BatchCounts", "Workload", "run"]
