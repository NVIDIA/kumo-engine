# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from io_utils import kumo_dataclass


@kumo_dataclass
class StatsConfig:
    """General configuration for a computed statistic."""
    value: Any
    approx: bool
    data_version: Any
    num_entries: Optional[int] = None
    timestamp: Optional[float] = None
