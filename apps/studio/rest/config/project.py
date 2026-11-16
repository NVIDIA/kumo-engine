# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project request models."""
from __future__ import annotations

from typing import Optional

from io_utils.dataclasses import kumo_dataclass


@kumo_dataclass
class UpdateProjectRequest:
    name: Optional[str] = None
    description: Optional[str] = None
    is_private: Optional[bool] = None
