# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils import kumo_dataclass


@kumo_dataclass
class SqlCastResult:
    rows_dropped: int = 0
