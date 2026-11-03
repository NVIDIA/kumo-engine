# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Text helpers shared by summaries and cards."""

import re
from typing import Optional


def entity(query: Optional[str]) -> Optional[str]:
    r"""The entity table a predictive query predicts for."""
    found = re.search(r'FOR\s+EACH\s+(\w+)', query or '', re.IGNORECASE)
    return found and found.group(1)


def metric_key(name: str) -> str:
    return name.replace('@', '_at_')
