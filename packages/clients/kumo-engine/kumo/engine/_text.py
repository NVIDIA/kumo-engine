# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Text helpers shared by summaries and cards."""

import math
import re
from typing import Any, Optional


def entity(query: Optional[str]) -> Optional[str]:
    r"""The entity table a predictive query predicts for."""
    found = re.search(r'FOR\s+EACH\s+(\w+)', query or '', re.IGNORECASE)
    return found and found.group(1)


def number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return f'{int(value):,}'
    if not math.isfinite(value) or abs(value) >= 1000:
        return f'{value:,.0f}'
    digits = 2 - math.floor(math.log10(abs(value)))
    return f'{value:.{digits}f}'.rstrip('0').rstrip('.')


def metric_key(name: str) -> str:
    return name.replace('@', '_at_')
