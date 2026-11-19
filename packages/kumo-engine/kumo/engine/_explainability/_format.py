# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Number formats shared by every card.

Magnitudes of 10 or more get one decimal, smaller values three significant
figures, shares one-decimal percentages; minus signs are typographic.
"""

import math
from collections.abc import Iterable

MINUS = '−'


def number(value: float) -> str:
    """One decimal at magnitude 10 or more, else three significant figures."""
    if not math.isfinite(value):
        return 'N/A'
    rounded = float(f'{value:.3g}')
    if abs(rounded) >= 10:
        text = f'{value:,.1f}'
    elif rounded == 0:
        text = '0'
    else:
        text = f'{rounded:.{2 - math.floor(math.log10(abs(rounded)))}f}'
    return text.replace('-', MINUS)


def signed(value: float) -> str:
    """`number` with an explicit sign, for differences."""
    text = number(value)
    if text in ('0', '0.0', 'N/A') or text.startswith(MINUS):
        return text
    return '+' + text


def numbers(values: Iterable[float]) -> list[str]:
    """Format values that share one row or axis on one scale.

    When any value reaches magnitude 10 the whole row gets one decimal, so a
    row such as ``-6.8 / 21.6 / 577.0`` keeps a single precision.
    """
    items = list(values)
    if max((abs(v) for v in items if math.isfinite(v)), default=0) < 10:
        return [number(v) for v in items]
    return [(f'{v:,.1f}'.replace('-', MINUS) if math.isfinite(v) else 'N/A')
            for v in items]


def percent(share: float) -> str:
    """A share in ``[0, 1]`` as a one-decimal percentage."""
    return f'{share * 100:.1f}%'.replace('-', MINUS)
