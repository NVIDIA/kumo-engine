# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from typing import Dict, List, Optional, Union

from io_utils import logging

logger = logging.getLogger(__name__)

FORMAT_MAP = {'b': 'B', 'k': 'KiB', 'm': 'MiB', 'g': 'GiB'}


def convert_bytes(value: int, output_unit: str = 'b') -> int:
    r"""Converts :obj:`value` from bytes to :obj:`output_unit`."""
    convert_map = {'b': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4}
    div_factor = 1024**convert_map[output_unit]
    return value / div_factor


def format_bytes(value: int, output_unit: Optional[str] = None) -> str:
    r"""Converts :obj:`value` from bytes to :obj:`output_unit`, formatting the
    resulting output."""
    if output_unit is not None and output_unit not in FORMAT_MAP:
        raise ValueError(f'Invalid output unit value {output_unit}')

    # Automatically convert:
    if output_unit is None:
        p = 2**10
        idx = 0
        labels = {0: '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti'}
        while value > p and idx < len(labels) - 1:
            value /= p
            idx += 1
        output_unit = labels[idx] + 'B'
        return f"{value:.2f} {output_unit}"

    # Convert from bytes:
    converted = f"{convert_bytes(value, output_unit):.2f}"
    return converted + ' ' + FORMAT_MAP[output_unit]


def convert_bytes_list(input: List, unit_letter: str):
    byte_col_index = input[0].index("B")
    input[0][byte_col_index] = FORMAT_MAP[unit_letter]
    for row in input[1:]:
        row[byte_col_index] = convert_bytes(row[byte_col_index], unit_letter)


def clean_for_json(input: Dict):
    r"""Method to recursively check if dictionary values can be written to json
    and replace with strings as needed.

    Args:
        input (Dict) : dictionary to convert into json-writable format
    """
    import json
    for key, value in input.items():
        try:
            json.dumps(value)
        except TypeError:
            if isinstance(value, Dict):
                clean_for_json(value)
            else:
                input[key] = str(value)


def format_exception(error: Union[Exception, BaseException]) -> str:
    return "".join(
        traceback.format_exception(  # type: ignore
            error, value=error, tb=error.__traceback__))
