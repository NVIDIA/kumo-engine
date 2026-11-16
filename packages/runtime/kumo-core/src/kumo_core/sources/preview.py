# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas


def sample_dataframe(sample_rows: str | None) -> 'pandas.DataFrame | None':
    """Decode a table-oriented source sample without rounding nullable IDs."""
    import json
    from io import StringIO

    import pandas as pd

    if not sample_rows:
        return None
    payload = json.loads(sample_rows)
    # Pandas can infer float64 before applying an integer extension dtype,
    # rounding distinct integers above 2**53. Decimal strings reach the same
    # schema-directed integer cast without that intermediate float array.
    integer_fields = []
    for field in payload.get('schema', {}).get('fields', []):
        if (field.get('type') in ('integer', 'any')
                and field.get('extDtype', '').startswith(('Int', 'UInt'))):
            integer_fields.append(field['name'])
            # Pandas 1.5 honors extDtype only for an "any" field. The
            # declared extension dtype still validates and casts the values.
            field['type'] = 'any'
    if integer_fields:
        for row in payload['data']:
            for name in integer_fields:
                if isinstance(row.get(name), int):
                    row[name] = str(row[name])
    encoded = json.dumps(payload) if integer_fields else sample_rows
    return pd.read_json(StringIO(encoded), orient='table')


def stringify_sample_dataframe(
        rows_df: 'pandas.DataFrame') -> 'pandas.DataFrame':
    r"""Stringify preview values without interpreting arbitrary binary data.

    Pandas' ``astype('str')`` decodes byte strings as UTF-8 and raises for
    valid, non-text ``BINARY`` values. Keep nulls as ``None``, render binary
    deterministically as hexadecimal, and use the ordinary string
    representation for every other connector value.

    Nullness is read before the cast, because a float column keeps its dtype
    through a ``None`` and hands ``str`` a ``NaN`` to render as ``"nan"``.
    """
    null_cells = rows_df.isnull()
    result = rows_df.astype(object).copy()
    for row_index in range(len(result.index)):
        for col_index in range(len(result.columns)):
            if null_cells.iat[row_index, col_index]:
                result.iat[row_index, col_index] = None
                continue
            value = result.iat[row_index, col_index]
            if isinstance(value, (bytes, bytearray, memoryview)):
                result.iat[row_index, col_index] = (f'0x{bytes(value).hex()}')
            else:
                result.iat[row_index, col_index] = str(value)
    return result
