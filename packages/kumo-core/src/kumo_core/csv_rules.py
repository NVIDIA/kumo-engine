# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rules that decide a CSV column's type, shared with the platform.

A CSV dtype reaches ``graph_id``, so both backends apply this one
implementation through their own reader -- pandas on the platform, Arrow here.
"""

from __future__ import annotations

import csv as stdlib_csv
import string
from typing import Any, Optional, Sequence

from kumoapi.typing import Dtype

#: pandas' ``STR_NA_VALUES``. Arrow's own defaults omit ``<NA>`` and ``None``.
NA_TOKENS = frozenset({
    '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
    '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a',
    'nan', 'null',
})

DEFAULT_SEPARATOR = ','

#: Text rather than ``unsupported``, which has no Arrow type.
MISSING_COLUMN_DTYPE = Dtype.string

PREFERRED_SEPARATORS = (',', '\t', ';', ':', '|')
DENIED_SEPARATORS = frozenset(string.ascii_letters + string.digits +
                              ' -._"\'\n')

SNIFF_CHARS = 50_000

#: Mirrors the platform's ``NUM_SAMPLE_ROWS * CSV_SAMPLE_MULTIPLIER`` rows over
#: at most its ``MAX_NUM_FILES_TO_SAMPLE`` files.
SAMPLE_ROWS = 100_000
MAX_SAMPLE_FILES = 10

#: Depth of the platform's first narrowing pass, which picks the candidate the
#: ``SAMPLE_ROWS`` pass is then read under.
NARROWING_ROWS = 1_000


def normalize_newlines(text: str) -> str:
    """``csv.Sniffer`` reads a Windows or classic-Mac line ending as content
    rather than as the end of a row."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


SNIFFED = 'sniffed'
UNSNIFFABLE = 'unsniffable'
DENIED = 'denied'


def separator_with_reason(text: str) -> tuple[Optional[str], str]:
    """The delimiter this text is written with and how it was arrived at, or
    ``None`` when the text is too short to hold one.
    """
    normalized = normalize_newlines(text)
    sniffer = stdlib_csv.Sniffer()
    sniffer.preferred = list(PREFERRED_SEPARATORS)
    reason = SNIFFED
    try:
        found = sniffer.sniff(normalized).delimiter
    except Exception:
        found, reason = DEFAULT_SEPARATOR, UNSNIFFABLE
    if found not in DENIED_SEPARATORS:
        return found, reason
    if len(normalized) >= 2:
        return DEFAULT_SEPARATOR, DENIED
    return None, DENIED


def separator_from(text: str) -> Optional[str]:
    return separator_with_reason(text)[0]


def has_leading_zero(value: str) -> bool:
    """A zip code read as a number loses the zero, irrecoverably."""
    text = str(value)
    return (text.startswith('0') and not text.startswith('0.')
            and len(text) > 1)


def whole_valued(values: Any) -> bool:
    """Whether a column of floats holds only integers, which is the integer
    column a warehouse ``NUMBER`` export writes as ``1.0``. Compared with
    ``np.isclose`` because decimal-to-binary need not land on the integer.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return False
    return bool(np.all(np.isclose(array, np.round(array))))


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value in NA_TOKENS)


def present(values: Sequence[Any]) -> list[Any]:
    return [value for value in values if not is_missing(value)]


TEXT = 'text'
MISSING = 'missing'
BOOLEAN = 'boolean'
INTEGER = 'integer'
FLOATING = 'floating'


class NoValidSeparatorFoundError(Exception):
    pass


#: All pandas reads as a boolean; Arrow's own cast is looser and takes
#: ``1``/``0`` too, so the rule is written against the literals.
BOOLEAN_LITERALS = frozenset({'true', 'false'})


def is_boolean(values: Sequence[Any]) -> bool:
    return all(
        isinstance(value, str) and value.lower() in BOOLEAN_LITERALS
        for value in values)


def _series(values: Sequence[Any]) -> Any:
    import pandas as pd

    return pd.Series(list(values), dtype=object)


def parses_as_float(values: Sequence[Any]) -> bool:
    try:
        _series(values).astype(float)
    except ValueError:
        return False
    return True


def parses_as_integer(values: Sequence[Any]) -> bool:
    """``OverflowError`` is deliberately not caught: a value too wide for
    ``int64`` is the caller's to report against a named column."""
    try:
        _series(values).astype('Int64')
    except ValueError:
        return False
    return True


#: The spellings ``int()`` accepts and Arrow's CSV parser does not. Held as
#: data because the scan rewrites them as a vectorized expression too.
NUMERIC_GROUPING = '_'
NUMERIC_SIGN = '+'


def normalize_numeric(value: str) -> str:
    return value.strip().replace(NUMERIC_GROUPING,
                                 '').removeprefix(NUMERIC_SIGN)


def column_kind(values: Sequence[Any],
                narrowing: Optional[Sequence[Any]] = None) -> str:
    """Which of text, boolean, integer or floating the platform would call
    this column, given the rows it reads and the shorter head it narrows on.

    An integer picked from the head and contradicted in the body is text: the
    pick made it an object column, and finding a float in it widens nothing.
    """
    kept = present(values)
    if not kept:
        return MISSING
    head = present(values[:NARROWING_ROWS] if narrowing is None else narrowing)
    if head and any(has_leading_zero(value) for value in head):
        return TEXT
    if head and parses_as_float(head) and parses_as_integer(head):
        return INTEGER if parses_as_integer(kept) else TEXT
    # pandas reads a column as `bool` only when every value is one, so a
    # boolean column with a single gap is text and not a nullable boolean.
    if len(kept) == len(values) and is_boolean(kept):
        return BOOLEAN
    if not parses_as_float(kept):
        return TEXT
    if parses_as_integer(kept):
        return INTEGER
    if whole_valued([float(normalize_numeric(value)) for value in kept]):
        return INTEGER
    return FLOATING


def validate_separator(text: str, delimiter: str) -> None:
    """The platform refuses a file whose head will not parse with the
    delimiter it sniffed. The last line is dropped because the sniffing window
    rarely ends on a row boundary.
    """
    from io import StringIO

    import pandas as pd

    lines = text.splitlines()
    if len(lines) > 1:
        text = '\n'.join(lines[:-1]) + '\n'
    try:
        pd.read_csv(StringIO(text), sep=delimiter, index_col=False,
                    on_bad_lines='error', skip_blank_lines=False)
    except Exception as error:
        raise NoValidSeparatorFoundError(
            f"Could not auto-detect a delimiter. Expected the delimiter: "
            f"{delimiter}, but ran into the following error when reading "
            f"with this delimiter: {error!r}")
