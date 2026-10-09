# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from kumoapi.typing import Dtype
from pyarrow import csv
from pyarrow.fs import FileSystem

from ..csv_rules import (
    BOOLEAN,
    DEFAULT_SEPARATOR,
    INTEGER,
    MAX_SAMPLE_FILES,
    NUMERIC_GROUPING,
    NUMERIC_SIGN,
    SAMPLE_ROWS,
    SNIFF_CHARS,
    TEXT,
    normalize_newlines,
    separator_from,
    validate_separator,
)
from .errors import SourceNotFoundError

_ARROW_TYPE = {
    TEXT: pa.string(),
    INTEGER: pa.int64(),
    BOOLEAN: pa.bool_(),
}

#: Decides *values*, where the wider :data:`kumo_core.csv_rules.NA_TOKENS`
#: decides *types*.
ARROW_NA_TOKENS = tuple(csv.ConvertOptions().null_values)


@dataclass(frozen=True)
class CsvOptions:
    delimiter: str | None = None
    escape_char: str | bool = False
    null_values: tuple[str, ...] | None = None


def readable_parts(found: Sequence[Any]) -> list[Any]:
    return [entry for entry in found if entry.size != 0]


def delimiter(filesystem: Any, path: str) -> str:
    with filesystem.open_input_stream(path) as handle:
        raw = handle.read(SNIFF_CHARS)
    text = normalize_newlines(raw.decode('utf-8', errors='replace'))
    sep = separator_from(text) or DEFAULT_SEPARATOR
    validate_separator(text, sep)
    return sep


def parse_options(sep: str, options: CsvOptions | None = None) -> Any:
    """A row with the wrong number of fields fails the read rather than being
    dropped, which is what the platform's Arrow export does too."""
    return csv.ParseOptions(
        delimiter=sep, escape_char=options.escape_char if options else False)


def convert_options(column_types: Mapping[str, pa.DataType] | None,
                    options: CsvOptions | None = None) -> Any:
    """Arrow's own null handling, unchanged, which is what the hosted export
    runs with unless the connector was configured with ``na_values``."""
    kwargs = {}
    if options is not None and options.null_values is not None:
        kwargs = {
            'null_values': list(options.null_values),
            'strings_can_be_null': True
        }
    return csv.ConvertOptions(column_types=dict(column_types or {}), **kwargs)


def file_format(sep: str, column_types: Mapping[str, pa.DataType] | None,
                options: CsvOptions | None = None) -> Any:
    return ds.CsvFileFormat(
        parse_options=parse_options(sep, options),
        convert_options=convert_options(column_types, options))


def pinned_types(columns: Mapping[str, Dtype]) -> dict[str, pa.DataType]:
    """Parser types for supported scalar declarations; absent columns remain
    inferred. The local reader uses these keys to select its conversion policy.
    """
    from ..types.conversions import TO_PYARROW_DTYPE
    pinned = {}
    for name, dtype in columns.items():
        dtype = Dtype(dtype)
        if dtype.is_list():
            continue
        # Decimal-formatted integers must not change schema between CSV parts.
        target = pa.float64() if dtype.is_int() else TO_PYARROW_DTYPE.get(dtype)
        if target is not None:
            pinned[name] = target
    return pinned


@contextmanager
def open_reader(
    filesystem: FileSystem, path: str, *, parse: csv.ParseOptions,
    convert: csv.ConvertOptions | None = None
) -> Iterator[csv.CSVStreamingReader]:
    # Arrow streams decompress by filename, including already-local .csv.gz.
    with filesystem.open_input_stream(path) as stream:
        with csv.open_csv(stream, parse_options=parse,
                          convert_options=convert) as reader:
            yield reader


def batches(
        filesystem: FileSystem, paths: Sequence[str], *,
        parse: csv.ParseOptions,
        convert: csv.ConvertOptions | None = None) -> Iterator[pa.RecordBatch]:
    for path in paths:
        with open_reader(filesystem, path, parse=parse,
                         convert=convert) as reader:
            yield from reader


def sampled(filesystem: Any, paths: Sequence[str], sep: str,
            options: CsvOptions | None = None) -> dict[str, pa.DataType]:
    """Per column, the type the platform reports for it, sampled across parts
    on one shared budget the way the platform spends its own."""
    from ..types.conversions import to_pyarrow_type
    from .csv_preview import column_dtypes, read_files

    frame = read_files(filesystem, list(paths[:MAX_SAMPLE_FILES]), sep=sep,
                       rows=SAMPLE_ROWS,
                       null_values=options.null_values if options else None,
                       escape_char=options.escape_char if options else False)
    if frame.empty:
        raise SourceNotFoundError("This table has a header and no rows.")
    return {
        name: to_pyarrow_type(dtype)
        for name, dtype in column_dtypes(frame).items()
    }


def text_format(sep: str, names: Sequence[str], unpinned: Sequence[str] = (),
                options: CsvOptions | None = None) -> Any:
    """Every column is read as text and converted afterwards: letting Arrow
    convert reads ``0x10`` as 16 and ``2026-01-01`` as a date, where the
    platform keeps both as text. ``unpinned`` columns have no Arrow spelling,
    so the hosted export leaves them to Arrow's inference too.
    """
    skip = set(unpinned)
    return file_format(
        sep, {name: pa.string()
              for name in names if name not in skip}, options)


def numeric(expression: Any) -> Any:
    """The two spellings Arrow's parser rejects, rewritten in the scan."""
    expression = pc.utf8_trim_whitespace(expression)
    expression = pc.replace_substring(expression, NUMERIC_GROUPING, '')
    return pc.replace_substring_regex(expression, f'^[{NUMERIC_SIGN}]', '')


def rounded(expression: Any) -> Any:
    return pc.round(expression.cast(pa.float64()))


#: What Arrow's own integer parser accepts.
_INTEGER_TEXT = r'^[+-]?[0-9]+$'


def _without_na(expression: Any, null_values: Sequence[str]) -> Any:
    """Arrow nulls these itself in a typed column, but these are read as text,
    so the substitution has to precede the cast."""
    return pc.if_else(expression.isin(null_values),
                      pa.scalar(None, pa.string()), expression)


def convert(expression: Any, target: pa.DataType, *,
            null_values: Sequence[str] = ARROW_NA_TOKENS) -> Any:
    """Text to the declared type, reaching the value the hosted export reaches
    from the same bytes.

    Integer-shaped text is cast directly, since routing it through a double
    would round an ID above 2**53; every other spelling takes the double,
    which is what makes ``10.0`` a 10 rather than a gap.
    """
    expression = _without_na(expression, null_values)
    if pa.types.is_integer(target):
        text = numeric(expression)
        exact = pc.match_substring_regex(text, _INTEGER_TEXT)
        return pc.coalesce(
            pc.if_else(exact, text, pa.scalar(None, pa.string())).cast(target),
            rounded(pc.if_else(exact, '0', text)).cast(target))
    if pa.types.is_floating(target):
        return numeric(expression).cast(target)
    return expression.cast(target)
