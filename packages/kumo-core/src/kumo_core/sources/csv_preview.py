# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import csv
import logging
import posixpath
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
from kumoapi.typing import Dtype
from pandas.api import types

from ..csv_rules import (
    MISSING_COLUMN_DTYPE,
    NA_TOKENS,
    has_leading_zero,
    parses_as_float,
    parses_as_integer,
    whole_valued,
)
from ..types.conversions import dtype_to_numpy, dtype_to_pandas, to_dtype
from .errors import PermanentSourceError

CSV_DTYPE_SAMPLE_ROWS = 1000


def read_pandas(read: Callable[[dict[str, Any]], pd.DataFrame],
                dtypes: Mapping[str, Dtype], *, index_col: str | None = None,
                dtype: Any = None, **kwargs: Any) -> pd.DataFrame:
    kwargs.setdefault('na_values', list(NA_TOKENS))
    orig_kwargs = copy.deepcopy(kwargs)

    # Prepare the dtype parsing args for `pandas.read_csv`.  We need to remove
    # datetimes from `kwargs.dtype`, because `read_csv` expects datetime
    # columns to be passed through `kwargs.parse_dates`
    kwargs['parse_dates'] = []
    kwargs['dtype'] = {}

    inferred_int_cols = []

    if len(dtypes) == 0 and dtype is None:
        orig_kwargs['nrows'] = CSV_DTYPE_SAMPLE_ROWS
        if 'dtype' in orig_kwargs:
            orig_kwargs.pop('dtype')

        df = read_pandas(read, dtypes, index_col=index_col, dtype=str,
                         **orig_kwargs)

        for col in df.columns:
            if not parses_as_float(df[col]):
                continue

            leading_zero = df[col].astype(str).apply(has_leading_zero).any()

            if leading_zero:
                kwargs['dtype'][col] = dtype_to_pandas(Dtype.string)
                continue

            if not parses_as_integer(df[col]):
                continue

            # Preserve large integer digits without rejecting mixed values
            # outside the inference sample; cast to Int64 after reading.
            kwargs['dtype'][col] = 'object'
            inferred_int_cols.append(col)

    elif len(dtypes) == 0:
        kwargs['dtype'] = dtype

    for col_name, dtype in dtypes.items():
        if types.is_datetime64_any_dtype(dtype_to_pandas(dtype)):
            kwargs['parse_dates'].append(col_name)
        elif index_col is not None and col_name == index_col:
            kwargs['dtype'][col_name] = dtype_to_numpy(dtype)
        else:
            kwargs['dtype'][col_name] = dtype_to_pandas(dtype)

    try:
        df = read(kwargs)
    except (ValueError, TypeError):
        kwargs.pop('dtype')
        df = read(kwargs)

    for col in inferred_int_cols:
        try:
            df[col] = df[col].astype('Int64')
        except ValueError:
            # mixed column, type is already object, no action needed
            pass

    if len(dtypes) == 0 and dtype is None:
        # inferred_int_cols only covers columns that looked int-like in the
        # string-sample pass; pandas can still infer floats (e.g., NaNs/mixed
        # formatting) even when all non-null values are integer-like. Normalize
        # such float columns to Int64 so inferred table config aligns with
        # the exported parquet schema.
        for col in df.columns:
            if not types.is_float_dtype(df[col].dtype):
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if whole_valued(series.to_numpy()):
                df[col] = df[col].round().astype('Int64')

    return df


class DuplicateColumnsError(PermanentSourceError):
    pass


def read_frames(filesystem: Any, paths: list[str], *,
                max_files: int | None = None, **options: Any) -> pd.DataFrame:
    from pandas.io.common import get_handle, infer_compression

    remaining = options.get('nrows')
    frames = []
    for path in paths:
        if frames and remaining is not None and remaining <= 0:
            break
        file_options = {**options, 'nrows': remaining}
        compression = file_options.get('compression', 'infer')
        if compression == 'infer':
            compression = infer_compression(path, 'infer')
        file_options['compression'] = compression
        header = []
        try:
            with filesystem.open_input_file(path) as source:
                with get_handle(source, 'r', compression=compression,
                                encoding=options.get('encoding')) as handle:
                    header = next(
                        csv.reader(handle.handle,
                                   delimiter=options.get('sep', ','),
                                   escapechar=options.get('escapechar')), [])
        except (ValueError, TypeError, UnicodeError, csv.Error):
            # Pandas below handles formats the optional header probe cannot parse.
            pass
        if len(header) != len(set(header)):
            raise DuplicateColumnsError(
                f'Duplicate columns found in file {posixpath.basename(path)}: {header}'
            )
        with filesystem.open_input_file(path) as source:
            frame = pd.read_csv(source, **file_options)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
        if max_files is not None and len(frames) >= max_files:
            break
    return pd.concat(frames, ignore_index=False) if frames else pd.DataFrame()


def read_files(filesystem: Any, paths: list[str], *, sep: str, rows: int,
               null_values: tuple[str, ...] | None = None,
               escape_char: str | bool = False) -> pd.DataFrame:
    def read(options: dict[str, Any]) -> pd.DataFrame:
        return read_frames(filesystem, paths, **options)

    try:
        return read_pandas(read, {}, sep=sep, nrows=rows,
                           na_values=list(NA_TOKENS | set(null_values or ())),
                           escapechar=escape_char or None)
    except pd.errors.ParserError as error:
        raise PermanentSourceError(str(error)) from error


def column_dtypes(frame: pd.DataFrame) -> dict[str, Dtype]:
    result = {}
    for name, dtype in frame.dtypes.items():
        if frame[name].isna().all():
            logging.getLogger(__name__).warning(
                "Unable to infer datatype of column %s: sampled rows are all "
                "N/A; defaulting to string. Drop the column or explicitly set "
                "its datatype to change this behavior.", name)
            result[name] = MISSING_COLUMN_DTYPE
        else:
            result[name] = (Dtype.string if isinstance(
                dtype, pd.CategoricalDtype) else to_dtype(dtype))
    return result
