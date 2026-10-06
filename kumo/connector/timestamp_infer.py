# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Temporal-hint type and the column-level temporal detector.

Decides, per column, which temporal hint the column carries: an epoch unit, a
strftime format, native (already-typed temporal), or unresolved. The matching
apply-time conversion to epoch seconds/days lives in
:mod:`kumo.connector.timestamp_convert`.

Every value-level decision is DiskGraph's: :func:`infer_temporal_hint` hands the
column to ``resolve_temporal`` and stores whichever hint comes back. The format
string is an opaque token here -- stored, hashed and displayed, never parsed --
so the hint kumo persists is by construction the hint ingest will apply.
"""
import dataclasses
import enum
from typing import Optional

import pandas as pd
import pyarrow as pa
from diskgraph import resolve_temporal
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype

# Valid epoch-unit tokens, matching the four Arrow TimeUnit variants.
_VALID_UNITS = ('s', 'ms', 'us', 'ns')

# Aliases a declared unit may arrive as (from a config, a UI, or a caller),
# normalized to the four canonical tokens. A spelled-out unit like "seconds" /
# "sec" / "milliseconds" maps to s/ms/us/ns rather than being misread.
_TIMESTAMP_UNIT_ALIASES = {
    's': 's', 'sec': 's', 'secs': 's', 'second': 's', 'seconds': 's',
    'ms': 'ms', 'milli': 'ms', 'millis': 'ms', 'millisecond': 'ms',
    'milliseconds': 'ms',
    'us': 'us', 'µs': 'us', 'micro': 'us', 'micros': 'us', 'microsecond': 'us',
    'microseconds': 'us',
    'ns': 'ns', 'nano': 'ns', 'nanos': 'ns', 'nanosecond': 'ns',
    'nanoseconds': 'ns',
}


def map_timestamp_unit(unit: Optional[str]) -> Optional[str]:
    r"""Normalize a declared timestamp unit to a canonical s/ms/us/ns token.

    Accepts the canonical tokens, their common spelled-out aliases, and an
    already-``TimestampUnit``-valued input (matched by its ``.value``). Returns the
    canonical token, or ``None`` when the input is ``None`` or an unrecognized
    string. An unrecognized unit is never fatal here; the caller degrades to
    inference.
    """
    if unit is None:
        return None
    key = getattr(unit, 'value', unit)
    key = str(key).strip().lower()
    return _TIMESTAMP_UNIT_ALIASES.get(key)


# ---------------------------------------------------------------------------
# The temporal-hint type.
# ---------------------------------------------------------------------------
class HintKind(enum.Enum):
    r"""Which kind of temporal hint a column carries.

    A column carries exactly one of: an epoch unit, a strftime format, native, or
    unresolved.
    """
    UNIT = 'unit'              # epoch integer/float column: carries s/ms/us/ns
    FORMAT = 'format'          # string or encoded-date integer: carries strftime
    NATIVE = 'native'          # already-typed temporal: unit lives on the dtype
    UNRESOLVED = 'unresolved'  # no hint learned: apply layer defaults to seconds


@dataclasses.dataclass(frozen=True)
class TemporalHint:
    r"""A resolved per-column temporal hint.

    Exactly one of ``unit`` (iff ``kind`` is ``UNIT``) or ``strftime`` (iff ``kind``
    is ``FORMAT``) is set; ``NATIVE`` and ``UNRESOLVED`` set neither. This mutual
    exclusion disambiguates the overloaded v2
    ``Column.timestamp_format: Union[TimestampUnit, str]``. Frozen so it is
    hashable.
    """
    kind: HintKind
    unit: Optional[str] = None      # "s"|"ms"|"us"|"ns"; set iff kind is UNIT
    strftime: Optional[str] = None  # strftime format string; set iff FORMAT

    def __post_init__(self) -> None:
        if self.kind is HintKind.UNIT:
            if self.unit not in _VALID_UNITS:
                raise ValueError(
                    f"UNIT hint requires unit in {_VALID_UNITS}, got {self.unit!r}")
            if self.strftime is not None:
                raise ValueError("UNIT hint must not carry a strftime format")
        elif self.kind is HintKind.FORMAT:
            if self.strftime is None:
                raise ValueError("FORMAT hint requires a strftime format")
            if self.unit is not None:
                raise ValueError("FORMAT hint must not carry a unit")
        else:  # NATIVE / UNRESOLVED
            if self.unit is not None or self.strftime is not None:
                raise ValueError(
                    f"{self.kind.name} hint must not carry a unit or format")

    @classmethod
    def of_unit(cls, unit: str) -> 'TemporalHint':
        return cls(HintKind.UNIT, unit=unit)

    @classmethod
    def of_format(cls, strftime: str) -> 'TemporalHint':
        return cls(HintKind.FORMAT, strftime=strftime)

    @classmethod
    def native(cls) -> 'TemporalHint':
        return cls(HintKind.NATIVE)

    @classmethod
    def unresolved(cls) -> 'TemporalHint':
        return cls(HintKind.UNRESOLVED)


# ---------------------------------------------------------------------------
# Storage mapping: where a resolved hint lands in the config.
# ---------------------------------------------------------------------------
# A UNIT hint writes ``BaseColumnConfig.timestamp_unit``.
TIMESTAMP_UNIT_CONFIG_FIELD = 'timestamp_unit'
# A FORMAT hint writes ``metadata['TS_FORMAT']``; there is no ``timestamp_format``
# field on ``BaseColumnConfig``.
TS_FORMAT_METADATA_KEY = 'TS_FORMAT'


@dataclasses.dataclass(frozen=True)
class HintStorage:
    r"""Where a resolved hint is written in the kumo config.

    ``timestamp_unit`` maps to ``BaseColumnConfig.timestamp_unit``. ``ts_format``
    maps to ``metadata['TS_FORMAT']``. A ``NATIVE`` or ``UNRESOLVED`` hint writes
    neither (both stay ``None``); the unit for a native column rides its dtype, and
    an unresolved column defaults to seconds at apply time.
    """
    timestamp_unit: Optional[str] = None
    ts_format: Optional[str] = None


def hint_to_storage(hint: TemporalHint) -> HintStorage:
    r"""Map a resolved hint onto its config storage slots (no config mutation).

    Call sites use this to persist a hint; this module never writes a config itself.
    """
    if hint.kind is HintKind.UNIT:
        return HintStorage(timestamp_unit=hint.unit)
    if hint.kind is HintKind.FORMAT:
        return HintStorage(ts_format=hint.strftime)
    return HintStorage()


# ---------------------------------------------------------------------------
# Column-level entry point.
# ---------------------------------------------------------------------------
def _is_native_temporal_series(series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series):
        return True
    dtype = series.dtype
    if isinstance(dtype, pd.ArrowDtype):
        try:
            pa_type = dtype.pyarrow_dtype
            return pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type)
        except Exception:
            return False
    # An object series can hold already-typed temporal scalars (e.g. a Snowflake
    # read of ``pd.Timestamp`` objects, or a parquet column surfaced as
    # ns-precision ``np.datetime64`` objects). Treating these as NATIVE keeps a
    # ns-precision scalar from being read as an epoch integer.
    if is_object_dtype(series):
        try:
            return pd.api.types.infer_dtype(series, skipna=True) in {
                'datetime', 'date', 'datetime64'}
        except (TypeError, ValueError):
            return False
    return False


def _as_arrow(series: pd.Series) -> pa.Array:
    r"""Hand a column to DiskGraph as an Arrow array.

    Physical layout only, never value meaning: a dictionary-encoded
    (``category``) column is decoded, and a mixed-object column -- which has no
    Arrow type -- is stringified, which is what the row reader parses anyway.
    """
    try:
        array = pa.Array.from_pandas(series)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        array = pa.Array.from_pandas(series.astype('string'))
    if pa.types.is_dictionary(array.type):
        return array.dictionary_decode()
    return array


def infer_temporal_hint(
    sample: pd.Series,
    *,
    declared_unit: Optional[str] = None,
    declared_format: Optional[str] = None,
    is_create_time: bool = False,
) -> TemporalHint:
    r"""Infer the temporal hint for a column from a sample of its values.

      1. Gate first: if ``declared_unit`` or ``declared_format`` is already set,
         return that hint unchanged; never re-infer a column that already carries a
         hint. A declared format takes precedence over a declared unit if both are
         given. A declared unit is normalized through :func:`map_timestamp_unit`,
         so an alias such as "seconds" is accepted; an unrecognized unit falls
         through to inference rather than raising.
      2. If the sample is already a native temporal dtype, return ``NATIVE``.
      3. Otherwise DiskGraph classifies and votes on the column and returns at
         most one of a format or a unit.

    Messy data never raises: a column DiskGraph cannot classify is ``UNRESOLVED``.

    If ``is_create_time`` is ``True`` and the column produces no resolvable hint
    (entirely null or entirely unresolved), raise ``ValueError``: a create_time
    column must contain parseable timestamp values.
    """
    if declared_format is not None:
        return TemporalHint.of_format(declared_format)
    if declared_unit is not None:
        normalized_unit = map_timestamp_unit(declared_unit)
        if normalized_unit is not None:
            return TemporalHint.of_unit(normalized_unit)
        # An unrecognized declared unit is not fatal: fall through to inference
        # rather than raising on a caller-supplied alias.

    if not isinstance(sample, pd.Series):
        sample = pd.Series(sample)

    if _is_native_temporal_series(sample):
        return TemporalHint.native()

    _, strftime, unit = resolve_temporal(_as_arrow(sample))
    if strftime is not None:
        return TemporalHint.of_format(strftime)
    if unit is not None:
        return TemporalHint.of_unit(unit)

    if is_create_time:
        raise ValueError(
            "create_time column is entirely null or unresolved across the "
            "sample; a create_time column must contain parseable timestamp "
            "values.")
    return TemporalHint.unresolved()
