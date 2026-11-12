# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoders over stored table columns, fitted from DiskGraph's column statistics at a cutoff."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from kumo.pytorch.contract.batch import (
    EntitySource,
    EventsOf,
    EventSource,
    Fields,
    Graph,
    UseEncoding,
)
from kumo.pytorch.contract.batch.encodings import Buckets, ScalarEncoding, Text, Vocabulary, ZScore
from kumo.pytorch.contract.batch.validation import resolved_fields
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import Field, destination_table
from kumo.pytorch.data.records import cutoff_to_epoch
from kumo.pytorch.encoding import TEXT_RECIPE, FittedEncoder, learned_entries, typed_key

__all__ = [
    "EMPTY_STATS",
    "LATEST_ANCHOR",
    "ColumnStats",
    "StatRequest",
    "TableColumnEncoder",
    "cutoff_seconds",
    "fit_from_stats",
    "stats_request",
    "stype_groups",
    "table_column_encoders",
]

logger = logging.getLogger(__name__)

Column = tuple[str, str]
#: ``(table, column, semantic type)``.
StatRequest = tuple[str, str, str]

# Quantiles each numerical column's t-digest publishes; Buckets cuts interpolate them.
QUANTILE_GRID = [index / 1024 for index in range(1025)]
_TICKS_PER_SECOND = {
    "seconds": 1,
    "milliseconds": 10**3,
    "microseconds": 10**6,
    "nanoseconds": 10**9,
}
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
#: DiskGraph's statistics of a column with no rows.
EMPTY_STATS = {
    "count": 0,
    "null_count": 0,
    "invalid_count": 0,
    "mean": 0.0,
    "std": 0.0,
    "quantiles": [],
    "value_bins": [],
    "category_counts": [],
    "category_counts_capped": False,
}


class _LatestAnchor:
    def __repr__(self) -> str:
        return "LATEST_ANCHOR"


LATEST_ANCHOR = _LatestAnchor()


@dataclass(frozen=True)
class ColumnStats:
    #: Each request's entry in DiskGraph's ``col_stats.json``.
    columns: Mapping[StatRequest, Mapping[str, Any]]
    cutoff: Mapping[str, Any]
    #: The ``col_stats/`` directories read.
    dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableColumnEncoder:
    config: ScalarEncoding
    dtype: DType
    columns: tuple[Column, ...]

    @property
    def requests(self) -> tuple[StatRequest, ...]:
        if isinstance(self.config, Text):
            return ()
        stype = "categorical" if isinstance(self.config, Vocabulary) else "numerical"
        return tuple((table, column, stype) for table, column in self.columns)

    def tokenizer_dir(self, cutoff: int | None, unit: str) -> str:
        import tokenizers

        settings = [
            cutoff,
            unit,
            self.config.vocabulary_size,
            self.config.min_frequency,
            self.columns,
            tokenizers.__version__,
            TEXT_RECIPE,
        ]
        return "text-" + hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:16]


def stats_request(
    requests: Sequence[StatRequest], cutoffs: Sequence[int | None], unit: str
) -> dict:
    """DiskGraph's ``cutoff_col_stats`` request; each cutoff's directory names its inputs."""

    columns: dict[str, dict[str, str]] = {}
    for table, column, stype in sorted(requests):
        columns.setdefault(table, {})[column] = stype
    quantiles = {
        table: {column: QUANTILE_GRID for column, stype in types.items() if stype == "numerical"}
        for table, types in columns.items()
    }
    dirs = [
        hashlib.sha256(json.dumps([cutoff, unit, columns]).encode()).hexdigest()[:16]
        for cutoff in cutoffs
    ]
    return {"cutoffs": list(cutoffs), "dirs": dirs, "columns": columns, "quantiles": quantiles}


def stype_groups(requests: Sequence[StatRequest]) -> list[list[StatRequest]]:
    """Requests split so no group asks one column for two semantic types."""

    groups: list[list[StatRequest]] = []
    for request in sorted(set(requests)):
        group = next((g for g in groups if all(r[:2] != request[:2] for r in g)), None)
        if group is None:
            groups.append(group := [])
        group.append(request)
    return groups


def table_column_encoders(contract: Any) -> dict[str, TableColumnEncoder]:
    from kumo.pytorch.assembly import _encoder_identity, _graph_encoder_identity

    spec, sample, schema = contract.batch, contract.sample, contract.schema
    uses: dict[str, tuple[ScalarEncoding, DType, set[Column] | None]] = {}

    def add(identity: str, encoding: Any, dtype: DType, columns: set[Column] | None) -> None:
        config = spec.encodings[encoding.name] if isinstance(encoding, UseEncoding) else encoding
        _, _, known = uses.get(identity, (config, dtype, set()))
        uses[identity] = (
            config,
            dtype,
            None if known is None or columns is None else known | columns,
        )

    tables = {table.name: table for table in schema.tables}
    for name, value in sorted(spec.inputs.items()):
        if isinstance(value.representation, Graph):
            for table, group in value.representation.node_fields.items():
                for use in group.fields:
                    add(
                        _graph_encoder_identity(name, table, use),
                        use.encoding,
                        tables[table].columns[use.field],
                        {(table, use.field)},
                    )
            continue
        if not isinstance(value.representation, Fields):
            continue
        direct = isinstance(value.source, (EntitySource, EventSource, EventsOf))
        for use in value.representation.fields:
            variants = resolved_fields(contract, name, use.field)
            columns = {_stored_column(sample, schema, variant) for variant, _, _ in variants}
            add(
                _encoder_identity(name, use),
                use.encoding,
                variants[0][1],
                columns if direct and None not in columns else None,
            )
    return {
        identity: TableColumnEncoder(config, dtype, tuple(sorted(columns)))
        for identity, (config, dtype, columns) in sorted(uses.items())
        if columns and _learns_from_stats(config, dtype)
    }


def _stored_column(sample: Any, schema: Any, identity: tuple[str, ...]) -> Column | None:
    if identity[0] == "entity":
        declaration, table = sample.entity_values[identity[1]], sample.entity.table
    elif identity[0] in {"events", "current_events"}:
        component = (sample.events if identity[0] == "events" else sample.current_events)[
            identity[1]
        ]
        declaration = component.fields[identity[2]]
        table = destination_table(schema, sample.entity.table, component.rows.path)
    else:
        return None
    if not isinstance(declaration, Field) or declaration.path:
        return None
    return table, declaration.column


def _learns_from_stats(config: Any, dtype: DType) -> bool:
    if isinstance(config, Vocabulary):
        return config.values is None
    if isinstance(config, ZScore):
        return config.mean is None and dtype.numeric
    if isinstance(config, Buckets):
        return config.bins is not None and dtype.numeric
    if isinstance(config, Text):
        return config.state is None and dtype.name == "string"
    return False


def cutoff_seconds(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"encoder_cutoff {value!r} is not an ISO 8601 timestamp") from None
    if isinstance(value, dt.date):
        return cutoff_to_epoch(value)
    raise TypeError(f"encoder_cutoff must be a timestamp string or datetime, not {value!r}")


def fit_from_stats(identity: str, encoder: TableColumnEncoder, stats: ColumnStats) -> FittedEncoder:
    config, dtype = encoder.config, encoder.dtype
    datas = [stats.columns[request] for request in encoder.requests]
    if isinstance(config, Vocabulary):
        counts: Counter = Counter()
        actual = {}
        for (table, column), data in zip(encoder.columns, datas, strict=True):
            if data["category_counts_capped"]:
                logger.warning(
                    "%s.%s has more distinct values than DiskGraph counts; encoder %s learns "
                    "the ones it counted",
                    table,
                    column,
                    identity,
                )
            for raw, count in data["category_counts"]:
                value = dtype.normalize(_category(dtype, raw, stats.cutoff["unit"]))
                key = typed_key(dtype, value)
                counts[key] += count
                actual.setdefault(key, value)
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            entries=learned_entries(config, counts, actual),
        )
    if any(data["invalid_count"] for data in datas):
        raise ValueError(f"encoder {identity}: numeric transforms require finite values")
    finite = [data["count"] - data["null_count"] for data in datas]
    if isinstance(config, ZScore):
        count, mean, m2 = 0, 0.0, 0.0
        for data, n in zip(datas, finite, strict=True):
            if n:
                total = count + n
                delta = data["mean"] - mean
                mean += delta * n / total
                m2 += data["std"] ** 2 * (n - 1) + delta * delta * count * n / total
                count = total
        scale = math.sqrt(m2 / count) if count else 1.0
        if not math.isfinite(mean) or not math.isfinite(scale):
            raise ValueError(f"encoder {identity}: fitted statistics overflow")
        return FittedEncoder(
            identity=identity, config=config, dtype=dtype, mean=mean, scale=scale or 1.0
        )
    return FittedEncoder(
        identity=identity,
        config=config,
        dtype=dtype,
        boundaries=_bucket_cuts(config, datas, finite),
    )


def _category(dtype: DType, raw: Any, unit: str) -> Any:
    if dtype.name == "string":
        return raw
    if dtype.name == "bytes":
        return bytes.fromhex(raw)
    if dtype.name.startswith("float"):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{dtype.name} vocabulary values must be finite")
        return value
    if dtype.name.startswith("decimal"):
        return Decimal(raw).quantize(Decimal(1).scaleb(-dtype.scale))
    value = int(raw)
    if dtype.name == "bool":
        return bool(value)
    if dtype.name == "uint64" and value < 0:
        return value + 2**64
    if dtype.name == "date32":
        return dt.date(1970, 1, 1) + dt.timedelta(days=value)
    if dtype.name == "timestamp":
        return _EPOCH + dt.timedelta(seconds=value / _TICKS_PER_SECOND[unit])
    return value


def _bucket_cuts(config: Buckets, datas: Sequence[Mapping[str, Any]], finite: Sequence[int]):
    """``np.quantile``'s cuts of the columns' combined values, from each one's t-digest grid."""

    grids = [(np.asarray(data["quantiles"]), n) for data, n in zip(datas, finite) if n]
    if not grids:
        return ()
    zeros = (
        sum(count for data in datas for start, count in data["value_bins"] if start == 0.0)
        if config.separate_zero
        else 0
    )
    values = np.unique(np.concatenate([grid for grid, _ in grids]))
    below = sum(n * np.interp(values, grid, QUANTILE_GRID) for grid, n in grids)
    total = sum(n for _, n in grids) - zeros
    if total <= 0:
        return ()
    below = (below - zeros * (values >= 0)) / total
    ranges = config.bins - int(config.separate_zero)
    cuts = np.interp(np.linspace(0, 1, ranges + 1)[1:-1], np.maximum.accumulate(below), values)
    return tuple(float(value) for value in np.unique(cuts))
