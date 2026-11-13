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
from pathlib import Path
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
from kumo.pytorch.contract.dtypes import DType, _since_epoch
from kumo.pytorch.contract.sample import Field, destination_table
from kumo.pytorch.encoding import TEXT_RECIPE, FittedEncoder, learned_entries, typed_key

logger = logging.getLogger(__name__)

Column = tuple[str, str]
#: ``(table, column, semantic type)``.
StatRequest = tuple[str, str, str]

# Bump when the statistics a request asks for change, so a stale directory is never reused.
COL_STATS_RECIPE = 1
# Buckets cuts interpolate the t-digest quantiles at these probabilities.
QUANTILE_GRID = [index / 1024 for index in range(1025)]
#: A key each semantic type's statistics add to a column's entry.
_STAT_KEYS = {"categorical": "category_counts", "numerical": "mean"}
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
    #: Each request's column entry, in DiskGraph's ``col_stats.json`` format.
    columns: Mapping[StatRequest, Mapping[str, Any]]
    cutoff: Mapping[str, Any]
    #: The ``col_stats/`` files read.
    files: tuple[str, ...] = ()


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

    def tokenizer_file(self, ticks: int | None, unit: str) -> str:
        import diskgraph

        settings = [
            self.config.vocabulary_size,
            self.config.min_frequency,
            self.columns,
            TEXT_RECIPE,
            diskgraph.TOKENIZER_TRAINER,
        ]
        digest = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:16]
        table, column = (file_name_part(name) for name in self.columns[0])
        return f"{_cutoff_dir(ticks, unit)}/{table}.{column}.text-{digest}.tokenizer.json"


#: A stored column the model reads, with its fitted tokenizer's JSON if it is ``Text``.
LengthColumn = tuple[str, str | None]


def token_length_columns(contract: Any, plan: Any) -> dict[str, tuple[LengthColumn, ...]]:
    """Each event table's stored columns that the inputs of a fitted plan read."""

    columns: dict[str, set[LengthColumn]] = {}
    for name, value in sorted(contract.batch.inputs.items()):
        if not isinstance(value.representation, Fields):
            continue
        for use in value.representation.fields:
            encoder = plan.output(name=name).field(name=use.field).encoder
            tokenizer = encoder.tokenizer_json if isinstance(encoder.config, Text) else None
            for variant, _, _ in resolved_fields(contract, name, use.field):
                stored = _stored_column(contract.sample, contract.schema, variant)
                if stored is not None and variant[0] in {"events", "current_events"}:
                    columns.setdefault(stored[0], set()).add((stored[1], tokenizer))
    return {
        table: tuple(sorted(items, key=lambda item: (item[0], item[1] or "")))
        for table, items in sorted(columns.items())
    }


def token_lengths_file(
    table: str, columns: Sequence[LengthColumn], ticks: int | None, unit: str
) -> str:
    import diskgraph

    settings = [
        [
            [column, None if tokenizer is None else hashlib.sha256(tokenizer.encode()).hexdigest()]
            for column, tokenizer in columns
        ],
        diskgraph.TOKENIZER_TRAINER,
    ]
    digest = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:16]
    return f"{_cutoff_dir(ticks, unit)}/{file_name_part(table)}.lengths-{digest}.bin"


def read_token_lengths(path: str | Path) -> np.ndarray:
    """A token lengths file's per-node counts, memory-mapped, indexed by node ID."""

    if not Path(path).stat().st_size:
        return np.zeros(0, dtype="<u2")
    return np.memmap(path, dtype="<u2", mode="r")


def stats_request(
    requests: Sequence[StatRequest], cutoffs: Sequence[int | None], unit: str
) -> dict:
    """DiskGraph's ``cutoff_col_stats`` request, into each cutoff's own directory."""

    columns: dict[str, dict[str, list[str]]] = {}
    for table, column, stype in sorted(set(requests)):
        columns.setdefault(table, {}).setdefault(column, []).append(stype)
    return {
        "cutoffs": list(cutoffs),
        "dirs": [_cutoff_dir(cutoff, unit) for cutoff in cutoffs],
        "columns": columns,
        "quantiles": QUANTILE_GRID,
    }


def read_column_stats(
    index: str | Path,
    requests: Sequence[StatRequest],
    ticks: int | None,
    unit: str,
    num_threads: int | None = None,
) -> ColumnStats:
    """The requested statistics at a cutoff, first computing into the index those it lacks."""

    import diskgraph

    directory = Path(index, "col_stats", _cutoff_dir(ticks, unit))
    paths = {
        table: directory / f"{file_name_part(table)}.col_stats.json" for table, _, _ in requests
    }

    def read() -> dict:
        return {
            table: json.loads(path.read_text())[table]
            for table, path in paths.items()
            if path.is_file()
        }

    found = read()
    missing = [
        (table, column, stype)
        for table, column, stype in requests
        if _STAT_KEYS[stype] not in found.get(table, {}).get(column, {})
    ]
    if missing:
        request = stats_request(missing, [ticks], unit)
        diskgraph.cutoff_col_stats(str(index), json.dumps(request), num_threads=num_threads)
        found = read()
    return ColumnStats(
        {item: found[item[0]][item[1]] for item in requests},
        {"value": ticks, "unit": unit},
        tuple(f"{directory.name}/{path.name}" for _, path in sorted(paths.items())),
    )


def file_name_part(name: str) -> str:
    """``name`` with every byte but ASCII letters, digits, ``-`` and ``_`` percent-encoded."""

    return "".join(
        chr(byte) if bytes([byte]).isalnum() or byte in b"-_" else f"%{byte:02X}"
        for byte in name.encode()
    )


def _cutoff_dir(ticks: int | None, unit: str) -> str:
    return hashlib.sha256(json.dumps([COL_STATS_RECIPE, ticks, unit]).encode()).hexdigest()[:16]


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


def cutoff_nanos(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"encoder_cutoff {value!r} is not an ISO 8601 timestamp") from None
    if isinstance(value, dt.date):
        return _since_epoch(value) // dt.timedelta(microseconds=1) * 1_000
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
        return Decimal(raw)
    value = int(raw)
    if dtype.name == "bool":
        return bool(value)
    if dtype.name == "uint64" and value < 0:
        return value + 2**64
    if dtype.name == "date32":
        return dt.date(1970, 1, 1) + dt.timedelta(days=value)
    if dtype.name == "timestamp":
        import diskgraph

        return _EPOCH + dt.timedelta(seconds=diskgraph.convert_ticks(value, unit, "seconds"))
    return value


def _bucket_cuts(config: Buckets, datas: Sequence[Mapping[str, Any]], finite: Sequence[int]):
    """``np.quantile``'s cuts of the columns' combined values, from each one's t-digest grid."""

    grids = [(np.asarray(data["quantiles"]), n) for data, n in zip(datas, finite) if n]
    if not grids:
        return ()
    # Interpolation slopes overflow for values past about 1e289; scale by an exact power of 2.
    scale = 2.0 ** max(0, math.frexp(max(np.abs(grid).max() for grid, _ in grids))[1] - 960)
    grids = [(grid / scale, n) for grid, n in grids]
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
    return tuple(float(value * scale) for value in np.unique(cuts))
