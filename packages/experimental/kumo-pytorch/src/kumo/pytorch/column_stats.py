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
    Composite,
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
from kumo.pytorch.encoding import (
    TEXT_RECIPE,
    FittedEncoder,
    learned_entries,
    learns_from_values,
    typed_key,
)

logger = logging.getLogger(__name__)

Column = tuple[str, str]
#: ``(table, column, semantic type)``.
StatRequest = tuple[str, str, str]

# Bump when the statistics a request asks for change, so a stale directory is never reused.
COL_STATS_RECIPE = 1
# Bump when the token-length rules change, so a stale lengths file is never reused.
TOKEN_LENGTHS_RECIPE = 1
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


#: A per-node column the model reads: its name, its fitted tokenizer's JSON if it is ``Text``,
#: and the count of a present value without pieces, 1 under ``Text(empty="unknown")``.
LengthColumn = tuple[str, str | None, int]

#: The per-node count of a row delivery drops for a NULL in ``Composite.required_fields``.
DROPPED = 0xFFFF


def token_length_columns(
    contract: Any, plan: Any
) -> dict[str, tuple[int, tuple[LengthColumn, ...], tuple[str, ...]]]:
    """Each event table's per-event constant, per-node columns and required columns under
    ``Batching.length_input``.

    ``Text`` and variable-length list columns vary per node; every other stored column the input
    reads adds 1 to the constant. Lists can't be declared yet (DG-NESTED-COLUMN), so only ``Text``
    is per-node today. The required columns are those of the ``Composite.required_fields`` of the
    input, or of its ``EventsOf`` parent, since delivery drops those rows for both.
    """

    name = contract.batch.batching.length_input
    if name is None or not isinstance(contract.batch.inputs[name].representation, Fields):
        return {}
    group = contract.batch.inputs[name].representation
    columns: dict[str, dict[str, tuple[str, int] | None]] = {}
    for use in group.fields:
        encoder = plan.output(name=name).field(name=use.field).encoder
        text = (
            (encoder.tokenizer_json, int(encoder.config.empty == "unknown"))
            if isinstance(encoder.config, Text)
            else None
        )
        for variant, _, _ in resolved_fields(contract, name, use.field):
            stored = _stored_column(contract.sample, contract.schema, variant)
            if stored is None or variant[0] not in {"events", "current_events"}:
                continue
            table, column = stored
            known = columns.setdefault(table, {}).get(column)
            if text is not None and known is not None and known != text:
                raise ValueError(
                    f"{table}.{column} is read by Text encoders with different tokenizers or "
                    "empty policies, so its token length is ambiguous"
                )
            columns[table][column] = text or known
    required = _required_columns(contract, name)
    lengths = {}
    for table in sorted(columns.keys() | required.keys()):
        read = columns.get(table, {})
        per_node = tuple(
            (column, *text) for column, text in sorted(read.items()) if text is not None
        )
        lengths[table] = (len(read) - len(per_node), per_node, required.get(table, ()))
    return lengths


def _required_columns(contract: Any, name: str) -> dict[str, tuple[str, ...]]:
    while isinstance(contract.batch.inputs[name].source, EventsOf):
        name = contract.batch.inputs[name].source.input
    value = contract.batch.inputs[name]
    row = getattr(value.representation, "row", None)
    if not isinstance(row, Composite) or not isinstance(value.source, EventSource):
        return {}
    collection = "current_events" if value.source.collection == "current" else "events"
    stored = {
        field: {
            variant[:2]: _stored_column(contract.sample, contract.schema, variant)
            for variant, _, _ in resolved_fields(contract, name, field)
        }
        for field in row.required_fields
    }
    required: dict[str, dict[str, str]] = {}
    for event in value.source.events:
        rows = getattr(contract.sample, collection)[event].rows
        table = destination_table(contract.schema, contract.sample.entity.table, rows.path)
        found = {field: stored[field].get((collection, event)) for field in row.required_fields}
        missing = [field for field, column in found.items() if column is None]
        if missing:
            raise ValueError(
                f"{table} has no stored column for required field(s) {', '.join(missing)} of "
                f"{name!r}, so token lengths can't count the events delivery keeps"
            )
        for field, (_, column) in found.items():
            if required.setdefault(table, {}).setdefault(field, column) != column:
                raise ValueError(
                    f"required field {field!r} of {name!r} reads several columns of {table}, so "
                    "token lengths can't count the events delivery keeps"
                )
    return {table: tuple(sorted(set(fields.values()))) for table, fields in required.items()}


def token_lengths_file(
    table: str,
    columns: Sequence[LengthColumn],
    required: Sequence[str],
    ticks: int | None,
    unit: str,
) -> str:
    import diskgraph

    settings = [
        [
            [
                column,
                None if tokenizer is None else hashlib.sha256(tokenizer.encode()).hexdigest(),
                empty,
            ]
            for column, tokenizer, empty in columns
        ],
        sorted(required),
        TOKEN_LENGTHS_RECIPE,
        diskgraph.TOKENIZER_TRAINER,
    ]
    digest = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:16]
    return f"{_cutoff_dir(ticks, unit)}/{file_name_part(table)}.lengths-{digest}.bin"


def read_token_lengths(
    index: str | Path, entry: Mapping[str, Any]
) -> tuple[int, np.ndarray | None]:
    """A table's per-event constant and its read-only per-node counts, indexed by node ID.

    ``entry`` is the table's ``FitRecord["statistics"]["token_lengths"]`` value and ``index``
    the DiskGraph index directory. The per-node counts are ``None`` when ``entry`` has no file;
    a count of ``DROPPED`` marks a row delivery drops, which adds neither tokens nor an event.
    """

    if entry["file"] is None:
        return entry["constant"], None
    path = Path(index, "col_stats", entry["file"])
    if not path.stat().st_size:
        return entry["constant"], np.frombuffer(b"", dtype="<u2")
    return entry["constant"], np.memmap(path, dtype="<u2", mode="r")


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
        if columns and learns_from_values(config, dtype)
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


def cutoff_nanos(value: Any) -> int | None:
    """Epoch nanoseconds of a cutoff: an ISO 8601 string, a date or datetime, or epoch seconds.

    A string is validated as `datetime.fromisoformat` reads it; DiskGraph supplies the exact value
    when it parses the string too, keeping digits past the microsecond.
    """

    if value is None:
        return None
    if type(value) is int:
        return value * 10**9
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"encoder_cutoff {value!r} is not an ISO 8601 timestamp") from None
        exact = _resolve_nanos(value)
        return _resolve_nanos(parsed) if exact is None else exact
    if isinstance(value, dt.date):
        return _resolve_nanos(value)
    raise TypeError(
        f"encoder_cutoff must be a timestamp string, datetime or epoch seconds, not {value!r}"
    )


def _resolve_nanos(value: str | dt.date) -> int | None:
    import diskgraph
    import pyarrow as pa

    if isinstance(value, str):
        array = pa.array([value])
    else:
        if not isinstance(value, dt.datetime):
            value = dt.datetime.combine(value, dt.time())
        array = pa.array([value], pa.timestamp("ns"))
    return diskgraph.resolve_temporal(array, timestamp_unit="nanoseconds")[0][0]


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
