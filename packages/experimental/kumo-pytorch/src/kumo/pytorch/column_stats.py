# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoders over stored table columns, fitted from column statistics up to a cutoff."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    "LATEST_ANCHOR",
    "Column",
    "ColumnStats",
    "StatRequest",
    "TableColumnEncoder",
    "cutoff_seconds",
    "data_sha256",
    "fit_from_stats",
    "table_column_encoders",
]

Column = tuple[str, str]
StatRequest = tuple[str, str, str]

_STATS = {Vocabulary: ("counts",), ZScore: ("moments",), Buckets: ("counts", "bins"), Text: ()}
_SUPPORTED = frozenset(
    {
        "bool",
        *(f"int{bits}" for bits in (8, 16, 32, 64)),
        *(f"uint{bits}" for bits in (8, 16, 32, 64)),
        "float32",
        "float64",
        "string",
        "date32",
        "timestamp",
    }
)
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


class _LatestAnchor:
    def __repr__(self) -> str:
        return "LATEST_ANCHOR"


LATEST_ANCHOR = _LatestAnchor()


@dataclass(frozen=True)
class ColumnStats:
    files: Mapping[StatRequest, Mapping[str, Any]]
    cutoff: Mapping[str, Any]
    cutoff_id: str | None = None
    index_id: str | None = None

    def provenance(self, files: Sequence[StatRequest]) -> dict[str, Any]:
        return {
            "cutoff": dict(self.cutoff),
            "cutoff_id": self.cutoff_id,
            "index_id": self.index_id,
            "files": {
                f"{table}/{column}.{stat}": self.files[(table, column, stat)]["sha256"]
                for table, column, stat in files
            },
        }


@dataclass(frozen=True)
class TableColumnEncoder:
    config: ScalarEncoding
    dtype: DType
    columns: tuple[Column, ...]

    @property
    def stats(self) -> tuple[str, ...]:
        return _STATS[type(self.config)]

    @property
    def requests(self) -> tuple[StatRequest, ...]:
        return tuple((table, column, stat) for table, column in self.columns for stat in self.stats)

    @property
    def text(self) -> StatRequest:
        import tokenizers

        settings = [
            self.config.vocabulary_size,
            self.config.min_frequency,
            self.columns,
            tokenizers.__version__,
            TEXT_RECIPE,
        ]
        digest = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:16]
        return (*self.columns[0], f"text-{digest}")

    @property
    def files(self) -> tuple[StatRequest, ...]:
        return (self.text,) if isinstance(self.config, Text) else self.requests


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
    if dtype.name not in _SUPPORTED:
        return False
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


def data_sha256(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def fit_from_stats(
    identity: str, encoder: TableColumnEncoder, files: Mapping[StatRequest, Mapping[str, Any]]
) -> FittedEncoder | None:
    """``None`` when a vocabulary's counts are truncated."""

    config, dtype = encoder.config, encoder.dtype
    if isinstance(config, Text):
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            tokenizer_json=files[encoder.text]["tokenizer"],
        )
    datas = {
        stat: [files[request]["data"] for request in encoder.requests if request[2] == stat]
        for stat in encoder.stats
    }
    if sum(data["non_finite"] for data in datas[encoder.stats[0]]):
        raise ValueError(
            f"encoder {identity}: {dtype.name} vocabulary values must be finite"
            if isinstance(config, Vocabulary)
            else f"encoder {identity}: numeric transforms require finite values"
        )
    if isinstance(config, ZScore):
        count, mean, m2 = 0, 0.0, 0.0
        for data in datas["moments"]:
            if data["count"]:
                total = count + data["count"]
                delta = data["mean"] - mean
                mean += delta * data["count"] / total
                m2 += data["m2"] + delta * delta * count * data["count"] / total
                count = total
        scale = math.sqrt(m2 / count) if count else 1.0
        if not math.isfinite(mean) or not math.isfinite(scale):
            raise ValueError(f"encoder {identity}: fitted statistics overflow")
        return FittedEncoder(
            identity=identity, config=config, dtype=dtype, mean=mean, scale=scale or 1.0
        )
    exact = not any(data["truncated"] for data in datas["counts"])
    if isinstance(config, Vocabulary):
        if not exact:
            return None
        counts: Counter = Counter()
        actual = {}
        for data in datas["counts"]:
            for raw, count in data["values"]:
                value = _value(dtype, raw)
                key = typed_key(dtype, value)
                counts[key] += count
                actual.setdefault(key, value)
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            entries=learned_entries(config, counts, actual),
        )
    ranges = config.bins - int(config.separate_zero)
    probabilities = np.linspace(0, 1, ranges + 1)[1:-1]
    boundaries = (
        _exact_quantiles(datas["counts"], probabilities, config.separate_zero)
        if exact
        else _binned_quantiles(datas["bins"], probabilities, config.separate_zero)
    )
    return FittedEncoder(
        identity=identity,
        config=config,
        dtype=dtype,
        boundaries=tuple(float(value) for value in np.unique(boundaries)),
    )


def _value(dtype: DType, raw: Any) -> Any:
    if dtype.name == "timestamp":
        return _EPOCH + dt.timedelta(seconds=raw)
    if dtype.name == "date32":
        return dt.date(1970, 1, 1) + dt.timedelta(days=raw)
    return raw


def _interpolated(total: int, order_statistic: Any, probabilities: np.ndarray) -> list[float]:
    """``np.quantile``'s linear interpolation."""

    result = []
    for probability in probabilities:
        position = (total - 1) * probability
        below = math.floor(position)
        low = order_statistic(below)
        if position == below:
            result.append(low)
            continue
        high, weight = order_statistic(below + 1), position - below
        result.append(
            high - (high - low) * (1 - weight) if weight >= 0.5 else low + (high - low) * weight
        )
    return result


def _exact_quantiles(
    datas: Sequence[Mapping[str, Any]], probabilities: np.ndarray, skip_zero: bool
):
    counts: Counter = Counter()
    for data in datas:
        counts.update({float(value): count for value, count in data["values"]})
    if skip_zero:
        counts.pop(0.0, None)
    if not counts:
        return []
    values = np.asarray(sorted(counts))
    ends = np.cumsum([counts[value] for value in values])
    return _interpolated(
        int(ends[-1]),
        lambda rank: values[np.searchsorted(ends, rank, side="right")],
        probabilities,
    )


def _bin_edges(key: int, bits: int) -> tuple[float, float]:
    def magnitude(index: int) -> float:
        return struct.unpack("<d", struct.pack("<Q", index << bits))[0]

    if key >= 0:
        return magnitude(key), magnitude(key + 1)
    return -magnitude(-key), -magnitude(-key - 1)


def _binned_quantiles(
    datas: Sequence[Mapping[str, Any]], probabilities: np.ndarray, skip_zero: bool
):
    """Ranks spread evenly within a bin, clamped to the exact minimum and maximum."""

    bins: Counter = Counter()
    zeros = 0
    for data in datas:
        bits = 52 - int(math.log2(data["bins_per_octave"]))
        bins.update({key: count for key, count in data["bins"]})
        zeros += data["zeros"]
    lowest = min((data["min"] for data in datas if data["min"] is not None), default=None)
    highest = max((data["max"] for data in datas if data["max"] is not None), default=None)
    bins[0] -= zeros
    pieces = [(*_bin_edges(key, bits), count) for key, count in sorted(bins.items()) if count]
    if not skip_zero and zeros:
        pieces.append((0.0, 0.0, zeros))
    pieces.sort(key=lambda piece: (piece[0], piece[1]))
    if not pieces:
        return []
    starts = np.cumsum([0] + [count for _, _, count in pieces])

    def order_statistic(rank: int) -> float:
        index = int(np.searchsorted(starts, rank, side="right")) - 1
        low, high, count = pieces[index]
        low, high = max(low, lowest), min(high, highest)
        return low + (high - low) * (rank - starts[index] + 0.5) / count

    return _interpolated(int(starts[-1]), order_statistic, probabilities)
