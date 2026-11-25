# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an anchor table stored as local Parquet, for engines whose data is Parquet.

Anchor columns are held as Arrow arrays (millions of anchors cost a few hundred
megabytes) and turned into `SampleRequest` values only when a batch fetches
them. Split membership is indexed once with NumPy, so a split of millions of
rows is selected without a Python pass per row.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np

from kumo.pytorch.anchors import (
    ALL_ANCHORS,
    AllAnchors,
    AnchorTable,
    PositionalAnchors,
    SplitSelection,
)
from kumo.pytorch.contract.dtypes import DType, _arrow_scalar_value, _arrow_values
from kumo.pytorch.contract.sample import SampleRequest, TargetValue
from kumo.pytorch.dataset import _can_coerce_to_int64, _MissingParquetColumns
from kumo.pytorch.spec import ValidationIssue

__all__ = ["ParquetAnchors", "check_parquet_anchors"]


def _files(path: str) -> list[str]:
    from kumo.pytorch.materialization import _local_parquet_files

    return _local_parquet_files(path)


def _physical(table: AnchorTable) -> dict[str, str]:
    """Logical anchor column -> physical Parquet column (honouring `Parquet.columns` renames)."""

    return table.source._bind_columns(tuple(table.columns))


def _anchor_key_array(values: Any, dtype: DType) -> Any:
    """Keep already-typed integer/string keys columnar; normalize other inputs."""
    import pyarrow as pa

    arrow_type = dtype.to_arrow()
    if (dtype.integer or dtype.name == "string") and values.type == arrow_type:
        return values
    return pa.array(
        [dtype.normalize(value, path="anchors.entity_id") for value in values.to_pylist()],
        type=arrow_type,
    )


def _entity_id_type(physical: Any, key: DType) -> Any:
    """The Arrow type that entity ids stored as ``physical`` are read as, for this key.

    An ``int64`` key reads narrower signed integers and scale-zero decimals as
    ``int64``, the same physical types a declared ``int64`` table column
    accepts. Every other type is kept, and its values are checked one by one.
    """

    try:
        coerced = _can_coerce_to_int64(DType.from_arrow(physical), key)
    except (TypeError, ValueError):
        return physical
    return key.to_arrow() if coerced else physical


def _entity_ids(values: Any, key: DType, file: str) -> Any:
    """Read one batch of entity ids as `_entity_id_type` says, refusing values outside the key."""

    import pyarrow as pa
    import pyarrow.compute as pc

    target = _entity_id_type(values.type, key)
    if target == values.type:
        return values
    try:
        return pc.cast(values, target, safe=True)
    except pa.ArrowInvalid as error:
        raise ValueError(
            f"anchors.entity_id in {file!r}: a {values.type} id is outside the "
            f"{key.name} primary key's range ({error})"
        ) from error


def check_parquet_anchors(table: AnchorTable) -> tuple[ValidationIssue, ...]:
    """Refuse what this reader cannot serve, before any task is built."""

    if getattr(table.source, "kind", None) != "parquet":
        return ()  # DiskGraphEngine.check_sources refuses it (DG-SOURCE-KIND)
    if "://" in table.source.path:
        return ()  # DiskGraphEngine.check_sources refuses it (DG-REMOTE-PARQUET)
    try:
        files = _files(table.source.path)
    except (FileNotFoundError, ValueError) as error:
        return (ValidationIssue("anchors.source.path", str(error), "point at local Parquet files"),)
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        schemas = {file: pq.read_schema(file) for file in files}
        required = table.source._bind_columns(
            tuple(table.columns),
            file_columns={file: schema.names for file, schema in schemas.items()},
        )
    except _MissingParquetColumns as error:
        return (
            ValidationIssue(
                "anchors.columns",
                f"anchor Parquet file {error.path!r} lacks columns {error.columns}",
                "write those columns or rename them through Parquet(columns=...)",
            ),
        )
    except (ValueError, OSError) as error:
        return (
            ValidationIssue(
                "anchors.columns",
                f"anchor Parquet {error}",
                "write those columns or rename them through Parquet(columns=...)",
            ),
        )
    for file, schema in schemas.items():
        for name, binding in table.candidates.items():
            dtype = schema.field(required[binding.column]).type
            if not (
                (
                    pa.types.is_list(dtype)
                    or pa.types.is_large_list(dtype)
                    or pa.types.is_fixed_size_list(dtype)
                )
                and pa.types.is_struct(dtype.value_type)
                and all(field in dtype.value_type.names for field in binding.fields)
            ):
                return (
                    ValidationIssue(
                        f"anchors.candidates.{name}",
                        f"{binding.column!r} in {file!r} must be a list of structs "
                        f"with fields {binding.fields}, got {dtype}",
                        "supply the candidate ID and bound label fields in each struct",
                    ),
                )
    return ()


def _candidate_row(table, columns, index, position):
    """Decode nested IDs/labels field by field to preserve nanosecond Arrow scalars."""
    result = {}
    for name, binding in table.candidates.items():
        value = columns[binding.column][index]
        entries = (
            (
                {field: _arrow_scalar_value(entry[field]) for field in binding.fields}
                if entry.is_valid
                else None
                for entry in value.values
            )
            if value.is_valid
            else None
        )
        result[name] = binding._requests(entries, position)
    return result


class ParquetAnchors(PositionalAnchors):
    """Anchors from local Parquet, in file-then-row order (the canonical order)."""

    def __init__(self, table: AnchorTable):
        issues = check_parquet_anchors(table)
        if issues:
            raise ValueError(issues[0].describe())
        import pyarrow.dataset as pads

        self._table = table
        physical = _physical(table)
        data = pads.dataset(_files(table.source.path), format="parquet").to_table(
            columns=sorted(set(physical.values()))
        )
        self._columns = {
            logical: data.column(column).combine_chunks() for logical, column in physical.items()
        }
        self._n = data.num_rows
        self._all_rows = np.arange(self._n, dtype=np.int64)
        if table.split is None:
            self._rows_by_split = {None: self._all_rows}
        else:
            codes = self._columns[table.split].dictionary_encode()
            index = np.asarray(codes.indices.fill_null(-1).to_numpy(zero_copy_only=False))
            names = codes.dictionary.to_pylist()
            self._rows_by_split = {
                (names[code] if code >= 0 else None): np.flatnonzero(index == code)
                for code in np.unique(index)
            }
        self._selected: dict[Any, np.ndarray] = {}

    def _row_count(self) -> int:
        return self._n

    def _split_of(self, row: int) -> str | None:
        if self._table.split is None:
            return None
        return self._columns[self._table.split][row].as_py()

    def _rows(self, selection: AllAnchors | Collection[str | None]) -> np.ndarray:  # type: ignore[override]
        if isinstance(selection, AllAnchors):
            return self._all_rows
        cached = self._selected.get(selection)
        if cached is not None:
            return cached
        unknown = sorted(
            name for name in selection if name is not None and name not in self._rows_by_split
        )
        if unknown:
            present = list(self.splits())
            raise ValueError(
                f"no anchors in split(s) {unknown}; splits present: {present}"
                if present
                else f"no anchors in split(s) {unknown}; these anchors declare no splits"
            )
        parts = [self._rows_by_split[name] for name in selection if name in self._rows_by_split]
        rows = np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
        self._selected[selection] = rows
        return rows

    def fetch(self, selection: SplitSelection, positions: Sequence[int]) -> Sequence[SampleRequest]:
        """Gather each Arrow column once, preserving selection order and duplicates."""
        import pyarrow as pa

        rows = self._rows(selection)
        selected = [int(rows[position]) for position in positions]
        if not selected:
            return []
        indices = pa.array(selected, type=pa.int64())
        selected_columns = {name: values.take(indices) for name, values in self._columns.items()}
        table = self._table
        columns = {
            name: _arrow_values(selected_columns[name]) for name in set(table._scalar_columns)
        }
        targets = sorted(table.targets.items())
        return [
            SampleRequest(
                sample_id=columns[table.sample_id][index] if table.sample_id else row,
                entity_id=columns[table.entity_id][index],
                as_of_time=columns[table.as_of_time][index] if table.as_of_time else 0,
                targets=tuple(
                    TargetValue(name, columns[column][index]) for name, column in targets
                ),
                candidates=_candidate_row(table, selected_columns, index, row),
                split=columns[table.split][index] if table.split else None,
                metadata={name: columns[name][index] for name in table.metadata},
            )
            for index, row in enumerate(selected)
        ]

    def latest_time(self, selection: SplitSelection = ALL_ANCHORS) -> Any:
        """The latest ``as_of_time`` of the selected anchors, as `fetch` returns it."""
        import pyarrow.compute as pc

        rows = self._rows(selection)
        if not len(rows):
            raise ValueError("no anchors selected")
        if self._table.as_of_time is None:
            return 0
        times = self._columns[self._table.as_of_time]
        if not isinstance(selection, AllAnchors):
            times = times.take(rows)
        if times.null_count:
            raise ValueError("anchor prediction times cannot be NULL")
        return _arrow_scalar_value(pc.max(times))

    def _value(self, name: str, row: int) -> Any:
        return _arrow_scalar_value(self._columns[name][row])

    def _request(self, position: int) -> SampleRequest:
        row = int(position)
        table = self._table
        return SampleRequest(
            sample_id=self._value(table.sample_id, row) if table.sample_id else row,
            entity_id=self._value(table.entity_id, row),
            as_of_time=self._value(table.as_of_time, row) if table.as_of_time else 0,
            targets=tuple(
                TargetValue(name, self._value(column, row))
                for name, column in sorted(table.targets.items())
            ),
            candidates=_candidate_row(table, self._columns, row, row),
            split=self._value(table.split, row) if table.split else None,
            metadata={name: self._value(name, row) for name in table.metadata},
        )
