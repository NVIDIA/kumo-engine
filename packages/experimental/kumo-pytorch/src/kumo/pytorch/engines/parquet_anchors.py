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

from kumo.pytorch.anchors import AllAnchors, AnchorTable, PositionalAnchors, SplitSelection
from kumo.pytorch.contract.sample import SampleRequest, TargetValue
from kumo.pytorch.engines.unsupported import DISKGRAPH_UNSUPPORTED
from kumo.pytorch.spec import ValidationIssue

__all__ = ["ParquetAnchors", "check_parquet_anchors"]


def _files(path: str) -> list[str]:
    from kumo.pytorch.materialization import _local_parquet_files

    return _local_parquet_files(path)


def _physical(table: AnchorTable) -> dict[str, str]:
    """Logical anchor column -> physical Parquet column (honouring `Parquet.columns` renames)."""

    logical_of = dict(getattr(table.source, "columns", {}) or {})
    physical_of = {logical: physical for physical, logical in logical_of.items()}
    return {name: physical_of.get(name, name) for name in table.columns}


def check_parquet_anchors(table: AnchorTable) -> tuple[ValidationIssue, ...]:
    """Refuse what this reader cannot serve, before any task is built."""

    if getattr(table.source, "kind", None) != "parquet":
        return ()  # DiskGraphEngine.check_sources refuses it (DG-SOURCE-KIND)
    if table.candidates:
        entry = DISKGRAPH_UNSUPPORTED["DG-CANDIDATE-SETS"]
        return (
            ValidationIssue(
                "anchors.candidates",
                "the Parquet anchor reader does not serve candidate sets yet",
                entry.next_step,
                entry.code,
            ),
        )
    if "://" in table.source.path:
        return ()  # DiskGraphEngine.check_sources refuses it (DG-REMOTE-PARQUET)
    try:
        files = _files(table.source.path)
    except (FileNotFoundError, ValueError) as error:
        return (ValidationIssue("anchors.source.path", str(error), "point at local Parquet files"),)
    import pyarrow.parquet as pq

    required = _physical(table)
    for file in files:
        present = set(pq.read_schema(file).names)
        missing = sorted(
            logical for logical, physical in required.items() if physical not in present
        )
        if missing:
            return (
                ValidationIssue(
                    "anchors.columns",
                    f"anchor Parquet file {file!r} lacks columns {missing}",
                    "write those columns or rename them through Parquet(columns=...)",
                ),
            )
    return ()


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
        columns = {name: values.take(indices).to_pylist() for name, values in self._columns.items()}
        table = self._table
        targets = sorted(table.targets.items())
        return [
            SampleRequest(
                sample_id=columns[table.sample_id][index] if table.sample_id else row,
                entity_id=columns[table.entity_id][index],
                as_of_time=columns[table.as_of_time][index] if table.as_of_time else 0,
                targets=tuple(
                    TargetValue(name, columns[column][index]) for name, column in targets
                ),
                split=columns[table.split][index] if table.split else None,
                metadata={name: columns[name][index] for name in table.metadata},
            )
            for index, row in enumerate(selected)
        ]

    def _value(self, name: str, row: int) -> Any:
        return self._columns[name][row].as_py()

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
            split=self._value(table.split, row) if table.split else None,
            metadata={name: self._value(name, row) for name in table.metadata},
        )
