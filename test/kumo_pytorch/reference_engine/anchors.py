"""Arena-owned Parquet anchor access using Kumo's public positional order."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
from kumo.pytorch.anchors import (
    ALL_ANCHORS,
    AllAnchors,
    AnchorTable,
    PositionalAnchors,
    SplitSelection,
)
from kumo.pytorch.contract.sample import SampleRequest, TargetValue

from .parquet import anchor_columns, check_parquet_anchors, local_parquet_files


class ParquetAnchors(PositionalAnchors):
    """Projected Arrow columns in deterministic file/row order, gathered per batch.

    Only request objects for the fetched positions are decoded into Python.
    Split membership is indexed once. Kumo supplies its standard rank/shuffle
    algorithm; this reader owns the arrays and rejects reads after close.
    """

    def __init__(self, table: AnchorTable):
        issues = check_parquet_anchors(table)
        if issues:
            raise ValueError("\n".join(issue.describe() for issue in issues))
        self._closed = False
        self._table = table
        self._physical = anchor_columns(table)
        self._targets = tuple(sorted(table.targets.items()))
        self._selected: dict[SplitSelection, np.ndarray] = {}
        self._rows_by_split: dict[str | None, np.ndarray] = {}
        self._data = None
        self._n = 0
        self._all_rows = np.empty(0, dtype=np.int64)
        try:
            files = local_parquet_files(table.source.path)
            self._data = pads.dataset(files, format="parquet").to_table(
                columns=sorted(set(self._physical.values()))
            )
            self._n = self._data.num_rows
            self._all_rows = np.arange(self._n, dtype=np.int64)
            if table.split is None:
                self._rows_by_split = {None: self._all_rows}
            else:
                values = self._data.column(self._physical[table.split]).combine_chunks()
                if pa.types.is_null(values.type):
                    self._rows_by_split = {None: self._all_rows}
                else:
                    codes = values.dictionary_encode()
                    names = codes.dictionary.to_pylist()
                    if any(not isinstance(name, str) or not name for name in names):
                        raise ValueError(
                            "anchor split values must be nonempty strings or null"
                        )
                    indexes = codes.indices.fill_null(-1).to_numpy(zero_copy_only=False)
                    self._rows_by_split = {
                        names[code] if code >= 0 else None: np.flatnonzero(
                            indexes == code
                        )
                        for code in np.unique(indexes)
                    }
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("this Parquet anchor reader is closed")

    def _row_count(self) -> int:
        self._require_open()
        return self._n

    def _split_of(self, row: int) -> str | None:
        self._require_open()
        if self._table.split is None:
            return None
        return self._data.column(self._physical[self._table.split])[row].as_py()

    def splits(self) -> tuple[str, ...]:
        self._require_open()
        return super().splits()

    def _rows(self, selection: SplitSelection) -> np.ndarray:
        self._require_open()
        if isinstance(selection, AllAnchors):
            return self._all_rows
        cached = self._selected.get(selection)
        if cached is not None:
            return cached
        unknown = sorted(
            name
            for name in selection
            if name is not None and name not in self._rows_by_split
        )
        if unknown:
            raise ValueError(
                f"no anchors in split(s) {unknown}; splits present: {list(self.splits())}"
            )
        parts = [
            self._rows_by_split[name]
            for name in selection
            if name in self._rows_by_split
        ]
        rows = np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        self._selected[selection] = rows
        return rows

    def order(
        self,
        selection: SplitSelection,
        *,
        seed: int,
        epoch: int,
        shuffle: bool,
        rank: int,
        world_size: int,
    ) -> Iterator[int]:
        positions = super().order(
            selection,
            seed=seed,
            epoch=epoch,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
        )

        def consume():
            self._require_open()
            for position in positions:
                self._require_open()
                yield position

        return consume()

    def fetch(
        self, selection: SplitSelection, positions: Sequence[int]
    ) -> list[SampleRequest]:
        rows = self._rows(selection)
        selected = []
        for position in positions:
            if isinstance(position, (bool, np.bool_)):
                raise TypeError("anchor positions must be integers, not booleans")
            index = operator.index(position)
            if not 0 <= index < len(rows):
                raise IndexError(f"anchor position {index} outside [0, {len(rows)})")
            selected.append(int(rows[index]))
        if not selected:
            return []
        # One Arrow gather for the complete batch, including repeated positions.
        columns = self._data.take(pa.array(selected, type=pa.int64())).to_pydict()
        logical = {name: columns[physical] for name, physical in self._physical.items()}
        table = self._table
        return [
            SampleRequest(
                sample_id=logical[table.sample_id][index] if table.sample_id else row,
                entity_id=logical[table.entity_id][index],
                as_of_time=logical[table.as_of_time][index] if table.as_of_time else 0,
                targets=tuple(
                    TargetValue(name, logical[column][index])
                    for name, column in self._targets
                ),
                split=logical[table.split][index] if table.split else None,
                metadata={name: logical[name][index] for name in table.metadata},
            )
            for index, row in enumerate(selected)
        ]

    def _request(self, row: int) -> SampleRequest:
        return self.fetch(ALL_ANCHORS, [row])[0]

    def close(self) -> None:
        self._closed = True
        self._data = None
        self._n = 0
        self._all_rows = np.empty(0, dtype=np.int64)
        self._rows_by_split.clear()
        self._selected.clear()
