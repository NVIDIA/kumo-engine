# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an anchor table held as Python mappings.

The reference reader: obvious scans, one decode path, no format handling. It is
what an optimized reader is compared against, so it is written to be believed
rather than to be fast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kumo.pytorch.anchors import AnchorTable, CandidateColumn, PositionalAnchors
from kumo.pytorch.contract.sample import CandidateRequest, SampleRequest, TargetValue
from kumo.pytorch.dataset import InMemoryRows

__all__ = ["InMemoryAnchors"]


class InMemoryAnchors(PositionalAnchors):
    """Anchors from a sequence of Python mappings, in the order given.

    Rows are decoded on demand rather than up front: the table is already in
    memory, and a second copy as request objects would buy nothing.
    """

    def __init__(self, table: AnchorTable):
        if not isinstance(table.source, InMemoryRows):
            raise TypeError(
                f"the in-memory engine reads in-memory anchors, not {table.source.kind!r}"
            )
        self._table = table
        self._table_rows: Sequence[Mapping[str, Any]] = tuple(table.source.rows)
        super().__init__()

    def _row_count(self) -> int:
        return len(self._table_rows)

    def _split_of(self, row: int) -> str | None:
        if self._table.split is None:
            return None
        return self._table_rows[row].get(self._table.split)

    def _request(self, position: int) -> SampleRequest:
        row = self._table_rows[position]
        table = self._table
        missing = [name for name in table.columns if name not in row]
        if missing:
            raise ValueError(f"anchor row {position} is missing columns {sorted(missing)}")
        return SampleRequest(
            sample_id=row[table.sample_id] if table.sample_id else position,
            entity_id=row[table.entity_id],
            # A non-temporal contract has no prediction time to carry, and
            # validation has already refused an as_of_time column for one.
            as_of_time=row[table.as_of_time] if table.as_of_time else 0,
            targets=tuple(
                TargetValue(name, row[column]) for name, column in sorted(table.targets.items())
            ),
            split=row[table.split] if table.split else None,
            metadata={name: row[name] for name in table.metadata},
            candidates={
                name: self._candidates(position, row, binding)
                for name, binding in sorted(table.candidates.items())
            },
        )

    @staticmethod
    def _candidates(
        position: int,
        row: Mapping[str, Any],
        binding: CandidateColumn,
    ) -> tuple[CandidateRequest, ...]:
        entries = row[binding.column]
        if entries is None:
            # A declared set with nothing for this anchor is empty, not an
            # error: the contract already permits an empty candidate set.
            return ()
        requests = []
        for index, entry in enumerate(entries):
            missing = [field for field in binding.fields if field not in entry]
            if missing:
                raise ValueError(
                    f"candidate {index} of anchor row {position} is missing {sorted(missing)}"
                )
            requests.append(
                CandidateRequest(
                    entity_id=entry[binding.entity_id],
                    targets=tuple(
                        TargetValue(name, entry[field])
                        for name, field in sorted(binding.targets.items())
                    ),
                )
            )
        # Order is the column's order, not a sort: candidate labels align
        # positionally with the candidate input.
        return tuple(requests)
