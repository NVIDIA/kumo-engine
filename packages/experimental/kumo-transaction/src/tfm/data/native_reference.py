# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded history reads for optional Python reference baselines after inference.

The native model path never uses this reader. Hand-engineered reference models
consume one source history at a time and retain only their feature matrices;
there is no event corpus or tokenization fallback in the training iterator.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import overload

import pyarrow.parquet as pq

from tfm.contracts.schema import EntityProfile, LifelongItem, TransactionEvent, TransactionSequence
from tfm.data.native_inputs import NativeInputs
from tfm.data.native_source import source_entity_id

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class ReferenceRows:
    """Disk-backed scalar features, shared by lazily generated reference rows."""

    def __init__(self, inputs: NativeInputs) -> None:
        self.inputs = inputs
        self.db = inputs.normalized.connection
        self.db.execute(
            "CREATE TABLE reference_rows(table_name TEXT,code INTEGER,numbers TEXT,categories TEXT,texts TEXT,event_id TEXT,PRIMARY KEY(table_name,code))"
        )
        for table in inputs.source.tables:
            if table.role != "input":
                continue
            for block in pq.ParquetFile(inputs.source.files[table.name]).iter_batches(
                batch_size=4096
            ):
                rows = []
                original = next(t for t in inputs.original.tables if t.name == table.name)
                for row in block.to_pylist():
                    fields = [
                        {
                            alias: row[column]
                            for alias, column in mapping.items()
                            if not alias.startswith("flat_") and row[column] is not None
                        }
                        for mapping in (table.numeric, table.categorical, table.text)
                    ]
                    rows.append(
                        (
                            table.name,
                            row["__tfm_row_id"],
                            *(json.dumps(value) for value in fields),
                            source_entity_id(tuple(row[column] for column in original.key)),
                        )
                    )
                self.db.executemany("INSERT INTO reference_rows VALUES (?,?,?,?,?,?)", rows)
        self.db.commit()

    def rows(self, sample_ids: Sequence[str]) -> ReferenceSequences:
        return ReferenceSequences(self, tuple(sample_ids))

    def profile(self, entity: str) -> EntityProfile:
        code = self.inputs.normalized.code("users", (entity,))
        row = self.db.execute(
            "SELECT numbers,categories,texts FROM reference_rows WHERE table_name='users' AND code=?",
            (code,),
        ).fetchone()
        attributes = {key: str(value) for raw in row for key, value in json.loads(raw).items()}
        facts = tuple(
            LifelongItem(
                key=json.loads(cat)["milestone"], timestamp=EPOCH + timedelta(microseconds=at)
            )
            for cat, at in self.db.execute(
                "SELECT r.categories,i.event_us FROM links l JOIN reference_rows r ON r.table_name='facts' AND r.code=l.child JOIN identities i ON i.table_name='facts' AND i.code=l.child WHERE l.role='facts' AND l.parent=? ORDER BY i.event_us,i.code",
                (code,),
            )
        )
        return EntityProfile(entity, EPOCH, attributes, facts)

    def sequence(self, sample_id: str) -> TransactionSequence:
        root, cutoff, split = self.db.execute(
            "SELECT root,cutoff,split FROM anchors WHERE sample_id=?", (sample_id,)
        ).fetchone()
        view = (
            "train"
            if split == "train"
            else "validation"
            if split == "validation"
            else self.inputs.fit_view
            if split == self.inputs.fit_split
            else "full"
        )
        rules = self.inputs.views.get(view, {})
        rule = rules.get("events")
        conditions, parameters = [], [root, cutoff * 1_000_000]
        if rule is not None:
            if rule.before_us is not None:
                conditions.append("i.event_us<?")
                parameters.append(rule.before_us)
            if rule.partition_column is not None:
                conditions.append("p.partition IN (" + ",".join("?" for _ in rule.partitions) + ")")
                parameters.extend(rule.partitions)
        window = self.inputs._history_options["observation_seconds"]
        if window is not None:
            conditions.append("i.event_us>=?")
            parameters.append((cutoff - window) * 1_000_000)
        extra = "" if not conditions else "AND " + " AND ".join(conditions)
        parameters.append(self.inputs.max_events)
        history = list(
            self.db.execute(
                f"""
            SELECT i.source_key,i.event_us,r.numbers,r.categories,r.texts
            FROM links l JOIN identities i ON i.table_name='events' AND i.code=l.child
            JOIN reference_rows r ON r.table_name='events' AND r.code=i.code
            JOIN event_partitions p ON p.code=i.code
            WHERE l.role='events' AND l.parent=? AND i.event_us<=? {extra}
            ORDER BY i.event_us DESC,r.event_id DESC LIMIT ?
        """,
                parameters,
            )
        )
        owner_key = self.db.execute(
            "SELECT source_key FROM identities WHERE table_name='users' AND code=?", (root,)
        ).fetchone()[0]
        owner = str(json.loads(owner_key)[0])
        events = tuple(
            TransactionEvent(
                entity_id=owner,
                event_id=str(json.loads(key)[0]),
                event_timestamp=EPOCH + timedelta(microseconds=at),
                timezone="UTC",
                numeric=json.loads(nums),
                categorical=json.loads(cats),
                text=json.loads(texts),
            )
            for key, at, nums, cats, texts in reversed(history)
        )
        return TransactionSequence(
            owner,
            events,
            EPOCH + timedelta(seconds=cutoff),
            observation_window=None,
            sequence_id=sample_id,
        )


class ReferenceSequences(Sequence[TransactionSequence]):
    """An optional baseline may make several bounded passes without retaining histories."""

    def __init__(self, source: ReferenceRows, sample_ids: tuple[str, ...]) -> None:
        self.source, self.sample_ids = source, sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    @overload
    def __getitem__(self, index: int) -> TransactionSequence: ...
    @overload
    def __getitem__(self, index: slice) -> ReferenceSequences: ...
    def __getitem__(self, index: int | slice) -> TransactionSequence | ReferenceSequences:
        if isinstance(index, slice):
            return ReferenceSequences(self.source, self.sample_ids[index])
        return self.source.sequence(self.sample_ids[index])


class ReferenceProfiles(Mapping[str, EntityProfile]):
    """Static profile metadata is read on demand, independently of tensor streams."""

    def __init__(self, rows: ReferenceRows, entities: Sequence[str]) -> None:
        self.rows, self.entities = rows, tuple(dict.fromkeys(entities))
        self._entity_set = frozenset(self.entities)

    def __getitem__(self, entity: str) -> EntityProfile:
        if entity not in self._entity_set:
            raise KeyError(entity)
        return self.rows.profile(entity)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self) -> Iterator[str]:
        return iter(self.entities)
