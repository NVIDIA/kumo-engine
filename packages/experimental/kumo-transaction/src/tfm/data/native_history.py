# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native history retrieval with the existing tokenizers and label contracts.

Object adapters still import a bounded corpus. Only keys and clocks enter the
native index; labels remain in the validated source records. Exact residual
filtering preserves microsecond and inclusive-window semantics even when an
index uses a coarser clock. This is not a full-book columnar ingestion path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, overload

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tfm.contracts.graph import utc_us
from tfm.contracts.schema import TransactionEvent, TransactionSequence, ensure_utc
from tfm.data.collate import collate_batch
from tfm.data.sequence_builders import entity_sequences as reference
from tfm.data.validation.schema import validate_events
from tfm.extras.kumo_pytorch import native_api, temporary_engine

Loader = Literal["kumo_pytorch", "python"]


class NativeHistories:
    """One owned DiskGraph index and materialized anchor table, closed explicitly."""

    def __init__(self, events: Sequence[TransactionEvent]) -> None:
        self.api = native_api()
        self.events = tuple(
            replace(e)
            for e in sorted(
                validate_events(events), key=lambda e: (e.event_timestamp, e.entity_id, e.event_id)
            )
        )
        self.owners = {key: i for i, key in enumerate(sorted({e.entity_id for e in self.events}))}
        self.stack = ExitStack()
        try:
            self.root = Path(
                self.stack.enter_context(TemporaryDirectory(prefix="tfm-native-history-"))
            )
            api = self.api
            clock = api.DType("timestamp", unit="us", timezone="UTC")
            schema = api.DataSchema(
                (
                    api.TableSpec(
                        "users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL
                    ),
                    api.TableSpec(
                        "events",
                        "id",
                        {"id": api.DType("int64"), "owner": api.DType("int64"), "at": clock},
                        time_column="at",
                    ),
                ),
                (api.RelationSpec("activity", "users", "id", "events", "owner"),),
            )
            pq.write_table(
                pa.table({"id": pa.array(list(self.owners.values()), type=pa.int64())}),
                self.root / "users.parquet",
            )
            pq.write_table(
                pa.table(
                    {
                        "id": pa.array(range(len(self.events)), type=pa.int64()),
                        "owner": pa.array(
                            [self.owners[e.entity_id] for e in self.events], type=pa.int64()
                        ),
                        "at": pa.array(
                            [e.event_timestamp for e in self.events],
                            type=pa.timestamp("us", tz="UTC"),
                        ),
                    }
                ),
                self.root / "events.parquet",
            )
            # Retrieve candidates through the cutoff without a lossy event cap.
            # Inclusive lower bounds and tied/subsecond limits are applied below.
            contract = api.DataLoaderSpec(
                schema,
                api.SampleSpec(
                    api.EntitySpec("users"),
                    events={
                        "history": api.EventSpec(
                            api.Rows(("activity",), api.TimeWindow("at", stop=0)),
                            {"row": api.Field(column="id")},
                        )
                    },
                ),
                api.BatchSpec(
                    {
                        "history": api.Input(
                            api.EventSource(
                                ("history",),
                                max_events=max(
                                    Counter(e.entity_id for e in self.events).values(), default=1
                                ),
                                keep="last",
                                order="ascending",
                            ),
                            api.Fields((api.Encode("row", api.Raw()),), row=api.Columns()),
                            api.Padded(),
                        )
                    }
                ),
            )
            dataset = api.Dataset(
                {
                    name: api.Parquet(str(self.root / f"{name}.parquet"))
                    for name in ("users", "events")
                }
            )
            self.plan = self.stack.enter_context(
                api.prepare(
                    contract,
                    dataset,
                    temporary_engine(self.stack),
                )
            )
        except BaseException:
            self.stack.close()
            raise

    def __enter__(self) -> NativeHistories:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release native tasks before deleting their source/index directories."""
        self.stack.close()

    def bind(self, requests: Sequence[tuple[str, datetime]], splits: Sequence[str]) -> Any:
        """Bind exact caller identities and cutoffs through an AnchorTable, not requests."""
        self.requests = tuple((owner, ensure_utc(at)) for owner, at in requests)
        if len(splits) != len(requests) or not requests:
            raise ValueError("Native anchors need a nonempty, aligned split column")
        path = self.root / "anchors.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_id": [str(i) for i in range(len(requests))],
                    "owner": pa.array(
                        [self.owners[owner] for owner, _ in requests], type=pa.int64()
                    ),
                    "at": pa.array(
                        [at for _, at in self.requests], type=pa.timestamp("us", tz="UTC")
                    ),
                    "cutoff_us": pa.array([utc_us(at) for _, at in self.requests], type=pa.int64()),
                    "split": list(splits),
                }
            ),
            path,
        )
        anchors = self.api.AnchorTable(
            self.api.Parquet(str(path)),
            entity_id="owner",
            as_of_time="at",
            sample_id="sample_id",
            split="split",
            metadata=("cutoff_us",),
        )
        return self.stack.enter_context(self.plan.materialize(anchors, fit_split=splits[0]))

    def decode(self, batch: Any) -> dict[int, tuple[TransactionEvent, ...]]:
        """Check native bindings and map raw row IDs back to immutable source snapshots."""
        values = batch.inputs["history"]["row"]
        out = {}
        units = {"seconds": 1_000_000, "milliseconds": 1000, "microseconds": 1}
        clock_unit = batch.metadata.get("timestamp_unit", "seconds")
        for pos, sid in enumerate(batch.metadata["sample_ids"]):
            i = int(sid)
            if str(i) != str(sid) or i in out or not 0 <= i < len(self.requests):
                raise ValueError("Unknown or duplicate native sample ID")
            owner, cutoff = self.requests[i]
            if int(batch.metadata["entity_ids"][pos]) != self.owners[owner]:
                raise ValueError("Native entity/sample alignment changed")
            raw = batch.metadata["request_metadata"][pos]["cutoff_us"]
            if int(raw) != utc_us(cutoff):
                raise ValueError("Native cutoff binding changed")
            clock = int(batch.metadata["as_of_time"][pos])
            expected = (
                utc_us(cutoff) * 1000
                if clock_unit == "nanoseconds"
                else utc_us(cutoff) // units[clock_unit]
            )
            if clock != expected:
                raise ValueError("Native sampling cutoff differs from the bound anchor")
            ids = values["values"][pos][values["valid"][pos].bool()].tolist()
            if len(set(ids)) != len(ids) or any(not 0 <= x < len(self.events) for x in ids):
                raise ValueError("Invalid or duplicate native event ID")
            events = tuple(self.events[x] for x in ids)
            if any(e.entity_id != owner for e in events):
                raise ValueError("Native history crosses entity boundaries")
            # Native clocks can be coarser than the source. No candidate beyond
            # the exact cutoff is passed to a tokenizer or model.
            out[i] = tuple(e for e in events if e.event_timestamp <= cutoff)
        return out


def _histories(
    events: Sequence[TransactionEvent], requests: Sequence[tuple[str, datetime]]
) -> list[tuple[TransactionEvent, ...]]:
    if not events or not requests:
        return [() for _ in requests]
    result: list[tuple[TransactionEvent, ...]] = [() for _ in requests]
    with NativeHistories(events) as source:
        present = [i for i, (owner, _) in enumerate(requests) if owner in source.owners]
        if not present:
            return result
        task = source.bind([requests[i] for i in present], ["rows"] * len(present))
        seen = set()
        for batch in task.batches(
            split="rows", batch_size=128, shuffle=False, output_format="model"
        ):
            for i, rows in source.decode(batch).items():
                if i in seen:
                    raise ValueError("Native loader repeated an anchor")
                seen.add(i)
                result[present[i]] = rows
        if seen != set(range(len(present))):
            raise ValueError("Native loader dropped an anchor")
    return result


def build_entity_sequences(
    events: Sequence[TransactionEvent],
    cutoff: datetime,
    *,
    max_events: int,
    observation_window: timedelta | None = None,
    causal: bool = True,
    loader: Loader = "kumo_pytorch",
) -> list[TransactionSequence]:
    """Retrieve histories natively by default; preserve the reference selection contract."""
    if loader not in {"python", "kumo_pytorch"}:
        raise ValueError(f"Unknown history loader {loader!r}")
    if loader == "python":
        return reference.build_entity_sequences(
            events,
            cutoff,
            max_events=max_events,
            observation_window=observation_window,
            causal=causal,
        )
    events = validate_events(events)
    owners = sorted({e.entity_id for e in events})
    rows = _histories(events, [(owner, cutoff) for owner in owners])
    if not any(rows):
        return []
    return reference.build_entity_sequences(
        [e for row in rows for e in row],
        cutoff,
        max_events=max_events,
        observation_window=observation_window,
        causal=causal,
    )


def build_entity_windows(
    events: Sequence[TransactionEvent],
    cutoff: datetime,
    *,
    window_events: int,
    observation_window: timedelta | None = None,
    causal: bool = True,
    loader: Loader = "kumo_pytorch",
) -> list[TransactionSequence]:
    """Tile the same native histories using the existing pretraining-window policy."""
    if loader not in {"python", "kumo_pytorch"}:
        raise ValueError(f"Unknown history loader {loader!r}")
    if loader == "python":
        return reference.build_entity_windows(
            events,
            cutoff,
            window_events=window_events,
            observation_window=observation_window,
            causal=causal,
        )
    events = validate_events(events)
    owners = sorted({e.entity_id for e in events})
    rows = _histories(events, [(owner, cutoff) for owner in owners])
    if window_events <= 0:
        raise ValueError("window_events must be positive")
    if not any(rows):
        return []
    return reference.build_entity_windows(
        [e for row in rows for e in row],
        cutoff,
        window_events=window_events,
        observation_window=observation_window,
        causal=causal,
    )


def build_sequences_at_label_cutoffs(
    events: Sequence[TransactionEvent],
    rows: Sequence[Any],
    *,
    max_events: int,
    observation_window: timedelta | None = None,
    causal: bool = True,
    loader: Loader = "kumo_pytorch",
) -> list[tuple[TransactionSequence, Any]]:
    """Retain label-row order while retrieving each history at its own prediction time."""
    if loader not in {"python", "kumo_pytorch"}:
        raise ValueError(f"Unknown history loader {loader!r}")
    if loader == "python":
        return reference.build_sequences_at_label_cutoffs(
            events,
            rows,
            max_events=max_events,
            observation_window=observation_window,
            causal=causal,
        )
    histories = _histories(events, [(r.entity_id, r.timestamp) for r in rows])
    out = []
    for row, history in zip(rows, histories, strict=True):
        if not history:
            continue
        selected = reference.build_entity_sequences(
            history,
            row.timestamp,
            max_events=max_events,
            observation_window=observation_window,
            causal=causal,
        )
        if selected:
            out.append((selected[0], row))
    return out


class NativeSequenceBatches(Sequence[Any]):
    """Lazy native retrieval per indexed minibatch; existing eager/DDP ordering is retained.

    The tokenizer is fitted before construction. Explicit window membership is
    retained, so tiled windows never silently become suffix windows. Native
    tensors are consumed before returning an owned, validated TFM batch.
    """

    def __init__(
        self,
        rows: Sequence[TransactionSequence],
        tokenizer: Any,
        *,
        batch_size: int,
        max_tokens: int,
        seed: int | None,
        amount_key: str,
        label_window: Any = None,
    ) -> None:
        if not rows:
            raise ValueError("Native training loader needs nonempty rows")
        order = (
            list(range(len(rows)))
            if seed is None
            else torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
        )
        self.rows = tuple(rows[i] for i in order)
        self.tokenizer, self.batch_size, self.max_tokens = tokenizer, batch_size, max_tokens
        self.amount_key, self.label_window = amount_key, label_window
        by_key = {}
        for row in self.rows:
            for e in row.events:
                key = (e.entity_id, e.event_id)
                if key in by_key and by_key[key] != e:
                    raise ValueError("Conflicting native source records")
                by_key[key] = e
        self.source = NativeHistories(list(by_key.values()))
        try:
            self.task = self.source.bind(
                [(r.entity_id, r.cutoff_time) for r in self.rows],
                [str(i // batch_size) for i in range(len(self.rows))],
            )
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Any]: ...

    @torch.inference_mode(False)
    @torch.no_grad()
    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        decoded = {}
        for native in self.task.batches(
            split=str(index), batch_size=self.batch_size, shuffle=False, output_format="model"
        ):
            for sid, history in self.source.decode(native).items():
                if sid in decoded:
                    raise ValueError("Native loader repeated a training sample")
                decoded[sid] = {(e.entity_id, e.event_id): e for e in history}
        expected = range(
            index * self.batch_size, min((index + 1) * self.batch_size, len(self.rows))
        )
        if set(decoded) != set(expected):
            raise ValueError("Native training batch membership changed")
        selected = []
        for sid in expected:
            original = self.rows[sid]
            if any((e.entity_id, e.event_id) not in decoded[sid] for e in original.events):
                raise ValueError("Native retrieval lost a required window event")
            selected.append(
                replace(
                    original,
                    events=tuple(decoded[sid][e.entity_id, e.event_id] for e in original.events),
                )
            )
        return collate_batch(
            selected,
            self.tokenizer,
            max_tokens=self.max_tokens,
            label_window=self.label_window,
            amount_key=self.amount_key,
        )

    def close(self) -> None:
        """Close the owned native task/index once training ends."""
        self.source.close()
