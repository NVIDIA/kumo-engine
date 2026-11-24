# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-wise reads of a pragmatiq `events.parquet`, in blocks of whole users.

`load_pragmatiq_dataset` builds one `TransactionEvent` per row, which is exact
but needs about 6 KB of memory per event. These reads keep Arrow and NumPy
columns instead and reproduce what the object path would build:

- events in `order_events` order (timestamp, then `event_id`);
- the `event_id` index and the microsecond value its string carries;
- each field's view: numeric when it parses as a finite float (its string is
  then Python's `str(float)`), text when longer than 64 characters,
  categorical otherwise, and `source` as one more categorical field.

The book must be sorted by `user_id`, as pragmatiq writes it, so a block of
rows holds whole users and blocks follow entity order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError
from tfm.data.adapters.pragmatiq import parse_float

FIELD_CATEGORICAL, FIELD_NUMERIC, FIELD_TEXT = 0, 1, 2
"""Field views, in the order `event_fields` lists them."""
SOURCE_MAP_POSITION = 127
"""`source` is added after an event's own categorical fields."""


@dataclass(frozen=True)
class BlockPlan:
    """Row ranges of whole users, in entity order."""

    starts: tuple[int, ...]
    stops: tuple[int, ...]
    n_users: int
    n_rows: int

    def __len__(self) -> int:
        return len(self.starts)


@dataclass(frozen=True)
class EventBlock:
    """One block of users, events in `order_events` order.

    Pair arrays list every field of every event (map fields and `source`),
    in no particular order; `pair_event` says whose they are.
    """

    user_ids: tuple[str, ...]
    user_offsets: NDArray[np.int64]
    ts_us: NDArray[np.int64]
    """True microseconds since the epoch."""
    id_us: NDArray[np.int64]
    """The microsecond value in `event_id`, `int(ts.timestamp() * 1_000_000)`."""
    loader_index: NDArray[np.int64]
    """The per-user index in `event_id`."""
    pair_event: NDArray[np.int64]
    pair_key: NDArray[np.int64]
    """Index into `keys`."""
    pair_value: NDArray[np.int64]
    """Index into `values`."""
    pair_kind: NDArray[np.int8]
    pair_map_position: NDArray[np.int64]
    keys: tuple[str, ...]
    values: tuple[str, ...]
    """Each field's string as `event_fields` would give it."""
    numbers: NDArray[np.float64]
    """`values` parsed as floats, NaN where a value is not numeric."""

    @property
    def n_events(self) -> int:
        return len(self.ts_us)


def book_fingerprint(root: Path) -> dict[str, Any]:
    """Enough to tell a cache it was built from another book."""
    events = pq.ParquetFile(root / "events.parquet")
    manifest = root / "manifest.json"
    out: dict[str, Any] = {
        "events_rows": events.metadata.num_rows,
        "events_bytes": (root / "events.parquet").stat().st_size,
        "events_row_groups": events.num_row_groups,
    }
    if manifest.exists():
        out["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in ("profiles.parquet",):
        path = root / name
        if path.exists():
            out[f"{name.split('.')[0]}_bytes"] = path.stat().st_size
    return out


def plan_blocks(
    root: Path, *, rows_per_block: int, users: Collection[str] | None = None
) -> BlockPlan:
    """Row ranges of about `rows_per_block` rows that never split a user.

    With `users`, a range may span rows of other users; `read_block` drops them.
    """
    if rows_per_block < 1:
        raise ValueError(f"rows_per_block must be positive, got {rows_per_block}.")
    ids = pq.read_table(root / "events.parquet", columns=["user_id"]).column("user_id")
    runs = pc.run_end_encode(ids.combine_chunks().cast(pa.string()))  # type: ignore[attr-defined]
    run_users = runs.values
    run_ends = runs.run_ends.to_numpy().astype(np.int64)
    if (
        len(run_users) > 1
        and not pc.all(  # type: ignore[attr-defined]
            pc.less(run_users[:-1], run_users[1:])  # type: ignore[attr-defined]
        ).as_py()
    ):
        raise SchemaError(
            f"{root / 'events.parquet'} is not sorted by user_id; the column-wise reader needs "
            "each user's rows together and users in order, as pragmatiq writes them."
        )
    run_starts = np.concatenate([[0], run_ends[:-1]])
    if users is not None:
        keep = pc.is_in(run_users, value_set=pa.array(sorted(users), pa.string()))  # type: ignore[attr-defined]
        keep_np = keep.to_numpy(zero_copy_only=False)
        missing = len(set(users)) - int(keep_np.sum())
        if missing:
            raise SchemaError(f"{missing} requested users have no events in the book.")
        run_starts, run_ends = run_starts[keep_np], run_ends[keep_np]
    starts: list[int] = []
    stops: list[int] = []
    block_start = block_stop = None
    for start, stop in zip(run_starts.tolist(), run_ends.tolist(), strict=True):
        if block_start is not None and stop - block_start > rows_per_block:
            starts.append(block_start)
            stops.append(int(block_stop))  # type: ignore[arg-type]
            block_start = None
        if block_start is None:
            block_start = start
        block_stop = stop
    if block_start is not None:
        starts.append(block_start)
        stops.append(int(block_stop))  # type: ignore[arg-type]
    return BlockPlan(
        starts=tuple(starts),
        stops=tuple(stops),
        n_users=len(run_starts),
        n_rows=int((run_ends - run_starts).sum()),
    )


def _read_rows(path: Path, start: int, stop: int) -> pa.Table:
    """Rows `[start, stop)`; a block may still be several non-adjacent user runs."""
    file = pq.ParquetFile(path)
    bounds = np.cumsum(
        [0] + [file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)]
    )
    first = int(np.searchsorted(bounds, start, side="right") - 1)
    last = int(np.searchsorted(bounds, stop, side="left"))
    table = file.read_row_groups(list(range(first, last)))
    return table.slice(start - int(bounds[first]), stop - start)


@lru_cache(maxsize=4)
def _string_order_rank(n: int) -> NDArray[np.int64]:
    """`rank[i]` orders `i` as `str(i)` sorts, which is how `event_id` ties break."""
    order = sorted(range(n), key=str)
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    return rank


def _classify(raw: str) -> tuple[int, str, float]:
    parsed = parse_float(raw)
    if parsed is not None:
        return FIELD_NUMERIC, str(parsed), parsed
    if len(raw) > 64:
        return FIELD_TEXT, raw, float("nan")
    return FIELD_CATEGORICAL, raw, float("nan")


def read_block(
    root: Path, start: int, stop: int, *, users: Collection[str] | None = None
) -> EventBlock:
    """Rows `[start, stop)` from `plan_blocks`, as column-wise events and fields."""
    table = _read_rows(root / "events.parquet", start, stop)
    if users is not None:
        keep = pc.is_in(table.column("user_id"), value_set=pa.array(sorted(users), pa.string()))  # type: ignore[attr-defined]
        table = table.filter(keep)
    n = table.num_rows
    user_col = table.column("user_id").combine_chunks().cast(pa.string())
    runs = pc.run_end_encode(user_col)  # type: ignore[attr-defined]
    user_ids = tuple(str(u) for u in runs.values.to_pylist())
    user_ends = runs.run_ends.to_numpy().astype(np.int64)
    user_offsets = np.concatenate([[0], user_ends]).astype(np.int64)
    user_code = np.repeat(np.arange(len(user_ids), dtype=np.int64), np.diff(user_offsets))
    ts_type = table.schema.field("ts").type
    if not pa.types.is_timestamp(ts_type) or ts_type.unit != "us":
        raise SchemaError(f"events.parquet `ts` must be timestamp[us], got {ts_type}.")
    ts_us = table.column("ts").combine_chunks().cast(pa.int64()).to_numpy().astype(np.int64)

    # The loader sorts rows by (user, ts), stably; its per-user rank is the event_id index.
    loader_order = np.lexsort((ts_us, user_code))
    loader_index = np.empty(n, np.int64)
    loader_index[loader_order] = np.arange(n) - user_offsets[user_code[loader_order]]
    # order_events sorts by (ts, event_id); equal timestamps compare the index as a string.
    most = int(np.diff(user_offsets).max()) if len(user_ids) else 0
    tie_rank = _string_order_rank(max(most, 1))[loader_index]
    final = np.lexsort((tie_rank, ts_us, user_code))
    final_of_row = np.empty(n, np.int64)
    final_of_row[final] = np.arange(n)

    fields = table.column("fields").combine_chunks()
    if not isinstance(fields, pa.MapArray):
        raise SchemaError("events.parquet `fields` must be a map<string, string> column.")
    offsets = fields.offsets.to_numpy().astype(np.int64)
    lengths = np.diff(offsets)
    if len(lengths) and int(lengths.max()) >= SOURCE_MAP_POSITION:
        raise SchemaError(
            f"An event has {int(lengths.max())} fields; the column-wise reader takes fewer "
            f"than {SOURCE_MAP_POSITION}."
        )
    row_of_pair = np.repeat(np.arange(n, dtype=np.int64), lengths)
    map_position = (
        offsets[0] + np.arange(len(row_of_pair), dtype=np.int64) - np.repeat(offsets[:-1], lengths)
    )
    keys_arr = fields.keys.slice(offsets[0], offsets[-1] - offsets[0])
    items_arr = fields.items.slice(offsets[0], offsets[-1] - offsets[0])
    present = items_arr.is_valid().to_numpy(zero_copy_only=False)
    key_dict = keys_arr.dictionary_encode()
    key_names = [str(k) for k in key_dict.dictionary.to_pylist()]
    if "source" in key_names:
        raise SchemaError(
            "An event's fields map has a `source` key; it would collide with the event source."
        )
    pair_key = key_dict.indices.to_numpy(zero_copy_only=False).astype(np.int64)[present]
    row_of_pair, map_position = row_of_pair[present], map_position[present]
    if len(pair_key):
        check = np.unique(row_of_pair * (len(key_names) + 1) + pair_key)
        if len(check) != len(pair_key):
            raise SchemaError("An event's fields map repeats a key.")
    value_dict = items_arr.filter(pa.array(present)).dictionary_encode()
    raw_values = value_dict.dictionary.to_pylist()
    classified = [_classify(str(v)) for v in raw_values]
    raw_code = value_dict.indices.to_numpy(zero_copy_only=False).astype(np.int64)

    # Canonical strings can coincide (`10.1` and `10.10`), so re-index on them.
    canonical = [c[1] for c in classified]
    uniq_values, value_of_raw = np.unique(np.asarray(canonical, dtype=object), return_inverse=True)
    kinds_raw = np.asarray([c[0] for c in classified], np.int8)
    numbers_raw = np.asarray([c[2] for c in classified], np.float64)
    pair_value = value_of_raw.astype(np.int64)[raw_code]
    pair_kind = kinds_raw[raw_code]
    values = [str(v) for v in uniq_values.tolist()]
    numbers = np.full(len(values), np.nan)
    numbers[pair_value[pair_kind == FIELD_NUMERIC]] = numbers_raw[raw_code][
        pair_kind == FIELD_NUMERIC
    ]

    # `source` is one more categorical field of every event.
    source_dict = table.column("source").combine_chunks().cast(pa.string()).dictionary_encode()
    source_names = [str(s) for s in source_dict.dictionary.to_pylist()]
    value_index = {v: i for i, v in enumerate(values)}
    for name in source_names:
        if name not in value_index:
            value_index[name] = len(values)
            values.append(name)
            numbers = np.append(numbers, np.nan)
    source_value = np.asarray([value_index[s] for s in source_names], np.int64)[
        source_dict.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    ]
    key_names.append("source")
    source_key = len(key_names) - 1

    return EventBlock(
        user_ids=user_ids,
        user_offsets=user_offsets,
        ts_us=ts_us[final],
        id_us=legacy_microseconds(ts_us[final]),
        loader_index=loader_index[final],
        pair_event=np.concatenate([final_of_row[row_of_pair], final_of_row]),
        pair_key=np.concatenate([pair_key, np.full(n, source_key, np.int64)]),
        pair_value=np.concatenate([pair_value, source_value]),
        pair_kind=np.concatenate([pair_kind, np.full(n, FIELD_CATEGORICAL, np.int8)]),
        pair_map_position=np.concatenate([map_position, np.full(n, SOURCE_MAP_POSITION, np.int64)]),
        keys=tuple(key_names),
        values=tuple(values),
        numbers=numbers,
    )


def legacy_microseconds(ts_us: NDArray[np.int64]) -> NDArray[np.int64]:
    """`int(ts.timestamp() * 1_000_000)` for true microseconds `ts_us`.

    The object path goes through float seconds, so its value can sit one below
    the true microsecond; event ids and `EncodedKVTRecord.event_ts_us` carry it.
    """
    return ((ts_us.astype(np.float64) / 1e6) * 1e6).astype(np.int64)


__all__ = [
    "FIELD_CATEGORICAL",
    "FIELD_NUMERIC",
    "FIELD_TEXT",
    "BlockPlan",
    "EventBlock",
    "book_fingerprint",
    "legacy_microseconds",
    "plan_blocks",
    "read_block",
]
