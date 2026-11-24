# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-wise reads of a pragmatiq `events.parquet`, in blocks of whole users.

`events.parquet` is one file, or a directory of parquet files read in string
order of their paths, as `pyarrow.dataset` lists them: `part-10` before `part-9`.

`load_pragmatiq_dataset` builds one `TransactionEvent` per row, which is exact
but needs about 6 KB of memory per event. These reads keep Arrow and NumPy
columns instead and reproduce what the object path would build:

- events in `order_events` order (timestamp, then `event_id`);
- the `event_id` index and the microsecond value its string carries;
- each field's view: numeric when it parses as a finite float (its string is
  then Python's `str(float)`), text when longer than 64 characters,
  categorical otherwise, and `source` as one more categorical field.

The book must be sorted by `user_id`, as pragmatiq writes it, so a block of
rows holds whole users and blocks follow entity order. It should also be
sorted by `ts` within each user; a block that is not is sorted in memory,
with a warning.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.tokenizers import (
    FIELD_CATEGORICAL,
    FIELD_NUMERIC,
    FIELD_TEXT,
    EventBlock,
)
from tfm.data.adapters.pragmatiq import parse_float

SOURCE_MAP_POSITION = 127
"""`source` is added after an event's own categorical fields."""


@dataclass(frozen=True)
class RowGroup:
    """One row group of an event file, as the planner saw it."""

    path: str
    size: int
    """The file's size in bytes when the blocks were planned."""
    mtime_ns: int
    """The file's modification time when the blocks were planned."""
    index: int
    first_row: int
    """Its first row's number in the whole book."""
    n_rows: int


@dataclass(frozen=True)
class PlannedBlock:
    """Book rows `[start, stop)` of whole users, and the row groups that hold them."""

    index: int
    start: int
    stop: int
    groups: tuple[RowGroup, ...]

    @property
    def n_rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class BlockPlan:
    """Blocks of whole users, in entity order."""

    blocks: tuple[PlannedBlock, ...]
    n_users: int
    n_rows: int
    longest_user_id: int
    """The longest planned user id, in UTF-8 bytes."""

    def __len__(self) -> int:
        return len(self.blocks)


def _event_files(root: Path) -> list[str]:
    """The book's event files, in read order."""
    path = root / "events.parquet"
    if not path.is_dir():
        return [str(path)]
    files = list(ds.dataset(str(path), format="parquet").files)
    if not files:
        raise SchemaError(f"{path} holds no parquet files.")
    return files


def _row_group_index(root: Path) -> tuple[list[RowGroup], dict[str, int]]:
    """Every row group of the book's event files in read order, and each file's size.

    Files must share a schema.
    """
    groups: list[RowGroup] = []
    sizes: dict[str, int] = {}
    first_row = 0
    schema = None
    files = _event_files(root)
    for path in files:
        stat = Path(path).stat()
        sizes[path] = stat.st_size
        meta = pq.ParquetFile(path).metadata
        if schema is None:
            schema = meta.schema
        elif not meta.schema.equals(schema):
            raise SchemaError(
                f"{path} has another schema than {files[0]}; write every event file with the "
                "same schema."
            )
        for g in range(meta.num_row_groups):
            n_rows = meta.row_group(g).num_rows
            groups.append(RowGroup(path, stat.st_size, stat.st_mtime_ns, g, first_row, n_rows))
            first_row += n_rows
    return groups, sizes


def book_fingerprint(root: Path) -> dict[str, Any]:
    """Enough to tell a cache it was built from another book: file names, sizes, row
    counts and the manifest's hash. Not the events' content: an edit that keeps every
    file's size and row count keeps the fingerprint; rebuild the cache after one."""
    groups, sizes = _row_group_index(root)
    manifest = root / "manifest.json"
    out: dict[str, Any] = {
        "events_rows": sum(g.n_rows for g in groups),
        "events_bytes": sum(sizes.values()),
        "events_row_groups": len(groups),
    }
    if (root / "events.parquet").is_dir():
        out["events_files"] = [[Path(p).name, size] for p, size in sizes.items()]
    if manifest.exists():
        out["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in ("profiles.parquet",):
        path = root / name
        if path.exists():
            out[f"{name.split('.')[0]}_bytes"] = path.stat().st_size
    return out


def _user_runs(
    groups: list[RowGroup],
) -> Iterator[tuple[pa.Array, NDArray[np.int64], NDArray[np.int64]]]:
    """Runs of equal `user_id` in read order, `(users, starts, stops)` per row group.

    Reads one row group of `user_id` at a time; a run that continues into the next
    row group or file is yielded once, whole.
    """
    offset = 0
    held: tuple[str, int, int] | None = None
    opened: tuple[str, pq.ParquetFile] | None = None
    for group in groups:
        path, g = group.path, group.index
        if opened is None or opened[0] != path:
            opened = (path, pq.ParquetFile(path))
        ids = opened[1].read_row_group(g, columns=["user_id"]).column("user_id")
        names = pc.run_end_encode(ids.combine_chunks().cast(pa.string()))  # type: ignore[attr-defined]
        if not len(names.values):
            continue
        users = names.values
        stops = offset + names.run_ends.to_numpy().astype(np.int64)
        starts = np.concatenate([[offset], stops[:-1]])
        offset = int(stops[-1])
        first = users[0].as_py()
        if (
            len(users) > 1 and not pc.all(pc.less(users[:-1], users[1:])).as_py()  # type: ignore[attr-defined]
        ) or (held is not None and first < held[0]):
            raise SchemaError(
                f"{path} row group {g} is not sorted by user_id. The column-wise reader "
                "needs each user's rows together and users ascending across row groups "
                "and files, which are read in string order of their paths (zero-pad "
                "numbered names). In Spark, write the book with "
                '`repartitionByRange("user_id").sortWithinPartitions("user_id", "ts")`.'
            )
        if held is not None and first == held[0]:
            starts[0] = held[1]
        elif held is not None:
            yield pa.array([held[0]], pa.string()), np.array([held[1]]), np.array([held[2]])
        held = (users[-1].as_py(), int(starts[-1]), int(stops[-1]))
        if len(users) > 1:
            yield users[:-1], starts[:-1], stops[:-1]
    if held is not None:
        yield pa.array([held[0]], pa.string()), np.array([held[1]]), np.array([held[2]])


def plan_blocks(
    root: Path, *, rows_per_block: int, users: Collection[str] | None = None
) -> BlockPlan:
    """Row ranges of at most `rows_per_block` rows that never split a user.

    Memory grows with the number of row groups and blocks, not rows. With `users`, a range may span
    rows of other users; `read_block` drops them.
    """
    if rows_per_block < 1:
        raise ValueError(f"rows_per_block must be positive, got {rows_per_block}.")
    wanted = None if users is None else pa.array(sorted(set(users)), pa.string())
    starts: list[int] = []
    stops: list[int] = []
    block_start: int | None = None
    block_stop = 0
    n_users = n_rows = matched = longest = 0
    groups, _ = _row_group_index(root)
    for names, run_starts, run_stops in _user_runs(groups):
        if wanted is not None:
            keep = pc.is_in(names, value_set=wanted)  # type: ignore[attr-defined]
            names = names.filter(keep)
            keep_np = keep.to_numpy(zero_copy_only=False)
            run_starts, run_stops = run_starts[keep_np], run_stops[keep_np]
            matched += len(run_starts)
        sizes = run_stops - run_starts
        big = np.flatnonzero(sizes > rows_per_block)
        if len(big):
            user, n = names[int(big[0])].as_py(), int(sizes[big[0]])
            raise UnsupportedCombinationError(
                f"User {user!r} has {n} events, more than rows_per_block={rows_per_block}, "
                f"and a block holds whole users. Raise rows_per_block to at least {n}."
            )
        n_users += len(run_starts)
        n_rows += int(sizes.sum())
        if len(names):
            longest = max(longest, pc.max(pc.binary_length(names)).as_py())  # type: ignore[attr-defined]
        i = 0
        while i < len(run_starts):
            if block_start is None:
                block_start = int(run_starts[i])
            fit = int(np.searchsorted(run_stops[i:], block_start + rows_per_block, side="right"))
            if not fit:
                starts.append(block_start)
                stops.append(block_stop)
                block_start = None
                continue
            block_stop = int(run_stops[i + fit - 1])
            i += fit
    if block_start is not None:
        starts.append(block_start)
        stops.append(block_stop)
    if wanted is not None and matched < len(wanted):
        raise SchemaError(f"{len(wanted) - matched} requested users have no events in the book.")
    first_rows = np.asarray([g.first_row for g in groups], np.int64)
    blocks = []
    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        a = int(np.searchsorted(first_rows, start, side="right")) - 1
        b = int(np.searchsorted(first_rows, stop, side="left"))
        blocks.append(PlannedBlock(i, start, stop, tuple(groups[a:b])))
    return BlockPlan(blocks=tuple(blocks), n_users=n_users, n_rows=n_rows, longest_user_id=longest)


def _changed(path: str, what: str) -> SchemaError:
    return SchemaError(
        f"{path} {what} than when the blocks were planned. An event file changed during the "
        "build; rebuild once the book is final."
    )


def _read_rows(block: PlannedBlock) -> pa.Table:
    """The block's rows; refuses a file that changed since the blocks were planned."""
    files: dict[str, pq.ParquetFile] = {}
    tables = []
    for group in block.groups:
        if group.path not in files:
            stat = Path(group.path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (group.size, group.mtime_ns):
                raise _changed(group.path, "has another size or modification time")
            files[group.path] = pq.ParquetFile(
                group.path, read_dictionary=list(_DICTIONARY_COLUMNS)
            )
        meta = files[group.path].metadata
        if (
            group.index >= meta.num_row_groups
            or meta.row_group(group.index).num_rows != group.n_rows
        ):
            raise _changed(group.path, f"has another row count in row group {group.index}")
        # Arrow cannot read a dictionary-encoded map across row groups in one call.
        tables.append(files[group.path].read_row_group(group.index))
    first = block.groups[0].first_row
    return pa.concat_tables(tables).slice(block.start - first, block.stop - block.start)


_DICTIONARY_COLUMNS = ("source", "fields.key_value.key", "fields.key_value.value")


def _distinct(strings: pa.Array) -> tuple[list[str], NDArray[np.int64]]:
    """The dictionary entries a non-null dictionary array uses, and each element's index."""
    entry = strings.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    used = np.flatnonzero(np.bincount(entry, minlength=len(strings.dictionary)))
    lookup = np.zeros(len(strings.dictionary), np.int64)
    lookup[used] = np.arange(len(used))
    return [str(s) for s in strings.dictionary.take(pa.array(used)).to_pylist()], lookup[entry]


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


def read_block(block: PlannedBlock, *, users: Collection[str] | None = None) -> EventBlock:
    """A block from `plan_blocks`, as column-wise events and fields."""
    table = _read_rows(block)
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
    position = np.arange(n, dtype=np.int64) - user_offsets[user_code]
    same_user = user_code[1:] == user_code[:-1]
    step = np.diff(ts_us)
    if (step[same_user] < 0).any():
        warnings.warn(
            f"Book rows {block.start}..{block.stop} ({block.groups[0].path}) are not sorted by "
            "(user_id, ts); "
            "sorting them in memory. Sort the book by (user_id, ts) to skip this.",
            stacklevel=2,
        )
        loader_index = np.empty(n, np.int64)
        loader_index[np.lexsort((ts_us, user_code))] = position
        rows = np.arange(n, dtype=np.int64)
    else:
        loader_index = position
        tie = np.zeros(n, bool)
        tied = np.flatnonzero(same_user & (step == 0))
        tie[tied] = tie[tied + 1] = True
        rows = np.flatnonzero(tie)
    # order_events sorts by (ts, event_id); equal timestamps compare the index as a string.
    most = int(np.diff(user_offsets).max()) if len(user_ids) else 0
    tie_rank = _string_order_rank(max(most, 1))[loader_index]
    final = np.arange(n, dtype=np.int64)
    final[rows] = rows[np.lexsort((tie_rank[rows], ts_us[rows], user_code[rows]))]
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
    key_names, pair_key = _distinct(keys_arr)
    if len(set(key_names)) != len(key_names):
        raise UnsupportedCombinationError(
            f"pyarrow {pa.__version__} left duplicate entries in a combined dictionary; "
            "the column-wise reader needs unified dictionaries. Install pyarrow 25.0.1, "
            "the version the SDK is tested with."
        )
    if "source" in key_names:
        raise SchemaError(
            "An event's fields map has a `source` key; it would collide with the event source."
        )
    pair_key = pair_key[present]
    row_of_pair, map_position = row_of_pair[present], map_position[present]
    code = np.sort(row_of_pair * (len(key_names) + 1) + pair_key, kind="stable")
    if (code[1:] == code[:-1]).any():
        raise SchemaError("An event's fields map repeats a key.")
    raw_values, raw_code = _distinct(items_arr.filter(pa.array(present)))
    classified = [_classify(v) for v in raw_values]

    # `source` is one more categorical field of every event.
    source_names, source_code = _distinct(table.column("source").combine_chunks())
    # Canonical strings can coincide (`10.1` and `10.10`), so re-index on them.
    canonical = [c[1] for c in classified]
    uniq_values, inverse = np.unique(
        np.asarray(canonical + source_names, dtype=object), return_inverse=True
    )
    value_of_raw, value_of_source = inverse[: len(canonical)], inverse[len(canonical) :]
    kinds_raw = np.asarray([c[0] for c in classified], np.int8)
    numbers_raw = np.asarray([c[2] for c in classified], np.float64)
    pair_value = value_of_raw.astype(np.int64)[raw_code]
    pair_kind = kinds_raw[raw_code]
    values = [str(v) for v in uniq_values.tolist()]
    numbers = np.full(len(values), np.nan)
    numbers[pair_value[pair_kind == FIELD_NUMERIC]] = numbers_raw[raw_code][
        pair_kind == FIELD_NUMERIC
    ]
    source_value = value_of_source.astype(np.int64)[source_code]
    key_names.append("source")
    source_key = len(key_names) - 1

    return EventBlock(
        index=block.index,
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


def copy_profiles(path: Path, users: Collection[str] | None, out: Path) -> None:
    """`path`'s rows for `users` (all when None), in row groups of 4,096 rows.

    Reads one record batch at a time and refuses rows not sorted by `user_id`, so each
    fit block reads only the few row groups that hold its users.
    """
    wanted = None if users is None else pa.array(sorted(users), pa.string())
    file = pq.ParquetFile(path)
    last: str | None = None
    with pq.ParquetWriter(out, file.schema_arrow) as writer:
        for batch in file.iter_batches(batch_size=4096):
            ids = batch.column("user_id")
            if len(ids) and (
                (last is not None and ids[0].as_py() < last)
                or not pc.all(pc.less_equal(ids[:-1], ids[1:])).as_py()  # type: ignore[attr-defined]
            ):
                raise SchemaError(
                    f"{path} is not sorted by user_id. Sort it, for example in Spark with "
                    '`df.sort("user_id")`, so each build block reads only its own users\' '
                    "profiles."
                )
            last = ids[-1].as_py() if len(ids) else last
            if wanted is not None:
                batch = batch.filter(pc.is_in(ids, value_set=wanted))  # type: ignore[attr-defined]
            if batch.num_rows:
                writer.write_batch(batch)


@dataclass(frozen=True)
class PragmatiqColumns:
    """The `ColumnarSource` of the pragmatiq book at `root`."""

    root: Path

    def __str__(self) -> str:
        return str(self.root)

    def fingerprint(self) -> dict[str, Any]:
        return {"root": str(self.root), **book_fingerprint(self.root)}

    def plan(self, *, rows_per_block: int, users: Collection[str] | None = None) -> BlockPlan:
        return plan_blocks(self.root, rows_per_block=rows_per_block, users=users)

    def read_block(
        self, block: PlannedBlock, *, users: Collection[str] | None = None
    ) -> EventBlock:
        return read_block(block, users=users)

    def write_profiles(self, users: Collection[str] | None, out: Path) -> bool:
        path = self.root / "profiles.parquet"
        if not path.exists():
            return False
        copy_profiles(path, users, out)
        return True


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
    "PlannedBlock",
    "RowGroup",
    "EventBlock",
    "PragmatiqColumns",
    "book_fingerprint",
    "copy_profiles",
    "legacy_microseconds",
    "plan_blocks",
    "read_block",
]
