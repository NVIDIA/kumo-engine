# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchors on the GPU: which (entity, prediction time) pairs a task serves, and in what order.

An anchor table is read once onto the GPU, in Kumo PyTorch's canonical order (files, then
rows, as Kumo PyTorch lists the files). A metadata column of a type Kumo PyTorch's types do not
describe (a struct, a list) is only ever handed back, as upstream's reader hands
it back: it is read on the host instead. Splits, counts and each rank's positions are
worked out on the GPU; nothing becomes a Python object per anchor except
through `GpuAnchors.fetch`, which serves Kumo PyTorch's `AnchorTask` protocol for
previews and checks.

The order is upstream Kumo PyTorch's `shard_positions`, computed for every position at
once: the same seeded Feistel permutation with cycle-walking, evaluated in
uint64 arithmetic on the GPU. Two engines given the same anchors, seed, epoch
and rank then consume the same anchors in the same sequence, as Kumo PyTorch intends.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.anchors import AnchorTable, SplitSelection
    from kumo.pytorch.contract.sample import SampleRequest

__all__ = ["GpuAnchors", "shard_positions"]

_ROUNDS = 10  # upstream Kumo PyTorch's Feistel rounds
_U64 = (1 << 64) - 1


def _mix(value: int) -> int:
    """splitmix64's finalizer on one Python int, exactly as upstream Kumo PyTorch's `anchors._mix`."""

    value = (value + 0x9E3779B97F4A7C15) & _U64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64
    return value ^ (value >> 31)


def _mix_gpu(value: Any) -> Any:
    """splitmix64's finalizer on a CuPy uint64 array; uint64 products wrap as Python's masks do."""

    import cupy

    value = value + cupy.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> cupy.uint64(30))) * cupy.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> cupy.uint64(27))) * cupy.uint64(0x94D049BB133111EB)
    return value ^ (value >> cupy.uint64(31))


def _permute(positions: Any, n: int, seed: int) -> Any:
    """Upstream's `_Permutation(n, seed)` applied to every position in `positions` at once."""

    import cupy

    bits = max(2, (n - 1).bit_length()) if n > 1 else 2
    half = (bits + 1) // 2
    mask, shift = cupy.uint64((1 << half) - 1), cupy.uint64(half)
    keys = [cupy.uint64(_mix(seed ^ _mix(round_index))) for round_index in range(_ROUNDS)]
    value = positions.astype(cupy.uint64)
    result = cupy.zeros_like(value)
    pending = cupy.ones(value.shape, dtype=bool)
    while True:
        left, right = value >> shift, value & mask
        for key in keys:
            left, right = right, left ^ (_mix_gpu(key ^ right) & mask)
        value = (left << shift) | right
        inside = value < cupy.uint64(n)
        result = cupy.where(pending & inside, value, result)
        pending &= ~inside
        if not bool(pending.any()):  # cycle-walk until every image is in [0, n); a few rounds
            return result.astype(cupy.int64)


def shard_positions(n: int, *, seed: int = 0, epoch: int = 0, shuffle: bool = False, rank: int = 0,
                    world_size: int = 1) -> Any:
    """The positions one rank consumes this epoch, as a CuPy int64 array: upstream's `shard_positions`."""

    import cupy

    if n < 0:
        raise ValueError("request count must be non-negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must identify one member of a positive world size")
    positions = cupy.arange(rank, n, world_size, dtype=cupy.int64)
    if not shuffle or positions.size == 0:
        return positions
    return _permute(positions, n, _mix(seed) ^ _mix(epoch))


class GpuAnchors:
    """An anchor table on the GPU, read by split and position (Kumo PyTorch's `AnchorTask` protocol).

    Built from a Parquet `AnchorTable` (`from_table`) or from `SampleRequest`
    values built in code (`from_requests`). Positions within a selection index
    its rows in canonical order, so a position means the same anchor for every
    rank and every epoch.
    """

    def __init__(self, columns: Any, table: AnchorTable | None, requests: Sequence[SampleRequest] | None,
                 split_codes: Any, split_names: Sequence[str | None], cutoffs: Any) -> None:
        self._columns = columns  # a GpuTable of the anchor table's columns, as read; None for requests
        self._arrow_types: dict[str, Any] = {}  # each column's type in the files (libcudf drops time zones)
        self._host: dict[str, Any] = {}  # metadata Kumo PyTorch's types do not describe: Arrow arrays on the host
        self._table = table
        self._requests = None if requests is None else tuple(requests)
        self._split_codes = split_codes  # CuPy int32 per row: an index into split_names
        self._split_names = tuple(split_names)
        #: Each anchor's cutoff, whole UTC seconds rounded down (Kumo PyTorch's `cutoff_to_epoch`), CuPy int64.
        self.cutoffs = cutoffs
        self._selected: dict[Any, Any] = {}
        self._entity: tuple[Any, Any] | None = None  # every anchor's entity ID (`entity_values`)
        self._hashes: Any = None  # each distinct text ID's masking key (`mask_keys`)
        self._kind: str | None = None  # what the entity IDs are (`_id_kind`)
        self._ids: list[Any] | None = None  # every anchor's entity ID as a Python value (`_python_ids`)

    @classmethod
    def from_table(cls, table: AnchorTable) -> GpuAnchors:
        """Read a Parquet anchor table onto the GPU, in Kumo PyTorch's canonical order."""

        from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import (
            decimal_stored,
        )
        from test.kumo_pytorch.reference_engine.gpu.data.sources import _reader
        from test.kumo_pytorch.reference_engine.gpu.data.tables import (
            GpuTable,
            _with_codes,
        )

        import pylibcudf as plc

        from kumo.pytorch.dataset import _local_parquet_files
        from kumo.pytorch.engines.parquet_anchors import (
            _physical,
            check_parquet_anchors,
        )

        issues = check_parquet_anchors(table)
        if issues:
            raise ValueError(issues[0].describe())
        import pyarrow.parquet as pq

        from kumo.pytorch.contract.dtypes import DType
        from kumo.pytorch.dataset import Parquet

        physical = _physical(table)
        files = _local_parquet_files(table.source.path)
        fields = {field.name: field.type for field in pq.read_schema(files[0])}
        host = [name for name in table.metadata if not _described(fields[physical[name]])]
        columns = {logical: DType.from_arrow(fields[name]) for logical, name in physical.items() if logical not in host}
        renames = {name: logical for logical, name in physical.items() if name != logical}
        source = Parquet(table.source.path, columns=renames)
        # Read as stored: a prediction time is not a stored clock, and rounds down, not up.
        options, physical_names = _reader("anchors", source, columns)
        result = plc.io.parquet.read_parquet(options)
        by_physical = dict(zip(result.column_names(), result.tbl.columns(), strict=True))
        loaded = _with_codes(GpuTable("anchors", result.tbl.num_rows(),
                                      {name: decimal_stored(by_physical[physical_names[name]], dtype, f"anchors.{name}")
                                       for name, dtype in columns.items()}, columns))
        codes, names = _splits(loaded, table.split)
        cutoffs = _cutoffs(loaded, table.as_of_time)
        anchors = cls(loaded, table, None, codes, names, cutoffs)
        anchors._arrow_types = {logical: fields[name] for logical, name in physical.items()}
        if host:  # upstream's own read of those columns: the same files, in the same order
            import pyarrow.dataset as pads

            data = pads.dataset(files, format="parquet").to_table(columns=sorted({physical[name] for name in host}))
            anchors._host = {name: data.column(physical[name]).combine_chunks() for name in host}
        return anchors

    @classmethod
    def from_requests(cls, requests: Sequence[SampleRequest]) -> GpuAnchors:
        """Anchors supplied in code, in the order given."""

        import cupy

        from kumo.pytorch.anchors import PADDING_KEY

        requests = tuple(requests)
        if any(PADDING_KEY in request.metadata for request in requests):
            raise ValueError(f"{PADDING_KEY!r} is reserved for the platform's padding marker")
        names = sorted({request.split for request in requests}, key=lambda name: (name is None, name or ""))
        index = {name: code for code, name in enumerate(names)}
        codes = cupy.asarray([index[request.split] for request in requests], dtype=cupy.int32)
        from kumo.pytorch.data.records import cutoff_to_epoch

        cutoffs = cupy.asarray([cutoff_to_epoch(request.as_of_time) for request in requests], dtype=cupy.int64)
        return cls(None, None, requests, codes, names, cutoffs)

    # -- Kumo PyTorch's AnchorTask protocol --------------------------------------------------

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._split_names if name is not None))

    def count(self, selection: SplitSelection) -> int:
        return int(self.rows(selection).size)

    def order(self, selection: SplitSelection, *, seed: int, epoch: int, shuffle: bool, rank: int,
              world_size: int) -> Iterator[int]:
        positions = self.positions(selection, seed=seed, epoch=epoch, shuffle=shuffle, rank=rank,
                                   world_size=world_size)
        return iter(positions.get().tolist())

    def fetch(self, selection: SplitSelection, positions: Sequence[int]) -> Sequence[SampleRequest]:
        """The anchors at `positions` within the selection, as Kumo PyTorch `SampleRequest` values (host copies)."""

        import cupy

        return self.fetch_rows(self.rows(selection)[cupy.asarray(list(positions), dtype=cupy.int64)])

    def fetch_rows(self, rows: Any) -> list[SampleRequest]:
        """The anchors at table `rows` (a CuPy array), as Kumo PyTorch `SampleRequest` values (host copies)."""

        if self._requests is not None:
            return [self._requests[row] for row in rows.get().tolist()]
        return _requests(self._columns, self._table, rows, self._arrow_types, self._host)

    def request_fields(self, rows: Any) -> tuple[list[Any], list[str | None], list[Any]]:
        """Each anchor at table `rows`' (CuPy) sample ID, split and metadata, as `fetch_rows` gives them, and
        nothing else: no prediction time or label becomes a Python object (a batch reads those on the GPU)."""

        if self._requests is not None:
            requests = [self._requests[row] for row in rows.get().tolist()]
            return ([request.sample_id for request in requests], [request.split for request in requests],
                    [request.metadata for request in requests])
        table = self._table
        names = [name for name in dict.fromkeys((table.sample_id, table.split, *table.metadata)) if name]
        columns, picked = _picked(self._columns, rows, self._arrow_types, self._host, names)
        count = len(picked)
        return (columns[table.sample_id] if table.sample_id else picked,
                columns[table.split] if table.split else [None] * count,
                [{name: columns[name][index] for name in table.metadata} for index in range(count)])

    def entity_values(self) -> tuple[Any, Any]:
        """Every anchor's entity ID on the GPU: (CuPy int64 IDs, None) for integer IDs, or (CuPy int32 codes, the
        sorted distinct texts they index) for text IDs. Worked out once; IDs of other types are refused."""

        import cupy

        if self._entity is not None:
            return self._entity
        if self._requests is not None:
            ids = [request.entity_id for request in self._requests]
            if all(type(value) is int for value in ids):  # not a bool, which upstream's integer keys refuse
                # a uint64's top half as its bits; an ID no integer key holds as 0, which names none either
                self._entity = (cupy.asarray([value - 2**64 if 2**63 <= value < 2**64 else
                                              value if -(2**63) <= value < 2**63 else 0 for value in ids],
                                             dtype=cupy.int64), None)
            elif all(isinstance(value, str) for value in ids):
                from test.kumo_pytorch.reference_engine.gpu.data.tables import (
                    GpuTable,
                    _with_codes,
                )

                import pyarrow as pa
                import pylibcudf as plc

                coded = _with_codes(GpuTable("anchors", len(ids), {"id": plc.Column.from_arrow(pa.array(ids, pa.string()))},
                                             {"id": None}))
                self._entity = (coded.values("id"), coded.dictionaries["id"])
            else:
                kinds = " and ".join(sorted({type(value).__name__ for value in ids}))
                raise ValueError(f"anchor entity IDs are {kinds}: test.kumo_pytorch.reference_engine.gpu takes integer or text IDs, all of one type")
            return self._entity
        column = self._table.entity_id
        if self._columns.present(column) is not None:
            raise ValueError(f"anchor column {column!r} holds NULL entity IDs")
        dtype, values = self._columns.dtypes[column], self._columns.values(column)
        if dtype.integer:
            self._entity = (values.astype(cupy.int64), None)
        elif dtype.name == "string":
            self._entity = (values, self._columns.dictionaries[column])
        else:
            raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not build anchors whose entity IDs are {dtype.name} yet")
        return self._entity

    def entity_identities(self, space: Any) -> tuple[Any, Any]:
        """Every anchor's entity ID as an identity of the entity's primary key `space` (`test.kumo_pytorch.reference_engine.gpu.data.keys`), as
        upstream finds the entity: the ID normalized to the key's type (an integer ID of a float key is that
        float), then compared by value. Returns CuPy int64 identities and which IDs can name an entity (CuPy
        bool; None for all): an ID upstream's normalization refuses, or NaN, names none."""

        from test.kumo_pytorch.reference_engine.gpu.data.keys import (
            Translation,
            dictionary_identities,
            key_space,
            python_identity,
        )
        from test.kumo_pytorch.reference_engine.gpu.sampling.calculations import (
            _BOUNDS,
        )

        import cupy

        dtype = space.dtype
        if self._requests is None:
            column = self._table.entity_id
            if self._columns.present(column) is not None:
                raise ValueError(f"anchor column {column!r} holds NULL entity IDs")
            own_dtype = self._columns.dtypes[column]
            # on the GPU, where upstream's normalization keeps the value as it is: the key's own type, or integers
            # of any widths (an integer outside the key's range names no row, and is refused with normalization's
            # verdict, as any ID naming none is; uint64 against a signed type through `Translation`'s range check).
            # Others normalize on the host: a boolean ID is refused for an integer key, a float64 rounded to float32
            if own_dtype == dtype or (own_dtype.integer and dtype.integer):
                ids = self._columns
                if own_dtype.name == "timestamp":  # read as stored: whole seconds, rounded up, as upstream normalizes
                    from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import (
                        _stored,
                    )
                    from test.kumo_pytorch.reference_engine.gpu.data.tables import (
                        GpuTable,
                    )

                    ids = GpuTable("anchors", ids.num_rows, {column: _stored(ids.columns[column], own_dtype,
                                                                             f"anchors.{column}")},
                                   {column: own_dtype})
                own = key_space(ids, column)
                values, found = Translation(own, space).keys()
                matchable = own.matchable_at()
                return values, (found if matchable is None else matchable if found is None else matchable & found)
            ids = self._python_ids()
        else:
            ids = self._python_ids()
            if dtype.integer:  # IDs within the key's range, as themselves; others normalized (refused, or uint64's top)
                low, high = _BOUNDS[dtype.name]
                if all(type(value) is int and low <= value <= min(high, 2**63 - 1) for value in ids):
                    return cupy.asarray(ids, dtype=cupy.int64), None
        normalized, valid = [], []
        for value in ids:
            try:
                normalized.append(dtype.normalize(value))
                valid.append(True)
            except ValueError:
                normalized.append(None)
                valid.append(False)
        if dtype.name in ("string", "bytes") or dtype.name.startswith("decimal"):
            held = dictionary_identities([value for value, ok in zip(normalized, valid, strict=True) if ok], space)
            identities = cupy.full(len(ids), -1, dtype=cupy.int64)
            identities[cupy.asarray(valid, dtype=bool)] = held
            return identities, cupy.asarray(valid, dtype=bool)
        found = [None if not ok else python_identity(value, space) for value, ok in zip(normalized, valid, strict=True)]
        return (cupy.asarray([0 if value is None else value for value in found], dtype=cupy.int64),
                cupy.asarray([value is not None for value in found], dtype=bool))

    def _python_ids(self) -> list[Any]:
        """Every anchor's entity ID as the Python value upstream's anchors give (a request's own; a Parquet anchor
        table's as upstream's reader reads it), worked out once."""

        if self._ids is None:
            if self._requests is not None:
                self._ids = [request.entity_id for request in self._requests]
            else:
                import cupy

                column = self._table.entity_id
                rows = cupy.arange(self._columns.num_rows, dtype=cupy.int64)
                self._ids = _picked(self._columns, rows, self._arrow_types, self._host, [column])[0][column]
        return self._ids

    def _id_kind(self) -> str:
        """"integer" when every entity ID is an int64 integer, "text" when every one is text, else "other"
        (uint64 IDs of 2**63 or more among them too). Worked out once."""

        if self._kind is None:
            if self._requests is not None:
                ids = [request.entity_id for request in self._requests]
                if all(type(value) is int and -(2**63) <= value < 2**63 for value in ids):
                    self._kind = "integer"
                else:
                    self._kind = "text" if all(isinstance(value, str) for value in ids) else "other"
            else:
                column = self._table.entity_id
                dtype = self._columns.dtypes[column]
                if dtype.name == "uint64":
                    self._kind = "other" if bool((self._columns.values(column).view("int64") < 0).any()) \
                        else "integer"
                else:
                    self._kind = "integer" if dtype.integer else "text" if dtype.name == "string" else "other"
        return self._kind

    def mask_keys(self, rows: Any) -> Any:
        """The entity of each anchor at table `rows` as upstream's masked-value draws read it, CuPy int64: an integer
        ID itself, a text ID the first 4 bytes of its UTF-8 SHA-256, big-endian (upstream's `assembly`), each
        distinct text hashed once."""

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _view,
        )

        import pylibcudf as plc

        if self._id_kind() == "other":  # upstream's rule, ID by ID: an integer or boolean itself, else its str's hash
            import cupy

            # for these rows only, as upstream works them out for a batch's: a uint64 ID of 2**63 or more elsewhere
            # (another split's) fails nothing here, and fails here as upstream's int64 tensor of them does
            ids = self._python_ids()
            return cupy.asarray([_mask_key(ids[row]) for row in rows.get().tolist()], dtype=cupy.int64)
        values, texts = self.entity_values()
        if texts is None:
            return values[rows]
        if self._hashes is None:
            width = plc.DataType(plc.TypeId.INT32)
            digests = plc.hashing.sha256(plc.Table([texts]))  # 64 hex digits each
            head = plc.strings.slice.slice_strings(digests, plc.Scalar.from_py(0, width), plc.Scalar.from_py(8, width))
            self._hashes = _view(plc.strings.convert.convert_integers.hex_to_integers(head,
                                                                                      plc.DataType(plc.TypeId.INT64)))
        return self._hashes[values[rows]]

    def entity_metadata(self, rows: Any) -> Any:
        """``metadata["entity_ids"]`` of the anchors at table `rows`, as upstream gives it: a tensor on the GPU for
        integer IDs, a tuple of Python strings for text IDs."""

        from test.kumo_pytorch.reference_engine.gpu.data.text import texts_at

        import torch

        if self._id_kind() == "other":  # a tuple of the Python IDs; each batch's decided by `batch_entity_ids`
            ids = self._python_ids()
            return tuple(ids[row] for row in rows.get().tolist())
        values, texts = self.entity_values()
        if texts is None:
            return torch.from_dlpack(values[rows])
        return tuple(texts_at(texts, values[rows]))

    def close(self) -> None:
        """Release the anchor table's GPU memory. Idempotent."""

        self._columns = None
        self._selected.clear()

    # -- on the GPU ---------------------------------------------------------------

    def rows(self, selection: SplitSelection) -> Any:
        """The selection's rows in canonical order, as a CuPy int64 array."""

        import cupy

        from kumo.pytorch.anchors import AllAnchors

        if isinstance(selection, AllAnchors):
            key: Any = "all"
        else:
            key = frozenset(selection)
            unknown = sorted(name for name in key if name is not None and name not in self._split_names)
            if unknown:
                present = list(self.splits())
                raise ValueError(
                    f"no anchors in split(s) {unknown}; splits present: {present}"
                    if present
                    else f"no anchors in split(s) {unknown}; these anchors declare no splits"
                )
        cached = self._selected.get(key)
        if cached is not None:
            return cached
        if key == "all":
            rows = cupy.arange(self._split_codes.size, dtype=cupy.int64)
        else:
            wanted = cupy.asarray([code for code, name in enumerate(self._split_names) if name in key],
                                  dtype=cupy.int32)
            rows = cupy.flatnonzero(cupy.isin(self._split_codes, wanted)).astype(cupy.int64)
        self._selected[key] = rows
        return rows

    def positions(self, selection: SplitSelection, *, seed: int, epoch: int, shuffle: bool, rank: int,
                  world_size: int) -> Any:
        """The positions within the selection one rank consumes this epoch, as a CuPy int64 array."""

        return shard_positions(self.count(selection), seed=seed, epoch=epoch, shuffle=shuffle, rank=rank,
                               world_size=world_size)


def _splits(loaded: Any, column: str | None) -> tuple[Any, list[str | None]]:
    """Each row's split as an index into the split names (None for a row without one)."""

    import cupy

    if column is None:
        return cupy.zeros(loaded.num_rows, dtype=cupy.int32), [None]
    names: list[str | None] = loaded.dictionaries[column].to_arrow().to_pylist()
    codes = loaded.values(column).astype(cupy.int32)
    present = loaded.present(column)
    if present is not None:
        codes = cupy.where(present, codes, cupy.int32(len(names)))
        names.append(None)
    return codes, names


def _cutoffs(loaded: Any, column: str | None) -> Any:
    """Each anchor's cutoff in whole UTC seconds, rounded toward negative infinity (Kumo PyTorch's `cutoff_to_epoch`).

    A timestamp is floored to its second, a date is its midnight, an integer is
    already seconds. Without a time column every cutoff is 0, as Kumo PyTorch's readers use.
    """

    import cupy
    import pylibcudf as plc

    if column is None:
        return cupy.zeros(loaded.num_rows, dtype=cupy.int64)
    times = loaded.columns[column]
    kind = times.type().id()
    if kind == plc.TypeId.TIMESTAMP_DAYS:
        seconds = loaded.values(column).astype(cupy.int64) * 86_400
    elif kind in (plc.TypeId.TIMESTAMP_MILLISECONDS, plc.TypeId.TIMESTAMP_MICROSECONDS,
                  plc.TypeId.TIMESTAMP_NANOSECONDS):
        floored = plc.datetime.floor_datetimes(times, plc.datetime.RoundingFrequency.SECOND)
        floored = plc.unary.cast(floored, plc.DataType(plc.TypeId.TIMESTAMP_SECONDS))
        seconds = type(loaded)("t", floored.size(), {"t": floored}, {"t": loaded.dtypes[column]}).values("t")
    else:  # whole seconds already: timestamps in seconds and integers
        seconds = loaded.values(column).astype(cupy.int64)
    if loaded.present(column) is not None:
        raise ValueError(f"anchor column {column!r} holds NULL prediction times")
    return seconds


def _described(arrow_type: Any) -> bool:
    """Whether Kumo PyTorch's types describe a column of `arrow_type` (`DType.from_arrow` resolves it)."""

    from kumo.pytorch.contract.dtypes import DType

    try:
        DType.from_arrow(arrow_type)
    except ValueError:
        return False
    return True


def batch_entity_ids(ids: Any, part: Any) -> Any:
    """A batch's ``metadata["entity_ids"]`` from `entity_metadata`'s for several (`part` of them), as upstream
    decides each batch's: a tensor when every ID is an int64 integer (a tuple of IDs of other kinds, some text,
    some uint64 of 2**63 or more, may hold such a batch), else a tuple."""

    if not isinstance(ids, tuple):
        return ids[part]
    picked = ids[part]
    if picked and all(type(value) is int and -(2**63) <= value < 2**63 for value in picked):
        import cupy
        import torch

        return torch.from_dlpack(cupy.asarray(picked, dtype=cupy.int64))
    return picked


def _mask_key(value: Any) -> int:
    """Upstream's masking key of one entity ID (`assembly`): an integer or boolean itself, anything else the first
    4 bytes of its ``str``'s UTF-8 SHA-256, big-endian."""

    import hashlib

    if isinstance(value, (int, bool)):
        return int(value)
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4], "big")


def _picked(loaded: Any, rows: Any, types: dict[str, Any], host: dict[str, Any],
            names: Any = None) -> tuple[dict[str, list[Any]], list[int]]:
    """Columns `names` (all, by default) of a loaded anchor table at table `rows` (CuPy), as Python values as
    upstream's Parquet reader gives them, and the rows as a list; `host` holds the metadata columns read on the
    host. The rows are picked on the GPU, so only theirs are copied to the host."""

    from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import (
        decimal_arrow,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _column,
    )

    import cupy
    import pylibcudf as plc

    names = list(loaded.columns) + list(host) if names is None else list(names)
    if not rows.size:
        return {name: [] for name in names}, []
    on_gpu = [name for name in names if name not in host]
    columns = {}
    if on_gpu:
        gathered = plc.copying.gather(plc.Table([loaded.columns[name] for name in on_gpu]),
                                      _column(rows.astype(cupy.int32)), plc.copying.OutOfBoundsPolicy.DONT_CHECK)
        for name, column in zip(on_gpu, gathered.columns(), strict=True):  # as stored, not as Kumo PyTorch's seconds
            if name in loaded.dictionaries:  # each picked row's text; a NULL row's code is undefined: masked, and
                codes = loaded.values(name)[rows]  # one out of range gives NULL rather than a read past the end
                texts = plc.copying.gather(plc.Table([loaded.dictionaries[name]]), _column(codes.astype(cupy.int32)),
                                           plc.copying.OutOfBoundsPolicy.NULLIFY).columns()[0]
                column = texts.with_mask(column.null_mask(), column.null_count()) if column.null_count() else texts
            dtype = loaded.dtypes[name]
            host_column = decimal_arrow(column, dtype) if dtype.name.startswith("decimal") else column.to_arrow()
            columns[name] = host_column.cast(types[name]).to_pylist()
    picked = rows.get().tolist()
    if len(on_gpu) < len(names):
        import pyarrow as pa

        at = pa.array(picked, pa.int64())
        for name in names:
            if name in host:
                columns[name] = host[name].take(at).to_pylist()
    return columns, picked


def _requests(loaded: Any, table: AnchorTable, rows: Any, types: dict[str, Any],
              host: dict[str, Any]) -> list[SampleRequest]:
    """`SampleRequest` values for table `rows` (CuPy) of a loaded anchor table, as upstream's Parquet reader
    builds them (`_picked`)."""

    from kumo.pytorch.contract.sample import SampleRequest, TargetValue

    columns, rows = _picked(loaded, rows, types, host)
    targets = sorted(table.targets.items())
    return [
        SampleRequest(
            sample_id=columns[table.sample_id][index] if table.sample_id else row,
            entity_id=columns[table.entity_id][index],
            as_of_time=columns[table.as_of_time][index] if table.as_of_time else 0,
            targets=tuple(TargetValue(name, columns[column][index]) for name, column in targets),
            split=columns[table.split][index] if table.split else None,
            metadata={name: columns[name][index] for name in table.metadata},
        )
        for index, row in enumerate(rows)
    ]
