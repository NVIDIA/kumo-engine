"""Which partitions own each row: joins along ownership chains, keys matched by value."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.dataset import Parquet

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable
from test.kumo_pytorch.reference_engine.gpu.split.hashing import partition_of_keys
from test.kumo_pytorch.reference_engine.gpu.split.partition_plan import PartitionPlan, Step


#: Integer key types, which the split compares across widths after a lossless cast (other pairs of number types,
#: by their exact forms: `_Owners._join`).
_SIGNED = frozenset({"int8", "int16", "int32", "int64"})

_UNSIGNED = frozenset({"uint8", "uint16", "uint32", "uint64"})

def _joinable(left: DType, right: DType) -> bool:
    """Whether keys of these types are matched as libcudf reads them: one type, or integers one type holds both
    of (cast to it, `_same_type`)."""

    if left == right:
        return True
    names = {left.name, right.name}
    return names <= _SIGNED | _UNSIGNED and not ("uint64" in names and names & _SIGNED)

def _same_type(*columns: Any) -> list[Any]:
    """`columns` cast to one type that holds every value of each: int64 for integers, uint64 if all are unsigned.

    G-ROWS-001 matches rows by equal key values, so an int32 5 matches an int64
    5. Columns of one type are returned unchanged. Callers pass only pairs
    `_joinable` accepts (others are joined by their exact forms, `_exact_join`);
    any other mix raises, as a guard.
    """

    import pylibcudf as plc

    types = {column.type().id() for column in columns}
    if len(types) == 1:
        return list(columns)
    TypeId = plc.TypeId
    signed = {TypeId.INT8, TypeId.INT16, TypeId.INT32, TypeId.INT64}
    unsigned = {TypeId.UINT8, TypeId.UINT16, TypeId.UINT32, TypeId.UINT64}
    if not types <= signed | unsigned or (TypeId.UINT64 in types and types & signed):
        raise ValueError(f"keys of types {sorted(t.name for t in types)} cannot be matched by value")
    common = plc.DataType(TypeId.UINT64 if types <= unsigned else TypeId.INT64)
    return [column if column.type().id() == common.id() else plc.unary.cast(column, common) for column in columns]

def _union(values: list[Any], parts: list[Any]) -> tuple[Any, Any]:
    """Distinct (value, partition) pairs of several pieces, without null values."""

    import pylibcudf as plc

    table = plc.Table([plc.concatenate.concatenate(_same_type(*values)), plc.concatenate.concatenate(parts)])
    table = plc.stream_compaction.drop_nulls(table, [0], 1)
    return _distinct_pairs(*table.columns())

class _Owners:
    """Which partition(s) each row belongs to, by joins up its chains on the GPU.

    For each table a chain passes through, a map from its primary key to the
    partition its chain reaches is built once (memoized per chain) and kept on
    the GPU: these maps must fit on one GPU.
    """

    def __init__(self, plan: PartitionPlan, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]],
                 num_partitions: int) -> None:
        self.plan, self.sources, self.k = plan, sources, num_partitions
        self._maps: dict[tuple[str, tuple[Step, ...]], tuple[Any, Any]] = {}
        self._forms: dict[tuple[str, tuple[Step, ...], DType], _Formed] = {}  # key maps in a child type's form

    def partition_of_keys(self, keys: Any, dtype: Any = None) -> Any:
        """The root partition of each root primary-key value (of `dtype`)."""

        return partition_of_keys(keys, self.k, dtype)

    def key_map(self, table: str, chain: tuple[Step, ...]) -> tuple[Any, Any]:
        """(primary-key values, partition) of `table`'s rows, through `chain` up to the root."""

        cached = self._maps.get((table, chain))
        if cached is not None:
            return cached
        from test.kumo_pytorch.reference_engine.gpu.data.tables import _load_parquet

        source, columns = self.sources[table]
        key = self.plan.primary_keys[table]
        if not chain:
            loaded = _load_parquet(table, source, {key: columns[key]})
            result = (loaded.columns[key], self.partition_of_keys(loaded.columns[key], columns[key]))
        else:
            step = chain[0]
            loaded = _load_parquet(table, source, {key: columns[key], step.key: columns[step.key]})
            rows, parts = self._join(loaded.columns[step.key], *self.key_map(step.parent, chain[1:]),
                                     columns[step.key], self._key_type(step.parent))
            result = _distinct_pairs(_gather(loaded.columns[key], rows), parts)
        self._maps[(table, chain)] = result
        return result

    def row_partitions(self, table: str, chunk: GpuTable) -> tuple[Any, Any]:
        """(row index, partition) pairs for a chunk of `table`, one per distinct pair."""

        import pyarrow as pa
        import pylibcudf as plc

        if table == self.plan.root:
            key = self.plan.primary_keys[table]
            rows = plc.filling.sequence(chunk.num_rows, plc.Scalar.from_arrow(pa.scalar(0, pa.int32())),
                                        plc.Scalar.from_arrow(pa.scalar(1, pa.int32())))
            return rows, self.partition_of_keys(chunk.columns[key], chunk.dtypes[key])
        pieces = [self._join(chunk.columns[chain[0].key], *self._parent(chain, chunk.dtypes[chain[0].key]),
                             chunk.dtypes[chain[0].key], self._key_type(chain[0].parent))
                  for chain in self.plan.owned[table]]
        rows = plc.concatenate.concatenate([piece[0] for piece in pieces])
        parts = plc.concatenate.concatenate([piece[1] for piece in pieces])
        return _distinct_pairs(rows, parts)

    def _key_type(self, table: str) -> DType:
        return self.sources[table][1][self.plan.primary_keys[table]]

    def _parent(self, chain: tuple[Step, ...], child_type: DType) -> tuple[Any, Any]:
        """`chain`'s first parent's key map, as a child key of `child_type` meets it: as it is, or (another number
        type) in the child type's exact form, made once for every chunk of the child (`_Formed`, parts None)."""

        from test.kumo_pytorch.reference_engine.gpu.data.keys import never_equal

        keys, parts = self.key_map(chain[0].parent, chain[1:])
        parent_type = self._key_type(chain[0].parent)
        if _joinable(child_type, parent_type) or never_equal(child_type, parent_type):
            return keys, parts
        key = (chain[0].parent, chain[1:], child_type)
        if key not in self._forms:
            self._forms[key] = _formed(keys, parts, parent_type, child_type)
        return self._forms[key], None

    @staticmethod
    def _join(child_keys: Any, parent_keys: Any, parent_parts: Any, child_type: DType | None = None,
              parent_type: DType | None = None) -> tuple[Any, Any]:
        """Rows of the child whose key matches a parent key, and that parent's partition, as upstream's Python
        values are equal (`test.kumo_pytorch.reference_engine.gpu.data.keys`). Nulls never match, nor NaNs (upstream's NaN equals nothing; libcudf's
        join would match them, and copy every NaN row to each partition a NaN reaches); -0.0 matches 0.0. Keys of
        types never equal (an integer and a text) match nothing; numbers of other types (an integer and a float,
        decimals of two scales) are matched by their exact forms (`_exact_join`)."""

        import pylibcudf as plc

        if isinstance(parent_keys, _Formed):  # already in the child's form
            return _formed_join(child_keys, child_type, parent_keys)
        if child_type is not None and parent_type is not None and not _joinable(child_type, parent_type):
            return _exact_join(child_keys, parent_keys, parent_parts, child_type, parent_type)
        child_keys, parent_keys = _same_type(child_keys, parent_keys)
        left, right = plc.join.inner_join(plc.Table([_nans_null(child_keys)]), plc.Table([_nans_null(parent_keys)]),
                                          plc.types.NullEquality.UNEQUAL)
        return left, _gather(parent_parts, right)

def _exact_join(child_keys: Any, parent_keys: Any, parent_parts: Any, child_type: DType,
                parent_type: DType) -> tuple[Any, Any]:
    """`_Owners._join` for keys libcudf does not compare as Python does: none for types never equal; numbers of
    different types each in the child's exact form (`test.kumo_pytorch.reference_engine.gpu.data.keys.exact_form`), keys with none (NaN, 1.5 against
    integers, 0.1 against decimals) left out."""

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column
    from test.kumo_pytorch.reference_engine.gpu.data.keys import never_equal

    if never_equal(child_type, parent_type):
        empty = _column(cupy.zeros(0, dtype=cupy.int32))
        return empty, _gather(parent_parts, empty)
    return _formed_join(child_keys, child_type, _formed(parent_keys, parent_parts, parent_type, child_type))

@dataclass(frozen=True)
class _Formed:
    """Keys in a column type's exact form (`_form`: those with none left out), and each one's partition: the
    other side of a join on a column of `target`, made once for every chunk it meets."""

    table: Any
    parts: Any
    target: DType

    def columns(self) -> list[Any]:
        return [*self.table.columns(), self.parts]

def _formed(keys: Any, parts: Any, dtype: DType, target: DType) -> _Formed:
    """Keys of `dtype` with their partitions, in `target`'s exact form (`_Formed`)."""

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    form, rows = _form(keys, dtype, target)
    return _Formed(form, _gather(parts, _column(cupy.ascontiguousarray(rows.astype(cupy.int32)))), target)

def _formed_join(child_keys: Any, child_type: DType, parent: _Formed) -> tuple[Any, Any]:
    """`_Owners._join` against keys already in the child's form: the child's keys in their own form (those with
    none, NaN, left out), joined on it."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view

    assert parent.target == child_type, (parent.target, child_type)
    child_form, child_rows = _form(child_keys, child_type, child_type)
    left, right = plc.join.inner_join(child_form, parent.table, plc.types.NullEquality.UNEQUAL)
    rows = _column(cupy.ascontiguousarray(child_rows[_view(left)].astype(cupy.int32)))
    return rows, _gather(parent.parts, right)

def _form(keys: Any, dtype: DType, target: DType) -> tuple[Any, Any]:
    """Key column `keys` (of number type `dtype`) in `target`'s exact form (a libcudf table: one int64 column, or a
    decimal's four words), the rows with one only, and which rows those are (CuPy int64)."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column
    from test.kumo_pytorch.reference_engine.gpu.data.keys import exact_form
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

    held = GpuTable("keys", keys.size(), {"k": keys}, {"k": dtype})
    if dtype.name.startswith("decimal"):
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import column_words

        values = column_words(keys)
    else:
        values = held.values("k")
    form, valid = exact_form(values, dtype, target)
    present = held.present("k")
    rows = cupy.flatnonzero(valid if present is None else valid & present)
    form = form[rows]
    columns = [form[:, k] for k in range(form.shape[1])] if form.ndim == 2 else [form]
    return plc.Table([_column(cupy.ascontiguousarray(column)) for column in columns]), rows

def _nans_null(column: Any) -> Any:
    """A float column with its NaNs NULL (they match nothing); others as they are."""

    import pylibcudf as plc

    if column.type().id() not in (plc.TypeId.FLOAT32, plc.TypeId.FLOAT64):
        return column
    mask, count = plc.transform.nans_to_nulls(column)
    return column.with_mask(mask, count) if count else column

def _gather(column: Any, rows: Any) -> Any:
    import pylibcudf as plc

    return plc.copying.gather(plc.Table([column]), rows, plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]

def _distinct_pairs(first: Any, second: Any) -> tuple[Any, Any]:
    import pylibcudf as plc

    pairs = plc.stream_compaction.distinct(
        plc.Table([first, second]), [0, 1], plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
        plc.types.NullEquality.EQUAL, plc.types.NanEquality.ALL_EQUAL,
    )
    return pairs.columns()[0], pairs.columns()[1]

def _places(name: str, chunk: GpuTable, owners: _Owners,
            selectors: Sequence[tuple[str, Any, Any, DType]]) -> tuple[Any, Any, Any, Any]:
    """A chunk's (row, partition) pairs: those its owners place, then the others its routes reach (None without routes)."""

    import pylibcudf as plc

    rows, parts = owners.row_partitions(name, chunk)
    if not selectors:
        return rows, parts, None, None
    pieces = [_Owners._join(chunk.columns[column], keys, key_parts, chunk.dtypes[column], dtype)
              for column, keys, key_parts, dtype in selectors]
    halo_rows, halo_parts = _distinct_pairs(plc.concatenate.concatenate([piece[0] for piece in pieces]),
                                            plc.concatenate.concatenate([piece[1] for piece in pieces]))
    others = plc.join.left_anti_join(plc.Table([halo_rows, halo_parts]), plc.Table([rows, parts]),
                                     plc.types.NullEquality.EQUAL)
    return rows, parts, _gather(halo_rows, others), _gather(halo_parts, others)
