"""Load a dataset's tables onto the GPU: one typed GPU column per declared column (`GpuTable`).

Load a dataset's tables onto the GPU: one typed GPU column per declared column.
The behavior follows Kumo PyTorch's G-TYPE-001 (``conformance/spec/guarantees/G-TYPE-001``
in the pinned Kumo PyTorch checkout) and Kumo PyTorch's own binding rules:

- The only way data enters is direct Parquet ingestion onto the GPU. A table
  bound to any other source is refused by name.
- Every column has one resolved Kumo PyTorch ``DType``. Omitted types come from file
  metadata through Kumo PyTorch's own ``Dataset.resolve_schema``. A declared type that
  differs from a file's type is refused from the files' metadata, before any row
  is read.
- NULL stays NULL (the column's validity mask); 0, ``False``, ``""`` and NaN are
  values.
- Integer width and signedness, and decimal digits, are kept exactly.
- A timestamp is a whole-second UTC instant; a fractional one is rounded toward
  positive infinity: -1 microsecond is second 0, +1 microsecond is second 1.
- The files are the ones Kumo PyTorch lists for the source (a file, every ``*.parquet``
  under a folder, or a glob), with its column renames applied.

A ``bytes`` column is read as text is (libcudf holds both as strings, byte for
byte) and handed back as bytes. libcudf has no 256-bit decimal: a ``decimal256``
column of precision up to 38 is read as libcudf's own decimals (up to 128 bits),
as ``decimal128`` is; a wider one is read as Parquet stores it, big-endian two's
complement bytes, and held as 32 bytes a value (`_wide_decimal`).

Stored forms, which later steps compute on (from the old GPU backend's measured
choices):

- A text column loaded by `load_tables` is held as int32 codes into its sorted
  distinct texts (`GpuTable.dictionaries`): a code's order is its text's byte
  order, NULL stays NULL, and ``""`` is a text like any other. Counting,
  grouping and matching then work on 4-byte integers, and each distinct text is
  handled once. `read_chunks` keeps text as text: separate pieces could not
  share one dictionary.
- A column (or a piece) without NULLs carries no validity mask: libcudf's
  reader leaves it out, which a test checks.
- `GpuTable.values` and `GpuTable.present` are CuPy views of a column's values
  and validity, without copying them.

libcudf reads each table straight onto the GPU, only the declared columns, in
one read. This route is built for the GPU and is to be tuned for speed on it,
measured on an idle GPU. Row order is the order read and carries no meaning:
no behavior may depend on it (G-GRAPH-001, for one, says so).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    import pyarrow as pa
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Dataset, Parquet
from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import _decimal_numbers, _stored, decimal_arrow
from test.kumo_pytorch.reference_engine.gpu.data.sources import _reader, resolve_sources

__all__ = ['GpuTable', 'load_tables', 'read_chunks']


@dataclass(frozen=True)
class GpuTable:
    """One table on the GPU.

    :param name: the table's name in the schema.
    :param num_rows: its number of rows.
    :param columns: logical column name to its ``pylibcudf.Column``.
    :param dtypes: logical column name to its resolved Kumo PyTorch ``DType``. It keeps
        what a GPU column's type does not, such as a decimal's precision.
    """

    name: str
    num_rows: int
    columns: Mapping[str, Any]
    dtypes: Mapping[str, DType]
    #: Text column name to its sorted distinct texts, which the column's int32 codes index.
    dictionaries: Mapping[str, Any] = field(default_factory=dict)
    _present: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_arrow(self) -> pa.Table:
        """Copy the table to the host, typed exactly as declared, text decoded. For checks only."""

        import pyarrow as pa

        def host(name: str, column: Any) -> pa.Array:
            if name in self.dictionaries:
                return self.dictionaries[name].to_arrow().take(column.to_arrow())
            if self.dtypes[name].name.startswith("decimal"):
                return decimal_arrow(column, self.dtypes[name])
            return column.to_arrow()

        return pa.table({name: host(name, column).cast(self.dtypes[name].to_arrow())
                         for name, column in self.columns.items()})

    def values(self, name: str) -> Any:
        """A CuPy view of column `name`'s values (a text column's codes), without a copy.

        A NULL row's value is undefined: read it only where `present` is true.
        Decimal columns have no CuPy type and are refused; so is text that is not
        held as codes (a piece from `read_chunks`).
        """

        import cupy
        import pylibcudf as plc

        column = self.columns[name]
        kind = _cupy_type(column.type().id())
        if kind is None:
            raise ValueError(f"{self.name}.{name} ({self.dtypes[name].name}, held as "
                             f"{column.type().id().name}) has no CuPy view")
        dtype = cupy.dtype(kind)
        if column.size() == 0:
            return cupy.empty(0, dtype)
        data = column.data()
        pointer = data.__cuda_array_interface__["data"][0] + column.offset() * dtype.itemsize
        memory = cupy.cuda.UnownedMemory(pointer, column.size() * dtype.itemsize, column)
        view = cupy.ndarray((column.size(),), dtype, cupy.cuda.MemoryPointer(memory, 0))
        return view if column.type().id() != plc.TypeId.BOOL8 else view.view(cupy.bool_)

    def numbers(self, name: str) -> Any:
        """Decimal column `name` as float64 numbers (CuPy), each rounded as upstream rounds a decimal (its
        ``arrow_float64``: as ``float(Decimal)``, correctly rounded), worked out once and kept. On the GPU,
        unscaled times or divided by a power of ten is exact, so correctly rounded, while the unscaled value
        fits in 53 bits and the scale in 22 digits; any other value is converted on the host by upstream's
        own function. A NULL row's number is undefined: read it only where `present` is true."""

        cache = self.__dict__.setdefault("_numbers", {})
        if name not in cache:
            cache[name] = _decimal_numbers(self.columns[name], self.present(name), self.dtypes[name])
        return cache[name]

    def present(self, name: str) -> Any:
        """A CuPy bool array, true where column `name` is not NULL; None when it has no NULLs."""

        import pylibcudf as plc

        column = self.columns[name]
        if column.null_count() == 0:
            return None
        if name not in self._present:
            mask = column.null_mask().__cuda_array_interface__["data"][0]
            bools = plc.transform.mask_to_bools(mask, column.offset(), column.offset() + column.size())
            self._present[name] = GpuTable(self.name, column.size(), {name: bools},
                                           {name: self.dtypes[name]}).values(name)
        return self._present[name]

def _cupy_type(type_id: Any) -> str | None:
    """The CuPy type of a libcudf type's values, or None when it has none (text, decimals)."""

    import pylibcudf as plc

    kinds = plc.TypeId
    return {
        kinds.INT8: "int8", kinds.INT16: "int16", kinds.INT32: "int32", kinds.INT64: "int64",
        kinds.UINT8: "uint8", kinds.UINT16: "uint16", kinds.UINT32: "uint32", kinds.UINT64: "uint64",
        kinds.FLOAT32: "float32", kinds.FLOAT64: "float64", kinds.BOOL8: "uint8",
        kinds.TIMESTAMP_SECONDS: "int64", kinds.TIMESTAMP_DAYS: "int32",
    }.get(type_id)

def load_tables(schema: DataSchema, dataset: Dataset, *,
                only: Sequence[str] | None = None, split: bool = False) -> dict[str, GpuTable]:
    """Load every table `schema` declares from `dataset` onto the GPU.

    :param only: load just these tables. The others need no source in `dataset`
        when their columns are all declared.
    :param split: the files are a split's own (`test.kumo_pytorch.reference_engine.gpu.split`), whose columns may be stored in another
        type than declared (`split_form`).
    """

    return {
        name: _with_codes(_load_parquet(name, source, columns, split=split))
        for name, (source, columns) in resolve_sources(schema, dataset, only=only).items()
    }

def _typed(table: str, result: Any, physical: Mapping[str, str], columns: Mapping[str, DType]) -> GpuTable:
    by_physical = dict(zip(result.column_names(), result.tbl.columns(), strict=True))
    gpu_columns = {name: _stored(by_physical[physical[name]], dtype, f"{table}.{name}")
                   for name, dtype in columns.items()}
    return GpuTable(table, result.tbl.num_rows(), gpu_columns, dict(columns))

def _load_parquet(table: str, source: Parquet, columns: Mapping[str, DType], *, split: bool = False) -> GpuTable:
    import pylibcudf as plc

    options, physical = _reader(table, source, columns, split=split)
    return _typed(table, plc.io.parquet.read_parquet(options), physical, columns)

def read_chunks(table: str, source: Parquet, columns: Mapping[str, DType], *,
                chunk_bytes: int) -> Iterator[GpuTable]:
    """Read a table in pieces of about `chunk_bytes` each, for tables bigger than the GPU.

    The same checks and stored forms as `load_tables`; at least one piece, which
    is empty for an empty table.
    """

    import pylibcudf as plc

    options, physical = _reader(table, source, columns)
    reader = plc.io.parquet.ChunkedParquetReader(
        options, chunk_read_limit=chunk_bytes, pass_read_limit=chunk_bytes
    )
    while True:
        yield _typed(table, reader.read_chunk(), physical, columns)
        if not reader.has_next():
            return

def _with_codes(table: GpuTable) -> GpuTable:
    """`table` with each text column held as int32 codes into its sorted distinct texts.

    The texts are sorted by bytes, so codes compare as their texts do. A NULL
    keeps its row's mask; ``""`` is a text. The text column itself is freed.
    """

    import pylibcudf as plc

    columns, dictionaries = dict(table.columns), {}
    for name, column in table.columns.items():
        dtype = table.dtypes.get(name)
        if column.type().id() != plc.TypeId.STRING or (dtype is not None and dtype.name.startswith("decimal")):
            continue  # a wide decimal's bytes are strings to libcudf, not text
        texts = plc.stream_compaction.drop_nulls(plc.Table([column]), [0], 1)
        texts = plc.stream_compaction.distinct(texts, [0], plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
                                               plc.types.NullEquality.EQUAL, plc.types.NanEquality.ALL_EQUAL)
        texts = plc.sorting.sort(texts, [plc.types.Order.ASCENDING], [plc.types.NullOrder.AFTER])
        codes = plc.search.lower_bound(texts, plc.Table([column]), [plc.types.Order.ASCENDING],
                                       [plc.types.NullOrder.AFTER])
        if column.null_count():
            codes = codes.with_mask(column.null_mask(), column.null_count())
        columns[name], dictionaries[name] = codes, texts.columns()[0]
    return GpuTable(table.name, table.num_rows, columns, table.dtypes, dictionaries)
