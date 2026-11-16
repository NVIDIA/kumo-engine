"""Aggregates over related rows: counts, sums, distinct counts, modes and groupings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
if TYPE_CHECKING:

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable


_U32 = (1 << 32) - 1

def _column_values(table: GpuTable, column: str, rows: Any, *, missing: bool = False) -> Values:
    """Column `column` of `table` at `rows`, as sampled values (decimals gathered to the host).

    With `missing`, a row of -1 means no row: NULL.
    """

    import cupy

    dtype = table.dtypes[column]
    present = table.present(column)
    if missing:
        has = rows >= 0
        if table.num_rows == 0:
            return _nulls(dtype, rows.size, table.dictionaries.get(column))
        rows = cupy.where(has, rows, 0)
        present = has if present is None else has & present[rows]
    else:
        present = None if present is None else present[rows]
    if dtype.name.startswith("decimal"):  # on the GPU: the values (an Arrow copy made when asked) and numbers
        import pylibcudf as plc

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

        picked = plc.copying.gather(plc.Table([table.columns[column]]), _column(rows.astype("int32")),
                                    plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
        return Values(dtype, None, present, numbers=table.numbers(column)[rows], column=picked)
    return Values(dtype, table.values(column)[rows], present, table.dictionaries.get(column))

def _nulls(dtype: Any, count: int, texts: Any) -> Values:
    import cupy

    absent = cupy.zeros(count, dtype=bool)
    if dtype.name.startswith("decimal"):
        import pyarrow as pa

        return Values(dtype, None, absent, host=pa.nulls(count, dtype.to_arrow()),
                      numbers=cupy.zeros(count, dtype=cupy.float64), words=cupy.zeros((count, 4), dtype=cupy.int64))
    return Values(dtype, cupy.zeros(count, dtype=_device_type(dtype)), absent, texts)

def _device_type(dtype: Any) -> Any:
    """The CuPy type a value of `dtype` is held in (timestamps: seconds; dates: days; texts: codes)."""

    import cupy

    if dtype.name == "timestamp":
        return cupy.int64
    if dtype.name in ("date32", "string", "bytes"):
        return cupy.int32
    if dtype.name == "bool":
        return cupy.bool_
    return cupy.dtype(dtype.name)

def _counts(owner: Any, count: int) -> Any:
    import cupy

    counts = cupy.zeros(count, dtype=cupy.int64)
    cupy.add.at(counts, owner, 1)
    return counts

def _bits(values: Any) -> Any:
    """Values as integers that are equal exactly when the stored values are (floats by their bits)."""

    import cupy

    if values.dtype == cupy.float64:
        return values.view(cupy.int64)
    if values.dtype == cupy.float32:
        return values.view(cupy.int32).astype(cupy.int64)
    return values.astype(cupy.int64)

def _count_distinct(owner: Any, keys: Any, count: int) -> Any:
    import cupy

    if owner.size == 0:
        return cupy.zeros(count, dtype=cupy.int64)
    order = cupy.lexsort(cupy.stack([keys, owner]))
    owner, keys = owner[order], keys[order]
    first = cupy.concatenate([cupy.ones(1, dtype=bool), (owner[1:] != owner[:-1]) | (keys[1:] != keys[:-1])])
    return _counts(owner[first], count)

def _exact_sum(where: str, owner: Any, values: Any, rows_per: Any, dtype: Any, flags: list) -> Values:
    """Exact integer sums per segment, in two 32-bit halves: int64 for signed values, uint64 for unsigned."""

    import cupy

    count = rows_per.size
    unsigned = dtype.name == "uint64"
    kind = cupy.uint64 if unsigned else cupy.int64
    wide = values.astype(kind)
    high_sum = cupy.zeros(count, dtype=kind)
    low_sum = cupy.zeros(count, dtype=kind)
    cupy.add.at(high_sum, owner, wide >> kind(32))
    cupy.add.at(low_sum, owner, wide & kind(_U32))
    high = high_sum + (low_sum >> kind(32))
    low = low_sum & kind(_U32)
    if unsigned:
        fits = high <= kind(_U32)
    else:
        fits = (high >= -(1 << 31)) & (high < (1 << 31))
    flags.append((f"{where}: an integer sum does not fit {dtype.name}", (~fits).any()))
    total = (high << kind(32)) | low
    valid_rows = _counts(owner, count)
    return Values(dtype, total.astype(kind), (rows_per == 0) | (valid_rows > 0))

def _grouped(owner: Any, values: Any, operation: str, count: int, *, ddof: int = 0) -> tuple[Any, Any]:
    """A reduction of `values` per segment (libcudf's sort-based groupby): (result, whether any value)."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view

    result = cupy.zeros(count, dtype=cupy.float64 if operation in ("mean", "variance", "std") else values.dtype)
    has = cupy.zeros(count, dtype=bool)
    if owner.size == 0:
        return result, has
    aggregation = {"sum": plc.aggregation.sum, "mean": plc.aggregation.mean, "min": plc.aggregation.min,
                   "max": plc.aggregation.max}.get(operation)
    if aggregation is None:
        aggregation = (plc.aggregation.variance if operation == "variance" else plc.aggregation.std)
        request = aggregation(ddof)
    else:
        request = aggregation()
    stored = values.astype(cupy.uint8) if values.dtype == cupy.bool_ else values
    groups = plc.groupby.GroupBy(plc.Table([_column(owner)]), plc.types.NullPolicy.EXCLUDE, plc.types.Sorted.YES)
    keys, results = groups.aggregate([plc.groupby.GroupByRequest(_column(cupy.ascontiguousarray(stored)), [request])])
    found = _view(keys.columns()[0])
    column = results[0].columns()[0]
    reduced = _view(column)
    valid = cupy.ones(found.size, dtype=bool) if column.null_count() == 0 else \
        _view(plc.transform.mask_to_bools(column.null_mask().__cuda_array_interface__["data"][0], 0,
                                          column.size())).view(cupy.bool_)
    result[found] = reduced.astype(result.dtype) if values.dtype != cupy.bool_ else reduced.astype(cupy.bool_)
    has[found] = valid
    return result, has

def _most_frequent(owner: Any, keys: Any, count: int) -> tuple[Any, Any]:
    """Per segment, the most frequent of `keys` (CuPy int64), ties to the smallest; and whether it has any."""

    import cupy

    best = cupy.zeros(count, dtype=cupy.int64)
    if owner.size == 0:
        return best, cupy.zeros(count, dtype=bool)
    order = cupy.lexsort(cupy.stack([keys, owner]))
    owner, keys = owner[order], keys[order]
    starts = cupy.flatnonzero(cupy.concatenate([cupy.ones(1, dtype=bool),
                                                (owner[1:] != owner[:-1]) | (keys[1:] != keys[:-1])]))
    lengths = cupy.diff(cupy.concatenate([starts, cupy.asarray([owner.size])]))
    run_owner, run_key = owner[starts], keys[starts]
    most = cupy.zeros(count, dtype=cupy.int64)
    cupy.maximum.at(most, run_owner, lengths)
    tied = lengths == most[run_owner]
    best[:] = cupy.iinfo(cupy.int64).max
    cupy.minimum.at(best, run_owner[tied], run_key[tied])
    has = most > 0
    return cupy.where(has, best, 0), has

def _printed_ranks(distinct: Any, dtype: Any) -> Any:
    """Each distinct decimal's place in upstream's tie order, its ``repr`` (CuPy int64): worked out on the host,
    for these values only."""

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.data.decimals import _arrow

    python = _arrow(distinct, dtype).to_pylist()
    order = sorted(range(len(python)), key=lambda i: repr(python[i]))
    ranks = cupy.empty(len(python), dtype=cupy.int64)
    ranks[cupy.asarray(order, dtype=cupy.int64)] = cupy.arange(len(python), dtype=cupy.int64)
    return ranks

def _sum_refusal(where: str, dtype: Any, limbs: Any, bad: Any) -> str:
    """Upstream's own message for the first refused decimal sum: its type's check of the exact sum."""

    import cupy
    from decimal import Context, Decimal

    from test.kumo_pytorch.reference_engine.gpu.data.decimals import limbs_int

    row = int(cupy.flatnonzero(bad)[0])
    value = Decimal(limbs_int(limbs[row].get().tolist())).scaleb(-dtype.scale, context=Context(prec=400))
    try:
        dtype.normalize(value, path=where)
    except ValueError as error:
        return str(error)
    return f"{where}: test.kumo_pytorch.reference_engine.gpu refused the sum {value!r}, which upstream Kumo PyTorch accepts (a bug in test.kumo_pytorch.reference_engine.gpu)"

def _floats(operand: Values) -> Values:
    """A decimal operand as its float64 numbers, as upstream takes ``float(value)``."""

    from kumo.pytorch.contract.dtypes import DType

    return Values(DType("float64"), operand.numbers, operand.present)
