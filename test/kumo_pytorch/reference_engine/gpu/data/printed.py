# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable printed-key ordering for rows with equal times or values.

Upstream sorts rows that tie (on time, on a value) by their primary key's
``(type name, repr)``: integers by their
decimal digits as text, texts by Python's ``repr``, dates and timestamps by their
numbers as printed. These are worked out here on the GPU, as columns a libcudf
sort takes (`key_columns`): an integer's from its value in one pass
(`integer_columns`), a text's by building its ``repr`` (`text_reprs`) and ranking
the distinct ones a sort meets (`text_ranks`), a date's and a timestamp's from
their calendar parts. A float's, a decimal's and bytes' ``repr`` are Python's own,
on the host, for the keys a sort meets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

__all__ = ["integer_columns", "key_columns", "on_gpu", "sort_key", "text_ranks", "text_reprs"]


def sort_key(value: Any) -> tuple[str, str]:
    """The declared host ordering for a scalar primary key or grouping value."""

    return type(value).__name__, repr(value)


#: One pass per value: its sign, its magnitude's digits as a 19-digit number padded with zeros on the right,
#: and its number of digits (a signed 64-bit magnitude has at most 19).
_PRINTED_SIGNED = """
    const bool negative = x < 0;
    const unsigned long long m = negative ? (unsigned long long)(-(x + 1)) + 1ULL : (unsigned long long)x;
    int d = 1;
    unsigned long long p = 10ULL;
    while (d < 19 && m >= p) { d++; p *= 10ULL; }
    unsigned long long scale = 1ULL;
    for (int i = d; i < 19; i++) scale *= 10ULL;
    sign = negative ? 0 : 1;
    padded = m * scale;
    digits = d;
"""

#: The same for unsigned values, of up to 20 digits: a 20-digit value pads its first 19, then gives its last.
_PRINTED_UNSIGNED = """
    int d = 1;
    unsigned long long p = 10ULL;
    for (int k = 1; k <= 19; k++) {
        if (x < p) break;
        d++;
        if (k < 19) p *= 10ULL;
    }
    if (d == 20) {
        padded = x / 10ULL;
        last = x % 10ULL;
    } else {
        unsigned long long scale = 1ULL;
        for (int i = d; i < 19; i++) scale *= 10ULL;
        padded = x * scale;
        last = 0;
    }
    digits = d;
"""

_PRINTED_KERNELS: dict[bool, Any] = {}


def integer_columns(values: Any) -> list[Any]:
    """Columns that sort integers as their printed decimal text sorts (Python's ``str``,
    which breaks ties between primary keys), from the values in one GPU pass, with no text.

    Text compares character by character: ``-`` sorts before every digit, so negatives come first, and a text
    that begins another sorts before it: ``"1"``, ``"10"``, ``"100"``, ``"19"``, ``"9"``. So a value sorts by
    its sign, then by its magnitude's digits read as a 19-digit number padded with zeros on the right (``1``
    and ``10`` both give 1000000000000000000), then by its number of digits; a 20-digit unsigned value by its
    first 19 digits so padded, then its digit count, then its last digit.
    """

    import cupy

    signed = values.dtype.kind == "i"
    kernel = _PRINTED_KERNELS.get(signed)
    if kernel is None:
        kernel = _PRINTED_KERNELS[signed] = (
            cupy.ElementwiseKernel("int64 x", "int8 sign, uint64 padded, int8 digits", _PRINTED_SIGNED,
                                   "kumo_pytorch_gpu_printed_signed") if signed else
            cupy.ElementwiseKernel("uint64 x", "uint64 padded, int8 digits, int8 last", _PRINTED_UNSIGNED,
                                   "kumo_pytorch_gpu_printed_unsigned"))
    count = values.size
    if signed:
        sign, padded, digits = (cupy.empty(count, cupy.int8), cupy.empty(count, cupy.uint64),
                                cupy.empty(count, cupy.int8))
        kernel(values.astype(cupy.int64, copy=False), sign, padded, digits)
        return [sign, padded, digits]
    padded, digits, last = cupy.empty(count, cupy.uint64), cupy.empty(count, cupy.int8), cupy.empty(count, cupy.int8)
    kernel(values.astype(cupy.uint64, copy=False), padded, digits, last)
    return [padded, digits, last]


#: ASCII characters Python's ``repr`` escapes, and how: a backslash doubled, tab, newline and return named, other
#: control characters as ``\xhh``. A quote is handled per text (`text_reprs`).
_CONTROL = [chr(code) for code in range(32)] + ["\x7f"]
_ESCAPES = {"\\": "\\\\", **{c: {"\t": "\\t", "\n": "\\n", "\r": "\\r"}.get(c, f"\\x{ord(c):02x}") for c in _CONTROL}}


def text_reprs(texts: Any) -> Any:
    """Python's ``repr`` of each text of `texts` (a libcudf strings column without NULLs), as a strings column.

    An ASCII text's is built on the GPU: escapes (`_ESCAPES`) inside quotes, double quotes when the text holds a
    single quote and no double one, single quotes otherwise (a single quote then escaped when the text holds
    both). Which other characters Python escapes depends on Unicode's categories, so a text with any is sent to
    the host, and only those texts.
    """

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _column,
        _view,
    )

    import cupy
    import pyarrow as pa
    import pylibcudf as plc

    def strings(values: list[str]) -> Any:
        return plc.Column.from_arrow(pa.array(values, pa.string()))

    def scalar(text: str) -> Any:
        return plc.Scalar.from_arrow(pa.scalar(text, pa.string()))

    count = texts.size()
    if count == 0:
        return texts
    flag = plc.DataType(plc.TypeId.BOOL8)
    wide = cupy.flatnonzero(_view(plc.strings.attributes.count_bytes(texts))
                            != _view(plc.strings.attributes.count_characters(texts))).astype(cupy.int32)
    body = plc.strings.replace.replace_multiple(texts, strings(list(_ESCAPES)), strings(list(_ESCAPES.values())))
    single, double = plc.strings.find.contains(texts, scalar("'")), plc.strings.find.contains(texts, scalar('"'))
    both = plc.binaryop.binary_operation(single, double, plc.binaryop.BinaryOperator.LOGICAL_AND, flag)
    if bool(_view(both).any()):  # both quotes: in single quotes, the single one escaped
        body = plc.copying.copy_if_else(plc.strings.replace.replace(body, scalar("'"), scalar("\\'")), body, both)
    only_single = plc.binaryop.binary_operation(single, double, plc.binaryop.BinaryOperator.GREATER, flag)
    del single, double, both
    quote = plc.Column.from_scalar(scalar("'"), count)
    if bool(_view(only_single).any()):  # a single quote and no double one: in double quotes
        quote = plc.copying.copy_if_else(plc.Column.from_scalar(scalar('"'), count), quote, only_single)
    del only_single
    reprs = plc.strings.combine.concatenate(plc.Table([quote, body, quote]), scalar(""))
    del quote, body
    if wide.size:  # beyond ASCII: Python's own repr, for those texts only
        picked = plc.copying.gather(plc.Table([texts]), _column(wide), plc.copying.OutOfBoundsPolicy.DONT_CHECK)
        host = strings([repr(text) for text in picked.columns()[0].to_arrow().to_pylist()])
        reprs = plc.copying.scatter(plc.Table([host]), _column(wide), plc.Table([reprs])).columns()[0]
    return reprs


def text_ranks(dictionary: Any, codes: Any) -> Any:
    """Each of `codes` (CuPy, into `dictionary`: sorted distinct texts) as its text's place in printed order
    (``repr``) among the distinct texts `codes` holds, CuPy int32: equal texts equal, so a sort by it is
    upstream's order. Only those texts are printed: found by marking them when `codes` are as many as the
    texts (an index over a whole table: every text, printed without a copy), else by sorting `codes`."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _column,
        _view,
    )

    import cupy
    import pylibcudf as plc

    size = dictionary.size()
    if codes.size == 0:
        return cupy.zeros(codes.shape, dtype=cupy.int32)
    if codes.size >= size:
        met = cupy.zeros(size, dtype=bool)
        met[codes] = True
        distinct = cupy.flatnonzero(met).astype(cupy.int32)
        del met
        inverse = None
    else:
        distinct, inverse = cupy.unique(codes, return_inverse=True)
        distinct = distinct.astype(cupy.int32)
    texts = dictionary if distinct.size == size else plc.copying.gather(
        plc.Table([dictionary]), _column(distinct), plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
    reprs = text_reprs(texts)
    del texts
    order = _view(plc.sorting.stable_sorted_order(plc.Table([reprs]), [plc.types.Order.ASCENDING],
                                                  [plc.types.NullOrder.AFTER]))
    del reprs
    if inverse is None:  # by code: each met text's rank
        rank = cupy.empty(size, dtype=cupy.int32)
        rank[distinct[order]] = cupy.arange(distinct.size, dtype=cupy.int32)
        return rank[codes]
    rank = cupy.empty(distinct.size, dtype=cupy.int32)
    rank[order] = cupy.arange(distinct.size, dtype=cupy.int32)
    return rank[inverse.reshape(codes.shape)]


#: Key types `key_columns` orders on the GPU; any other's order is worked out on the host (`_host_ranks`).
GPU_PRINTED = frozenset({"string", "bool", "date32", "timestamp"})


def on_gpu(dtype: Any) -> bool:
    """Whether `key_columns` orders keys of `dtype` on the GPU (integers, text, booleans, dates, timestamps)."""

    return dtype.integer or dtype.name in GPU_PRINTED


def key_columns(table: GpuTable, key: str, rows: Any) -> list[Any]:
    """Columns (CuPy) that sort the rows of `table` at `rows` as upstream sorts them by primary key `key`, its
    ``repr``:

    - an integer by its digits (`integer_columns`), a text by its ``repr`` (`text_ranks`);
    - a boolean by its value ("False" before "True");
    - a date, ``datetime.date(2024, 1, 5)``, by its year, month and day each as digits (unpadded: day 10 before
      day 5, as "10" before "5");
    - a timestamp, ``datetime.datetime(2024, 1, 5, 9, 30[, 15], tzinfo=...)``, by its year, month, day, hour and
      minute as digits, then its seconds, which Python prints only when not 0 (a digit sorts before "tzinfo");
    - a float, a decimal or bytes by Python's own ``repr``, on the host, for the distinct keys of these rows only
      (`_host_ranks`): Python's shortest float, ``Decimal('1.50')``, ``b'..'``. For a table's own primary key
      (a `NeighborIndex`), that is every row's (about 1.3 s a million floats or bytes, 2.4 decimals)."""

    dtype = table.dtypes[key]
    if not on_gpu(dtype):
        return _host_ranks(table, key, rows)
    values = table.values(key)[rows]
    if dtype.integer:
        return integer_columns(values)
    if dtype.name == "string":
        return [text_ranks(table.dictionaries[key], values)]
    if dtype.name == "bool":
        return [values.astype("int64")]
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
    from test.kumo_pytorch.reference_engine.gpu.sampling.calculations import (
        _calendar,
    )

    parts = ("year", "month", "day") if dtype.name == "date32" else ("year", "month", "day", "hour", "minute")
    columns = []
    for part in parts:
        columns += integer_columns(_calendar(part, Values(dtype, values)))
    if dtype.name == "timestamp":
        second = _calendar("second", Values(dtype, values))
        columns += [(second == 0).astype("int64"), *integer_columns(second)]
    return columns


def _host_ranks(table: GpuTable, key: str, rows: Any) -> list[Any]:
    """Each row's place among these rows' distinct keys in ``repr`` order (CuPy int64), by Python's ``repr`` on
    the host: one column. A decimal is gathered alone (not its float64 numbers, which `GpuTable.numbers` would
    work out for the whole column and keep)."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
    from test.kumo_pytorch.reference_engine.gpu.sampling.host_values import (
        _values_python,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
    )

    import cupy

    if table.dtypes[key].name.startswith("decimal"):
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _column,
        )

        import pylibcudf as plc

        picked = plc.copying.gather(plc.Table([table.columns[key]]), _column(cupy.ascontiguousarray(rows.astype("int32"))),
                                    plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
        values = Values(table.dtypes[key], None, column=picked)
    else:
        values = _column_values(table, key, rows)
    printed = [repr(value) for value in _values_python(values)]
    places = {text: place for place, text in enumerate(sorted(set(printed)))}
    return [cupy.asarray([places[text] for text in printed], dtype=cupy.int64)]
