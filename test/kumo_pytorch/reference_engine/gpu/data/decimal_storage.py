"""How decimal columns are stored on the GPU (256-bit ones as bytes) and turned into numbers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType


#: Bytes a wide decimal (a ``decimal256`` over precision 38) is held in, a value (`_wide_decimal`).
WIDE_DECIMAL_BYTES = 32

def _stored(column: Any, dtype: DType, where: str) -> Any:
    """A column as read, in its stored form: timestamps as whole UTC seconds, rounded up; wide decimals as 32
    bytes a value (`decimal_stored`)."""

    import pylibcudf as plc

    if dtype.name == "timestamp" and column.type().id() != plc.TypeId.TIMESTAMP_SECONDS:
        column = plc.datetime.ceil_datetimes(column, plc.datetime.RoundingFrequency.SECOND)
        column = plc.unary.cast(column, plc.DataType(plc.TypeId.TIMESTAMP_SECONDS))
    return decimal_stored(column, dtype, where)

def decimal_stored(column: Any, dtype: DType, where: str) -> Any:
    """A ``decimal256`` column libcudf read as strings (precision over 38) as a wide decimal (`_wide_decimal`);
    any other column as it is."""

    import pylibcudf as plc

    if dtype.name == "decimal256" and column.type().id() == plc.TypeId.STRING:
        return _wide_decimal(column, where)
    return column

def _decimal_numbers(column: Any, present: Any, dtype: Any = None) -> Any:
    """A decimal column's values as correctly rounded float64 (`GpuTable.numbers`): a libcudf decimal column,
    or a wide decimal of `dtype` (`_wide_decimal`)."""

    import cupy
    import pylibcudf as plc

    size = column.size()
    if size == 0:
        return cupy.zeros(0, dtype=cupy.float64)
    kind = column.type().id()
    if kind == plc.TypeId.STRING:  # a wide decimal: its low word, and whether the value fits in it
        scale = -dtype.scale  # libcudf's sense: value = unscaled * 10**scale; its strings type keeps none
        unscaled, fits = _wide_low_words(column)
    else:
        scale = column.type().scale()  # value = unscaled * 10**scale
        width = {plc.TypeId.DECIMAL32: 4, plc.TypeId.DECIMAL64: 8, plc.TypeId.DECIMAL128: 16}[kind]
        data = column.data()
        pointer = data.__cuda_array_interface__["data"][0] + column.offset() * width
        memory = cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(pointer, size * width, column), 0)
        if kind == plc.TypeId.DECIMAL128:  # little-endian words: an int64 holds it when the high word extends its sign
            words = cupy.ndarray((size, 2), cupy.int64, memory)
            unscaled = words[:, 0].copy()
            fits = words[:, 1] == (unscaled >> 63)
        else:
            unscaled = cupy.ndarray((size,), cupy.int32 if width == 4 else cupy.int64, memory).astype(cupy.int64)
            fits = cupy.ones(size, dtype=bool)

    def host(rows: Any) -> Any:
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

        picked = plc.copying.gather(plc.Table([column]), _column(rows.astype(cupy.int32)),
                                    plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
        return decimal_arrow(picked, dtype) if kind == plc.TypeId.STRING else picked.to_arrow()

    return rounded_numbers(unscaled, fits, scale, present, host)

def rounded_numbers(unscaled: Any, fits: Any, scale: int, present: Any, host: Any) -> Any:
    """Decimals as correctly rounded float64 (`GpuTable.numbers`), from each value's low 64 bits `unscaled`
    (CuPy int64), whether the value `fits` in them, and the `scale` in libcudf's sense (value = unscaled *
    10**scale). Unscaled times or divided by a power of ten is exact on the GPU, so correctly rounded, while the
    unscaled value fits in 53 bits and the scale in 22 digits; any other row is converted by upstream's own
    function from `host(rows)`, those rows as an Arrow decimal array."""

    import cupy

    exact = fits & (unscaled <= 2**53) & (unscaled >= -(2**53)) & (abs(scale) <= 22)
    factor = float(10 ** min(abs(scale), 22))
    numbers = unscaled.astype(cupy.float64) * factor if scale > 0 else unscaled.astype(cupy.float64) / factor
    slow = cupy.flatnonzero(~exact if present is None else ~exact & present)
    if slow.size:  # beyond exact float arithmetic: upstream's own conversion, on the host
        from kumo.pytorch.contract.dtypes import arrow_float64

        converted = arrow_float64(host(slow)).to_numpy(zero_copy_only=False)
        numbers[slow] = cupy.asarray(converted, dtype=cupy.float64)
    return numbers


# -- wide decimals ------------------------------------------------------------------------------------
#
# A ``decimal256`` of precision over 38 has no libcudf type: libcudf reads it as strings, each value the
# big-endian two's complement bytes Parquet stores, as few as the precision needs (17 for precision 39, 32 for
# 76). It is held as a strings column of 32 bytes a value, the sign extended: values are equal when their bytes
# are, it is gathered, joined and written as any strings column is, and a split's file of it (binary) reads back
# unchanged. A NULL row's bytes say nothing; libcudf gives it none after a gather, so every reader here goes by
# each row's own bytes, a row of none reading as 0.


def _wide_decimal(column: Any, where: str) -> Any:
    """A decimal column libcudf read as strings, as 32 bytes a value (big-endian, the sign extended)."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    size = column.size()
    if size == 0:
        return column
    _, lengths = _wide_bounds(column)
    if bool((lengths > WIDE_DECIMAL_BYTES).any()):
        raise ValueError(f"{where}: a decimal value of more than {WIDE_DECIMAL_BYTES} bytes")
    held = _wide_bytes(column)
    total = size * WIDE_DECIMAL_BYTES
    offsets = cupy.arange(size + 1, dtype=cupy.int64 if total >= 2**31 else cupy.int32) * WIDE_DECIMAL_BYTES
    mask = None
    if column.null_count():
        mask = column.null_mask() if column.offset() == 0 else plc.gpumemoryview(plc.null_mask.copy_bitmask(column))
    return plc.Column(plc.DataType(plc.TypeId.STRING), size, plc.gpumemoryview(held.reshape(-1)), mask,
                      column.null_count(), 0, [_column(offsets)])

def _wide_bounds(column: Any) -> tuple[Any, Any]:
    """Where each row's bytes start in a strings column's characters, and how many there are (CuPy int64)."""

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view

    if column.size() == 0:  # no rows: libcudf may give such a column no offsets at all (a gather of none)
        return cupy.zeros(0, dtype=cupy.int64), cupy.zeros(0, dtype=cupy.int64)
    bounds = _view(column.children()[0]).astype(cupy.int64)[column.offset():column.offset() + column.size() + 1]
    return bounds[:-1], bounds[1:] - bounds[:-1]

def _characters(column: Any) -> Any:
    """A strings column's characters (CuPy uint8, no copy); one zero byte when it has none."""

    import cupy

    data = column.data()
    interface = None if data is None else data.__cuda_array_interface__
    if interface is None or not interface["data"][0] or not interface["shape"][0]:  # no bytes (every row NULL)
        return cupy.zeros(1, dtype=cupy.uint8)
    size = int(interface["shape"][0])
    memory = cupy.cuda.UnownedMemory(interface["data"][0], size, column)
    return cupy.ndarray((size,), cupy.uint8, cupy.cuda.MemoryPointer(memory, 0))

def _wide_bytes(column: Any) -> Any:
    """Each row's value of a decimal strings column as 32 big-endian bytes, the sign extended (CuPy uint8
    (rows, 32)); a row of no bytes is 0."""

    import cupy

    starts, lengths = _wide_bounds(column)
    held = cupy.empty((column.size(), WIDE_DECIMAL_BYTES), dtype=cupy.uint8)
    if column.size():
        _wide_kernels()[0](starts, lengths, _characters(column), held)
    return held

def _wide_low_words(column: Any) -> tuple[Any, Any]:
    """Each row's value of a decimal strings column as its low 64 bits (CuPy int64), and whether the value
    fits in them (CuPy bool): its other 24 bytes and the low word's top bit all its sign."""

    import cupy

    starts, lengths = _wide_bounds(column)
    low, fits = cupy.empty(column.size(), dtype=cupy.int64), cupy.empty(column.size(), dtype=bool)
    if column.size():
        _wide_kernels()[1](starts, lengths, _characters(column), low, fits)
    return low, fits

def _wide_kernels() -> tuple[Any, Any]:
    """The kernels reading a decimal's big-endian bytes (`_wide_bytes`, `_wide_low_words`), compiled once."""

    global _WIDE_KERNELS
    if _WIDE_KERNELS is None:
        import cupy

        read = """
        const long long pad = 32 - length;
        const unsigned char sign = (length > 0 && (characters[start] & 0x80)) ? 0xff : 0;
        """
        widen = cupy.ElementwiseKernel(
            "raw int64 starts, raw int64 lengths, raw uint8 characters", "uint8 held",
            """
            const long long row = i / 32, place = i % 32;
            const long long start = starts[row], length = lengths[row];
            """ + read + """
            held = place >= pad ? characters[start + place - pad] : sign;
            """, "test.kumo_pytorch.reference_engine.gpu_wide_bytes")
        low = cupy.ElementwiseKernel(
            "int64 start, int64 length, raw uint8 characters", "int64 low, bool fits",
            read + """
            unsigned long long word = 0;
            bool extended = true;
            for (int place = 0; place < 32; ++place) {
                const unsigned char byte = place >= pad ? characters[start + place - pad] : sign;
                if (place < 24) extended = extended && byte == sign;
                else word = (word << 8) | byte;
            }
            low = (long long) word;
            fits = extended && (word >> 63) == (unsigned long long) (sign & 1);
            """, "test.kumo_pytorch.reference_engine.gpu_wide_low_word")
        _WIDE_KERNELS = (widen, low)
    return _WIDE_KERNELS

_WIDE_KERNELS: tuple[Any, Any] | None = None

def decimal_arrow(column: Any, dtype: Any) -> Any:
    """A decimal column as held (a libcudf decimal, or a wide decimal's bytes) as an Arrow array of `dtype`,
    NULLs kept: a host copy."""

    import pylibcudf as plc

    if column.type().id() != plc.TypeId.STRING:
        return column.to_arrow().cast(dtype.to_arrow())
    import numpy as np
    import pyarrow as pa

    little = np.ascontiguousarray(_wide_bytes(column).get()[:, ::-1])  # Arrow's: little-endian 64-bit words
    validity = None
    if column.null_count():
        mask = column.null_mask().__cuda_array_interface__["data"][0]
        bools = plc.transform.mask_to_bools(mask, column.offset(), column.offset() + column.size())
        validity = bools.to_arrow().buffers()[1]
    return pa.Array.from_buffers(dtype.to_arrow(), column.size(), [validity, pa.py_buffer(little)],
                                 null_count=column.null_count())
