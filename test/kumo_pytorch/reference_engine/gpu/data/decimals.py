"""Exact decimal values and arithmetic on the GPU.

Decimal calculations use 160 digits of precision and then check the
result against the result's type (`DType.normalize`): exact, never rounded, within the type's precision. Here a
decimal is its unscaled integer (value = unscaled * 10**-scale, the scale its type's) as 256-bit two's
complement: four little-endian 64-bit words a value, CuPy int64 ``(rows, 4)``. Every ``decimal256`` value fits
(10**76 < 2**253). `calculate` aligns scales by a power of ten (add, subtract) or adds them (multiply), and
marks a row as one upstream refuses when any step leaves 256 bits or the result has more digits than its type's
precision: a result that fits is exact, so it is upstream's.

A calculation reads a column's values (`column_words`: libcudf's decimals, and wide decimals), a constant's
(`constant_words`), an integer's (scale 0) or another calculation's (`Values.words`). Its result is held as
words, with float64 numbers worked out as a column's are (`test.kumo_pytorch.reference_engine.gpu.data.decimal_storage.rounded_numbers`) and Arrow values
made when something asks for them (`decimal_values`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values

__all__ = ["calculate", "column_words", "compare", "constant_words", "decimal_values", "python_decimal", "words"]

#: Words a value is held in, and the most digits a decimal holds (decimal256's precision).
WORDS = 4
MAX_DIGITS = 76

_OPERATIONS = {"add": 0, "subtract": 1, "multiply": 2, "abs": 3}


def is_decimal(dtype: Any) -> bool:
    return dtype is not None and dtype.name.startswith("decimal")


def words(values: Values) -> Any:
    """`values`' unscaled integers, CuPy int64 ``(rows, 4)``: a decimal's, or an integer's (scale 0), worked out
    once and kept (a field read by several calculations). A NULL row's words are any value."""

    if values.words is None:
        if values.values is not None:
            held = integer_words(values.values)
        elif values.column is not None:
            held = column_words(values.column)
        else:
            held = arrow_words(values.arrow())
        object.__setattr__(values, "words", held)  # kept, as `Values.arrow` keeps its copy
    return values.words


def integer_words(values: Any) -> Any:
    """Integers (CuPy, any width) as words: signed ones extend their sign, unsigned ones do not."""

    import cupy

    unsigned = values.dtype.kind == "u"
    low = values.view(cupy.int64) if values.dtype == cupy.uint64 else values.astype(cupy.int64)
    held = cupy.zeros((low.size, WORDS), dtype=cupy.int64)
    held[:, 0] = low
    if not unsigned:
        held[:, 1:] = (low >> 63)[:, None]
    return held


def column_words(column: Any) -> Any:
    """A libcudf decimal column's values (32, 64 or 128 bits), or a wide decimal's (`test.kumo_pytorch.reference_engine.gpu.data.decimal_storage._wide_decimal`),
    as words."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import _wide_bytes

    size = column.size()
    kind = column.type().id()
    if kind == plc.TypeId.STRING:  # big-endian bytes, reversed: little-endian words
        return cupy.ascontiguousarray(_wide_bytes(column)[:, ::-1]).view(cupy.int64).reshape(size, WORDS)
    held = cupy.zeros((size, WORDS), dtype=cupy.int64)
    if size == 0:
        return held
    width = {plc.TypeId.DECIMAL32: 4, plc.TypeId.DECIMAL64: 8, plc.TypeId.DECIMAL128: 16}[kind]
    pointer = column.data().__cuda_array_interface__["data"][0] + column.offset() * width
    memory = cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(pointer, size * width, column), 0)
    if kind == plc.TypeId.DECIMAL128:
        held[:, :2] = cupy.ndarray((size, 2), cupy.int64, memory)
    else:
        held[:, 0] = cupy.ndarray((size,), cupy.int32 if width == 4 else cupy.int64, memory).astype(cupy.int64)
        held[:, 1] = held[:, 0] >> 63
    held[:, 2:] = (held[:, 1] >> 63)[:, None]
    return held


def gathered_words(column: Any, rows: Any) -> Any:
    """A stored decimal column's words at `rows` (CuPy), only those rows read."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    picked = plc.copying.gather(plc.Table([column]), _column(cupy.ascontiguousarray(rows.astype(cupy.int32))),
                                plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
    return column_words(picked)


def distinct_index(held: Any) -> tuple[Any, Any]:
    """Each value's place among the distinct values of words `held`, those sorted by value (CuPy int64), and the
    distinct values' words: equal values share a place, and places order as their values do."""

    import cupy

    size = held.shape[0]
    if size == 0:
        return cupy.zeros(0, dtype=cupy.int64), cupy.zeros((0, WORDS), dtype=cupy.int64)
    flip = cupy.int64(-(2**63))  # the low words compare unsigned: their top bit flipped, they compare as signed
    order = cupy.lexsort(cupy.stack([held[:, 0] ^ flip, held[:, 1] ^ flip, held[:, 2] ^ flip, held[:, 3]]))
    ordered = held[order]
    new = cupy.ones(size, dtype=bool)
    new[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    index = cupy.empty(size, dtype=cupy.int64)
    index[order] = cupy.cumsum(new, dtype=cupy.int64) - 1
    return index, cupy.ascontiguousarray(ordered[new])


def segment_sums(owner: Any, held: Any, count: int) -> tuple[Any, Any]:
    """Exact sums of words `held` per segment (`owner`, `count` segments) in 288 bits: nine 32-bit limbs each
    summed (no int64 sum wraps under 2^31 rows), then carried. Returns each sum's limbs (CuPy int64 (count, 9),
    two's complement, the last limb its sign's 32 bits) and its words, the low 256 bits (CuPy int64 (count, 4))."""

    import cupy

    pieces = cupy.ascontiguousarray(held).view(cupy.uint32).reshape(held.shape[0], 2 * WORDS)
    sign = cupy.where(held[:, 3] < 0, cupy.int64(2**32 - 1), cupy.int64(0))
    limbs = cupy.zeros((count, 2 * WORDS + 1), dtype=cupy.int64)
    carry = cupy.zeros(count, dtype=cupy.int64)
    for k in range(2 * WORDS + 1):
        summed = cupy.zeros(count, dtype=cupy.int64)
        cupy.add.at(summed, owner, pieces[:, k].astype(cupy.int64) if k < 2 * WORDS else sign)
        total = summed + carry
        limbs[:, k] = total & (2**32 - 1)
        carry = total >> 32
    low = cupy.ascontiguousarray(limbs[:, :2 * WORDS].astype(cupy.uint32)).view(cupy.int64).reshape(count, WORDS)
    return limbs, low


def limbs_int(row: Any) -> int:
    """One sum's limbs (host, nine ints) as the Python integer they hold (288-bit two's complement)."""

    value = sum((int(limb) & (2**32 - 1)) << (32 * k) for k, limb in enumerate(row))
    return value - 2**288 if value >= 2**287 else value


def arrow_words(array: Any) -> Any:
    """An Arrow decimal array's unscaled integers as words (a host to GPU copy)."""

    import cupy
    import numpy as np
    import pyarrow as pa

    width = 4 if pa.types.is_decimal256(array.type) else 2
    size = len(array)
    held = np.zeros((size, WORDS), dtype=np.int64)
    if size:
        data = np.frombuffer(array.buffers()[1], dtype=np.int64)
        held[:, :width] = data[array.offset * width:(array.offset + size) * width].reshape(size, width)
        if width == 2:
            held[:, 2:] = (held[:, 1] >> 63)[:, None]
    return cupy.asarray(held)


def constant_words(value: Any, scale: int, count: int = 1) -> Any:
    """A Python value (a ``Decimal`` exact at `scale`, an ``int``; None as 0) as `count` rows of words."""

    import cupy

    unscaled = 0 if value is None else _unscaled(value, scale)
    row = [(unscaled >> (64 * k)) & (2**64 - 1) for k in range(WORDS)]
    row = [word - 2**64 if word >= 2**63 else word for word in row]
    return cupy.tile(cupy.asarray(row, dtype=cupy.int64), (count, 1))


def _unscaled(value: Any, scale: int) -> int:
    from decimal import Context, Decimal

    if isinstance(value, int):
        return value * 10**scale if scale >= 0 else value // 10**-scale
    shifted = Decimal(value).scaleb(scale, context=Context(prec=400))
    if shifted != shifted.to_integral_value():
        raise ValueError(f"{value!r} is not exact at scale {scale}")
    return int(shifted)


def python_decimal(row: Any, scale: int) -> Any:
    """One value's words (host, four ints) as the ``Decimal`` upstream holds: its unscaled integer at its type's
    scale (``Decimal('0.50')`` at scale 2)."""

    from decimal import Context, Decimal

    unscaled = sum((int(word) & (2**64 - 1)) << (64 * k) for k, word in enumerate(row))
    if unscaled >= 2**255:
        unscaled -= 2**256
    return Decimal(unscaled).scaleb(-scale, context=Context(prec=400))


def decimal_values(dtype: Any, held: Any, present: Any) -> Values:
    """Calculated decimals of `dtype` (words `held`) as sampled values: their words, float64 numbers, and Arrow
    values made when asked."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
    from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import rounded_numbers

    low = held[:, 0].copy()
    fits = (held[:, 1:] == (low >> 63)[:, None]).all(axis=1)
    numbers = rounded_numbers(low, fits, -dtype.scale, present, lambda rows: _arrow(held[rows], dtype))
    return Values(dtype, None, present, numbers=numbers, words=held, build=lambda: _arrow(held, dtype))


def _arrow(held: Any, dtype: Any) -> Any:
    """Words as an Arrow array of decimal `dtype` (no NULLs: `Values.arrow` applies them); a host copy."""

    import numpy as np
    import pyarrow as pa

    width = 2 if dtype.name == "decimal128" else WORDS
    data = np.ascontiguousarray(held.get()[:, :width])
    return pa.Array.from_buffers(dtype.to_arrow(), data.shape[0], [None, pa.py_buffer(data)])


def calculate(operation: str, operands: list[Values], dtype: Any) -> tuple[Any, Any]:
    """add, subtract, multiply or abs of decimals and integers, exact, as words of decimal `dtype`, and the rows
    upstream refuses: those whose result does not fit `dtype`'s precision (or leaves 256 bits on the way)."""

    import cupy
    import numpy as np

    scales = [operand.dtype.scale if is_decimal(operand.dtype) else 0 for operand in operands]
    scale = dtype.scale
    if operation in ("add", "subtract"):
        shifts = (scale - scales[0], scale - scales[1], 0)
    elif operation == "multiply":
        shifts = (0, 0, scale - scales[0] - scales[1])
    else:
        shifts = (0, 0, scale - scales[0])
    if min(shifts) < 0:  # upstream's types never ask for it: a result's scale is at least its operands'
        raise ValueError(f"test.kumo_pytorch.reference_engine.gpu cannot {operation} {[operand.dtype for operand in operands]} into {dtype}")
    first = cupy.ascontiguousarray(words(operands[0]))
    second = cupy.ascontiguousarray(words(operands[1])) if len(operands) > 1 else first
    held = cupy.empty_like(first)
    bad = cupy.empty(first.shape[0], dtype=bool)
    if bad.size:
        _kernels()[0](first, second, _powers(), np.int32(_OPERATIONS[operation]), *(np.int32(s) for s in shifts),
                      np.int32(dtype.precision), held, bad)
    return held, bad


def compare(first: Any, second: Any) -> Any:
    """-1, 0 or 1 (CuPy int8) as each value of words `first` is below, equal to or above `second`'s (the same
    rows, or one row for all), both at one scale."""

    import cupy
    import numpy as np

    first, second = cupy.ascontiguousarray(first), cupy.ascontiguousarray(second)
    order = cupy.empty(first.shape[0], dtype=cupy.int8)
    if order.size:
        _kernels()[1](first, second, np.int64(WORDS if second.shape[0] == first.shape[0] else 0), order)
    return order


def at_most(held: Any, scale: int, bound: int) -> Any:
    """Which decimals (words at `scale`) are at most the integer `bound`, exactly."""

    from decimal import Context, Decimal

    threshold = int(Decimal(bound).scaleb(scale, context=Context(prec=400)).to_integral_value(rounding="ROUND_FLOOR"))
    return compare(held, constant_words(threshold, 0)) <= 0


def rescaled(held: Any, shift: int) -> tuple[Any, Any]:
    """Decimals (words) times ``10**shift``, exactly: multiplied up, or divided down only where nothing is left
    over. Returns the words and which values have that form (CuPy bool): not beyond 256 bits, nothing left over."""

    import cupy
    import numpy as np

    held = cupy.ascontiguousarray(held)
    if shift == 0:
        return held, cupy.ones(held.shape[0], dtype=bool)
    result = cupy.empty_like(held)
    valid = cupy.empty(held.shape[0], dtype=bool)
    if valid.size:
        _conversions()[0](held, _powers(), np.int32(shift), result, valid)
    return result, valid


def float_words(values: Any, scale: int) -> tuple[Any, Any]:
    """Floats (CuPy, widened to float64 exactly) as decimals at `scale` (words), exactly: a value is valid only
    when it is a decimal of that scale (``Decimal(value)`` has at most `scale` places; 0.5 at scale 1, never 0.1,
    whose float is 0.1000000000000000055...), and none for NaN or infinity. -0.0 is 0."""

    import cupy
    import numpy as np

    from test.kumo_pytorch.reference_engine.gpu.data.keys import widened

    bits = cupy.ascontiguousarray(widened(values)).view(cupy.int64)
    held = cupy.empty((bits.size, WORDS), dtype=cupy.int64)
    valid = cupy.empty(bits.size, dtype=bool)
    if valid.size:
        _conversions()[1](bits, _powers_of_five(), np.int32(max(scale, 0)), held, valid)
    if scale < 0:  # a negative scale (in-memory decimals only): whole multiples of 10**-scale
        held, divided = rescaled(held, scale)
        valid &= divided
    return held, valid


def words_float(held: Any, scale: int) -> tuple[Any, Any]:
    """Decimals (words at `scale`) as float64 (CuPy), exactly: a value is valid only when a float64 equals it
    (``float(value) == value`` in Python: 0.5, never 0.1)."""

    import cupy

    if scale < 0:
        held, valid = rescaled(held, -scale)
        values, exact = words_float(held, 0)
        return values, valid & exact
    held = cupy.ascontiguousarray(held)
    bits = cupy.empty(held.shape[0], dtype=cupy.int64)
    valid = cupy.empty(held.shape[0], dtype=bool)
    if valid.size:
        import numpy as np

        _conversions()[2](held, np.int32(scale), bits, valid)
    return bits.view(cupy.float64), valid


_KERNELS: tuple[Any, Any] | None = None
_CONVERSIONS: tuple[Any, Any, Any] | None = None
_POWERS: Any = None
_FIVES: Any = None


def _powers_of_five() -> Any:
    """5**k for k = 0..76, as words (CuPy uint64 (77, 4))."""

    global _FIVES
    if _FIVES is None:
        import cupy

        _FIVES = cupy.asarray([[(5**k >> (64 * j)) & (2**64 - 1) for j in range(WORDS)]
                               for k in range(MAX_DIGITS + 1)], dtype=cupy.uint64)
    return _FIVES


def _powers() -> Any:
    """10**k for k = 0..76, as words (CuPy uint64 (77, 4))."""

    global _POWERS
    if _POWERS is None:
        import cupy

        _POWERS = cupy.asarray([[(10**k >> (64 * j)) & (2**64 - 1) for j in range(WORDS)]
                                for k in range(MAX_DIGITS + 1)], dtype=cupy.uint64)
    return _POWERS


_PREAMBLE = r"""
typedef unsigned long long kumo_u64;

// |w| into m (little-endian words); whether w is negative
__device__ bool kumo_magnitude(const long long* w, kumo_u64* m) {
    const bool negative = w[3] < 0;
    kumo_u64 carry = 1;
    for (int k = 0; k < 4; ++k) {
        kumo_u64 v = (kumo_u64) w[k];
        if (negative) { v = ~v + carry; carry = (carry && v == 0) ? 1 : 0; }
        m[k] = v;
    }
    return negative;
}

// r = a * b (r may be a); whether the product leaves 256 bits
__device__ bool kumo_multiply(const kumo_u64* a, const kumo_u64* b, kumo_u64* r) {
    kumo_u64 t[4] = {0, 0, 0, 0};
    bool over = false;
    for (int i = 0; i < 4; ++i) {
        kumo_u64 carry = 0;
        for (int j = 0; j < 4; ++j) {
            const kumo_u64 lo = a[i] * b[j], hi = __umul64hi(a[i], b[j]);
            if (i + j >= 4) { over = over || lo != 0 || hi != 0 || carry != 0; carry = 0; continue; }
            const kumo_u64 s = t[i + j] + lo;
            kumo_u64 c = s < lo;
            const kumo_u64 s2 = s + carry;
            c += s2 < carry;
            t[i + j] = s2;
            carry = hi + c;  // t + a*b + carry < 2^128: never wraps
        }
        over = over || carry != 0;
    }
    for (int k = 0; k < 4; ++k) r[k] = t[k];
    return over;
}

// m = m * 10**shift; whether it leaves 256 bits (10**77 and up does, but for 0)
__device__ bool kumo_scale(kumo_u64* m, int shift, const kumo_u64* powers) {
    if (shift == 0) return false;
    if (shift > 76) return (m[0] | m[1] | m[2] | m[3]) != 0;
    return kumo_multiply(m, &powers[4 * shift], m);
}

__device__ bool kumo_add(const kumo_u64* a, const kumo_u64* b, kumo_u64* r) {
    kumo_u64 carry = 0;
    for (int k = 0; k < 4; ++k) {
        const kumo_u64 s = a[k] + b[k];
        kumo_u64 c = s < a[k];
        const kumo_u64 s2 = s + carry;
        c += s2 < carry;
        r[k] = s2;
        carry = c;
    }
    return carry != 0;
}

__device__ int kumo_compare(const kumo_u64* a, const kumo_u64* b) {
    for (int k = 3; k >= 0; --k) if (a[k] != b[k]) return a[k] < b[k] ? -1 : 1;
    return 0;
}

// r = a - b, for a >= b
__device__ void kumo_subtract(const kumo_u64* a, const kumo_u64* b, kumo_u64* r) {
    kumo_u64 borrow = 0;
    for (int k = 0; k < 4; ++k) {
        const kumo_u64 d = a[k] - b[k];
        kumo_u64 o = a[k] < b[k];
        const kumo_u64 d2 = d - borrow;
        o += d < borrow;
        r[k] = d2;
        borrow = o;
    }
}
"""


_CONVERSION_PREAMBLE = r"""
// m /= d (d < 2**31); whether nothing was left over
__device__ bool kumo_divide_small(kumo_u64* m, kumo_u64 d) {
    kumo_u64 rest = 0;
    for (int k = 3; k >= 0; --k) {
        kumo_u64 current = (rest << 32) | (m[k] >> 32);
        const kumo_u64 high = current / d;
        rest = current % d;
        current = (rest << 32) | (m[k] & 0xffffffffULL);
        const kumo_u64 low = current / d;
        rest = current % d;
        m[k] = (high << 32) | low;
    }
    return rest == 0;
}

__device__ int kumo_length(const kumo_u64* m) {  // bits m takes
    for (int k = 3; k >= 0; --k) if (m[k]) return 64 * k + 64 - __clzll((long long) m[k]);
    return 0;
}

__device__ void kumo_store(const kumo_u64* m, bool negative, long long* out) {  // two's complement
    kumo_u64 carry = 1;
    for (int k = 0; k < 4; ++k) {
        kumo_u64 v = m[k];
        if (negative) { v = ~v + carry; carry = (carry && v == 0) ? 1 : 0; }
        out[k] = (long long) v;
    }
}
"""


def _conversions() -> tuple[Any, Any, Any]:
    """The exact conversion kernels (`rescaled`, `float_words`, `words_float`), compiled once."""

    global _CONVERSIONS
    if _CONVERSIONS is None:
        import cupy

        preamble = _PREAMBLE + _CONVERSION_PREAMBLE
        rescale = cupy.ElementwiseKernel(
            "raw int64 held, raw uint64 powers, int32 shift", "raw int64 result, bool valid",
            r"""
            kumo_u64 m[4];
            const bool negative = kumo_magnitude(&held[4 * i], m);
            bool ok = true;
            if (shift > 0) {
                ok = !kumo_scale(m, shift, &powers[0]) && kumo_length(m) <= 255;
            } else {
                for (int left = -shift; left > 0 && ok; left -= 9) {
                    kumo_u64 d = 1;
                    for (int k = 0; k < (left < 9 ? left : 9); ++k) d *= 10;
                    ok = kumo_divide_small(m, d);
                }
            }
            if (!ok) for (int k = 0; k < 4; ++k) m[k] = 0;
            kumo_store(m, negative, &result[4 * i]);
            valid = ok;
            """, "kumo_pytorch_gpu_decimal_rescale", preamble=preamble)
        to_words = cupy.ElementwiseKernel(
            "raw int64 bits, raw uint64 fives, int32 scale", "raw int64 held, bool valid",
            r"""
            const kumo_u64 b = (kumo_u64) bits[i];
            const bool negative = (b >> 63) != 0;
            const int exponent = (int) ((b >> 52) & 0x7FF);
            const kumo_u64 fraction = b & ((1ULL << 52) - 1);
            kumo_u64 r[4] = {0, 0, 0, 0};
            bool ok = exponent != 0x7FF;  // NaN and infinity are no decimal
            if (ok && (exponent != 0 || fraction != 0)) {
                kumo_u64 m = exponent ? (fraction | (1ULL << 52)) : fraction;
                int e = exponent ? exponent - 1075 : -1074;
                const int zeros = __ffsll((long long) m) - 1;
                m >>= zeros;
                e += zeros;  // the value is m * 2**e, m odd: times 10**scale, m * 5**scale * 2**(e + scale)
                const int shift = e + scale;
                if (shift < 0) {
                    ok = false;  // an odd multiple of a power of two below 2**-scale: more places than scale
                } else {
                    const kumo_u64 a[4] = {m, 0, 0, 0};
                    kumo_multiply(a, &fives[4 * scale], r);  // below 2**53 * 5**76 < 2**230
                    if (kumo_length(r) + shift > 255) {
                        ok = false;
                    } else if (shift > 0) {
                        const int words = shift / 64, rest = shift % 64;
                        for (int k = 3; k >= 0; --k) {
                            kumo_u64 v = k - words >= 0 ? r[k - words] << rest : 0;
                            if (rest && k - words - 1 >= 0) v |= r[k - words - 1] >> (64 - rest);
                            r[k] = v;
                        }
                    }
                }
            }
            if (!ok) for (int k = 0; k < 4; ++k) r[k] = 0;
            kumo_store(r, negative && ok, &held[4 * i]);
            valid = ok;
            """, "kumo_pytorch_gpu_decimal_float_words", preamble=preamble)
        to_float = cupy.ElementwiseKernel(
            "raw int64 held, int32 scale", "int64 bits, bool valid",
            r"""
            kumo_u64 m[4];
            const bool negative = kumo_magnitude(&held[4 * i], m);
            bool ok = true;
            kumo_u64 out = 0;
            if (kumo_length(m) != 0) {
                for (int left = scale; left > 0 && ok; left -= 13) {  // value = m / 10**scale: m / 5**scale first
                    kumo_u64 d = 1;
                    for (int k = 0; k < (left < 13 ? left : 13); ++k) d *= 5;
                    ok = kumo_divide_small(m, d);
                }
                if (ok) {  // value = m * 2**-scale: m, its trailing zeros off, must fit 53 bits
                    int zeros = 0;
                    for (int k = 0; k < 4; ++k) { if (m[k]) { zeros += __ffsll((long long) m[k]) - 1; break; } zeros += 64; }
                    const int words = zeros / 64, rest = zeros % 64;
                    for (int k = 0; k < 4; ++k) {
                        kumo_u64 v = k + words < 4 ? m[k + words] >> rest : 0;
                        if (rest && k + words + 1 < 4) v |= m[k + words + 1] << (64 - rest);
                        m[k] = v;
                    }
                    const int length = kumo_length(m);
                    if (length > 53) {
                        ok = false;
                    } else {
                        const int exponent = length - 1 + zeros - scale;  // within -77 .. 255: a normal float
                        const kumo_u64 fraction = (m[0] << (53 - length)) & ((1ULL << 52) - 1);
                        out = ((kumo_u64) negative << 63) | ((kumo_u64) (exponent + 1023) << 52) | fraction;
                    }
                }
            }
            bits = ok ? (long long) out : 0;
            valid = ok;
            """, "kumo_pytorch_gpu_decimal_words_float", preamble=preamble)
        _CONVERSIONS = (rescale, to_words, to_float)
    return _CONVERSIONS


def _kernels() -> tuple[Any, Any]:
    """The calculation and comparison kernels, compiled once."""

    global _KERNELS
    if _KERNELS is None:
        import cupy

        calculation = cupy.ElementwiseKernel(
            "raw int64 first, raw int64 second, raw uint64 powers, int32 operation, int32 shift_a, int32 shift_b, "
            "int32 shift_r, int32 digits",
            "raw int64 held, bool bad",
            r"""
            kumo_u64 m[4], n[4], r[4];
            bool negative = false, over = false;
            const bool mx = kumo_magnitude(&first[4 * i], m);
            bool my = kumo_magnitude(&second[4 * i], n);
            if (operation == 3) {  // abs
                for (int k = 0; k < 4; ++k) r[k] = m[k];
            } else if (operation == 2) {  // multiply
                over = kumo_multiply(m, n, r);
                negative = mx != my;
            } else {  // add, subtract: at one scale, then by sign
                over = kumo_scale(m, shift_a, &powers[0]);
                over = kumo_scale(n, shift_b, &powers[0]) || over;
                if (operation == 1) my = !my;
                if (mx == my) { over = kumo_add(m, n, r) || over; negative = mx; }
                else if (kumo_compare(m, n) >= 0) { kumo_subtract(m, n, r); negative = mx; }
                else { kumo_subtract(n, m, r); negative = my; }
            }
            over = kumo_scale(r, shift_r, &powers[0]) || over;
            bad = over || kumo_compare(r, &powers[4 * digits]) >= 0;
            kumo_u64 carry = 1;  // a negative result's two's complement (0's is 0)
            for (int k = 0; k < 4; ++k) {
                kumo_u64 v = r[k];
                if (negative) { v = ~v + carry; carry = (carry && v == 0) ? 1 : 0; }
                held[4 * i + k] = (long long) v;
            }
            """, "kumo_pytorch_gpu_decimal_calculation", preamble=_PREAMBLE)
        comparison = cupy.ElementwiseKernel(
            "raw int64 first, raw int64 second, int64 step", "int8 order",
            r"""
            const long long* x = &first[4 * i];
            const long long* y = &second[step * i];
            int result = 0;
            if (x[3] != y[3]) result = x[3] < y[3] ? -1 : 1;
            else for (int k = 2; k >= 0; --k) {
                if (x[k] != y[k]) { result = (kumo_u64) x[k] < (kumo_u64) y[k] ? -1 : 1; break; }
            }
            order = result;
            """, "kumo_pytorch_gpu_decimal_comparison", preamble=_PREAMBLE)
        _KERNELS = (calculation, comparison)
    return _KERNELS
