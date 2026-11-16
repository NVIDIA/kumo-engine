"""Keys of every type as int64 identities: equal exactly when upstream's Python values are.

A relation and an anchor lookup use Python value equality: two keys
match when their Python values are equal. Here every key column gets one int64 identity per row
(`identities`), and a key of one column is translated into another column's identities (`translate`) when the
two can be equal:

- integers: their values (any widths compare by value);
- booleans, dates (days) and timestamps (whole seconds): their stored integers;
- floats: their bits, -0.0 as 0.0 (they are equal); NaN equals nothing, so a NaN key matches no row;
- text and bytes: their codes among the column's sorted distinct values (`test.kumo_pytorch.reference_engine.gpu.data.tables`), a code translated
  by the bytes it stands for; text never equals bytes;
- decimals: their place among the column's sorted distinct values (`test.kumo_pytorch.reference_engine.gpu.data.decimals`), translated by value.

Keys whose identities line up (`comparable`) are matched directly: one type, integers and booleans of any
widths (but uint64 with a signed type, whose values above 2**63 would wrap), float32 and float64 (their float64
bits), decimals of one scale. A date never equals a timestamp, nor text bytes, nor either a number: those keys
match no row, as in upstream. Numbers of other types, which Python compares exactly (1 == 1.0 == True ==
Decimal('1.00'); 0.5 == Decimal('0.50'); 0.1 != Decimal('0.1'); 2**53 + 1 != float(2**53 + 1)), are converted
into the other column's type exactly, on the GPU (`Translation`): a key with no equal value of that type
matches no row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

__all__ = ["KeySpace", "Translation", "comparable", "identities", "key_space", "never_equal", "python_identity",
           "translate"]

#: Families whose values Python can find equal across them (numbers), and the others, each equal only within.
_NUMBERS = "number"


def family(dtype: Any) -> str:
    """The family of values a key of `dtype` holds: numbers (integers, booleans, floats, decimals) compare across
    their types in Python; text, bytes, dates and timestamps each only among themselves."""

    if dtype.integer or dtype.name == "bool" or dtype.name.startswith(("float", "decimal")):
        return _NUMBERS
    return dtype.name


def never_equal(left: Any, right: Any) -> bool:
    """Whether no key of `left` can equal a key of `right` (Python's equality): different families."""

    return family(left) != family(right)


def _whole(dtype: Any) -> bool:
    return dtype.integer or dtype.name == "bool"


def comparable(left: Any, right: Any) -> bool:
    """Whether keys of `left` and `right` have identities that line up (compared directly, or through a
    dictionary): one type; integers and booleans of any widths, but uint64 with a signed type; float32 and
    float64; decimals of one scale."""

    if left == right:
        return True
    if _whole(left) and _whole(right):
        signed = {"int8", "int16", "int32", "int64"}
        return not ({left.name, right.name} & {"uint64"} and {left.name, right.name} & signed)
    if left.name.startswith("float") and right.name.startswith("float"):
        return True
    return left.name.startswith("decimal") and right.name.startswith("decimal") and left.scale == right.scale


class KeySpace:
    """One key column as identities: `at` gives them (CuPy int64) for some rows, `matchable_at` which of those can
    match (CuPy bool, or None for all), and `dictionary` holds the sorted distinct values identities index (a
    text or bytes column's texts, a decimal column's words), or None where an identity is the value itself.
    Identities are worked out for the rows asked for only, but a decimal column's, kept: placing a decimal
    among the column's distinct values takes the whole column."""

    def __init__(self, table: GpuTable, column: str) -> None:
        self.table, self.column, self.dtype = table, column, table.dtypes[column]
        self.dictionary, self._places, self._searched = None, None, None
        if self.dtype.name in ("string", "bytes"):
            self.dictionary = table.dictionaries[column]
        elif self.dtype.name.startswith("decimal"):
            from test.kumo_pytorch.reference_engine.gpu.data.decimals import column_words, distinct_index

            self._places, self.dictionary = distinct_index(column_words(table.columns[column]))

    def at(self, rows: Any = None) -> Any:
        """The identities of `rows` (CuPy; every row when None), CuPy int64."""

        import cupy

        if self._places is not None:
            return self._places if rows is None else self._places[rows]
        values = self.table.values(self.column)
        values = values if rows is None else values[rows]
        if self.dtype.name.startswith("float"):
            return _float_identities(values)
        if values.dtype == cupy.uint64:  # its bits: 2**63 and up as negative int64s, one to one
            return values.view(cupy.int64)
        return values.astype(cupy.int64, copy=False)

    def searched(self) -> Any:
        """A decimal key's distinct values as `_search_table` gives them, made once and kept: keys of other number
        types are found among them each call (`Translation`)."""

        if self._searched is None:
            self._searched = _search_table(self.dictionary)
        return self._searched

    def values_at(self, rows: Any = None) -> Any:
        """The values of `rows` as held (CuPy; every row when None): numbers, or a decimal's words (its
        dictionary's)."""

        if self._places is not None:
            return self.dictionary if rows is None else self.dictionary[self._places[rows]]
        values = self.table.values(self.column)
        return values if rows is None else values[rows]

    def matchable_at(self, rows: Any = None) -> Any:
        """Which of `rows` (every row when None) hold a key that can match: not NULL, not NaN; None for all."""

        import cupy

        present = self.table.present(self.column)
        if present is not None and rows is not None:
            present = present[rows]
        if not self.dtype.name.startswith("float"):
            return present
        values = self.table.values(self.column)
        number = ~cupy.isnan(values if rows is None else values[rows])
        return number if present is None else present & number


def key_space(table: GpuTable, column: str) -> KeySpace:
    """`table.column` as a key (`KeySpace`), made once and kept on the table."""

    cache = table.__dict__.setdefault("_key_spaces", {})
    if column not in cache:
        cache[column] = KeySpace(table, column)
    return cache[column]


def identities(table: GpuTable, column: str) -> tuple[Any, Any]:
    """`table.column`'s identities (CuPy int64 per row) and which rows can match (CuPy bool, or None for all)."""

    space = key_space(table, column)
    return space.at(), space.matchable_at()


_WIDEN: Any = None


def widened(values: Any) -> Any:
    """Floats (CuPy) as float64, exactly: a float32 by its bits, with integer arithmetic. CuPy's float32
    operations (``astype``, ``==``, ``unique``) flush subnormals (1e-40f) to 0, which Python's exact widening never
    does; float64's do not."""

    import cupy

    global _WIDEN
    if values.dtype != cupy.float32:
        return values.astype(cupy.float64, copy=False)
    if _WIDEN is None:
        _WIDEN = cupy.ElementwiseKernel(
            "uint32 b", "int64 out",
            r"""
            const unsigned long long sign = (unsigned long long) (b >> 31) << 63;
            const int e = (int) ((b >> 23) & 0xFF);
            unsigned int m = b & 0x7FFFFF;
            if (e == 0xFF) {
                out = (long long) (sign | (0x7FFULL << 52) | ((unsigned long long) m << 29));
            } else if (e == 0) {
                if (m == 0) {
                    out = (long long) sign;
                } else {  // m * 2**-149: its leading bit to bit 23
                    const int shift = __clz(m) - 8;
                    m = (m << shift) & 0x7FFFFF;
                    out = (long long) (sign | ((unsigned long long) (1023 - 126 - shift) << 52)
                                       | ((unsigned long long) m << 29));
                }
            } else {
                out = (long long) (sign | ((unsigned long long) (e - 127 + 1023) << 52) | ((unsigned long long) m << 29));
            }
            """, "kumo_pytorch_gpu_widen_float32")
    return _WIDEN(cupy.ascontiguousarray(values).view(cupy.uint32)).view(cupy.float64)


def _float_identities(values: Any) -> Any:
    """Floats' identities: their float64 bits (a float32's exact widening), -0.0 as 0.0."""

    import cupy

    numbers = widened(values)
    return cupy.where(numbers == 0, cupy.int64(0), numbers.view(cupy.int64))


class Translation:
    """How keys of `source` are found among `destination`'s identities: `keys(rows)` gives the identities of
    source rows `rows` (CuPy int64) and which can match (CuPy bool, or None for all). Comparable keys translate
    directly or through their dictionaries (`translate`); numbers of other types are converted into the
    destination's type exactly (`_converted`): a key with no equal value of that type (1.5 for an integer key,
    2**53 + 1 for a float one, 0.1 for a decimal one) matches no row. A decimal source is converted once, its
    distinct values (kept per pair of columns); other sources, the rows asked for, each time."""

    def __init__(self, source: KeySpace, destination: KeySpace) -> None:
        self.source, self.destination = source, destination
        self.direct = comparable(source.dtype, destination.dtype)
        self._map, self._found = None, None
        if self.direct:  # its map only (-1 where the destination holds no equal value), as kept before
            self._map = translate(source, destination)
        elif source.dictionary is not None:  # a decimal: its distinct values, once, and which have an equal
            self._map, self._found = _converted(source.dictionary, source.dtype, destination)

    def keys(self, rows: Any = None) -> tuple[Any, Any]:
        import cupy

        if self._map is not None:
            places = self.source.at(rows)
            keys = self._map[places].astype(cupy.int64)  # one type: -1 where none is equal, which no place is
            return keys, (None if self._found is None else self._found[places])
        if self.direct:
            return self.source.at(rows), None
        return _converted(self.source.values_at(rows), self.source.dtype, self.destination)


def _converted(values: Any, dtype: Any, destination: KeySpace) -> tuple[Any, Any]:
    """Numbers `values` of `dtype` (CuPy; a decimal's words) as identities of `destination`, a number key of
    another type, exactly: (CuPy int64 identities, CuPy bool: which have an equal value of that type)."""

    form, valid = exact_form(values, dtype, destination.dtype)
    if destination.dtype.name.startswith("decimal"):  # words at the destination's scale: their places
        places = _word_places(form, destination.dictionary, destination.searched())
        return places, valid & (places >= 0)
    return form, valid


def exact_form(values: Any, dtype: Any, target: Any) -> tuple[Any, Any]:
    """Numbers `values` of `dtype` (CuPy; a decimal's words) in the form `target`'s keys are compared in, exactly,
    and which have an equal value of `target` (CuPy bool): two keys are equal exactly when their forms are.
    The form of an integer or boolean target is an integer identity (CuPy int64; uint64's bits), of a float
    target the float64 bits (-0.0 as 0.0; NaN has none), of a decimal target the words at its scale ((n, 4)).
    `target` may be `dtype` itself: the column's own form."""

    import cupy

    from test.kumo_pytorch.reference_engine.gpu.data.decimals import float_words, integer_words, rescaled, words_float

    decimal, floating = dtype.name.startswith("decimal"), dtype.name.startswith("float")
    if target.name.startswith("decimal"):
        if decimal:
            return rescaled(values, target.scale - dtype.scale)
        if floating:
            return float_words(values, target.scale)
        return rescaled(integer_words(_as_integers(values)), target.scale)
    if target.name.startswith("float"):
        if floating:
            numbers = widened(values)
            return _float_identities(numbers), ~cupy.isnan(numbers)
        if decimal:
            numbers, valid = words_float(values, dtype.scale)
        else:
            numbers, valid = _integer_floats(_as_integers(values))
        return _float_identities(numbers), valid
    # an integer or boolean target: integers in its range, as identities (uint64's bits)
    unsigned = target.name == "uint64"
    if decimal:
        held, valid = rescaled(values, -dtype.scale)
        low = held[:, 0]
        if unsigned:
            valid &= (held[:, 1:] == 0).all(axis=1)
        else:
            valid &= (held[:, 1:] == (low >> 63)[:, None]).all(axis=1)
        return cupy.where(valid, low, 0), valid
    if floating:
        numbers = widened(values)
        whole = cupy.isfinite(numbers) & (numbers == cupy.trunc(numbers))
        if unsigned:
            valid = whole & (numbers >= 0) & (numbers < 2.0**64)
            held = cupy.where(valid, numbers, 0).astype(cupy.uint64).view(cupy.int64)
        else:
            valid = whole & (numbers >= -(2.0**63)) & (numbers < 2.0**63)
            held = cupy.where(valid, numbers, 0).astype(cupy.int64)
        return held, valid
    integers = _as_integers(values)
    if integers.dtype == cupy.uint64:  # its bits; beside a signed target, only those below 2**63
        return integers.view(cupy.int64), (cupy.ones(integers.size, dtype=bool) if unsigned
                                           else integers < cupy.uint64(2**63))
    return integers.astype(cupy.int64), (integers >= 0 if unsigned else cupy.ones(integers.size, dtype=bool))


def _as_integers(values: Any) -> Any:
    """Integer or boolean values (CuPy) as integers: booleans as int64 0 and 1."""

    import cupy

    return values.astype(cupy.int64) if values.dtype == cupy.bool_ else values


def _integer_floats(integers: Any) -> tuple[Any, Any]:
    """Integers (CuPy) as float64, and which equal their float exactly (2**53 + 1 does not)."""

    import cupy

    numbers = integers.astype(cupy.float64)
    if integers.dtype == cupy.uint64:
        inside = numbers < 2.0**64
        back = cupy.where(inside, numbers, 0).astype(cupy.uint64)
    else:
        inside = numbers < 2.0**63
        back = cupy.where(inside, numbers, 0).astype(cupy.int64)
    return numbers, inside & (back == integers)


def translate(source: KeySpace, destination: KeySpace) -> Any:
    """A map from `source`'s identities to `destination`'s (CuPy integers indexed by a source identity, -1 where
    the destination holds no equal value; a text's int32 codes, as kept), for keys with dictionaries; None where
    identities are the values themselves and compare directly. Only for `comparable` types."""

    if source.dictionary is None:
        return None
    if source.dtype.name in ("string", "bytes"):
        from test.kumo_pytorch.reference_engine.gpu.data.text import text_codes

        return text_codes(source.dictionary, destination.dictionary)  # int32, as kept: cast where used
    return _word_places(source.dictionary, destination.dictionary)


def _word_places(words: Any, dictionary: Any, searched: Any = None) -> Any:
    """Each decimal of `words` by its place in `dictionary` (sorted distinct), -1 where absent; `searched` is
    `dictionary` as `_search_table` gives it, when kept."""

    import cupy

    if words.shape[0] == 0 or dictionary.shape[0] == 0:
        return cupy.full(words.shape[0], -1, dtype=cupy.int64)
    first = _word_bound(dictionary, words, "left", searched)
    found = first < dictionary.shape[0]
    at = cupy.minimum(first, dictionary.shape[0] - 1)
    found &= (dictionary[at] == words).all(axis=1)
    return cupy.where(found, first, -1).astype(cupy.int64)


def _search_table(held: Any) -> Any:
    """Decimals (words) as a libcudf table that sorts and searches in value order: the high word, then the low
    ones with their top bit flipped (they compare unsigned; flipped, as signed)."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    flip = cupy.int64(-(2**63))
    columns = [held[:, 3], held[:, 2] ^ flip, held[:, 1] ^ flip, held[:, 0] ^ flip]
    return plc.Table([_column(cupy.ascontiguousarray(column)) for column in columns])


def _word_bound(dictionary: Any, words: Any, side: str, searched: Any = None) -> Any:
    """`searchsorted` over decimals in value order: the place of each of `words` in sorted `dictionary`."""

    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view

    order, nulls = [plc.types.Order.ASCENDING] * 4, [plc.types.NullOrder.AFTER] * 4
    search = plc.search.lower_bound if side == "left" else plc.search.upper_bound
    return _view(search(searched if searched is not None else _search_table(dictionary), _search_table(words),
                        order, nulls)).astype("int64")


def python_identity(value: Any, space: KeySpace) -> int | None:
    """A Python value, already of `space`'s type (upstream's `normalize`), as an identity in `space`; None when it
    can match no key (NaN, or a text or decimal the column does not hold)."""

    import datetime as dt
    import math
    import struct

    dtype = space.dtype
    if dtype.name.startswith("float"):
        if math.isnan(value):
            return None
        return 0 if value == 0 else struct.unpack("<q", struct.pack("<d", value))[0]
    if dtype.name == "uint64" and value >= 2**63:
        return value - 2**64  # its bits, as `KeySpace.at` gives them
    if dtype.name == "date32":
        return (value - dt.date(1970, 1, 1)).days
    if dtype.name == "timestamp":
        return int((value - dt.datetime(1970, 1, 1, tzinfo=dt.UTC)).total_seconds())
    if dtype.name in ("string", "bytes") or dtype.name.startswith("decimal"):
        return None  # looked up in the column's dictionary: `dictionary_identities`
    return int(value)


def dictionary_identities(values: list[Any], space: KeySpace) -> Any:
    """Python text, bytes or decimal values (of `space`'s type) as identities in `space` (CuPy int64, -1 where the
    column holds no equal value)."""

    import cupy

    if not values:
        return cupy.zeros(0, dtype=cupy.int64)
    if space.dtype.name in ("string", "bytes"):
        from test.kumo_pytorch.reference_engine.gpu.data.text import text_codes, text_column

        return text_codes(text_column(values), space.dictionary).astype(cupy.int64)
    from test.kumo_pytorch.reference_engine.gpu.data.decimals import constant_words

    held = cupy.concatenate([constant_words(value, space.dtype.scale) for value in values])
    return _word_places(held, space.dictionary)
