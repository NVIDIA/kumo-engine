"""Row filters and comparisons against a constant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
from test.kumo_pytorch.reference_engine.gpu.data.text import text_column
if TYPE_CHECKING:

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable


def _equals(operand: Values, value: Any) -> Any:
    """Which of `operand`'s values equal the Python `value` (texts by their code; floats as IEEE compares)."""

    if operand.texts is not None:
        codes, code = _text_codes(operand.texts, operand.values, value)
        return codes == code
    if operand.dtype.name.startswith("decimal"):  # exactly, at its scale (map members are normalized to it)
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import compare, constant_words, words

        return compare(words(operand), constant_words(value, operand.dtype.scale)) == 0
    return operand.values == _scalar(operand.dtype, value)

def _scalar(dtype: Any, value: Any) -> Any:
    """A Python value as the number its column holds on the GPU."""

    import datetime as dt

    if dtype.name == "timestamp":
        return int(value.timestamp())
    if dtype.name == "date32":
        return (value - dt.date(1970, 1, 1)).days
    return value

def _text_codes(texts: Any, codes: Any, text: Any) -> tuple[Any, float]:
    """Codes and a code-space stand-in for `text` (a ``str``, or ``bytes`` for a bytes column): equal codes match
    it, ordering is the texts'."""

    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view
    from test.kumo_pytorch.reference_engine.gpu.data.text import text_codes

    probe = text_column([text])
    low = int(_view(plc.search.lower_bound(plc.Table([texts]), plc.Table([probe]), [plc.types.Order.ASCENDING],
                                           [plc.types.NullOrder.AFTER]))[0])
    exact = int(text_codes(probe, texts)[0]) >= 0
    return codes.astype("float64"), float(low) if exact else low - 0.5

def _filter(table: GpuTable, condition: Any, rows: Any) -> Any:
    """Which of `rows` satisfy a `RowFilter`, with upstream's NULL rules."""

    import cupy

    column, operation = condition.column, condition.operation
    dtype = table.dtypes[column]
    present = table.present(column)
    valid = cupy.ones(rows.size, dtype=bool) if present is None else present[rows]
    if operation == "is_null":
        return ~valid
    if operation == "not_null":
        return valid
    texts = table.dictionaries.get(column)
    if dtype.name.startswith("decimal"):  # exactly: each row's order against the operand's
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import compare, constant_words, gathered_words

        held = gathered_words(table.columns[column], rows)

        def against(operand: Any) -> tuple[Any, Any]:
            return compare(held, constant_words(operand, dtype.scale)), 0
    else:
        values = table.values(column)[rows]

        def against(operand: Any) -> tuple[Any, Any]:
            if texts is not None:
                return _text_codes(texts, values, operand)
            return values, _scalar(dtype, operand)

    def equal_to(operand: Any) -> Any:
        if operand is None:
            return ~valid
        left, right = against(operand)
        return valid & (left == right)

    if operation in ("in", "not_in"):
        operands = tuple(dtype.normalize(value, path=f"filter.{column}") for value in condition.value)
        matched = cupy.zeros(rows.size, dtype=bool)
        for operand in operands:
            matched |= equal_to(operand)
        return matched if operation == "in" else ~matched
    operand = dtype.normalize(condition.value, path=f"filter.{column}")
    if operation == "equal":
        return equal_to(operand)
    if operation == "not_equal":
        return ~equal_to(operand)
    left, right = against(operand)
    ordered = {"greater_than": cupy.greater, "greater_or_equal": cupy.greater_equal, "less_than": cupy.less,
               "less_or_equal": cupy.less_equal}[operation]
    return valid & ordered(left, right)
