"""Text columns: held as codes into sorted dictionaries, and turned back into Python or Arrow texts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    pass


#: Types held as text is, as codes into their sorted distinct values: text, and bytes (libcudf reads both as
#: strings, byte for byte, and sorts them as Python sorts ``str`` and ``bytes`` alike: by their bytes).
TEXT_TYPES = frozenset({"string", "bytes"})

def is_text(dtype: Any) -> bool:
    """Whether values of `dtype` are held as text is: `TEXT_TYPES`."""

    return dtype is not None and dtype.name in TEXT_TYPES

def arrow_texts(column: Any, dtype: Any = None) -> Any:
    """A libcudf strings column as an Arrow array: strings, or binary for `bytes` (its bytes as they are)."""

    import pyarrow as pa

    array = column.to_arrow()
    return array.cast(pa.binary()) if dtype is not None and dtype.name == "bytes" else array

def python_texts(column: Any, dtype: Any = None) -> list[Any]:
    """A libcudf strings column's values as Python ``str``, or ``bytes`` for `bytes`."""

    return arrow_texts(column, dtype).to_pylist()

def text_column(values: Sequence[Any]) -> Any:
    """Python ``str`` or ``bytes`` values (None for NULL) as a libcudf strings column, byte for byte."""

    import pyarrow as pa

    binary = any(isinstance(value, bytes) for value in values)
    return arrow_column(pa.array(values, pa.binary() if binary else pa.string()))

def arrow_column(array: Any) -> Any:
    """An Arrow string or binary array as a libcudf strings column, byte for byte (libcudf takes no binary
    type: a binary array's buffers are read as strings)."""

    import pyarrow as pa
    import pylibcudf as plc

    if pa.types.is_binary(array.type):
        array = pa.Array.from_buffers(pa.string(), len(array), array.buffers(), null_count=array.null_count,
                                      offset=array.offset)
    return plc.Column.from_arrow(array)

def byte_order(value: Any) -> bytes:
    """The bytes a text (``str``, as UTF-8) or ``bytes`` value sorts by, as `_with_codes` sorts a column."""

    return value if isinstance(value, bytes) else value.encode()

def text_codes(texts: Any, dictionary: Any) -> Any:
    """The code of each text of `texts` (a libcudf strings column without NULLs) in `dictionary` (sorted distinct
    texts, as `_with_codes` builds them), -1 where `dictionary` does not hold it; CuPy int32.

    Texts are equal when their bytes are, as Python compares the strings upstream reads.
    """

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _view

    size = texts.size()
    if size == 0 or dictionary.size() == 0:
        return cupy.full(size, -1, dtype=cupy.int32)
    bounds = (plc.Table([dictionary]), plc.Table([texts]), [plc.types.Order.ASCENDING], [plc.types.NullOrder.AFTER])
    first, last = _view(plc.search.lower_bound(*bounds)), _view(plc.search.upper_bound(*bounds))
    return cupy.where(last > first, first, -1).astype(cupy.int32)  # held: its one place, between its bounds

def texts_at(dictionary: Any, codes: Any, dtype: Any = None) -> list[Any]:
    """The texts of `dictionary` at `codes` (CuPy), as Python strings (``bytes`` for `bytes`): a host copy of
    those texts only."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    if codes.size == 0:
        return []
    picked = plc.copying.gather(plc.Table([dictionary]), _column(cupy.ascontiguousarray(codes.astype(cupy.int32))),
                                plc.copying.OutOfBoundsPolicy.DONT_CHECK)
    return python_texts(picked.columns()[0], dtype)
