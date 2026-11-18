"""Shared native arrays encoding; Python handles the small request envelope."""
from dataclasses import dataclass
import json
import pyarrow as pa
import pyarrow.compute as pc

from ._phase import phase


@dataclass(frozen=True)
class EncodedRows:
    """Internal JSON fragment produced only by the native Arrow encoder."""
    data: bytes


def encode_rows(table, max_bytes):
    # Fail explicitly if a source checkout was not built; never silently time
    # the reference encoder when native encoding was requested.
    from ._native_json import encode_rows as native_encode_rows
    # Arrow's calendar formatter has implementation-specific behavior outside
    # Python's year 1..9999 range. Preserve that existing behavior (including
    # errors) with its native string kernel for unusual columns, while normal
    # dates/timestamps use the direct C++ formatter. No per-cell Python values.
    with phase('native_arrays_prepare', 'type_conversion'):
        for index, field in enumerate(table.schema):
            timestamp = pa.types.is_timestamp(field.type)
            date = pa.types.is_date32(field.type)
            if not (timestamp or date):
                continue
            column = table.column(index)
            bounds = pc.min_max(column.cast(pa.int64() if timestamp else pa.int32())).as_py()
            low, high = ((-62135596800000000, 253402300799999999) if timestamp else (-719162, 2932896))
            if bounds['min'] is not None and (bounds['min'] < low or bounds['max'] > high):
                values = pc.strftime(column, format='%Y-%m-%dT%H:%M:%S' if timestamp else '%Y-%m-%d')
                if timestamp:
                    values = pc.binary_join_element_wise(values, 'Z', '')
                table = table.set_column(index, field.name, values)
    return EncodedRows(native_encode_rows(table.__arrow_c_stream__(), max_bytes))


def dumps(payload, max_bytes):
    """Preserve json.dumps ordering/escaping while inserting encoded row arrays.

    Walk only the envelope dictionaries. Ordinary lists (including legacy row
    lists) retain the reference JSON implementation. Native rows stay bytes.
    """
    def parts(value):
        if isinstance(value, EncodedRows):
            yield value.data
        elif isinstance(value, dict):
            yield b'{'
            for i, (key, item) in enumerate(value.items()):
                if i:
                    yield b','
                yield json.dumps(key, ensure_ascii=True).encode('ascii')
                yield b':'
                yield from parts(item)
            yield b'}'
        else:
            yield json.dumps(value, separators=(',', ':'), allow_nan=False).encode('ascii')

    fragments, size = [], 0
    for fragment in parts(payload):
        size += len(fragment)
        if size > max_bytes:
            raise ValueError('Payload exceeds byte budget; reduce the workload batch')
        fragments.append(fragment)
    return b''.join(fragments)
