# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP encodings over retained Arrow tables."""
import base64

import pyarrow as pa
import pyarrow.compute as pc

from . import native_json
from ._phase import phase
from .bundle import TYPES


def encode_arrow_table(table, columns, encoding, budget, *, native_arrays=False):
    """One final scalar conversion for arrays; Arrow IPC retains typed columns."""
    if len(columns) > budget["max_columns"]:
        raise ValueError("Payload table exceeds column budget")
    with phase("normalize_wire_values", "type_conversion"):
        schema = pa.schema([pa.field(c, TYPES[s["dtype"]], nullable=s["nullable"])
                            for c, s in columns.items()])
        table = table.select(list(columns)).cast(schema)
        for field in schema:
            if not field.nullable and table[field.name].null_count:
                raise ValueError(f"Null in required Arrow field: {field.name}")
    if encoding == "arrays":
        if native_arrays:
            # Formatting and serialization are fused; charge the combined work
            # to JSON rather than pretending to split a single native call.
            with phase("native_arrays_encode", "json"):
                rows = native_json.encode_rows(table, budget["max_payload_bytes"])
            return {"format": "arrays", "columns": list(columns), "rows": rows}, 0
        with phase("arrays_conversion", "type_conversion"):
            values = []
            for field, column in zip(schema, table.columns, strict=True):
                if pa.types.is_timestamp(field.type):
                    # Arrow's %S retains the timestamp unit's fractional digits.
                    with phase("arrays_temporal_format", "type_conversion", column=field.name):
                        column = pc.binary_join_element_wise(
                            pc.strftime(column, format="%Y-%m-%dT%H:%M:%S"), "Z", "")
                elif pa.types.is_date(field.type):
                    with phase("arrays_temporal_format", "type_conversion", column=field.name):
                        column = pc.strftime(column, format="%Y-%m-%d")
                with phase("arrays_to_python", "type_conversion", column=field.name):
                    cells = column.to_pylist()
                if pa.types.is_int64(field.type):
                    with phase("arrays_integer_bounds", "type_conversion", column=field.name):
                        bounds = pc.min_max(column).as_py()
                    if ((bounds["min"] is not None and bounds["min"] < -(2**53 - 1))
                            or (bounds["max"] is not None and bounds["max"] > 2**53 - 1)):
                        with phase("arrays_large_integer_format", "type_conversion", column=field.name):
                            cells = [str(v) if v is not None and abs(v) > 2**53 - 1 else v for v in cells]
                values.append(cells)
            with phase("arrays_row_transpose", "type_conversion"):
                return {"format": "arrays", "columns": list(columns),
                        "rows": list(zip(*values, strict=True))}, 0
    return (table, table.nbytes) if encoding == "arrow" else encode_ipc_table(table)


def encode_ipc_table(table):
    """Shared IPC/base64 boundary for row and columnar inputs."""
    with phase("arrow_ipc_encode", "arrow_ipc"):
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema, options=pa.ipc.IpcWriteOptions(compression=None)) as writer:
            writer.write_table(table)
        raw = sink.getvalue().to_pybytes()
    with phase("base64_encode", "base64"):
        encoded = base64.b64encode(raw).decode()
    return {"format": "arrow_ipc", "transport": "base64", "payload_base64": encoded}, len(raw)
