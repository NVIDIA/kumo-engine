# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire encodings over normalized, retained Arrow tables."""
import base64

import pyarrow as pa

from . import native_json


def encode_arrow_table(table, encoding, max_bytes):
    """Encode tables normalized once when their PayloadBatch was constructed."""
    if encoding == "arrays":
        return {"format": "arrays", "columns": table.column_names,
                "rows": native_json.encode_rows(table, max_bytes)}
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema, options=pa.ipc.IpcWriteOptions(compression=None)) as writer:
        writer.write_table(table)
    return {"format": "arrow_ipc", "transport": "base64",
            "payload_base64": base64.b64encode(sink.getvalue().to_pybytes()).decode()}
