# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
from test_batch import request, workload

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/local_sdm"))
from adapter import _compatible_buffers, arrow_to_sdm  # noqa: E402


def test_sdm_bridge_uses_public_arrow_conversion(monkeypatch):
    calls = []

    class TableTensor:
        def __init__(self, table, stypes):
            self.table, self.stypes = table, stypes

        @classmethod
        def from_arrow(cls, table, *, stypes, device):
            assert device == "cpu"
            calls.append(table)
            return cls(table, stypes)

        def drop_columns(self, name):
            return TableTensor(self.table.drop([name]), {
                key: value for key, value in self.stypes.items() if key != name})

        def __getitem__(self, names):
            return TableTensor(self.table.select(names), {
                key: self.stypes[key] for key in names})

    class RelatedTables:
        def __init__(self, *, tables, relationships, task_links):
            self.tables, self.relationships, self.task_links = tables, relationships, task_links

    monkeypatch.setitem(sys.modules, "sdm", SimpleNamespace(
        TableTensor=TableTensor, RelatedTables=RelatedTables))
    batch = request(workload())
    source = batch.tables("context")["related_tables"]["events"]
    result = arrow_to_sdm(batch)
    assert len(calls) == 6
    assert source.equals(batch.tables("context")["related_tables"]["events"])
    assert result["y_context"].stypes == {"__fm_target": "categorical"}
    assert "__fm_target" not in result["x_query"].stypes
    assert result["related_context_tables"].relationships[0]["left_columns"] == [
        "instance_id", "__fm_edge_event_user"]
    assert result["related_query_tables"].task_links[0]["task_columns"] == [
        "instance_id", "__fm_entity_key"]


def test_buffer_compatibility_only_repairs_incompatible_columns():
    writable = pa.table({"name": pa.array(["a", None], type=pa.string())})
    assert _compatible_buffers(writable) is writable
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, writable.schema) as writer:
        writer.write_table(writable)
    read_only = pa.ipc.open_stream(sink.getvalue()).read_all()
    assert not read_only["name"].chunk(0).buffers()[2].is_mutable
    repaired = _compatible_buffers(read_only)
    assert repaired.equals(read_only)
    assert repaired["name"].chunk(0).buffers()[2].is_mutable
