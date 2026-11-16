# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.sources.types import ReadMode, StagedTable
from kumo_databricks.backend import DatabricksSourceBackend


@pytest.mark.parametrize("rows", [0, 3])
def test_zero_row_read_is_an_empty_source(rows, tmp_path, monkeypatch):
    part = tmp_path / "part-0.parquet"
    part.write_bytes(b"schema only")
    staged = StagedTable(directory=str(tmp_path), paths=(str(part), ),
                         version="v1", source_size=1,
                         mode_used=ReadMode.STREAMED, num_rows=rows)
    monkeypatch.setattr("kumo_core.sources.prepare.read_table",
                        lambda *args, **kwargs: staged)
    backend = DatabricksSourceBackend()
    for name in ("_declared", "_reader"):
        monkeypatch.setattr(backend, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(backend, "_bare", lambda connector, table: table)
    monkeypatch.setattr(backend, "_connected",
                        lambda connector: nullcontext(None))
    connector = SimpleNamespace(id="c")

    def prepare():
        return backend.prepare_ingest(connector, "t", str(tmp_path),
                                      dtypes={})

    if rows:
        assert prepare().source["paths"] == [str(part)]
    else:
        with pytest.raises(PermanentSourceError, match="no rows"):
            prepare()
