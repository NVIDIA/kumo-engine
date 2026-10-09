# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from diskgraph._c import DiskgraphConfigError, DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError

import json
import pyarrow as pa
import pyarrow.parquet as pq
import diskgraph as diskgraph_py
from diskgraph import DiskgraphConfigError

def test_export_tt_artifacts_pybind(tmp_path_factory, ifood_graph):
    """
    Tests the PyBind exposure of the export_tt_artifacts Rust utility using
    the standard ifood mock index.
    """
    root = tmp_path_factory.mktemp("export_test")

    # Create mock train table pointing to users and merchants in the ifood graph
    tt_dir = root / "train_table"
    tt_dir.mkdir()
    tt_path = str(tt_dir / "part-00000.parquet")
    schema = pa.schema([
        ("arange", pa.int64()),
        ("USER_FKEY", pa.string()),
        ("MERCHANT_FKEY", pa.string())
    ])

    # 3 rows:
    # 0: u1, m1 (Both valid)
    # 1: u2, m_invalid (One valid)
    # 2: u3, None (One valid, one null)
    table = pa.Table.from_arrays([
        pa.array([100, 101, 102]),
        pa.array(["u1", "u2", "u3"]),
        pa.array(["m1", "m_invalid", None])
    ], schema=schema)

    pq.write_table(table, tt_path)

    output_dir = str(root / "output")

    config = diskgraph_py.ExportConfig(
        train_table_path=str(tt_dir),
        index_path=ifood_graph,
        output_dir=output_dir,
        fkeys=[
            diskgraph_py.FkeyConfig("USER_FKEY", "users"),
            diskgraph_py.FkeyConfig("MERCHANT_FKEY", "merchants")
        ]
    )

    # Call the PyBind function!
    diskgraph_py.export_tt_artifacts(config)
    # Verification happens by asserting no exception was thrown,
    # but let's do a quick schema sanity check on the output.
    mask_table = pq.read_table(f"{output_dir}/custom_train_table_mask.parquet")
    assert mask_table.num_rows == 3

    # Row 0 is the only fully valid row
    assert mask_table.column("__mask__")[0].as_py() == True
    assert mask_table.column("__mask__")[1].as_py() == False
    assert mask_table.column("__mask__")[2].as_py() == False


def test_export_tt_artifacts_pybind_multi_file(tmp_path_factory, ifood_graph):
    """
    Tests the PyBind exposure of the export_tt_artifacts Rust utility using
    multiple parquet files in a directory.
    """
    root = tmp_path_factory.mktemp("export_test_multi")

    tt_dir = root / "train_table_dir"
    tt_dir.mkdir()

    schema = pa.schema([
        ("arange", pa.int64()),
        ("USER_FKEY", pa.string()),
        ("MERCHANT_FKEY", pa.string())
    ])

    # File 1: 2 rows
    table1 = pa.Table.from_arrays([
        pa.array([100, 101]),
        pa.array(["u1", "u2"]),
        pa.array(["m1", "m_invalid"])
    ], schema=schema)
    pq.write_table(table1, str(tt_dir / "part-00000.parquet"))

    # File 2: 1 row
    table2 = pa.Table.from_arrays([
        pa.array([102]),
        pa.array(["u3"]),
        pa.array([None])
    ], schema=schema)
    pq.write_table(table2, str(tt_dir / "part-00001.parquet"))

    output_dir = str(root / "output")

    config = diskgraph_py.ExportConfig(
        train_table_path=str(tt_dir),
        index_path=ifood_graph,
        output_dir=output_dir,
        fkeys=[
            diskgraph_py.FkeyConfig("USER_FKEY", "users"),
            diskgraph_py.FkeyConfig("MERCHANT_FKEY", "merchants")
        ]
    )

    diskgraph_py.export_tt_artifacts(config)

    mask_table = pq.read_table(f"{output_dir}/custom_train_table_mask.parquet")
    assert mask_table.num_rows == 3

    assert mask_table.column("__mask__")[0].as_py() == True
    assert mask_table.column("__mask__")[1].as_py() == False
    assert mask_table.column("__mask__")[2].as_py() == False
