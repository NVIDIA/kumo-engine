import pytest
import json
import pyarrow as pa
import pyarrow.parquet as pq
import diskgraph as diskgraph_py

def test_export_tt_artifacts_pybind(tmp_path_factory, ifood_graph):
    """
    Tests the PyBind exposure of the export_tt_artifacts Rust utility using
    the standard ifood mock index.
    """
    root = tmp_path_factory.mktemp("export_test")

    # Create mock train table pointing to users and merchants in the ifood graph
    tt_path = str(root / "train_table.parquet")
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
        train_table_path=tt_path,
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
