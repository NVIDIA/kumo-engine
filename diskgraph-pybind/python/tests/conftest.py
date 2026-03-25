import pytest
from diskgraph._c import DiskgraphConfigError, DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError

import os
import json
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq

@pytest.fixture(scope="session")
def ifood_graph(tmp_path_factory):
    """
    Generates a portable mock index that mimics the ifood graph structure
    required by the sampling tests, removing the dependency on external data.
    """
    root = tmp_path_factory.mktemp("mock_ifood")
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    # 1. Create mock Parquet data
    users_schema = pa.schema([
        ("user_id", pa.string()),
        ("age", pa.int64())
    ])
    users_table = pa.Table.from_arrays(
        [pa.array(["u1", "u2", "u3"]), pa.array([25, 30, 35])],
        schema=users_schema
    )
    users_path = str(data_dir / "users.parquet")
    pq.write_table(users_table, users_path)

    orders_schema = pa.schema([
        ("order_id", pa.string()),
        ("user_id", pa.string()),
        ("merchant_id", pa.string()),
        ("timestamp", pa.int64())
    ])
    orders_table = pa.Table.from_arrays([
        pa.array(["o1", "o2", "o3"]),
        pa.array(["u1", "u2", "u1"]),
        pa.array(["m1", "m2", "m1"]),
        pa.array([1609459200000000, 1609545600000000, 1609632000000000])
    ], schema=orders_schema)
    orders_path = str(data_dir / "orders.parquet")
    pq.write_table(orders_table, orders_path)

    merchants_schema = pa.schema([("merchant_id", pa.string())])
    merchants_table = pa.Table.from_arrays([pa.array(["m1", "m2"])], schema=merchants_schema)
    merchants_path = str(data_dir / "merchants.parquet")
    pq.write_table(merchants_table, merchants_path)

    # 2. Create DiskGraph Config mimicking ifood schema
    config = {
        "dataset_name": "mock_ifood",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "age", "ingested_dtype": "int64"}
                ],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [users_path]}
            },
            {
                "name": "merchants",
                "columns": [{"name": "merchant_id", "ingested_dtype": "string"}],
                "primary_key": "merchant_id",
                "source": {"type": "parquet", "paths": [merchants_path]}
            },
            {
                "name": "orders_parquet",
                "columns": [
                    {"name": "order_id", "ingested_dtype": "string"},
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "merchant_id", "ingested_dtype": "string"},
                    {"name": "timestamp", "ingested_dtype": "timestamp"}
                ],
                "primary_key": "order_id",
                "create_time": "timestamp",
                "fkeys": [
                    {"src_column": "user_id", "dst_table": "users", "reverse": {}},
                    {"src_column": "merchant_id", "dst_table": "merchants", "reverse": {}}
                ],
                "source": {"type": "parquet", "paths": [orders_path]}
            }
        ]
    }

    config_path = root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # 3. Build the Index using cargo run (assumes we are in the workspace)
    try:
        # Find cargo project root (assuming tests are run from diskgraph-pybind or root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        print(f"\nBuilding mock index at {index_dir} using cargo run...")
        subprocess.run(
            ["cargo", "run", "--release", "--bin", "ingest", "--",
             "--source", str(config_path),
             "--dest", "local",
             "--output", str(index_dir)],
            cwd=project_root,
            check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Index ingestion failed with exit code {e.returncode}")

    return str(index_dir)
