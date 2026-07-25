from diskgraph import DiskgraphConfigConverter


def test_converter_accepts_direct_databricks_source_without_parquet():
    config = {
        "name": "orders_graph",
        "tables": [{
            "table_name": "orders",
            "data_source": "databricks",
            "source_name": "orders",
            "data_version": "v1",
            "pkey": {
                "name": "id",
                "dtype": "int64",
                "stype": "ID",
            },
            "cols": [],
        }],
    }
    source = {
        "type": "databricks",
        "server_hostname": "workspace.cloud.databricks.com",
        "warehouse_id": "warehouse",
        "catalog": "catalog",
        "schema": "schema",
        "sql": "SELECT `id` FROM `catalog`.`schema`.`orders`",
        "auth_profile": "kumo_connector",
    }

    converted = DiskgraphConfigConverter(
        config,
        "/path/that/does/not/exist",
        {"orders": "v1"},
        source_specs={
            "orders": {
                "source": source,
                "source_size": 123_456,
            },
        },
    ).convert()

    assert converted["tables"][0]["source"] == source
    assert converted["tables"][0]["source_size"] == 123_456
