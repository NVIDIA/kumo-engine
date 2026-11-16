# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom SQL snapshots against live warehouses; skipped without credentials.

``KUMO_SNOWFLAKE_CREDS`` (account, user, private_key or password) reads the
CI test account's financial tables. ``KUMO_DATABRICKS_CREDS`` is the file the
Databricks live conformance tier reads. ``KUMO_BIGQUERY_CREDS`` is a service
account key; BigQuery reads public data billed to that key's project.
"""

import json
import os

import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import PermanentSourceError
from kumoapi.graph import Edge

from kumo.engine import (
    BigQueryConnector,
    BigQueryIdentity,
    DatabricksConnector,
    DatabricksIdentity,
    Graph,
    SnowflakeConnector,
    SnowflakeIdentity,
    Table,
)


def _secrets(variable):
    path = os.environ.get(variable)
    if not path or not os.path.exists(path):
        pytest.skip(f"{variable} is not set")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _snowflake():
    found = _secrets("KUMO_SNOWFLAKE_CREDS")
    return SnowflakeConnector(
        warehouse="WH_XS", database="KUMO", schema_name="FINANCIAL_PYTEST",
        identity=SnowflakeIdentity(account=found["account"],
                                   username=found["user"],
                                   password=found.get("password"),
                                   private_key=found.get("private_key")))


def _databricks():
    found = _secrets("KUMO_DATABRICKS_CREDS")
    return DatabricksConnector(
        catalog="samples", schema_name="tpch",
        warehouse_id=found["warehouse_id"],
        identity=DatabricksIdentity(host=found["host"],
                                    client_id=found["client_id"],
                                    client_secret=found["client_secret"]))


def _bigquery():
    found = _secrets("KUMO_BIGQUERY_CREDS")
    return BigQueryConnector(
        project_id="bigquery-public-data", dataset_id="usa_names",
        billing_project_id=found["project_id"],
        identity=BigQueryIdentity(service_account_key=found))


CASES = {
    "bigquery": (
        _bigquery,
        ("SELECT state, SUM(number) AS total FROM usa_1910_2013 "
         "GROUP BY state", "state", None),
        ("SELECT state, name, gender, number FROM usa_1910_2013 "
         "WHERE year = 2000", None, None),
        "state",
        "SELECT COUNT(*) AS n FROM usa_1910_2013 WHERE year = 2000",
    ),
    "snowflake": (
        _snowflake,
        ("SELECT ACCOUNT_ID, DISTRICT_ID, FREQUENCY FROM ACCOUNT "
         "WHERE DISTRICT_ID < 40", "ACCOUNT_ID", None),
        ("SELECT LOAN_ID, ACCOUNT_ID, AMOUNT, STATUS FROM LOAN "
         "WHERE AMOUNT > 100000", "LOAN_ID", None),
        "ACCOUNT_ID",
        "SELECT COUNT(*) AS N FROM LOAN WHERE AMOUNT > 100000",
    ),
    "databricks": (
        _databricks,
        ("SELECT c_custkey, c_acctbal, c_mktsegment FROM customer "
         "WHERE c_acctbal > 9000", "c_custkey", None),
        ("SELECT o_orderkey, o_custkey AS c_custkey, o_totalprice, o_orderdate"
         " FROM orders WHERE o_orderdate >= '1998-07-01'", "o_orderkey",
         "o_orderdate"),
        "c_custkey",
        "SELECT COUNT(*) AS n FROM orders WHERE o_orderdate >= '1998-07-01'",
    ),
}


def _count(connector, sql):
    source = connector.sql(sql)
    response = connector.source_backend.get_sample(
        connector, source.table_name, 1, sql=source.sql)
    from kumo_core.sources.preview import sample_dataframe
    return int(sample_dataframe(response.sample_rows).iloc[0, 0])


@pytest.mark.parametrize("warehouse", sorted(CASES))
def test_snapshot_from_sql_tables(warehouse, monkeypatch, tmp_path):
    make, parent, child, fkey, count_sql = CASES[warehouse]
    connector = make()
    from kumo.engine import configure
    configure(scratch_dir=str(tmp_path / "scratch"))
    parents = Table.from_sql(connector, parent[0], primary_key=parent[1])
    children = Table.from_sql(connector, child[0], primary_key=child[1],
                              time_column=child[2])

    from kumo_core.diskgraph import ingest
    rows = {}
    real_ingest = ingest.execute_diskgraph_ingest

    def spy(graph_config, spec):
        for alias, path in spec.source_paths.items():
            rows[alias] = pq.ParquetDataset(path).read().num_rows
        return real_ingest(graph_config, spec)

    monkeypatch.setattr(ingest, "execute_diskgraph_ingest", spy)
    snapshot = Graph(
        tables={
            "parents": parents,
            "children": children
        },
        edges=[Edge(src_table="children", fkey=fkey,
                    dst_table="parents")]).snapshot()

    assert (snapshot.index_path / "_MANIFEST.json").exists()
    assert rows["children"] == _count(connector, count_sql) > 0
    assert rows["parents"] > 0


@pytest.mark.parametrize("warehouse", sorted(CASES))
def test_zero_row_query_is_an_empty_source(warehouse, tmp_path):
    connector = CASES[warehouse][0]()
    sql = CASES[warehouse][2][0] + " AND 1 = 0"
    source = connector.sql(sql)
    with pytest.raises(PermanentSourceError, match="no (rows|files)"):
        connector.source_backend.prepare_ingest(connector, source.table_name,
                                                str(tmp_path), sql=source.sql)


@pytest.mark.parametrize("warehouse", sorted(CASES))
def test_out_of_scope_query_never_reaches_the_warehouse(warehouse):
    connector = CASES[warehouse][0]()
    sql = "SELECT * FROM other_catalog.information_schema.tables"
    with pytest.raises(ValueError, match="namespace"):
        connector.sql(sql)
    with pytest.raises(ValueError, match="namespace"):
        connector.source_backend.get_columns(connector, "t", sql=sql)
