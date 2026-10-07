# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom SQL source tables, from definition through a real snapshot."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sqlglot
from kumo_core.custom_sql import custom_sql_synthetic_name
from kumo_core.sources.backend import sampled_table, source_columns
from kumo_core.types import ConnectorType
from kumoapi.graph import Edge
from kumoapi.plugin import PrepareIngestResult
from kumoapi.source_table import CustomSQLSourceTable
from kumoapi.typing import Dtype, Stype

from kumo.engine import (
    BigQueryConnector,
    DatabricksConnector,
    DatabricksIdentity,
    FileConnector,
    Graph,
    SnowflakeConnector,
    Table,
    configure,
)
from kumo.engine.pquery._custom_source import stage_source_table

SQL = "SELECT user_id, age FROM users WHERE active = 1"


class SqliteWarehouse:
    """Runs the query the engine hands it, so assertions see its result."""
    def __init__(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript(
            "CREATE TABLE users (user_id INTEGER, age INTEGER, active INTEGER);"
            "INSERT INTO users VALUES (1, 30, 1), (2, 40, 0), (3, 50, 1);")
        self.queries = []

    def _run(self, table_name, sql):
        assert table_name == custom_sql_synthetic_name(sql,
                                                       ConnectorType.SNOWFLAKE)
        self.queries.append(sql)
        query = sqlglot.transpile(sql, read='snowflake', write='sqlite')[0]
        return pd.read_sql_query(query, self.db)

    def _columns(self, frame):
        return source_columns((name, Dtype.int) for name in frame.columns)

    def get_columns(self, connector, table_name, project_id=None, sql=None):
        frame = self._run(table_name, sql)
        return sampled_table(table_name, self._columns(frame), [])

    def get_sample(self, connector, table_name, sample_rows, project_id=None,
                   sql=None, timeout=None):
        frame = self._run(table_name, sql).head(sample_rows)
        return sampled_table(table_name, self._columns(frame),
                             frame.itertuples(index=False, name=None))

    def prepare_ingest(self, connector, table_name, dest_dir, dtypes=None,
                       project_id=None, data_version=None, sql=None):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        path = Path(dest_dir) / 'part-0.parquet'
        pq.write_table(pa.Table.from_pandas(self._run(table_name, sql)), path)
        return PrepareIngestResult(
            source={'type': 'parquet', 'paths': [str(path)]},
            source_size=path.stat().st_size, version='v1')


@pytest.fixture
def warehouse(tmp_path):
    configure(scratch_dir=str(tmp_path / 'scratch'))
    connector = SnowflakeConnector(warehouse='WH', database='DB',
                                   schema_name='PUBLIC')
    connector.__dict__['source_backend'] = SqliteWarehouse()
    return connector


def test_sql_table_snapshot_ingests_the_query_result(warehouse, tmp_path,
                                                     monkeypatch):
    root = tmp_path / 'files'
    root.mkdir()
    pq.write_table(pa.table({
        'order_id': [10, 11, 12],
        'user_id': [1, 3, 3]
    }), root / 'orders.parquet')
    orders = Table.from_source_table(
        FileConnector(root_dir=str(root))['orders.parquet'],
        primary_key='order_id')
    inferred = Table.from_sql(warehouse, SQL, primary_key='user_id')
    columns = {col.name: col for col in inferred.definition.cols}
    assert columns['user_id'].stype == Stype.ID
    assert inferred.table_name == custom_sql_synthetic_name(
        SQL, ConnectorType.SNOWFLAKE)
    assert inferred.definition.source_table.referenced_tables == ['users']
    users = Table(warehouse, 'users', definition=inferred.definition)
    assert users.infer_metadata() == users

    from kumo_core.diskgraph import ingest
    staged = {}
    real_ingest = ingest.execute_diskgraph_ingest

    def spy(graph_config, spec):
        for alias, path in spec.source_paths.items():
            staged[alias] = pq.ParquetDataset(path).read().to_pydict()
        return real_ingest(graph_config, spec)

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', spy)
    orders = replace(orders, table_name='orders')
    graph = Graph(
        tables=[users, orders],
        edges=[Edge(src_table='orders', fkey='user_id', dst_table='users')])
    snapshot = graph.snapshot()

    assert staged['users'] == {'user_id': [1, 3], 'age': [30, 50]}
    assert (snapshot.index_path / '_MANIFEST.json').exists()
    table_names = {
        table['table_name']: table['source_table_name']
        for table in snapshot.graph_config['tables']
    }
    assert table_names['users'].startswith('custom_sql.')
    assert len(table_names['users'].split('.')) == 3
    assert set(warehouse.source_backend.queries) == {SQL}


def test_custom_training_rows_stage_the_query_result(warehouse, tmp_path):
    staged = stage_source_table(warehouse.sql(SQL), tmp_path / 'staged')
    assert pq.read_table(staged).to_pydict() == {
        'user_id': [1, 3],
        'age': [30, 50]
    }


@pytest.mark.parametrize(('sql', 'match'), [
    ("DELETE FROM users", "read-only"),
    ("SELECT 1 AS a; SELECT 2 AS b", "single SQL statement"),
    ("SELECT age + 1 FROM users", "explicit alias"),
    ("SELECT * FROM system.information_schema.tables", "namespace"),
    ("SELECT * FROM main.other.users", "namespace"),
])
def test_unsafe_sql_is_rejected_before_the_warehouse_is_reached(sql, match):
    connector = DatabricksConnector(
        catalog='main', schema_name='sales', warehouse_id='w',
        identity=DatabricksIdentity(host='https://example.invalid',
                                    client_id='id', client_secret='secret'))
    with pytest.raises(ValueError, match=match):
        connector.sql(sql)
    with pytest.raises(ValueError, match=match):
        connector.source_backend.get_columns(connector, 't', sql=sql)


@pytest.mark.parametrize('connector', [
    SnowflakeConnector(warehouse='WH', database='DB', schema_name='PUBLIC'),
    BigQueryConnector(project_id='project', dataset_id='dataset'),
])
def test_readers_refuse_out_of_scope_sql(connector, tmp_path):
    sql = "SELECT * FROM other.information_schema.tables"
    backend = connector.source_backend
    for read in (
            lambda: backend.get_columns(connector, 't', sql=sql),
            lambda: backend.get_sample(connector, 't', 1, sql=sql),
            lambda: backend.prepare_ingest(connector, 't', str(tmp_path),
                                           sql=sql),
    ):
        with pytest.raises(ValueError, match='namespace'):
            read()


def test_sql_needs_a_warehouse_connector(tmp_path):
    with pytest.raises(ValueError, match='warehouse connectors'):
        FileConnector(root_dir=str(tmp_path)).sql(SQL)


def test_snapshot_regates_a_hand_built_definition(warehouse):
    source = warehouse.sql(SQL).to_source_table()
    escaped = CustomSQLSourceTable(connector_id=source.connector_id,
                                   sql="SELECT * FROM OTHER.PUBLIC.users",
                                   table=source.table)
    table = Table.from_sql(warehouse, SQL)
    table = replace(table, definition=replace(table.definition,
                                              source_table=escaped))
    with pytest.raises(ValueError, match='namespace'):
        Graph(tables=[table]).snapshot()
