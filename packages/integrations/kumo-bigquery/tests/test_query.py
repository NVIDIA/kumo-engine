# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from fakes import FakeBigQueryClient, FakeField, FakeStorageClient
from kumo_bigquery import streaming
from kumo_bigquery.sql import home_dataset
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.sources.types import ReadMode
from kumoapi.typing import Dtype
from test_source import (
    COLUMNS,
    STAGING,
    build,
    fake_workers,
    parquet_bytes,
    read_client,
)

SQL = "SELECT state, number FROM usa_1910_2013 WHERE number > 0"
NAME = "custom_sql_0123456789abcdef"
EXPORTED = {
    f"ingest/{NAME}/part-000000000000.parquet": parquet_bytes(["CA", "NY"],
                                                              [7, 8]),
}


@pytest.fixture(autouse=True)
def workers_are_fakes(monkeypatch):
    monkeypatch.setattr(streaming, "read_clients", fake_workers)


def staged(**client_kwargs):
    client = FakeBigQueryClient(missing=True, **client_kwargs)
    storage = FakeStorageClient({})
    client.on_export = lambda: storage.bodies.update(EXPORTED)
    return client, build(client, staging=STAGING, storage=storage)


def test_the_query_supplies_the_schema_no_table_could(tmp_path):
    client, reader = staged()
    reader.prepare(NAME, str(tmp_path), columns=COLUMNS, mode=ReadMode.STAGED,
                   sql=SQL)
    assert client.dry_runs == [SQL]


@pytest.mark.parametrize("suffix", ["", "; -- trailing comment"])
def test_the_query_is_exported_in_subquery_position_and_not_rewritten(
        tmp_path, suffix):
    client, reader = staged()
    reader.prepare(NAME, str(tmp_path), columns=COLUMNS, mode=ReadMode.STAGED,
                   sql=SQL + suffix)
    exported = client.matching("EXPORT DATA")
    assert len(exported) == 1
    assert exported[0].endswith(f"FROM ({SQL})")


def test_a_query_is_streamed_through_its_own_result_table(tmp_path):
    client = FakeBigQueryClient(missing=True)
    reader = build(client, reads=read_client())
    reader.prepare(NAME, str(tmp_path), columns=COLUMNS, mode=ReadMode.STREAMED,
                   sql=SQL)
    assert client.matching(f"FROM ({SQL})")


def test_a_statement_the_warehouse_does_not_call_a_select_is_refused(tmp_path):
    client = FakeBigQueryClient(missing=True, statement_type="SCRIPT")
    with pytest.raises(PermanentSourceError, match="read-only SELECT"):
        build(client).prepare(NAME, str(tmp_path), columns=COLUMNS,
                              mode=ReadMode.STREAMED, sql=SQL)


def test_a_query_is_stamped_by_the_tables_it_reads():
    reader = build(FakeBigQueryClient())
    assert reader.data_version(NAME,
                               sql=SQL) == reader.data_version(NAME, sql=SQL)
    assert reader.data_version(NAME, sql=SQL) != reader.data_version(NAME)


def test_the_schema_the_dry_run_reports_orders_the_projection(tmp_path):
    client, reader = staged(query_schema=(FakeField("number"),
                                          FakeField("state")))
    reader.prepare(NAME, str(tmp_path), columns={
        "state": Dtype.string,
        "number": Dtype.int64
    }, mode=ReadMode.STAGED, sql=SQL)
    assert "SELECT `number`, `state`" in client.matching("EXPORT DATA")[0]


def test_an_option_nobody_set_is_left_out_rather_than_sent_as_the_text_none():
    """``QueryJobConfig`` stringifies ``maximum_bytes_billed``, so a ``None``
    reaches BigQuery as ``"None"`` and every job carrying the config is
    rejected -- including the plain-table reads that share it."""
    assert home_dataset("p", "d").to_api_repr() == {
        "query": {
            "defaultDataset": {
                "projectId": "p",
                "datasetId": "d"
            }
        }
    }
    bounded = home_dataset("p", "d", 1 << 30, 5000).to_api_repr()
    assert bounded["query"]["maximumBytesBilled"] == str(1 << 30)
    assert bounded["jobTimeoutMs"] == "5000"


def test_dry_run_preserves_job_and_only_compiles_in_home_dataset():
    from kumo_bigquery.sql import dry_run

    client = Mock()
    job = dry_run(client, SQL, project='billing', dataset='sources')
    assert job is client.query.return_value
    args, kwargs = client.query.call_args
    assert args == (SQL, )
    assert kwargs['job_config'].to_api_repr() == {
        'dryRun': True,
        'query': {
            'defaultDataset': {
                'projectId': 'billing',
                'datasetId': 'sources',
            },
            'useQueryCache': False,
        },
    }
