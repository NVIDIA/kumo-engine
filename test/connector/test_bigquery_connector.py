# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import tempfile
import uuid
from dataclasses import asdict

import google.cloud.bigquery as bigquery
import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype, Stype

import kumo.pquery.AST.ast_node as ast_node
from io_utils import fsutil
from kumo.config.graph import BaseColumnConfig
from kumo.connector.bigquery_connector import (
    ENV_GOOGLE_CREDENTIALS_JSON,
    BigQueryConnector,
)
from kumo.connector.connector import (
    Connector,
    ConnectorError,
    SourceTableNotFoundError,
)
from kumo.stats.compute_stats_python import compute_col_stats_python
from kumo.testing.bigquery import (
    FAIL_PROJECT_PERMISSION_BIGQUERY_KEY,
    FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
    onlyGCPTests,
    withBigQueryCredentials,
)


def _write_df_to_bigquery(connector, table_name, df, replace=False):
    r"""Test helper: write a pandas DataFrame to a BigQuery table."""
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition=(bigquery.WriteDisposition.WRITE_APPEND if not replace
                           else bigquery.WriteDisposition.WRITE_TRUNCATE),
    )

    job = connector.context.load_table_from_dataframe(
        df, connector.full_table_name(table_name), job_config=job_config)
    job.result()


@withBigQueryCredentials
def test_bigquery_connector_config():
    bq_connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    config = asdict(bq_connector.config())
    assert config == {
        '_target_': 'kumo.connector.bigquery_connector.BigQueryConnector',
        'name': 'bigquery',
        'project_id': 'kumo-engineering',
        'dataset_id': 'financial',
        'location': 'US',
        'billing_project_id': 'kumo-engineering',
    }
    assert bq_connector == Connector.from_config(bq_connector.config())
    assert len(bq_connector.list_tables()) == 8


@withBigQueryCredentials
def test_bigquery_read_table():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    df = connector.read_table('LOAN', pkey=BaseColumnConfig('LOAN_ID'))
    assert df.shape == (682, 6)

    chunk = connector.read_table('LOAN', pkey=BaseColumnConfig('LOAN_ID'),
                                 num_rows=5)
    assert df[:5].equals(chunk)

    chunk = connector.read_table('LOAN', pkey=BaseColumnConfig('LOAN_ID'),
                                 num_rows=3, offset=2)
    assert df[2:5].equals(chunk)


@withBigQueryCredentials
def test_bigquery_overwrite_table():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    assert connector.table_exists("customers")
    assert connector.table_exists("articles")
    overwrite_table_name = f"test_{uuid.uuid4().hex}"

    try:
        # Test overwrite to a new table.
        connector.overwrite_table("customers", overwrite_table_name)
        assert connector._get_table_num_rows(
            "customers") == connector._get_table_num_rows(overwrite_table_name)

        # Test overwrite existing table.
        connector.overwrite_table("articles", overwrite_table_name)
        assert connector._get_table_num_rows(
            "articles") == connector._get_table_num_rows(overwrite_table_name)
    finally:
        connector.drop_table_if_exists(overwrite_table_name)
        assert not connector.table_exists(overwrite_table_name)


@withBigQueryCredentials
def test_get_exact_col_stats():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )

    out = connector._get_exact_col_stats(
        table_name='LOAN',
        stats_dict={
            BaseColumnConfig.cast('AMOUNT'): [
                ColStatType.MEAN,
                ColStatType.MEDIAN,
                ColStatType.MIN,
                ColStatType.MAX,
                ColStatType.STD,
            ],
            BaseColumnConfig.cast('STATUS'): [
                ColStatType.NUM_UNIQUE,
                ColStatType.UNIQUE_FRACTION,
                ColStatType.CATEGORY_COUNTS,
            ],
        },
    )
    assert len(out) == 2

    stats = out[BaseColumnConfig.cast('AMOUNT')]
    assert len(stats) == 5
    assert stats[ColStatType.MEAN].value == pytest.approx(151410.1)
    assert stats[ColStatType.MEAN].num_entries == 682
    assert stats[ColStatType.MEDIAN].value == 116928
    assert stats[ColStatType.MEDIAN].num_entries == 682
    assert stats[ColStatType.MIN].value == 4980
    assert stats[ColStatType.MIN].num_entries == 682
    assert stats[ColStatType.MAX].value == 590820
    assert stats[ColStatType.MAX].num_entries == 682
    assert stats[ColStatType.STD].value == pytest.approx(113372.4)
    assert stats[ColStatType.STD].num_entries == 682

    stats = out[BaseColumnConfig.cast('STATUS')]
    assert len(stats) == 3
    assert stats[ColStatType.NUM_UNIQUE].value == 4
    assert stats[ColStatType.NUM_UNIQUE].num_entries == 682
    assert stats[ColStatType.UNIQUE_FRACTION].value == pytest.approx(
        0.0058, abs=1e-2)
    assert stats[ColStatType.UNIQUE_FRACTION].num_entries == 682
    assert stats[ColStatType.CATEGORY_COUNTS].value == [('C', 403), ('A', 203),
                                                        ('D', 45), ('B', 31)]
    assert stats[ColStatType.CATEGORY_COUNTS].num_entries == 682

    out = connector._get_exact_col_stats(
        'DISTRICT',
        {
            BaseColumnConfig.cast('A12'): [
                ColStatType.NUM_NA,
                ColStatType.NA_FRACTION,
            ],
        },
    )
    assert len(out) == 1

    stats = out[BaseColumnConfig.cast('A12')]
    assert stats[ColStatType.NUM_NA].value == 0
    assert stats[ColStatType.NUM_NA].num_entries == 77
    # TODO: The district table in BigQuery financial dataset is marked as '?'
    # instead of a NULL value. Hence NA_FRACTION=0.
    # assert stats[ColStatType.NA_FRACTION].value ==
    # pytest.approx(1 / 77, 0.001)
    # assert stats[ColStatType.NA_FRACTION].num_entries == 77
    assert stats[ColStatType.NA_FRACTION].value == 0
    assert stats[ColStatType.NA_FRACTION].num_entries == 77


@withBigQueryCredentials
def test_bigquery_data_version():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    expected_table_data_versions = {
        'TRANS': 1695079444,
        'ORDERS': 1695077884,
    }
    for table_name, version in expected_table_data_versions.items():
        assert connector._get_data_version(table_name) == str(version)


@withBigQueryCredentials
def test_bigquery_table_info():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )

    # num rows
    assert connector._get_table_num_rows('ACCOUNT') == 4500
    assert connector._get_table_num_rows('CARD') == 892
    assert connector._get_table_num_rows('CLIENT') == 5369
    assert connector._get_table_num_rows('DISP') == 5369
    assert connector._get_table_num_rows('DISTRICT') == 77
    assert connector._get_table_num_rows('LOAN') == 682
    assert connector._get_table_num_rows('ORDERS') == 6471

    # table size (in bytes)
    assert connector._get_table_source_size('ACCOUNT') == 188706
    assert connector._get_table_source_size('CARD') == 37055
    assert connector._get_table_source_size('CLIENT') == 144963
    assert connector._get_table_source_size('DISP') == 169915
    assert connector._get_table_source_size('DISTRICT') == 10317
    assert connector._get_table_source_size('LOAN') == 34782
    assert connector._get_table_source_size('ORDERS') == 270796


@withBigQueryCredentials
def test_bigquery_table_dtype():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )

    metadata = connector._get_one_table_metadata('embeddings2')
    assert metadata.cols[0].name == 'article_id'
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].name == 'embedding'
    assert metadata.cols[1].dtype == Dtype.floatlist

    metadata = connector._get_one_table_metadata('embeddings')
    assert metadata.cols[0].name == 'article_id'
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].name == 'embedding'
    # Return unsupported data type because it is a record.
    assert metadata.cols[1].dtype == Dtype.unsupported


@withBigQueryCredentials
def test_bigquery_connector_list_types(monkeypatch):
    monkeypatch.setattr(
        ast_node,
        'ALLOW_ARRAY_TARGETS',
        True,
    )

    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='movielens',
    )
    metadata = connector.get_table_metadata(["movie_stringlist"])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.stringlist

    metadata = connector.get_table_metadata(["movie_intlist"])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.intlist


@withBigQueryCredentials
def test_emb_col_stats():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='test',
    )
    key_col = BaseColumnConfig('A', stype=Stype.numerical, dtype=Dtype.int64)
    emb_col = BaseColumnConfig('EMB', stype=Stype.sequence, dtype=Dtype.intlist)
    connector.open()
    df = connector.read_table('TEST_EMPTY_EMB', cols=[key_col, emb_col])
    stats = compute_col_stats_python(df['EMB'], [
        ColStatType.SEQUENCE_MAX_LENGTH,
        ColStatType.SEQUENCE_MIN_LENGTH,
        ColStatType.NUM_NA,
    ])
    assert stats[ColStatType.SEQUENCE_MAX_LENGTH].value == 3
    assert stats[ColStatType.SEQUENCE_MIN_LENGTH].value == 3
    assert stats[ColStatType.NUM_NA].value == 0


@withBigQueryCredentials
def test_bq_non_existent_table_data_version():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # The table is legit - no error should be thrown.
    connector._get_data_version('LOAN')

    with pytest.raises(SourceTableNotFoundError) as missing_table_error:
        connector._get_data_version('NON_EXISTENT_TABLE')
    message = str(missing_table_error.value)
    assert "Source table 'NON_EXISTENT_TABLE'" in message


@withBigQueryCredentials
def test_read_table_sample():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # Since BigQuery sampling operation is not deterministic, we check for
    # approximate correctness
    # Sample with num_rows:
    sample_df = connector.read_table_sample(table_name='LOAN', num_rows=100)
    assert sample_df.shape[0] == pytest.approx(100, rel=0.1)
    # Sample with frac:
    sample_df = connector.read_table_sample(table_name='LOAN', frac=0.1)
    total_num_rows = connector._get_table_num_rows('LOAN') * 1.0
    assert sample_df.shape[0] / total_num_rows == pytest.approx(0.1, abs=1e-2)


@withBigQueryCredentials
def test_get_table_num_rows():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    total_rows = connector._get_table_num_rows("transactions")
    assert total_rows == 31788324

    view_total_rows = connector._get_table_num_rows("transactions_view")
    assert view_total_rows == total_rows


@withBigQueryCredentials
def test_set_credentials_and_open():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # This will get credentials from the JSON key file.
    connector.open()
    # Corrupt the env variable so that we don't get credentials from there:
    os.environ[ENV_GOOGLE_CREDENTIALS_JSON] = "random"

    connector2 = BigQueryConnector(
        name='bigquery2',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # Test the workflow of explicitly setting credentials and opening the
    # connector.
    connector2.set_credentials(connector._credentials)
    connector.open()


@onlyGCPTests
@withBigQueryCredentials(service_account_key=FAIL_TABLE_PERMISSION_BIGQUERY_KEY)
def test_bigquery_table_permission_fail_read_table():
    connector = BigQueryConnector(
        name='bigquery_fail',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    with pytest.raises(ConnectorError):
        connector.check_table_read_permissions(table_name='LOAN')
    with pytest.raises(ConnectorError) as caught:
        connector.read_table(table_name='LOAN', num_rows=100)
    message = str(caught.value)
    assert message == (
        'The selected identity does not have the required BigQuery dataset '
        'read permissions.')
    assert 'LOAN' not in message


@onlyGCPTests
@withBigQueryCredentials(
    service_account_key=FAIL_PROJECT_PERMISSION_BIGQUERY_KEY)
def test_bigquery_project_permission_fail_read_table():
    connector = BigQueryConnector(
        name='bigquery_fail',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    with pytest.raises(ConnectorError):
        connector.check_table_read_permissions(table_name='LOAN')
    with pytest.raises(ConnectorError, match=r"does not have .+ permission"):
        connector.read_table(table_name='LOAN', num_rows=100)


@withBigQueryCredentials
def test_invalid_project():
    with pytest.raises(ConnectorError) as caught:
        connector = BigQueryConnector(
            name='bigquery',
            project_id='MXYZPTLK',
            dataset_id='hnm_integ_test',
            location='us-west1',
        )
        connector.open()
    message = str(caught.value)
    assert message == 'BigQuery connector access validation failed.'
    assert 'MXYZPTLK' not in message
    assert 'hnm_integ_test' not in message


@withBigQueryCredentials
def test_invalid_dataset():
    with pytest.raises(ConnectorError) as caught:
        connector = BigQueryConnector(
            name='bigquery',
            project_id='kumo-engineering',
            dataset_id='mxyzptlk',
        )
        connector.open()
    message = str(caught.value)
    assert message == (
        'The configured BigQuery dataset was not found or is not accessible '
        'to the selected identity.')
    assert 'mxyzptlk' not in message
    assert 'kumo-engineering' not in message


_BQ_EXPORT_TEST_STAGING_PATH = os.environ.get(
    'BIGQUERY_EXPORT_TEST_STAGING_PATH',
    'gs://kumo-test-bucket/bigquery-integ-test/')


@withBigQueryCredentials
def test_export_table_to_blob_storage():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_export_{uuid.uuid4().hex}"
    dest_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                         f'export_{uuid.uuid4().hex}/')

    nrows = 100
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
        'STATUS': ['A'] * nrows,
    })
    _write_df_to_bigquery(connector, table_name, source_df, replace=True)

    try:
        connector.export_table_to_blob_storage(table_name, dest_path)

        exported_files = fsutil.listdir(dest_path, suffix='parquet')
        assert exported_files, f"No parquet files at {dest_path}"

        exported = pd.concat([pd.read_parquet(f) for f in exported_files])
        exported = exported.sort_values('LOAN_ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(exported[source_df.columns], source_df,
                                      check_dtype=False)
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(dest_path, ignore_errors=True)


@withBigQueryCredentials
def test_export_table_to_blob_storage_with_columns():
    r"""Functional test: exporting with the ``columns`` argument should
    produce parquet with exactly the requested columns (and none of the
    others). Asserts that unselected columns are absent from the output
    schema.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_export_cols_{uuid.uuid4().hex}"
    dest_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                         f'export_cols_{uuid.uuid4().hex}/')

    nrows = 50
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
        'STATUS': ['A'] * nrows,
        'EXTRA_COL': ['should_not_appear'] * nrows,
    })
    _write_df_to_bigquery(connector, table_name, source_df, replace=True)

    try:
        connector.export_table_to_blob_storage(table_name, dest_path,
                                               columns=['LOAN_ID', 'AMOUNT'])

        exported_files = fsutil.listdir(dest_path, suffix='parquet')
        assert exported_files, f"No parquet files at {dest_path}"

        exported = pd.concat([pd.read_parquet(f) for f in exported_files])
        exported = exported.sort_values('LOAN_ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(
            exported[['LOAN_ID', 'AMOUNT']],
            source_df[['LOAN_ID', 'AMOUNT']],
            check_dtype=False,
        )
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(dest_path, ignore_errors=True)


@withBigQueryCredentials
def test_export_table_to_blob_storage_error_nonexistent_table():
    r"""Exporting a table that doesn't exist must raise ConnectorError
    (not silently succeed).
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    missing_table = f"does_not_exist_{uuid.uuid4().hex}"
    dest_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                         f'missing_{uuid.uuid4().hex}/')

    try:
        with pytest.raises(ConnectorError):
            connector.export_table_to_blob_storage(missing_table, dest_path)
    finally:
        # Defensive cleanup in case the export partially succeeded.
        fsutil.rm(dest_path, ignore_errors=True)


def _seed_gcs_parquet(df: pd.DataFrame, gcs_path: str) -> None:
    r"""Write ``df`` to a single parquet file under ``gcs_path`` so the
    gs:// fast path has something to load.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_parquet(osp.join(tmpdir, 'part-0.parquet'), index=False)
        fsutil.copy(tmpdir, gcs_path)


@withBigQueryCredentials
def test_bigquery_write_artifacts_to_table_from_gcs():
    r"""Functional test: seed parquet in a Kumo-owned GCS bucket, then
    load it into a fresh BigQuery table via the gs:// fast path. Verifies
    the rows and columns round-trip.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_gcs_write_{uuid.uuid4().hex}"
    gcs_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                        f'gcs_write_{uuid.uuid4().hex}/')

    nrows = 100
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
        'STATUS': ['A'] * nrows,
    })
    try:
        _seed_gcs_parquet(source_df, gcs_path)

        connector.write_artifacts_to_table(table_name, gcs_path)

        loaded = connector.read_table(table_name)
        loaded = loaded.sort_values('LOAN_ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(loaded[source_df.columns], source_df,
                                      check_dtype=False)
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(gcs_path, ignore_errors=True)


@withBigQueryCredentials
def test_bigquery_write_artifacts_to_table_from_gcs_with_embedding_column():
    r"""Verify pandas-generated parquet with an embedding (list) column
    round-trips through BigQuery's ``enable_list_inference=True``.

    This test uses ``pandas.to_parquet``, matching the artifact writer used
    by the current pipeline. Succeeding here means BigQuery's list inference
    recognizes its standard parquet LIST logical type.

    Failure here means the comment "only works for PySpark generated
    parquet files" is empirically load-bearing and the gs:// fast path
    can't be used with pandas-written embedding columns until we add
    a conversion step.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_gcs_embed_{uuid.uuid4().hex}"
    gcs_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                        f'gcs_embed_{uuid.uuid4().hex}/')

    nrows = 50
    emb_size = 5
    source_df = pd.DataFrame({
        'LOAN_ID':
        list(range(nrows)),
        'EMBEDDING':
        [[float(i + k) for k in range(emb_size)] for i in range(nrows)],
    })
    try:
        _seed_gcs_parquet(source_df, gcs_path)
        connector.write_artifacts_to_table(table_name, gcs_path)

        # ARRAY_LENGTH only succeeds when EMBEDDING is a BQ ARRAY
        # column. Without enable_list_inference (or if it's ignored),
        # BQ would read the parquet LIST as a nested STRUCT and this
        # query would fail with a SQL type error.
        query = (f'SELECT COUNT(*) AS n FROM '
                 f'{connector.full_table_name(table_name)} '
                 f'WHERE ARRAY_LENGTH(EMBEDDING) = {emb_size}')
        result_df = connector.context.query(query).result().to_dataframe()
        assert result_df['n'].iloc[0] == nrows, (
            f"Expected {nrows} rows with EMBEDDING of length {emb_size}; "
            f"got {result_df['n'].iloc[0]}. pandas-generated LIST parquet "
            f"may not be read as a BQ ARRAY via enable_list_inference.")
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(gcs_path, ignore_errors=True)


@withBigQueryCredentials
def test_bigquery_write_artifacts_to_table_from_gcs_append_and_replace():
    r"""Functional test: the ``replace`` parameter controls
    ``WriteDisposition``. Seeds parquet, writes once (append on a new
    table), writes again with ``replace=False`` and confirms rows double,
    then writes again with ``replace=True`` and confirms rows are
    truncated back to the original count.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_gcs_replace_{uuid.uuid4().hex}"
    gcs_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                        f'gcs_replace_{uuid.uuid4().hex}/')

    nrows = 50
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
    })
    try:
        _seed_gcs_parquet(source_df, gcs_path)

        # 1st write: creates the table, loads nrows rows.
        connector.write_artifacts_to_table(table_name, gcs_path)
        assert connector._get_table_num_rows(table_name) == nrows

        # 2nd write (replace=False): appends, row count doubles.
        connector.write_artifacts_to_table(table_name, gcs_path, replace=False)
        assert connector._get_table_num_rows(table_name) == 2 * nrows

        # 3rd write (replace=True): truncates, back to nrows.
        connector.write_artifacts_to_table(table_name, gcs_path, replace=True)
        assert connector._get_table_num_rows(table_name) == nrows
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(gcs_path, ignore_errors=True)
