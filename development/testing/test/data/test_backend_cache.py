# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import pathlib
import shutil
import threading
import uuid
from types import SimpleNamespace
from typing import Dict

import pandas as pd
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from kumo_core.types import ColStatType, EdgeStatType, TableStatType
from kumo_core.types.schema import (
    BaseColumnConfig,
    IngestedTableFingerprint,
    SourceTableID,
    VersionedSourceTableID,
)
from kumoapi.typing import Dtype

from io_utils import fsutil
from io_utils.fsutil import s3_filesystem
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.data import (
    MaterializationStage,
    get_persist_dir,
    reset_active_cache_for_test,
)
from kumo.data.backend_cache import (
    CACHE_DIR,
    AsyncDataTaskStatus,
    BackendCacheManager,
    DirectDiskgraphSourceDecision,
    ExactStats,
    TableDataTaskResult,
    _load_col_stats_json,
    _minify_stats,
)
from kumo.data.cache import SourceTableColumn, table_source_path_component
from kumo.data.cache import table_version_cache as table_version_cache_module
from kumo.data.cache.table_version_cache import TableIngestionStatus
from kumo.data.common import TableRefreshStage
from kumo.data.utils import get_materialize_path
from kumo.graph.graph import Graph
from kumo.stats.stats import Stats
from kumo.stats.train_table_stats import (
    ITERATION_DATA,
    NUM_EXPECTED_ITERATIONS,
    NUM_PRED_ROWS,
    NUM_TARGETS,
    NUM_TOTAL_ROWS,
    NUM_TRAIN_ROWS,
    TemporalNodeTimeFrameStats,
)
from kumo.validation.validation import ValidationResponse


def setup_function():
    # Clean up the data backend cache since this test is to test backend cache.
    # This runs after `run_before_and_after_tests` in conftest.py.
    backend_cache_dir = osp.join(get_persist_dir(), 'data_backend_cache')
    shutil.rmtree(backend_cache_dir, ignore_errors=True)
    os.makedirs(backend_cache_dir, exist_ok=True)


def test_load_col_stats_cache_stages_s3_object_on_local_disk(
        monkeypatch, tmp_path):
    backend_cache = BackendCacheManager('s3://bucket/cache')
    table_fingerprint = IngestedTableFingerprint(
        'connector',
        'table',
        cols={},
        pkey=None,
        time_col=None,
        end_time_col=None,
    )
    data_version = 'version'
    table_dir = backend_cache.create_table_cache_dir(
        table_fingerprint.connector_name,
        table_fingerprint.source_name,
        data_version,
        table_fingerprint,
    )
    real_exists = fsutil.exists
    monkeypatch.setattr(
        fsutil,
        'exists',
        lambda path: True
        if str(path).startswith('s3://') else real_exists(path),
    )
    local_targets = []

    def managed_download(src, dst):
        assert src == table_dir.col_stats
        local_targets.append(pathlib.Path(dst))
        pathlib.Path(dst).write_text('{}')

    monkeypatch.setattr(fsutil, 'copy_file', managed_download)

    assert backend_cache.load_col_stats_cache(table_fingerprint, data_version)
    cache_key = BackendCacheManager._get_col_stats_cache_key(
        table_fingerprint, data_version)
    assert backend_cache._col_stats_cache[cache_key] == {}
    assert len(local_targets) == 1
    assert not local_targets[0].exists()


def test_load_col_stats_json_skips_missing_s3_object(monkeypatch):
    monkeypatch.setattr(fsutil, 'exists', lambda _: False)

    def unexpected_download(*_):
        raise AssertionError('missing objects must not be downloaded')

    monkeypatch.setattr(fsutil, 'copy_file', unexpected_download)

    assert _load_col_stats_json('s3://bucket/missing') is None


@pytest.mark.parametrize('anonymous', [False, True])
@pytest.mark.parametrize('error_code', ['404', 'NoSuchKey', 'AccessDenied'])
def test_load_col_stats_json_handles_provider_download_errors(
        monkeypatch, error_code, anonymous):
    real_exists = fsutil.exists
    monkeypatch.setattr(
        fsutil, 'exists', lambda path: True
        if str(path).startswith('s3://') else real_exists(path))
    local_parents = []

    def failed_download(bucket, key, dst):
        local_parents.append(pathlib.Path(dst).parent)
        raise ClientError({'Error': {'Code': error_code}}, 'HeadObject')

    client = SimpleNamespace(download_file=failed_download)
    if anonymous:

        def unsigned_download(*_):
            raise NoCredentialsError()

        authenticated = SimpleNamespace(download_file=unsigned_download)
        monkeypatch.setattr(s3_filesystem, '_unsigned_s3_client',
                            lambda: client)
        monkeypatch.setattr(s3_filesystem, 'credentials_for_provider',
                            lambda _: None)
    else:
        authenticated = client
    monkeypatch.setattr(s3_filesystem, 's3_ctx',
                        SimpleNamespace(_s3_client=authenticated))

    if error_code == 'AccessDenied':
        with pytest.raises(RuntimeError, match='File copy failed'):
            _load_col_stats_json('s3://bucket/deleted')
    else:
        assert _load_col_stats_json('s3://bucket/deleted') is None
    assert len(set(local_parents)) == 1
    assert not local_parents[0].exists()


def test_table_data_task_result_tracks_explicitly_skipped_stats():
    legacy_completed = TableDataTaskResult(
        status=AsyncDataTaskStatus.DONE,
        validation=ValidationResponse(),
        stage=TableRefreshStage.COMPLETED,
    )
    assert legacy_completed.min_col_stats_computed
    assert legacy_completed.full_col_stats_computed

    completed_without_stats = TableDataTaskResult(
        status=AsyncDataTaskStatus.DONE,
        validation=ValidationResponse(),
        stage=TableRefreshStage.COMPLETED,
        col_stats_computed=False,
    )
    assert not completed_without_stats.min_col_stats_computed
    assert not completed_without_stats.full_col_stats_computed


def test_reset_clears_direct_diskgraph_source_decisions(tmp_path):
    backend_cache = BackendCacheManager(osp.join(tmp_path, CACHE_DIR))
    decision_cache = backend_cache.direct_diskgraph_source_decision
    decision_cache.put(
        'connector',
        'source',
        'version',
        DirectDiskgraphSourceDecision(source=None),
    )
    assert len(decision_cache._cache) == 1

    backend_cache.reset()

    assert len(decision_cache._cache) == 0


def test_table_cache_dirs_hash_arbitrary_object_store_paths(tmp_path):
    cache_dir = osp.join(tmp_path, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)
    connector_name = 'connector_11111111111111111111111111111111'
    data_version = 'version_1'
    source_paths = [
        's3://bucket/one/two/three/table.parquet',
        'gs://bucket/one/two/three/table.parquet',
        'abfss://container@account.dfs.core.windows.net/one/two/table.parquet',
    ]

    table_dirs = [
        backend_cache.create_table_cache_dir(connector_name, source_path,
                                             data_version)._table_dir
        for source_path in source_paths
    ]

    assert len(set(table_dirs)) == len(source_paths)
    for source_path, table_dir in zip(source_paths, table_dirs):
        assert table_dir == osp.join(cache_dir, 'tables', connector_name,
                                     table_source_path_component(source_path),
                                     data_version)
        assert backend_cache.create_table_cache_dir(
            connector_name, source_path, data_version)._table_dir == table_dir


def test_table_stats_require_explicit_approximate_fallback(tmp_path):
    backend_cache = BackendCacheManager(osp.join(tmp_path, CACHE_DIR))
    fingerprint = IngestedTableFingerprint('databricks', 'orders', cols={},
                                           pkey=None, time_col=None,
                                           end_time_col=None)
    version = 'estimated'
    estimated = Stats(value=123, approx=True, data_version=version)
    backend_cache.save_table_stats(
        fingerprint,
        version,
        {TableStatType.SOURCE_SIZE: estimated},
    )

    assert backend_cache.get_table_stats(fingerprint,
                                         [TableStatType.SOURCE_SIZE],
                                         version) == {}
    assert backend_cache.get_table_stats(fingerprint,
                                         [TableStatType.SOURCE_SIZE], version,
                                         approx=True) == {
                                             TableStatType.SOURCE_SIZE:
                                             estimated,
                                         }


def test_materialization_stage_cache_is_schema_aware(tmp_path):
    cache_dir = osp.join(tmp_path, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)
    common = dict(connector_name='snowflake', source_name='orders', pkey=None,
                  time_col=None, end_time_col=None)
    narrow = IngestedTableFingerprint(cols={'id': Dtype.int}, **common)
    wider = IngestedTableFingerprint(
        cols={
            'id': Dtype.int,
            'amount': Dtype.float,
        }, **common)
    version = 'v1'

    backend_cache.materialization_stage.put(narrow, version,
                                            MaterializationStage.INGEST,
                                            save=True)

    reloaded = BackendCacheManager(cache_dir)
    assert reloaded.materialization_stage.get_stage(
        narrow, version, check_local=False) is MaterializationStage.INGEST
    assert reloaded.materialization_stage.get_stage(wider, version,
                                                    check_local=False) is None

    reloaded.materialization_stage.remove_stage(narrow, version)
    assert reloaded.materialization_stage.get_stage(narrow, version,
                                                    check_local=False) is None


def test_approximate_table_stats_merge_preserves_exact_stats(tmp_path):
    cache_dir = osp.join(tmp_path, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)
    fingerprint = IngestedTableFingerprint('databricks', 'orders', cols={},
                                           pkey=None, time_col=None,
                                           end_time_col=None)
    version = 'shared-version'
    exact_source_size = Stats(value=4096, approx=False, data_version=version)
    backend_cache.save_table_stats(
        fingerprint,
        version,
        {TableStatType.SOURCE_SIZE: exact_source_size},
    )

    backend_cache.add_table_stats(
        fingerprint, {
            TableStatType.SOURCE_SIZE:
            Stats(value=8192, approx=True, data_version=version),
            TableStatType.NUM_ROWS:
            Stats(value=100, approx=True, data_version=version),
        })

    reloaded = BackendCacheManager(cache_dir)
    assert reloaded.get_table_stats(
        fingerprint,
        [TableStatType.SOURCE_SIZE],
        version,
        approx=True,
    ) == {
        TableStatType.SOURCE_SIZE: exact_source_size
    }


def test_table_cache_roundtrip(tmp_path: pathlib.Path):
    cache_dir = osp.join(tmp_path, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    connector = ParquetConnector('parquet', root_dir)
    graph.connect(connector)
    # Create two tables from same source
    graph.add('TRANS')
    graph.add('TRANS_2', source_name='TRANS', cols=['K_SYMBOL', 'BANK'])

    root_dir2 = osp.dirname(osp.realpath(__file__))
    root_dir2 = osp.join(root_dir2, '..', 'parquet_data', 'SYNTHETIC')
    connector2 = ParquetConnector('parquet_synthetic', root_dir2)
    graph.connect(connector2)
    graph.add('small_t1', connector2.name)

    graph.infer_metadata()

    data_version = connector.get_data_version('TRANS', skip_cache=True)
    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {
        graph["TRANS"]["K_SYMBOL"].base_config(): {
            ColStatType.KURTOSIS:
            Stats(value=-1.093549928703169, approx=True,
                  data_version=data_version, num_entries=5000),
        }
    }
    synthetic_data_version = connector2.get_data_version(
        'small_t1', skip_cache=True)
    col_stats_synthetic = {
        graph["small_t1"]["test__col__name"].base_config(): {
            ColStatType.NUM_UNIQUE:
            Stats(value=642, approx=True, data_version=synthetic_data_version,
                  num_entries=1000),
            ColStatType.NA_FRACTION:
            Stats(value=0, approx=True, data_version=synthetic_data_version,
                  num_entries=1000),
            ColStatType.COUNT:
            Stats(value=1000, approx=True, data_version=synthetic_data_version,
                  num_entries=1000),
        }
    }
    table_stats = graph['TRANS'].compute_table_stats()
    table_stats_synthetic = graph['small_t1'].compute_table_stats()
    table_samples = connector.read_table_sample('TRANS')
    table_samples_synthetic = connector2.read_table_sample('small_t1')

    # Test the scenario of mulitple concurrent writers for colstats of the same
    # source table key.
    backend_cache2 = BackendCacheManager(cache_dir)
    col_stats2: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {
        graph["TRANS_2"]["BANK"].base_config(): {
            ColStatType.NUM_UNIQUE:
            Stats(value=13.0, approx=True, data_version=data_version,
                  num_entries=5000),
        }
    }
    trans_fp = graph['TRANS'].fingerprint()
    trans_2_fp = graph['TRANS_2'].fingerprint()
    small_t1_fp = graph['small_t1'].fingerprint()
    backend_cache2.add_col_stats(trans_fp, col_stats2)
    backend_cache2.save_col_stats_cache(trans_fp, data_version)

    backend_cache.add_col_stats(trans_fp, col_stats)
    backend_cache.add_col_stats(small_t1_fp, col_stats_synthetic)
    backend_cache.add_table_stats(graph['TRANS'].fingerprint(), table_stats)
    backend_cache.add_table_stats(graph['small_t1'].fingerprint(),
                                  table_stats_synthetic)
    backend_cache.table_samples_cache.put('parquet', 'TRANS', data_version,
                                          table_samples, save=True)
    backend_cache.table_samples_cache.put('parquet_synthetic', 'small_t1',
                                          synthetic_data_version,
                                          table_samples_synthetic, save=True)
    backend_cache.materialization_stage.put(trans_fp, data_version,
                                            MaterializationStage.INGEST,
                                            save=True)
    backend_cache.materialization_stage.put(small_t1_fp, synthetic_data_version,
                                            MaterializationStage.INGEST,
                                            save=True)

    materialization_path = get_materialize_path(graph["TRANS"].fingerprint(),
                                                data_version,
                                                MaterializationStage.INGEST)
    fsutil.makedirs(materialization_path, exist_ok=True)

    backend_cache.save_col_stats_cache(trans_fp, data_version)
    backend_cache.save_col_stats_cache(trans_2_fp, data_version)
    backend_cache.save_col_stats_cache(small_t1_fp, synthetic_data_version)

    new_backend_cache = BackendCacheManager(cache_dir)

    # Try to load table cache of a data version that doesn't exist
    # Create a modified fingerprint for non-existent version test
    # This is a bit of a hack for testing purposes
    class TestFingerprint(type(trans_fp)):
        def __init__(self, original_fp):
            for key, value in vars(original_fp).items():
                setattr(self, key, value)

        def _fpid(self) -> str:
            # Return a unique fingerprint id that
            # won't match any existing cache
            return "test_nonexistent_fingerprint"

    nonexistent_fp = TestFingerprint(trans_fp)

    new_backend_cache.load_col_stats_cache(nonexistent_fp, '-1')
    assert len(new_backend_cache._col_stats_cache) == 0
    assert len(new_backend_cache._table_stats_cache) == 0

    new_backend_cache.load_col_stats_cache(trans_fp, data_version)
    new_backend_cache.load_col_stats_cache(small_t1_fp, synthetic_data_version)
    stats_dict = {key: list(value.keys()) for key, value in col_stats.items()}
    assert new_backend_cache.get_col_stats(trans_fp, stats_dict, data_version,
                                           approx=True) == col_stats
    stats_dict2 = {k: list(v.keys()) for k, v in col_stats2.items()}

    assert new_backend_cache.get_col_stats(trans_2_fp, stats_dict2,
                                           data_version,
                                           approx=True) == col_stats2
    stats_dict_synthetic = {
        key: list(value.keys())
        for key, value in col_stats_synthetic.items()
    }
    assert new_backend_cache.get_col_stats(small_t1_fp, stats_dict_synthetic,
                                           synthetic_data_version,
                                           approx=True) == col_stats_synthetic

    # Check min col stats exists
    table_dir = new_backend_cache.create_table_cache_dir(
        'parquet_synthetic', 'small_t1', synthetic_data_version)
    assert osp.exists(table_dir.min_col_stats)

    # Try to get stats associated with a data version that doesn't exist
    assert not new_backend_cache.get_col_stats(trans_fp, stats_dict,
                                               data_version='-1', approx=True)
    stats_list = list(table_stats.keys())
    assert new_backend_cache.get_table_stats(graph['TRANS'].fingerprint(),
                                             stats_list,
                                             data_version) == table_stats
    pd.testing.assert_frame_equal(
        new_backend_cache.table_samples_cache.get_v1('parquet', 'TRANS',
                                                     data_version),
        table_samples, check_dtype=False)
    pd.testing.assert_frame_equal(
        new_backend_cache.table_samples_cache.get_v1('parquet_synthetic',
                                                     'small_t1',
                                                     synthetic_data_version),
        table_samples_synthetic, check_dtype=False)
    assert new_backend_cache.materialization_stage.get_stage(
        graph["TRANS"].fingerprint(),
        data_version) == MaterializationStage.INGEST

    # Verify that if the materializatio path for a table does not exist,
    # backend cache does not load table materialization from cache dir.
    fsutil.rm(materialization_path)
    another_backend_cache = BackendCacheManager(cache_dir)
    parquet_trans_fp = IngestedTableFingerprint('parquet', 'TRANS', cols={},
                                                pkey=None, time_col=None,
                                                end_time_col=None)
    another_backend_cache.load_col_stats_cache(parquet_trans_fp, data_version)
    assert another_backend_cache.materialization_stage.get_stage(
        graph["TRANS"].fingerprint(), data_version) is None

    # Test get_many_exact_stats
    tables = [
        # Should return None for invalid table name
        (VersionedSourceTableID(connector.name, 'INVALID_TABLE',
                                data_version), None),
        (VersionedSourceTableID(connector.name, 'TRANS',
                                data_version), trans_fp),
        (VersionedSourceTableID(connector2.name, 'small_t1',
                                synthetic_data_version), small_t1_fp),
    ]
    got = backend_cache.get_many_min_exact_stats(tables)
    assert got == [None, None, None]

    # Should return exact stats, once we upload exact stats to the cache
    for ts in (table_stats, table_stats_synthetic):
        for _, stat in ts.items():
            stat.approx = False

    for cs in (col_stats, col_stats2, col_stats_synthetic):
        for _, stats in cs.items():
            for _, stat in stats.items():
                stat.approx = False

    backend_cache.add_col_stats(trans_fp, col_stats)
    backend_cache.add_col_stats(trans_fp, col_stats2)
    backend_cache.add_table_stats(graph['TRANS'].fingerprint(), table_stats)
    backend_cache.add_col_stats(small_t1_fp, col_stats_synthetic)
    backend_cache.add_table_stats(graph['small_t1'].fingerprint(),
                                  table_stats_synthetic)
    backend_cache.save_col_stats_cache(trans_fp, data_version)
    backend_cache.save_col_stats_cache(small_t1_fp, synthetic_data_version)
    tables = [(VersionedSourceTableID(connector.name, 'TRANS', data_version),
               graph['TRANS'].fingerprint()),
              (VersionedSourceTableID(connector2.name, 'small_t1',
                                      synthetic_data_version),
               graph['small_t1'].fingerprint())]
    got = backend_cache.get_many_min_exact_stats(tables)
    # The response data structure should have plain dataclasses instead of Stat
    # objects (for performance reasons)
    expect = ExactStats(
        {
            stattype: stat.config()
            for stattype, stat in table_stats.items()
        },
        {
            "K_SYMBOL": {
                stattype: stat.config()
                for stattype, stat in col_stats[
                    graph['TRANS']['K_SYMBOL'].base_config()].items()
            },
            "BANK": {
                stattype: stat.config()
                for stattype, stat in col_stats2[
                    graph['TRANS']['BANK'].base_config()].items()
            },
        },
    )
    expect_synthetic = ExactStats(
        {
            stattype: stat.config()
            for stattype, stat in table_stats_synthetic.items()
        },
        {
            "test__col__name": {
                stattype: stat.config()
                for stattype, stat in col_stats_synthetic[
                    graph['small_t1']['test__col__name'].base_config()].items()
            },
        },
    )

    assert got == [expect, expect_synthetic]


def test_graph_cache(tempdir):
    cache_dir = osp.join(tempdir.name, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    connector = ParquetConnector('parquet', root_dir)
    graph.connect(connector)
    graph.add(name='LOAN', pkey='LOAN_ID', fkeys=[('ACCOUNT_ID')])
    graph.add(name='ACCOUNT', pkey='ACCOUNT_ID')
    graph.add(name='ORDERS', pkey=None, fkeys=[('ACCOUNT_ID')])
    graph.infer_links()
    graph.infer_metadata()

    graph_version = 'testversion'
    test_edge_stats = {
        EdgeStatType.NUM_SRC_NODES:
        Stats(value=15, approx=False, data_version=graph_version),
        EdgeStatType.NUM_DST_NODES:
        Stats(value=1000, approx=False, data_version=graph_version)
    }

    src = SourceTableColumn('parquet', 'ACCOUNT', 'ACCOUNT_ID')
    dst = SourceTableColumn('parquet', 'LOAN', 'ACCOUNT_ID')
    backend_cache.graph_cache.add_edge_stats(graph.name, src, dst,
                                             test_edge_stats, graph_version)

    new_backend_cache = BackendCacheManager(cache_dir)
    edge_stats = new_backend_cache.graph_cache.get_edge_stats(
        graph.name, src, dst, test_edge_stats, graph_version)
    assert edge_stats == test_edge_stats


def test_training_table_stats_cache(tempdir):
    cache_dir = osp.join(tempdir.name, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)
    dummy_training_table_stats = {
        NUM_TARGETS: Stats(1000, False),
        NUM_TRAIN_ROWS: Stats(800, False),
        NUM_TOTAL_ROWS: Stats(1000, False),
        NUM_PRED_ROWS: Stats(200, False),
    }
    op_id = uuid.uuid4().hex
    backend_cache.pquery_table_stats_cache.update(op_id,
                                                  dummy_training_table_stats)

    new_backend_cache = BackendCacheManager(cache_dir)
    assert new_backend_cache.pquery_table_stats_cache.get_stats(
        op_id, [
            NUM_TARGETS,
            NUM_TRAIN_ROWS,
            NUM_TOTAL_ROWS,
            NUM_PRED_ROWS,
        ]) == dummy_training_table_stats


def test_training_table_progress_cache(tempdir):
    cache_dir = osp.join(tempdir.name, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)
    dummy_timeframe_stats = TemporalNodeTimeFrameStats(
        min_date=pd.Timestamp('2020-01-01'),
        max_date=pd.Timestamp('2020-02-01'),
        anchor_date=pd.Timestamp('2020-01-01'),
        num_train_rows=3,
        num_targets=2,
        num_entities=2,
        target_distribution={
            'A': 0.5,
            'B': 0.5
        },
    )
    dummy_training_table_progress = {
        NUM_EXPECTED_ITERATIONS:
        Stats(5, False),
        ITERATION_DATA:
        Stats([dummy_timeframe_stats, dummy_timeframe_stats], False),
    }
    op_id = uuid.uuid4().hex
    backend_cache.pquery_table_progress_cache.put(
        op_id, dummy_training_table_progress)

    new_backend_cache = BackendCacheManager(cache_dir)
    loaded_progress = new_backend_cache.pquery_table_progress_cache.get_stats(
        op_id)
    assert loaded_progress[NUM_EXPECTED_ITERATIONS].value == 5
    assert len(loaded_progress[ITERATION_DATA].value) == 2
    for item in loaded_progress[ITERATION_DATA].value:
        assert pd.Timestamp(
            item['min_date']) == pd.Timestamp('2020-01-01T00:00:00')
        assert pd.Timestamp(
            item['max_date']) == pd.Timestamp('2020-02-01T00:00:00')
        assert pd.Timestamp(
            item['anchor_date']) == pd.Timestamp('2020-01-01T00:00:00')
        assert item['num_train_rows'] == 3
        assert item['num_targets'] == 2
        assert item['target_distribution'] == {'A': 0.5, 'B': 0.5}


def test_table_version_cache(tempdir):
    cache_dir = osp.join(tempdir.name, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)
    reset_active_cache_for_test(backend_cache)

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')
    connector = CSVConnector('csv', root_dir, na_values='?')
    graph = Graph(name='FINANCIAL')
    graph.connect(connector)
    graph.add('ORDERS')
    graph.add('TRANS')
    dummy_data_version = 'dummy_data_version'

    data_version = connector.get_data_version('ORDERS', skip_cache=True)
    assert backend_cache.table_version_cache.get(connector.name,
                                                 'ORDERS') == data_version
    assert connector.get_data_version('ORDERS') == data_version

    backend_cache.table_version_cache.save(connector.name, 'TRANS',
                                           dummy_data_version)
    assert backend_cache.table_version_cache.get(connector.name,
                                                 'TRANS') == dummy_data_version
    assert connector.get_data_version('TRANS') == dummy_data_version

    got_version = backend_cache.table_version_cache.get_many(
        [SourceTableID(connector.name, 'TRANS')],
        TableIngestionStatus.SUCCESSFUL)
    expect_version = [
        VersionedSourceTableID(connector.name, 'TRANS', dummy_data_version)
    ]
    assert got_version == expect_version
    got_completed_version = (backend_cache.table_version_cache.get_many(
        [SourceTableID(connector.name, 'TRANS')],
        TableIngestionStatus.COMPLETED))
    assert got_completed_version == expect_version
    another_connector = CSVConnector('csv', root_dir, na_values='?')
    assert another_connector.get_data_version('ORDERS') == data_version
    assert another_connector.get_data_version('TRANS') == dummy_data_version

    # Validate the saving of failed data versions
    failed_data_version = 'failed_data_version'
    backend_cache.table_version_cache.save(
        connector.name, 'TRANS', failed_data_version,
        is_version_ingestion_successful=False)
    trans_id = SourceTableID(connector.name, 'TRANS')
    assert backend_cache.table_version_cache.get_many(
        [trans_id], TableIngestionStatus.SUCCESSFUL) == expect_version
    expect_failed_version = [
        VersionedSourceTableID(connector.name, 'TRANS', failed_data_version)
    ]
    assert backend_cache.table_version_cache.get_many(
        [trans_id], TableIngestionStatus.FAILED) == expect_failed_version
    assert backend_cache.table_version_cache.get_many(
        [trans_id], TableIngestionStatus.COMPLETED) == expect_failed_version

    job_id = 'dummy_job_id'
    data_version = 'dummy_data_version'
    backend_cache.table_version_cache.save_for_job(job_id, 'csv', 'ORDERS',
                                                   data_version)
    backend_cache.table_version_cache.save_for_job(job_id, 'csv', 'TRANS',
                                                   data_version)

    graph.set_table_data_versions(job_id)
    assert graph['ORDERS'].data_version == data_version
    assert graph['TRANS'].data_version == data_version
    # Verify that when connector gets garbage collected, the entry in the
    # hashmap is also purged.
    os.environ['CLEAN_TABLE_VERSION_CACHE'] = 'true'
    del connector
    del graph
    assert len(backend_cache.table_version_cache._table_versions) == 0
    os.environ['CLEAN_TABLE_VERSION_CACHE'] = ''


def test_table_version_persistence_does_not_hold_in_memory_cache_lock(
        monkeypatch, tempdir):
    cache_dir = osp.join(tempdir.name, CACHE_DIR)
    cache = BackendCacheManager(cache_dir).table_version_cache
    write_started = threading.Event()
    cleanup_finished = threading.Event()

    def save_while_connector_cleanup_runs(*args, **kwargs):
        del args, kwargs
        write_started.set()
        assert cleanup_finished.wait(timeout=2)

    monkeypatch.setattr(table_version_cache_module, 'save_json',
                        save_while_connector_cleanup_runs)

    def cleanup_connector_cache():
        assert write_started.wait(timeout=2)
        cache.remove('connector')
        cleanup_finished.set()

    cleanup_thread = threading.Thread(target=cleanup_connector_cache)
    cleanup_thread.start()
    cache.save_for_job('job', 'connector', 'table', 'version')
    cleanup_thread.join(timeout=2)
    assert not cleanup_thread.is_alive()


def test_minify_stats():
    # test nop
    in_nop = {
        "key_CATEGORY_COUNTS__True": None,
        "key_CATEGORY_COUNTS__False": [(str(x), x) for x in range(5)],
        "key2_CATEGORY_COUNTS__False": [(str(x), x) for x in range(20)],
        "key_nop": "value_nop",
    }
    out_nop: Dict = _minify_stats(in_nop)
    assert in_nop == out_nop

    in_filter = {
        "key_CATEGORY_COUNTS__True": [(str(x), x) for x in range(21)],
        "key2_CATEGORY_COUNTS__False": [(str(x), x) for x in range(99)],
        "key_nop": "value_nop",
    }
    exp_filter = {
        "key_CATEGORY_COUNTS__True": [(str(x), x) for x in range(20)],
        "key2_CATEGORY_COUNTS__False": [(str(x), x) for x in range(20)],
        "key_nop": "value_nop",
    }
    # test filter
    out_filter: Dict = _minify_stats(in_filter)
    assert exp_filter == out_filter


def test_merge_col_stats(tmp_path: pathlib.Path):
    cache_dir = osp.join(tmp_path, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    connector = ParquetConnector('parquet', root_dir)
    graph.connect(connector)
    graph.add('TRANS')
    graph.infer_metadata()

    data_version = connector.get_data_version('TRANS', skip_cache=True)
    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {
        graph["TRANS"]["K_SYMBOL"].base_config(): {
            ColStatType.KURTOSIS:
            Stats(value=-1.093549928703169, approx=False,
                  data_version=data_version, num_entries=5000),
        }
    }
    trans_fp = graph['TRANS'].fingerprint()
    backend_cache.add_col_stats(trans_fp, col_stats)
    backend_cache.save_col_stats_cache(trans_fp, data_version)

    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {
        graph["TRANS"]["K_SYMBOL"].base_config(): {
            ColStatType.KURTOSIS:
            Stats(value=1.093549928703169, approx=False,
                  data_version=data_version, num_entries=5000),
            ColStatType.COUNT:
            Stats(value=5000, approx=False, data_version=data_version,
                  num_entries=5000),
        }
    }
    backend_cache.add_col_stats(trans_fp, col_stats)
    # The value in the memory cache get updated.
    cache_key = BackendCacheManager._get_col_stats_cache_key(
        trans_fp, data_version)
    assert (backend_cache._col_stats_cache[cache_key]
            ['K_SYMBOL__kumo.categorical__KURTOSIS__False'].value ==
            1.093549928703169)

    backend_cache.save_col_stats_cache(trans_fp, data_version)
    backend_cache.load_col_stats_cache(trans_fp, data_version)
    # Merge order is memory-wins: after save+load the in-memory cache
    # keeps the merged value (+1.0935...) rather than the disk's legacy
    # value (-1.0935...).
    assert (backend_cache._col_stats_cache[cache_key]
            ['K_SYMBOL__kumo.categorical__KURTOSIS__False']['value'] ==
            1.093549928703169)


def test_backward_compatibility_dropped_rows_table_stat(tmp_path: pathlib.Path):
    """Test backward compatibility due to DROPPED_ROWS being moved from a
     TableStat to a ColStat.

    Any table ingested via release version <= v2.13 will have DROPPED_ROWS as a
     TableStat.
    """
    cache_dir = osp.join(tmp_path, CACHE_DIR)
    backend_cache = BackendCacheManager(cache_dir)

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    connector = ParquetConnector('parquet', root_dir)
    graph.connect(connector)
    graph.add('TRANS')

    data_version = connector.get_data_version('TRANS', skip_cache=True)
    trans_fp = graph['TRANS'].fingerprint()

    # Create a mock table stats JSON that includes the old DROPPED_ROWS
    # table stat
    # This simulates old cached data that has DROPPED_ROWS as a table stat
    mock_table_stats = {
        "NUM_ROWS__False": {
            "value": 5000,
            "approx": False,
            "data_version": data_version,
            "num_entries": 5000
        },
        "SOURCE_SIZE__False": {
            "value": 1024000,
            "approx": False,
            "data_version": data_version,
            "num_entries": 5000
        },
        # This is the old format that should be handled by backward
        # compatibility
        "DROPPED_ROWS__False": {
            "value": 100,
            "approx": False,
            "data_version": data_version,
            "num_entries": 5000
        }
    }

    # Create the cache directory structure
    table_cache = backend_cache.create_table_cache_dir(connector.name, 'TRANS',
                                                       data_version, trans_fp)

    # Write the mock data to files
    import json
    with open(table_cache.table_stats, 'w') as f:
        json.dump(mock_table_stats, f)

    # Test that get_many_min_exact_stats can parse the old format without
    # crashing
    tables = [(VersionedSourceTableID(connector.name, 'TRANS',
                                      data_version), trans_fp)]

    stats = backend_cache.get_many_min_exact_stats(tables, skip_col_stats=True)

    assert len(stats) == 1
    assert stats[0] is not None

    # The result should contain the valid table stats but not DROPPED_ROWS
    exact_stats = stats[0]
    assert len(exact_stats.table_stats) == 2
    assert TableStatType.NUM_ROWS in exact_stats.table_stats
    assert TableStatType.SOURCE_SIZE in exact_stats.table_stats

    # Verify the TableStats are correct
    assert exact_stats.table_stats[TableStatType.NUM_ROWS].value == 5000
    assert exact_stats.table_stats[TableStatType.SOURCE_SIZE].value == 1024000


def test_copying_stats_a_job_never_wrote_is_a_no_op(tmp_path):
    r"""A job that generated nothing wrote no stats.

    Every reader of this cache already answers ``None``/``{}`` for a key it
    does not hold, so the absence is readable. ``fsutil.copy`` is not: it
    raises for a source that is not there. The guard keeps the absence an
    absence rather than a failure -- which is what a registered training
    table, generated by nothing, leaves behind.
    """
    cache = BackendCacheManager(str(tmp_path / 'cache'))

    cache.copy_pquery_table_stats_cache('gen-traintable-job-registered',
                                        'trainingjob-1')

    assert not fsutil.exists(
        osp.join(cache.cache_dir, 'train_tables', 'trainingjob-1'))
    assert cache.pquery_table_stats_cache.get_stats('trainingjob-1') == {}


def test_copying_stats_a_job_did_write_still_copies_them(tmp_path):
    cache = BackendCacheManager(str(tmp_path / 'cache'))
    source = osp.join(cache.cache_dir, 'train_tables', 'gen-traintable-job-1')
    fsutil.makedirs(source, exist_ok=True)
    with open(osp.join(source, 'pquery_table_stats'), 'w') as handle:
        handle.write('{}')

    cache.copy_pquery_table_stats_cache('gen-traintable-job-1', 'trainingjob-1')

    assert fsutil.exists(
        osp.join(cache.cache_dir, 'train_tables', 'trainingjob-1',
                 'pquery_table_stats'))
