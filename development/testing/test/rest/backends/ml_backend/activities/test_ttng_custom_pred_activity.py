# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from importlib import import_module
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

import pytest
from kumoapi.typing import Dtype

mpt = import_module(
    'rest.backends.ml_backend.activities.materialize_prediction_task')


def _install_common_fakes(monkeypatch, *, task_type=None, entity_time_col=None,
                          target_stype='numerical', query_type=None):
    from kumoapi.pquery import QueryType
    if query_type is None:
        query_type = QueryType.STATIC
    monkeypatch.setattr(mpt, 'open_connectors', lambda *_a, **_k: None)
    monkeypatch.setattr(mpt, 'get_diskgraph_index_path_for_job',
                        lambda *_a, **_k: '/tmp/index')

    if task_type is None:
        task_type = SimpleNamespace(value='node_prediction', is_link_pred=False)

    entity_time = (SimpleNamespace(
        name=entity_time_col) if entity_time_col else None)
    entity_table = SimpleNamespace(pkey_name='customer_id',
                                   pkey=SimpleNamespace(dtype=Dtype.string),
                                   time_col=entity_time)

    graph = SimpleNamespace(
        set_table_data_versions=lambda *_a, **_k: None,
        table=lambda _name: entity_table,
    )
    validated = SimpleNamespace(
        entity_column='Customer.customer_id',
        target_stype=(SimpleNamespace(
            value=target_stype) if target_stype else None),
    )
    pquery = SimpleNamespace(
        graph=graph,
        validated_predictive_query=validated,
        get_task_type=lambda: task_type,
        get_batch_pred_table_generator=(
            lambda: SimpleNamespace(query_type=query_type)),
    )

    class _Config:
        def with_credentials(self, credentials):
            return self

    monkeypatch.setattr(mpt.PredictiveQueryV2, 'from_config',
                        staticmethod(lambda *_a, **_k: pquery))
    monkeypatch.setattr(
        mpt,
        'hydrate_predictive_query_for_activity',
        lambda config, _references: (config, {}),
    )
    return pquery, _Config()


def _job_input(config):
    return SimpleNamespace(
        id='batchpredictionjob-bp1',
        predictive_query=config,
        graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                 connectors={}),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )


def test_custom_pred_activity_builds_bundle_and_does_not_ingest(monkeypatch):
    """The activity derives args from the pquery, calls the build+persist
    helper, and does NOT ingest."""
    _pquery, config = _install_common_fakes(monkeypatch)

    build_calls = []
    monkeypatch.setattr(mpt, '_build_ttng_custom_pred_table_artifacts',
                        lambda **kw: build_calls.append(kw))

    ingest_calls = []
    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(execute_tt_ingest=lambda **kw: ingest_calls.append(kw)))

    mpt.build_ttng_custom_pred_table_bundle(
        job_input=_job_input(config),
        settings=SimpleNamespace(),
        pred_table_path='/tmp/custom/pred_table',
    )

    assert len(build_calls) == 1
    kw = build_calls[0]
    assert kw['pred_table_path'] == '/tmp/custom/pred_table'
    assert kw['diskgraph_index_path'] == '/tmp/index'
    # tt_id == pred_<bp_job_id> (matches predict_staging_tt_id the consumer
    # resolves from the bundle spec).
    assert kw['ttng_tt_id'] == 'pred_batchpredictionjob-bp1'
    assert kw['pquery'] is _pquery
    assert ingest_calls == [], 'custom-pred activity must not ingest'


def test_custom_pred_artifacts_persists_bundle_and_calls_dg_predict(
        monkeypatch, tmp_path):
    """Integration: with a fake DG PREDICT builder, the real helper persists
    the bundle to ttng_persisted_pred_staging_dir and passes is_prediction
    args (no split_config)."""
    import os
    from datetime import datetime, timezone

    from kumoapi.pquery import QueryType

    from kumo.data.generic import diskgraph_train_table
    from kumo.data.generic.diskgraph_staging import (
        ttng_persisted_pred_staging_dir,
    )
    pquery, _config = _install_common_fakes(monkeypatch,
                                            query_type=QueryType.TEMPORAL)

    local_pred_table_path = str(tmp_path / 'custom' / 'pred_table')
    # Real parquet so the schema validation can read it.
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Named as DiskGraph reads it. Nothing renames the graph's own primary
    # key onto ENTITY1 on the way in any more.
    for shard, customer_id in enumerate(('c1', 'c2')):
        partition = os.path.join(local_pred_table_path, f'shard={shard}')
        os.makedirs(partition, exist_ok=True)
        pq.write_table(
            pa.table({
                'ENTITY1': [customer_id],
                'TIMESTAMP':
                pa.array(
                    [datetime(2026, 7, 11, shard, tzinfo=timezone.utc)],
                    type=pa.timestamp('ms', tz='UTC'),
                ),
            }), os.path.join(partition, 'part-00000.parquet'))

    predict_calls = []

    def _build_predict(**kwargs):
        normalized = pq.read_table(kwargs['output_path'])
        assert normalized.schema.names == ['ENTITY1', 'TIMESTAMP', 'shard']
        assert normalized.schema.field('TIMESTAMP').type == pa.timestamp('ns')
        assert sorted(normalized['ENTITY1'].to_pylist()) == ['c1', 'c2']
        predict_calls.append(kwargs)
        # The native builder puts the output beside its normalized input. This
        # directory is owned by normalized_scratch, so the production helper
        # must persist it before leaving that context.
        bundle_dir = write_v3_bundle(
            os.path.dirname(kwargs['output_path']),
            kwargs['tt_id'],
            rows=2,
            ingest_spec_schema_version='diskgraph.tt_ingest_spec.v3',
        )
        return {'staging_dir': bundle_dir}

    import diskgraph

    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(
            build_prediction_table_artifacts_for_existing_table=_build_predict,
            # The column contract is read off DiskGraph, so a stand-in for
            # DiskGraph has to carry the real one rather than a made-up one.
            CUSTOM_TT_CONTRACT=diskgraph.CUSTOM_TT_CONTRACT))

    pred_table_path = local_pred_table_path
    durable = ttng_persisted_pred_staging_dir(pred_table_path)
    original_persist = diskgraph_train_table.persist_durable_ttng_bundle
    persisted = []

    def _persist(source, destination, *, expected_tt_id=None):
        assert os.path.exists(os.path.join(source, 'tt.parquet'))
        assert os.path.exists(os.path.join(source, 'tt_ingest_spec.json'))
        assert expected_tt_id == 'pred_bp1'
        persisted.append((source, destination, expected_tt_id))
        original_persist(
            source,
            destination,
            expected_tt_id=expected_tt_id,
        )

    monkeypatch.setattr(diskgraph_train_table, 'persist_durable_ttng_bundle',
                        _persist)

    mpt._build_ttng_custom_pred_table_artifacts(
        pred_table_path=pred_table_path,
        pquery=pquery,
        diskgraph_index_path='/tmp/index',
        ttng_tt_id='pred_bp1',
    )

    assert len(predict_calls) == 1
    call = predict_calls[0]
    # PREDICT entry: no split_config / split_family_hint.
    assert 'split_config' not in call
    assert 'split_family_hint' not in call
    assert call['tt_id'] == 'pred_bp1'
    assert call['index_path'] == '/tmp/index'
    assert call['split_stats_config'] == {
        'entity_table': 'Customer',
        'entity_pk_column': 'customer_id'
    }
    assert not call['output_path'].startswith('abfss://')
    assert len(persisted) == 1
    assert persisted[0][1] == durable
    # Bundle persisted beside the pred table.
    published = diskgraph_train_table.resolve_durable_ttng_bundle(durable)
    assert os.path.exists(os.path.join(published, 'tt.parquet'))
    assert os.path.exists(os.path.join(published, 'tt_ingest_spec.json'))


def test_remote_custom_pred_input_localizes_for_dg_builder(monkeypatch):
    from kumo.data.generic import diskgraph_train_table

    remote = ('abfss://container@account.dfs.core.windows.net/jobs/bp1/'
              'inputs/custom_prediction_table')
    copies = []
    monkeypatch.setattr(diskgraph_train_table.fsutil, 'is_local',
                        lambda _path: False)
    monkeypatch.setattr(
        diskgraph_train_table.fsutil,
        'copy_tree_threaded',
        lambda source, destination: copies.append((source, destination)),
    )

    with diskgraph_train_table.local_ttng_builder_input(
            remote, 'custom_prediction_table.parquet') as localized:
        assert localized.endswith('/custom_prediction_table.parquet')
        assert not localized.startswith('abfss://')

    assert copies == [(remote, localized)]


def _write_pred_parquet(path, cols):
    import pyarrow as pa
    import pyarrow.parquet as pq
    arrays = {c: pa.array([1, 2], pa.int64()) for c in cols}
    pq.write_table(pa.table(arrays), path)


def test_validation_rejects_target_column(monkeypatch, tmp_path):
    part = str(tmp_path / 'p.parquet')
    _write_pred_parquet(part, ['ENTITY1', 'TIMESTAMP', 'TARGET1'])
    with pytest.raises(mpt.PredictionDataMaterializationError,
                       match='must not carry a target'):
        mpt._validate_custom_pred_table_schema(parquet_path=part,
                                               entity_pk_dtype=Dtype.int64,
                                               is_temporal=False)


def test_validation_rejects_an_entity_column_by_any_other_name(
        monkeypatch, tmp_path):
    """The graph's own primary-key column name is not an accepted spelling,
    and neither is a singular ``ENTITY``. DiskGraph looks this column up as
    ``ENTITY1``."""
    part = str(tmp_path / 'p.parquet')
    _write_pred_parquet(part, ['customer_id', 'ENTITY', 'TIMESTAMP'])
    with pytest.raises(mpt.PredictionDataMaterializationError,
                       match="missing 'ENTITY1'"):
        mpt._validate_custom_pred_table_schema(parquet_path=part,
                                               entity_pk_dtype=Dtype.int64,
                                               is_temporal=False)


def test_validation_rejects_missing_entity_pkey(monkeypatch, tmp_path):
    part = str(tmp_path / 'p.parquet')
    _write_pred_parquet(part, ['other_col', 'TIMESTAMP'])
    with pytest.raises(mpt.PredictionDataMaterializationError,
                       match="missing 'ENTITY1'"):
        mpt._validate_custom_pred_table_schema(parquet_path=part,
                                               entity_pk_dtype=Dtype.int64,
                                               is_temporal=False)


def test_validation_temporal_rejects_missing_timestamp(monkeypatch, tmp_path):
    """A TEMPORAL task needs a per-row anchor time, so a missing TIMESTAMP is
    rejected."""
    part = str(tmp_path / 'p.parquet')
    _write_pred_parquet(part, ['ENTITY1'])
    with pytest.raises(mpt.PredictionDataMaterializationError,
                       match='no TIMESTAMP'):
        mpt._validate_custom_pred_table_schema(parquet_path=part,
                                               entity_pk_dtype=Dtype.int64,
                                               is_temporal=True)


def test_validation_static_accepts_missing_timestamp(monkeypatch, tmp_path):
    """A STATIC task may omit TIMESTAMP -- DG synthesizes a sentinel per-row
    timestamp at generation (mirrors the native PQL/custom-train static path).
    """
    part = str(tmp_path / 'p.parquet')
    _write_pred_parquet(part, ['ENTITY1'])
    # Should not raise.
    mpt._validate_custom_pred_table_schema(parquet_path=part,
                                           entity_pk_dtype=Dtype.int64,
                                           is_temporal=False)


def test_validation_accepts_valid_table(monkeypatch, tmp_path):
    part = str(tmp_path / 'p.parquet')
    _write_pred_parquet(part, ['ENTITY1', 'TIMESTAMP'])
    mpt._validate_custom_pred_table_schema(parquet_path=part,
                                           entity_pk_dtype=Dtype.int64,
                                           is_temporal=True)


def test_validation_rejects_an_entity_type_diskgraph_cannot_resolve(
        monkeypatch, tmp_path):
    """A boolean entity key survives ``cell_to_value`` and then resolves to no
    node, so every row is silently skipped."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    part = str(tmp_path / 'p.parquet')
    pq.write_table(
        pa.table({
            'ENTITY1': pa.array([True, False], pa.bool_()),
            'TIMESTAMP': pa.array([0, 0], pa.int64()),
        }), part)

    with pytest.raises(mpt.PredictionDataMaterializationError,
                       match='An entity key is read as one of'):
        mpt._validate_custom_pred_table_schema(parquet_path=part,
                                               entity_pk_dtype=Dtype.int64,
                                               is_temporal=False)


def test_validation_rejects_an_entity_the_graph_key_cannot_resolve(tmp_path):
    """The second question, after the contract's: DiskGraph reads a string
    ``ENTITY1`` fine and then resolves it against an int64 graph key, which
    matches nothing -- every row skipped, success reported. Asked through
    ``kumo_core.training.custom_table``, which the training path asks too."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    part = str(tmp_path / 'p.parquet')
    pq.write_table(
        pa.table({
            'ENTITY1': pa.array(['a', 'b'], pa.string()),
            'TIMESTAMP': pa.array([0, 0], pa.int64()),
        }), part)

    with pytest.raises(mpt.PredictionDataMaterializationError,
                       match='graph primary key requires'):
        mpt._validate_custom_pred_table_schema(parquet_path=part,
                                               entity_pk_dtype=Dtype.int64,
                                               is_temporal=False)


def test_validation_accepts_a_string_entity_against_a_string_key(tmp_path):
    """The same pairing, agreeing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    part = str(tmp_path / 'p.parquet')
    pq.write_table(
        pa.table({
            'ENTITY1': pa.array(['a', 'b'], pa.string()),
            'TIMESTAMP': pa.array([0, 0], pa.int64()),
        }), part)

    mpt._validate_custom_pred_table_schema(parquet_path=part,
                                           entity_pk_dtype=Dtype.string,
                                           is_temporal=True)


# ---------------------------------------------------------------------------
# generate_prediction_config: custom-pred relocation gate
# ---------------------------------------------------------------------------


def _stub_generate_prediction_config_deps(monkeypatch, *, task_type, saved):
    """Stub every collaborator of generate_prediction_config so the test only
    exercises pred-config assembly. ``saved`` collects the PredictionConfig
    that would be persisted."""
    monkeypatch.setattr(mpt, 'get_diskgraph_index_path_for_job',
                        lambda *_a, **_k: '/tmp/index')

    graph = SimpleNamespace(
        set_table_data_versions=lambda *_a, **_k: None,
        table_names=['Customer'],
    )
    pquery = SimpleNamespace(graph=graph, name='pq')

    class _Config:
        name = 'pq'

        def with_credentials(self, credentials):
            return self

    monkeypatch.setattr(mpt.PredictiveQueryV2, 'from_config',
                        staticmethod(lambda *_a, **_k: pquery))
    monkeypatch.setattr(
        mpt,
        'hydrate_predictive_query_for_activity',
        lambda config, _references: (config, {}),
    )

    # I/O + metadata collaborators -> no-ops / trivial returns.
    monkeypatch.setattr(mpt, 'estimate_input_data_stats',
                        lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(mpt, 'save_json', lambda *_a, **_k: None)
    monkeypatch.setattr(mpt, 'deploy_env', lambda *_a, **_k: 'test')
    monkeypatch.setattr(mpt, 'get_job_dir', lambda *_a, **_k: '/tmp/job')
    monkeypatch.setattr(mpt, 'get_job_base_dir', lambda *_a, **_k: '/tmp')
    monkeypatch.setattr(mpt, 'get_persist_dir', lambda *_a, **_k: '/tmp/d')
    monkeypatch.setattr(mpt, 'return_embeddings', lambda *_a, **_k: False)
    monkeypatch.setattr(mpt, 'return_predictions', lambda *_a, **_k: True)
    monkeypatch.setattr(mpt, '_configure_prediction_output_fields',
                        lambda *_a, **_k: None)

    class _FakeHomeDirMgr:
        def __init__(self, *_a, **_k):
            self.estimated_input_data_stats_path = '/tmp/stats.json'
            self.prediction_output_dir = '/tmp/out'
            self.jobs_dir = '/tmp/jobs'

        def get_best_trainer_dir(self, *_a, **_k):
            return '/tmp/model'

        def job_config_path(self, *_a, **_k):
            return '/tmp/pred_config.json'

    monkeypatch.setattr(mpt, 'HomeDirManager', _FakeHomeDirMgr)
    monkeypatch.setattr(mpt.fsutil, 'makedirs', lambda *_a, **_k: None)

    # Capture the PredictionConfig instead of constructing the real one.
    def _fake_pred_config(**kwargs):
        cfg = SimpleNamespace(**kwargs)
        saved.append(cfg)
        return cfg

    monkeypatch.setattr(mpt, 'PredictionConfig', _fake_pred_config)

    def _job_input_full():
        cfg = SimpleNamespace(
            task_type=task_type,
            binary_classification_threshold=None,
            num_classes_to_return=1,
            explanations=False,
        )
        return SimpleNamespace(
            id='batchpredictionjob-bp1',
            training_job_id='trainingjob-t1',
            predictive_query=_Config(),
            config=cfg,
            graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                     connectors={}),
            graph_version_info=SimpleNamespace(graph_data_version='v1'),
        )

    return _job_input_full


def _run_generate_prediction_config(monkeypatch, *, task_type, pred_table_path):
    saved: list = []
    make_job_input = _stub_generate_prediction_config_deps(
        monkeypatch, task_type=task_type, saved=saved)
    mpt.generate_prediction_config(
        job_input=make_job_input(),
        settings=SimpleNamespace(),
        serving=False,
        pred_table_path=pred_table_path,
    )
    assert len(saved) == 1
    return saved[0].pred_table_path


def test_generate_pred_config_ttng_custom_does_not_relocate(monkeypatch):
    """TT-NG custom pred keeps pred_table_path STABLE (bundle resolved at the
    original location); no relocation happens."""
    from kumo_core.types import TaskType
    original = '/customer/data/my_pred_table'
    result_path = _run_generate_prediction_config(
        monkeypatch, task_type=TaskType.BINARY_CLASSIFICATION,
        pred_table_path=original)
    assert result_path == original, 'TT-NG must not relocate the pred table'


def test_generate_pred_config_preserves_embeddings_only_request(
        monkeypatch):
    """The failed replay requested temporal-LP embeddings with no top-k."""
    from kumo_core.types import TaskType

    saved: list = []
    make_job_input = _stub_generate_prediction_config_deps(
        monkeypatch,
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        saved=saved,
    )
    monkeypatch.setattr(mpt, 'return_predictions', lambda *_a, **_k: False)
    monkeypatch.setattr(mpt, 'return_embeddings', lambda *_a, **_k: True)
    job_input = make_job_input()
    job_input.config.num_classes_to_return = None

    mpt.generate_prediction_config(
        job_input=job_input,
        settings=SimpleNamespace(),
        serving=False,
        pred_table_path='/tmp/embeddings-only-pred-table',
    )

    assert len(saved) == 1
    assert saved[0].return_predictions is False
    assert saved[0].return_embeddings is True
    assert saved[0].top_k is None


def test_rehydration_normalizes_the_anchor_and_renames_nothing(tmp_path):
    """The bridge used to map ``ENTITY``, ``TARGET`` and the graph's own
    primary-key column onto ``ENTITY1`` / ``TARGET1`` on the way in. That was
    the rename that made a misnamed table validate on this one path and fail
    everywhere else. What is left is the anchor cast.
    """
    from datetime import datetime, timezone

    import pyarrow as pa
    import pyarrow.parquet as pq

    from kumo.data.generic import _diskgraph_legacy_graphstore_bridge as bridge

    source = str(tmp_path / 'in.parquet')
    pq.write_table(
        pa.table({
            'customer_id': pa.array([1], pa.int64()),
            'ENTITY': pa.array([1], pa.int64()),
            'TARGET': pa.array([1], pa.int64()),
            'TIMESTAMP':
            pa.array([datetime(2026, 7, 11, tzinfo=timezone.utc)],
                     type=pa.timestamp('ms', tz='UTC')),
        }), source)

    out = bridge.normalize_legacy_parquet_for_rehydration(
        source, output_dir=str(tmp_path / 'out'), normalize_timestamp=True)

    schema = pq.read_schema(out)
    assert schema.names == ['customer_id', 'ENTITY', 'TARGET', 'TIMESTAMP']
    assert pa.types.is_timestamp(schema.field('TIMESTAMP').type)
    assert schema.field('TIMESTAMP').type.unit == 'ns'
