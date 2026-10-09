# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

import pytest

mtt = import_module(
    'rest.backends.ml_backend.activities.materialize_training_task')


class _DiskgraphDataError(RuntimeError):
    pass


def _install_common_fakes(monkeypatch):
    monkeypatch.setattr(mtt, 'open_connectors', lambda *_a, **_k: None)
    monkeypatch.setattr(mtt, 'get_job_base_dir', lambda: '/tmp/jobs')
    monkeypatch.setattr(mtt, 'get_job_dir', lambda *_a, **_k: '/tmp/jobs/j1')
    monkeypatch.setattr(mtt, 'get_diskgraph_index_path_for_job',
                        lambda *_a, **_k: '/tmp/index')

    class _Cache:
        table_version_cache = SimpleNamespace(remove=lambda *_a, **_k: None)

    monkeypatch.setattr(mtt, 'get_active_cache', lambda: _Cache())

    graph = SimpleNamespace(set_table_data_versions=lambda *_a, **_k: None)
    pquery = SimpleNamespace(
        graph=graph,
        validated_predictive_query=SimpleNamespace(target_stype=SimpleNamespace(
            value='numerical')),
    )
    monkeypatch.setattr(mtt, '_create_pquery_with_credentials',
                        lambda *_a, **_k: (pquery, {}))

    class _Split:
        def config(self):
            return {'_target_': 'kumo.task.split.FixedSplit'}

    train_table = SimpleNamespace(
        data_source='src',
        split=_Split(),
        time_col_name='TIMESTAMP',
        get_train_table_stats=lambda after_split=False: None,
    )
    task = SimpleNamespace(train_table=train_table, task_type='NODE')
    monkeypatch.setattr(mtt.AutoTrainer, 'parse_pquery', lambda *_a, **_k:
                        (task, None))

    class _FakeModelPlan:
        pass

    monkeypatch.setattr(mtt, 'ModelPlan', _FakeModelPlan)
    return _FakeModelPlan, pquery, task


def _job_input(model_plan_cls):
    return SimpleNamespace(
        id='trainingjob-j1',
        spec=SimpleNamespace(
            advanced_options=model_plan_cls(),
            split=None,
        ),
        graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                 connectors={}),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )


def test_custom_gen_activity_builds_bundle_and_does_not_ingest(
        monkeypatch, tmp_path):
    """C2: the new activity derives the build args from the parsed pquery,
    calls the shared build+persist helper, and does NOT ingest."""
    model_plan_cls, pquery, task = _install_common_fakes(monkeypatch)

    train_table_path = '/tmp/jobs/gen-traintable-j1/train_table'

    build_calls = []
    monkeypatch.setattr(
        mtt, '_build_ttng_custom_train_table_artifacts', lambda **kw:
        (build_calls.append(kw) or '/published/bundle/tt.parquet'))

    # No split_stats.json -> C3 sub-step is a no-op (assert no crash).
    monkeypatch.setattr(mtt.fsutil, 'exists', lambda _p: False)

    ingest_calls = []
    import diskgraph
    monkeypatch.setattr(
        diskgraph,
        'execute_tt_ingest',
        lambda **kw: ingest_calls.append(kw),
    )

    mtt.build_ttng_custom_train_table_bundle(
        job_input=_job_input(model_plan_cls),
        settings=SimpleNamespace(),
        train_table_path=train_table_path,
        weight_col_name=None,
    )

    assert len(
        build_calls) == 1, 'custom-gen must call the build+persist helper'
    kw = build_calls[0]
    assert kw['train_table_path'] == train_table_path
    assert kw['diskgraph_index_path'] == '/tmp/index'
    assert kw['ttng_tt_id'] == 'trainingjob-j1'
    assert kw['split_family'] == 'FixedSplit'
    assert kw['pquery'] is pquery
    assert kw['task'] is task
    # C2 invariant: the custom-gen activity must NOT ingest.
    assert ingest_calls == [], 'custom-gen activity must not call ingest'


def test_custom_gen_activity_builds_even_when_pre_split_stats_present(
        monkeypatch, tmp_path):
    """Regression (was in test_diskgraph_materialize_training_table.py): a
    custom TT exported WITH pre-split stats must STILL build+persist its durable
    bundle. The custom-gen activity has no missing_pre_split_stats gate, so the
    build always runs -- pin that here."""
    model_plan_cls, _pquery, task = _install_common_fakes(monkeypatch)
    # Pre-split stats ARE present.
    task.train_table.get_train_table_stats = lambda after_split=False: {'ok': 1}

    monkeypatch.setattr(mtt.fsutil, 'exists', lambda _p: False)

    build_calls = []
    monkeypatch.setattr(
        mtt, '_build_ttng_custom_train_table_artifacts', lambda **kw:
        (build_calls.append(kw) or '/published/bundle/tt.parquet'))

    mtt.build_ttng_custom_train_table_bundle(
        job_input=_job_input(model_plan_cls),
        settings=SimpleNamespace(),
        train_table_path='/tmp/jobs/gen-traintable-j1/train_table',
        weight_col_name=None,
    )

    assert len(build_calls) == 1, (
        'custom-gen must build the bundle even when pre-split stats present')


def test_custom_gen_activity_persists_real_bundle(monkeypatch, tmp_path):
    """C2 integration: with a fake DG builder, the real helper persists
    and returns one immutable tt.parquet + tt_ingest_spec.json bundle."""
    import os

    model_plan_cls, _pquery, _task = _install_common_fakes(monkeypatch)

    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
        ttng_persisted_staging_dir,
    )
    train_table_path = str(tmp_path / 'jobs' / 'gen-traintable-j1' /
                           'train_table')
    os.makedirs(os.path.dirname(train_table_path), exist_ok=True)

    # Real task carries the fkeys + graph the helper reads for entity metadata.
    task = SimpleNamespace(
        graph=SimpleNamespace(tables=[
            SimpleNamespace(table_name='Customer', name='Customer',
                            pkey_name='customer_id')
        ]),
        train_table=SimpleNamespace(
            data_source='src',
            split=SimpleNamespace(
                config=lambda: {'_target_': 'kumo.task.split.FixedSplit'}),
            time_col_name='TIMESTAMP',
            fkeys=[SimpleNamespace(dst_table='Customer')],
            get_train_table_stats=lambda after_split=False: None,
        ),
        task_type=SimpleNamespace(is_link_pred=False),
    )
    monkeypatch.setattr(mtt.AutoTrainer, 'parse_pquery', lambda *_a, **_k:
                        (task, None))

    bundle_dir = write_v3_bundle(str(tmp_path), 'trainingjob-j1')

    build_calls = []

    def _build_artifacts(**kwargs):
        build_calls.append(kwargs)
        return {'manifest_path': '/tmp/m.json', 'staging_dir': bundle_dir}

    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(
            DiskgraphDataError=_DiskgraphDataError,
            SplitSpec=lambda **kw: SimpleNamespace(kind=kw.get('kind'), **kw),
            build_train_table_artifacts_for_existing_table=_build_artifacts,
        ))

    import kumo.data.generic.diskgraph_train_table as dtt
    monkeypatch.setattr(
        dtt, '_build_split_spec',
        lambda cfg, fam, spec_cls: SimpleNamespace(kind='FixedSplit'))

    persisted = ttng_persisted_staging_dir(train_table_path)

    published_path = mtt.build_ttng_custom_train_table_bundle(
        job_input=_job_input(model_plan_cls),
        settings=SimpleNamespace(),
        train_table_path=train_table_path,
        weight_col_name=None,
    )

    published = ttng_bundle_dir_for_train_table(published_path)
    assert os.path.dirname(published) == os.path.join(persisted, '_bundles')
    assert published_path == os.path.join(published, 'tt.parquet')
    assert os.path.exists(os.path.join(published, 'tt.parquet'))
    assert os.path.exists(os.path.join(published, 'tt_ingest_spec.json'))
    assert len(build_calls) == 1
    assert build_calls[0]['output_path'] == train_table_path


def _remote_lane_task(is_link_pred=False):
    return SimpleNamespace(
        graph=SimpleNamespace(tables=[
            SimpleNamespace(table_name='Customer', name='Customer',
                            pkey_name='customer_id')
        ]),
        train_table=SimpleNamespace(
            data_source='src',
            split=SimpleNamespace(
                config=lambda: {'_target_': 'kumo.task.split.FixedSplit'}),
            time_col_name='TIMESTAMP',
            fkeys=[SimpleNamespace(dst_table='Customer')],
            get_train_table_stats=lambda after_split=False: None,
        ),
        task_type=SimpleNamespace(is_link_pred=is_link_pred),
    )


def _install_fake_dg_builder(monkeypatch):
    """Fake DG builder that writes ``<output_path>.artifacts`` LOCALLY (like
    the native wheel), so passing it a remote output_path would fail. Returns
    the recorded call list."""
    import os

    build_calls = []

    def _build_artifacts(**kwargs):
        build_calls.append(kwargs)
        staging_dir = write_v3_bundle(os.path.dirname(kwargs['output_path']),
                                      kwargs['tt_id'])
        return {'manifest_path': '/tmp/m.json', 'staging_dir': staging_dir}

    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(
            DiskgraphDataError=_DiskgraphDataError,
            SplitSpec=lambda **kw: SimpleNamespace(kind=kw.get('kind'), **kw),
            build_train_table_artifacts_for_existing_table=_build_artifacts,
        ))
    import kumo.data.generic.diskgraph_train_table as dtt
    monkeypatch.setattr(
        dtt, '_build_split_spec',
        lambda cfg, fam, spec_cls: SimpleNamespace(kind='FixedSplit'))
    return build_calls


def test_custom_gen_stages_remote_train_table_to_local_output(
        monkeypatch, tmp_path):
    """Regression: on the S3 storage lane ``train_table_path`` is remote, but
    the DG builder writes its ``<output_path>.artifacts`` bundle with local FS
    ops (fails ``os error 2`` on an ``s3://`` path). The helper must stage the
    customer parquet into a LOCAL scratch dir and pass that LOCAL path as
    ``output_path``, then persist the bundle to the durable (remote) location.
    """
    import os

    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
        ttng_persisted_staging_dir,
    )

    # A real local dir backs the "remote" table so copy_tree_threaded works;
    # ``is_local`` is forced to report it as remote to take the staging branch.
    remote_train_table_path = str(tmp_path / 'remote' / 'custom_train_table')
    os.makedirs(remote_train_table_path, exist_ok=True)
    (Path(remote_train_table_path) /
     'part-00000.parquet').write_bytes(b'PAR1data')

    build_calls = _install_fake_dg_builder(monkeypatch)
    monkeypatch.setattr(mtt, '_ttng_custom_tt_entity_metadata', lambda _task:
                        ('Customer', 'customer_id'))
    monkeypatch.setattr(mtt, '_ttng_custom_tt_target_stype',
                        lambda _pq: 'numerical')

    monkeypatch.setattr(
        mtt.fsutil, 'is_local',
        lambda p: not str(p).startswith(remote_train_table_path))

    published_path = mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=remote_train_table_path,
        split_config={'_target_': 'kumo.task.split.FixedSplit'},
        split_family='FixedSplit',
        task=_remote_lane_task(),
        pquery=SimpleNamespace(graph=SimpleNamespace(tables=[])),
        diskgraph_index_path='/tmp/index',
        ttng_tt_id='gen-traintable-j1',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    assert len(build_calls) == 1
    passed_output = build_calls[0]['output_path']
    # The DG builder must NOT receive the remote path (would fail os error 2).
    assert passed_output != remote_train_table_path
    assert mtt.fsutil.is_local(passed_output), (
        'the DG builder must receive a LOCAL output_path on the remote lane')

    persisted = ttng_persisted_staging_dir(remote_train_table_path)
    published = ttng_bundle_dir_for_train_table(published_path)
    assert os.path.dirname(published) == os.path.join(persisted, '_bundles')
    assert published_path == os.path.join(published, 'tt.parquet')
    assert os.path.exists(os.path.join(published, 'tt.parquet'))
    assert os.path.exists(os.path.join(published, 'tt_ingest_spec.json'))


def test_custom_gen_local_train_table_passes_path_through(
        monkeypatch, tmp_path):
    """On the local lane ``output_path`` must be forwarded unchanged (no
    needless staging copy) -- pins the no-regression side of the fix."""
    import os

    local_train_table_path = str(tmp_path / 'jobs' / 'custom_train_table')
    os.makedirs(local_train_table_path, exist_ok=True)
    (Path(local_train_table_path) / 'part-00000.parquet').write_bytes(b'PAR1d')

    build_calls = _install_fake_dg_builder(monkeypatch)
    monkeypatch.setattr(mtt, '_ttng_custom_tt_entity_metadata', lambda _task:
                        ('Customer', 'customer_id'))
    monkeypatch.setattr(mtt, '_ttng_custom_tt_target_stype',
                        lambda _pq: 'numerical')

    mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=local_train_table_path,
        split_config={'_target_': 'kumo.task.split.FixedSplit'},
        split_family='FixedSplit',
        task=_remote_lane_task(),
        pquery=SimpleNamespace(graph=SimpleNamespace(tables=[])),
        diskgraph_index_path='/tmp/index',
        ttng_tt_id='gen-traintable-j1',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    assert len(build_calls) == 1
    assert build_calls[0]['output_path'] == local_train_table_path


@pytest.mark.parametrize(
    'unit', ['seconds', 'milliseconds', 'microseconds', 'nanoseconds'])
def test_custom_gen_forwards_declared_timestamp_unit(monkeypatch, unit):
    model_plan_cls, _, _ = _install_common_fakes(monkeypatch)
    job_input = _job_input(model_plan_cls)
    job_input.spec.custom_train_table = SimpleNamespace(timestamp_unit=unit)
    calls = []
    monkeypatch.setattr(
        mtt, '_build_ttng_custom_train_table_artifacts', lambda **kw:
        (calls.append(kw) or '/published/bundle/tt.parquet'))
    monkeypatch.setattr(mtt, '_mirror_ttng_split_stats_to_job_dir',
                        lambda **kw: None)
    mtt.build_ttng_custom_train_table_bundle(
        job_input=job_input, settings=SimpleNamespace(),
        train_table_path='/tmp/custom.parquet')
    assert calls[0]['timestamp_unit'] == unit
    assert 'timestamp_reference' not in calls[0]
