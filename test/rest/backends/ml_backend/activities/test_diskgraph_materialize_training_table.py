# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

import pytest
from kumo_core.train.autoregression import (
    AUTOREGRESSION_COL,
    FUTURE_LABELS_COL,
    YOY_COL,
)
from kumoapi.encoder import Numerical, NumericalList
from kumoapi.typing import Dtype, Stype

mtt = import_module(
    'rest.backends.ml_backend.activities.materialize_training_task')
split_mod = import_module('rest.backends.ml_backend.activities.split')


def _write_current_v3_bundle_at(destination, tt_id):
    source = write_v3_bundle(os.path.dirname(destination), tt_id)
    os.replace(source, destination)
    return os.path.join(destination, 'tt.parquet')


def test_ttng_col_stats_sidecar_is_in_published_extension(tmp_path):
    train_table_path = _write_current_v3_bundle_at(
        str(tmp_path / 'published'),
        'trainingjob-1',
    )
    assert mtt._ttng_train_table_col_stats_sidecar_path(
        train_table_path,
        '/tmp/dg/graph/v1') == ('/tmp/dg/graph/v1/train_tables/trainingjob-1/'
                                'train_table_col_stats.json')


def test_custom_tt_split_override_defaults_to_original_split():
    class _Split:
        def config(self):
            return {'_target_': 'kumo.task.split.FixedSplit'}

    split = _Split()
    split_out, family = mtt._custom_tt_split_override(split, None)

    assert split_out is split
    assert family == 'FixedSplit'


def test_custom_tt_split_override_uses_the_column_the_spec_names():
    split_out, family = mtt._custom_tt_split_override(object(), 'fold')

    assert family == 'ManualSplit'
    assert split_out.col == 'fold'
    assert split_out.train_value == 0
    assert split_out.val_value == 1
    assert split_out.test_value == 2


def test_custom_tt_split_column_comes_from_the_job_spec():
    """A cloud-registered table naming its own split column used to have that
    name silently ignored: hosted read a deployment-wide environment variable
    and nothing else."""
    job_input = SimpleNamespace(spec=SimpleNamespace(
        custom_train_table=SimpleNamespace(split_col_name='fold')))

    assert mtt._custom_tt_split_col_name(job_input) == 'fold'


def test_a_job_with_no_custom_table_names_no_split_column():
    assert mtt._custom_tt_split_col_name(
        SimpleNamespace(spec=SimpleNamespace(custom_train_table=None))) is None


def test_ttng_task_parse_preserves_generated_logical_name():
    resolved = '/immutable/version/tt.parquet'
    assert mtt._ttng_train_table_parse_target(resolved, ) == (resolved,
                                                              'train_table')


def test_ttng_task_parse_preserves_custom_logical_name():
    resolved = '/immutable/version/tt.parquet'
    assert mtt._ttng_train_table_parse_target(
        resolved,
        is_custom_train_table=True,
    ) == (resolved, 'tt')


# ---------------------------------------------------------------------------
# custom-TT durable-bundle contract (spec kumo 6/8/9).
#
# The snapshot-restore bug: custom-TT fuses gen+ingest and never persists a
# re-ingestable bundle at ``ttng_persisted_staging_dir(custom_train_table_path)``.
# On a restored train pod the in-place ``train_tables/<tt_id>/`` extension is
# gone and ``_ttng_ensure_train_table_extension`` finds no durable bundle to
# re-ingest from -> raises. Normal gen persists that bundle
# (``generate_train_table.py:247-251``); custom-TT must do the same.
#
# These tests drive the REAL ``_build_ttng_custom_train_table_artifacts`` with
# a fake DG builder that writes a realistic bundle (``tt.parquet`` +
# ``tt_ingest_spec.json``) into a temp ``staging_dir`` and returns it, then
# assert the TARGET invariant: the custom step copies that bundle to
# ``ttng_persisted_staging_dir(train_table_path)``. RED today (the custom path
# never persists it), GREEN once the fix mirrors the normal-gen persist.
# ---------------------------------------------------------------------------


def _install_custom_tt_bundle_fakes(monkeypatch, tmp_path, *, is_link_pred):
    """Fake ``diskgraph`` + helpers so the real custom build runs offline.

    The fake DG builder materializes a self-describing bundle
    (``tt.parquet`` + ``tt_ingest_spec.json``) into ``<tmp>/dg_staging`` --
    exactly the bundle ``execute_tt_ingest`` consumes -- and returns its dir
    as ``staging_dir`` (mirroring the gen-only ``execute_tt_gen`` result).
    Returns ``(bundle_dir, calls)`` where ``calls['build']`` captures kwargs.
    """
    calls: dict = {}

    class _SplitSpec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    bundle_parent = str(tmp_path)

    def _build_artifacts(**kwargs):
        calls['build'] = kwargs
        bundle_dir = write_v3_bundle(
            bundle_parent,
            kwargs['tt_id'],
            ingest_spec_schema_version='diskgraph.tt_ingest_spec.v3',
        )
        return {
            'manifest_path': '/tmp/train_table.manifest.json',
            'staging_dir': bundle_dir,
            'tt_parquet_path': os.path.join(bundle_dir, 'tt.parquet'),
            'ingest_spec_path': os.path.join(bundle_dir, 'tt_ingest_spec.json'),
        }

    fake_dg = SimpleNamespace(
        DiskgraphDataError=type('DiskgraphDataError', (Exception, ), {}),
        TimeUnit=object,
        SplitSpec=_SplitSpec,
        build_train_table_artifacts_for_existing_table=_build_artifacts,
    )
    if is_link_pred:
        from kumo.pquery import utils as pquery_utils
        monkeypatch.setattr(
            pquery_utils,
            'get_rhs_lp_table',
            lambda *_a, **_k: 'Product',
        )
    monkeypatch.setitem(sys.modules, 'diskgraph', fake_dg)
    return bundle_parent, calls


def _custom_tt_task(is_link_pred=False):
    return SimpleNamespace(
        graph=SimpleNamespace(tables=[
            SimpleNamespace(table_name='Customer', name='Customer',
                            pkey_name='customer_id')
        ]),
        train_table=SimpleNamespace(
            fkeys=[SimpleNamespace(dst_table='Customer')]),
        task_type=SimpleNamespace(is_link_pred=is_link_pred),
    )


def _custom_tt_pquery():
    return SimpleNamespace(
        graph=SimpleNamespace(), validated_predictive_query=SimpleNamespace(
            target_stype=SimpleNamespace(value='categorical')))


def _custom_tt_split_config():
    return {
        '_target_': 'kumo.task.split.ManualSplit',
        'split_column': '__kumo_precomputed_split',
        'manual_train_values': ['0'],
        'manual_val_values': ['1'],
        'manual_test_values': ['2'],
    }


def test_custom_tt_persists_durable_bundle_to_persisted_staging_dir(
        monkeypatch, tmp_path):
    """Custom generation returns a selected, re-ingestable bundle."""
    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
        ttng_persisted_staging_dir,
    )
    _install_custom_tt_bundle_fakes(monkeypatch, tmp_path, is_link_pred=False)
    train_table_path = str(tmp_path / 'train_table')
    persisted = ttng_persisted_staging_dir(train_table_path)

    published_path = mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=train_table_path,
        split_config=_custom_tt_split_config(),
        split_family='ManualSplit',
        task=_custom_tt_task(),
        pquery=_custom_tt_pquery(),
        diskgraph_index_path=str(tmp_path / 'index'),
        ttng_tt_id='gen-traintable-1',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    published = ttng_bundle_dir_for_train_table(published_path)
    assert os.path.dirname(published) == os.path.join(persisted, '_bundles')
    assert os.path.exists(os.path.join(published, 'tt.parquet')), (
        'custom-TT must return an immutable tt.parquet selection so re-ingest '
        'works on a restored pod')
    assert os.path.exists(os.path.join(published, 'tt_ingest_spec.json')), (
        'custom-TT must persist tt_ingest_spec.json (hard-required by '
        'execute_tt_ingest) into ttng_persisted_staging_dir')

    # Native user-data errors must retain their actionable text across the
    # activity boundary instead of being sanitized as an internal failure.
    fake_dg = sys.modules['diskgraph']
    data_error = fake_dg.DiskgraphDataError(
        'custom-TT weight is null at source row 7')

    def _raise_data_error(**_kwargs):
        raise data_error

    fake_dg.build_train_table_artifacts_for_existing_table = _raise_data_error
    with pytest.raises(
            mtt.TrainingDataMaterializationError,
            match='weight is null at source row 7',
    ) as caught:
        mtt._build_ttng_custom_train_table_artifacts(
            train_table_path=train_table_path,
            split_config=_custom_tt_split_config(),
            split_family='ManualSplit',
            task=_custom_tt_task(),
            pquery=_custom_tt_pquery(),
            diskgraph_index_path=str(tmp_path / 'index'),
            ttng_tt_id='gen-traintable-1',
            time_col_name='TIMESTAMP',
            weight_col_name=None,
        )
    assert caught.value.get_user_message() == (
        'custom-TT weight is null at source row 7')


def test_custom_tt_build_routes_through_gen_persist_not_only_extension(
        monkeypatch, tmp_path):
    """Custom materialization routes gen -> immutable bundle -> ingest."""
    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
        ttng_persisted_staging_dir,
    )
    _install_custom_tt_bundle_fakes(monkeypatch, tmp_path, is_link_pred=False)
    train_table_path = str(tmp_path / 'train_table')
    persisted = ttng_persisted_staging_dir(train_table_path)

    published_path = mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=train_table_path,
        split_config=_custom_tt_split_config(),
        split_family='ManualSplit',
        task=_custom_tt_task(),
        pquery=_custom_tt_pquery(),
        diskgraph_index_path=str(tmp_path / 'index'),
        ttng_tt_id='gen-traintable-1',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    # The persisted bundle must exist and be re-ingestable (has the spec the
    # fused custom path never emits) -- i.e. the flow does NOT rely solely on
    # an in-place extension published by the build.
    assert os.path.isdir(persisted), (
        'custom-TT build must materialize a durable staging bundle dir')
    published = ttng_bundle_dir_for_train_table(published_path)
    assert os.path.dirname(published) == os.path.join(persisted, '_bundles')
    assert os.path.exists(os.path.join(published, 'tt_ingest_spec.json')), (
        'persisted custom bundle must be self-describing for re-ingest')


def test_custom_tt_build_persists_bundle_for_link_pred(monkeypatch, tmp_path):
    """LP custom-TT returns the immutable bundle selected for re-ingest."""
    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
        ttng_persisted_staging_dir,
    )
    _install_custom_tt_bundle_fakes(monkeypatch, tmp_path, is_link_pred=True)
    train_table_path = str(tmp_path / 'train_table')
    persisted = ttng_persisted_staging_dir(train_table_path)

    published_path = mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=train_table_path,
        split_config=_custom_tt_split_config(),
        split_family='ManualSplit',
        task=_custom_tt_task(is_link_pred=True),
        pquery=_custom_tt_pquery(),
        diskgraph_index_path=str(tmp_path / 'index'),
        ttng_tt_id='gen-traintable-1',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    published = ttng_bundle_dir_for_train_table(published_path)
    assert os.path.dirname(published) == os.path.join(persisted, '_bundles')
    assert os.path.exists(os.path.join(published, 'tt.parquet')), (
        'LP custom-TT must persist the durable bundle for re-ingest')
    assert os.path.exists(os.path.join(published, 'tt_ingest_spec.json')), (
        'LP custom-TT must persist the LP-resolution ingest spec')


def test_custom_tt_split_column_still_reaches_dg_builder(monkeypatch, tmp_path):
    """spec kumo 9 (split passthrough, keeps #29593 coverage) -- GREEN.

    ManualSplit / ``split_column`` must still reach the DG gen entry through
    the (now gen+persist) custom path. Mirrors the existing source-mode split
    test but runs against the persist-capable fixture so it stays green after
    the fix.
    """
    _bundle, calls = _install_custom_tt_bundle_fakes(monkeypatch, tmp_path,
                                                     is_link_pred=False)

    train_table_path = str(tmp_path / 'train_table')
    mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=train_table_path,
        split_config=_custom_tt_split_config(),
        split_family='ManualSplit',
        task=_custom_tt_task(),
        pquery=_custom_tt_pquery(),
        diskgraph_index_path=str(tmp_path / 'index'),
        ttng_tt_id='gen-traintable-1',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    split_spec = calls['build']['split_config']
    assert calls['build']['output_path'] == train_table_path
    assert calls['build']['tt_id'] == 'gen-traintable-1'
    assert calls['build']['split_family_hint'] == 'ManualSplit'
    assert calls['build']['input_timestamp_unit'] == 'nanoseconds'
    assert calls['build']['lp_rhs_is_temporal'] is False
    assert split_spec.kind == 'ManualSplit'
    assert split_spec.split_column == '__kumo_precomputed_split'
    assert split_spec.manual_train_values == ['0']
    assert split_spec.manual_val_values == ['1']
    assert split_spec.manual_test_values == ['2']


@pytest.mark.parametrize(
    ('rhs_has_time_col', 'expected'),
    [
        pytest.param(False, False, id='atemporal'),
        pytest.param(True, True, id='temporal'),
    ],
)
def test_custom_tt_link_pred_passes_rhs_temporal_flag(monkeypatch, tmp_path,
                                                      rhs_has_time_col,
                                                      expected):
    _bundle, calls = _install_custom_tt_bundle_fakes(monkeypatch, tmp_path,
                                                     is_link_pred=True)

    class _Graph:
        def __init__(self):
            self._tables = {
                'Customer':
                SimpleNamespace(
                    table_name='Customer',
                    name='Customer',
                    pkey_name='customer_id',
                ),
                'Product':
                SimpleNamespace(
                    table_name='Product',
                    name='Product',
                    pkey_name='product_id',
                    has_time_col=lambda: rhs_has_time_col,
                    has_end_time_col=lambda: False,
                ),
            }
            self.tables = list(self._tables.values())

        def __getitem__(self, name):
            return self._tables[name]

    graph = _Graph()
    task = SimpleNamespace(
        graph=graph,
        train_table=SimpleNamespace(
            fkeys=[SimpleNamespace(dst_table='Customer')]),
        task_type=SimpleNamespace(is_link_pred=True, value='link_prediction'),
    )
    pquery = SimpleNamespace(
        graph=graph,
        validated_predictive_query=SimpleNamespace(target_stype=SimpleNamespace(
            value='categorical')),
    )

    train_table_path = str(tmp_path / 'multipart_train_table')
    persisted = mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=train_table_path,
        split_config=_custom_tt_split_config(),
        split_family='ManualSplit',
        task=task,
        pquery=pquery,
        diskgraph_index_path=str(tmp_path / 'index'),
        ttng_tt_id='gen-traintable-lp',
        time_col_name='TIMESTAMP',
        weight_col_name='weight',
    )

    assert os.path.isdir(persisted)
    assert os.path.basename(persisted) == 'tt.parquet'
    assert calls['build']['output_path'] == train_table_path
    assert calls['build']['target_entity_table'] == 'Product'
    assert calls['build']['task_type'] == 'link_prediction'
    assert calls['build']['lp_rhs_is_temporal'] is expected
    assert calls['build']['train_table_col_stats_config'] == {
        'target_stype': 'categorical',
        'time_col': 'TIMESTAMP',
        'weight_col': 'weight',
    }


def test_ttng_materialize_validates_bundle_and_early_returns(monkeypatch):
    """C1 (was test_ttng_decoupled_materialize_writes_split_inputs): TT-NG
    materialize no longer writes legacy split inputs -- it validates the durable
    bundle and early-returns ("", ok). The persisted bundle (produced by gen)
    is present, so no error and no legacy materialization/save happens."""
    called = {'save_json': False}

    monkeypatch.setattr(mtt, 'open_connectors', lambda *_a, **_k: None)
    monkeypatch.setattr(mtt, 'get_job_base_dir', lambda: '/tmp/jobs')
    monkeypatch.setattr(mtt, 'get_job_dir', lambda *_a, **_k: '/tmp/jobs/j1')

    class _Cache:
        table_version_cache = SimpleNamespace(remove=lambda *_a, **_k: None)

    monkeypatch.setattr(mtt, 'get_active_cache', lambda: _Cache())
    graph = SimpleNamespace(set_table_data_versions=lambda *_a, **_k: None)
    monkeypatch.setattr(
        mtt,
        '_create_pquery_with_credentials',
        lambda *_a, **_k: (SimpleNamespace(graph=graph), {}),
    )

    monkeypatch.setattr(
        mtt.AutoTrainer,
        'parse_pquery',
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError('bundle validation must not parse the raw source')),
    )
    monkeypatch.setattr(mtt, 'save_json',
                        lambda *_a, **_k: called.update(save_json=True))

    import tempfile
    _root = tempfile.mkdtemp(prefix='c1_gate_')
    logical_path = os.path.join(_root, 'jobs', 'gen-traintable-job-j1',
                                'train_table')
    train_table_path = _write_current_v3_bundle_at(
        f'{logical_path}.ttng_staging',
        'gen-traintable-job-j1',
    )

    class _FakeModelPlan:
        pass

    monkeypatch.setattr(mtt, 'ModelPlan', _FakeModelPlan)

    job_input = SimpleNamespace(
        id='trainingjob-j1',
        spec=SimpleNamespace(advanced_options=_FakeModelPlan()),
        graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                 connectors={}),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )

    path_out, validation = mtt.materialize_training_table(
        job_input=job_input,
        settings=SimpleNamespace(),
        train_table_path=train_table_path,
        execute_split=False,
        weight_col_name=None,
    )

    assert path_out == ''
    assert validation.errors == []
    # No legacy split materialization / config-data save on TT-NG.
    assert called == {'save_json': False}


# NOTE: test_ttng_materialize_ingests_persisted_staging_bundle was removed.
# Under C1, materialize_training_table no longer ingests or applies split
# stats on TT-NG -- it validates the durable bundle and early-returns.
# No-ingest/early-return is covered by test_c1_ttng_* above; the gen-side
# split_stats.json -> job dir mirror is covered in
# test_ttng_gen_split_stats.py.


def test_configure_forecasting_sequence_columns():
    class _ForecastingTask:
        task_type = mtt.TaskType.FORECASTING

        def __init__(self):
            self.args = []

        def _set_forecast_col_encoder(self, year_over_year,
                                      include_future_labels=False):
            self.args.append((year_over_year, include_future_labels))

    task = _ForecastingTask()

    mtt._configure_forecasting_sequence_columns(task, year_over_year=True)

    assert task.args == [(True, False)]


def test_configure_forecasting_sequence_columns_can_include_future_labels():
    class _ForecastingTask:
        task_type = mtt.TaskType.FORECASTING

        def __init__(self):
            self.args = []

        def _set_forecast_col_encoder(self, year_over_year,
                                      include_future_labels=False):
            self.args.append((year_over_year, include_future_labels))

    task = _ForecastingTask()

    mtt._configure_forecasting_sequence_columns(
        task,
        year_over_year=True,
        include_future_labels=True,
    )

    assert task.args == [(True, True)]


def test_configure_forecasting_sequence_columns_ignores_other_tasks():
    class _BinaryClassificationTask:
        task_type = mtt.TaskType.BINARY_CLASSIFICATION

        def __init__(self):
            self.called = False

        def _set_forecast_col_encoder(self, year_over_year,
                                      include_future_labels=False):
            self.called = True

    task = _BinaryClassificationTask()

    mtt._configure_forecasting_sequence_columns(task, year_over_year=True)

    assert task.called is False


def test_ttng_forecasting_task_family_is_explicit():
    assert mtt.TaskType.FORECASTING in mtt._TTNG_TASK_DATASET_TASK_TYPES
    assert mtt._ttng_task_family(mtt.TaskType.FORECASTING) == 'forecasting'


def test_build_ttng_feature_schema_includes_forecasting_train_table_sequences():
    class _Col:
        def __init__(self, name, dtype, stype):
            self.name = name
            self.dtype = dtype
            self.stype = stype

    class _Table:
        table_name = 'User'
        name = 'connector.User'
        time_col_name = 'created_at'

        columns = {
            'age': _Col('age', Dtype.float, Stype.numerical),
        }

        @staticmethod
        def has_time_col():
            return True

        @classmethod
        def col(cls, name):
            return cls.columns[name]

    class _TrainTable:
        table_name = 'TRAIN_TABLE'
        name = 'connector.TRAIN_TABLE'
        columns = {
            name: _Col(name, Dtype.floatlist, Stype.sequence)
            for name in (FUTURE_LABELS_COL, AUTOREGRESSION_COL, YOY_COL)
        }

        @staticmethod
        def has_time_col():
            return False

        @classmethod
        def col(cls, name):
            return cls.columns[name]

    task = SimpleNamespace(
        task_type=mtt.TaskType.FORECASTING,
        graph=SimpleNamespace(tables=[_Table()]),
        train_table=_TrainTable(),
        exclude_cols_dict={},
    )
    fitted_encoders = SimpleNamespace(
        encoders={
            'User': {
                'age': Numerical(),
            },
            'TRAIN_TABLE': {
                FUTURE_LABELS_COL: NumericalList(),
                AUTOREGRESSION_COL: NumericalList(),
                YOY_COL: NumericalList(),
            },
        })

    schema = mtt._build_ttng_feature_schema(task, fitted_encoders)

    assert set(schema) == {'node_types', 'target_encoder_input'}
    user = schema['node_types']['User']
    assert user['feature_columns'] == ['age']
    assert user['mapper_key'] == 'User'
    assert user['time_column'] == 'created_at'
    train_table = schema['node_types']['TRAIN_TABLE']
    assert train_table['feature_columns'] == [
        FUTURE_LABELS_COL,
        AUTOREGRESSION_COL,
        YOY_COL,
    ]
    assert train_table['mapper_key'] == 'TRAIN_TABLE'
    assert [column['column_name'] for column in train_table['column_specs']
            ] == [FUTURE_LABELS_COL, AUTOREGRESSION_COL, YOY_COL]


def test_ttng_task_dataset_uses_existing_extension_manifest(monkeypatch):
    calls = {'materialize': None, 'saved': []}

    monkeypatch.setattr(mtt.fsutil, 'exists', lambda _path: True)
    # tt_id now comes from the durable bundle spec; stub it to the value the old
    # path-parse produced for this train_table_path (gen-traintable-job-1).
    monkeypatch.setattr(mtt, '_ttng_train_table_tt_id',
                        lambda _p: 'gen-traintable-job-1')
    monkeypatch.setattr(mtt, '_ttng_ensure_train_table_extension',
                        lambda *_args, **_kwargs: None)

    import kumo.task.task as task_mod
    monkeypatch.setattr(task_mod, 'collect_encoders',
                        lambda _task: SimpleNamespace(encoders={}))

    class _FakeService:
        def materialize_task_dataset(self, handle, spec):
            calls['materialize'] = (handle, spec)
            return SimpleNamespace(
                bundle_dir='/dg/graph-v1/task_datasets/trainingjob-1',
                manifest_path=(
                    '/dg/graph-v1/task_datasets/trainingjob-1/manifest.json'),
                storage_kind='diskgraph_table',
            )

    import kumo.diskgraph as dg_mod
    monkeypatch.setattr(dg_mod, 'get_service', lambda: _FakeService())
    monkeypatch.setattr(
        mtt.JobConfig,
        'load',
        lambda _path: SimpleNamespace(diskgraph_task_dataset_bundle_dir=None),
    )
    monkeypatch.setattr(mtt, 'save_json',
                        lambda obj, path: calls['saved'].append((obj, path)))

    train_table = SimpleNamespace(
        table_name='train_table',
        name='connector.train_table',
        columns=[],
        fkeys=[SimpleNamespace(dst_table='connector.users')],
    )
    graph = SimpleNamespace(
        tables=[
            SimpleNamespace(
                table_name='users',
                fkeys=[],
                time_col_name=None,
                has_time_col=lambda: False,
            ),
        ], )
    task = SimpleNamespace(
        task_type=mtt.TaskType.FORECASTING,
        train_table=train_table,
        graph=graph,
        exclude_cols_dict={},
    )
    job_input = SimpleNamespace(
        id='trainingjob-1',
        graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                 connectors={}),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )

    mtt._maybe_materialize_ttng_task_dataset(
        task=task,
        pquery=SimpleNamespace(validated_predictive_query=None, graph=graph),
        train_table_path=(
            '/kumo/jobs/gen-traintable-job-1/train_table/_bundles/b1/'
            'tt.parquet'),
        job_input=job_input,
        diskgraph_index_path='/dg/graph-v1',
        job_config_path='/job/config.json',
    )

    assert calls['materialize'] is not None
    _, spec = calls['materialize']
    assert spec.tt_manifest_path == (
        '/dg/graph-v1/train_tables/gen-traintable-job-1/manifest.json')
    saved_config, saved_path = calls['saved'][0]
    assert saved_config.diskgraph_task_dataset_bundle_dir == (
        '/dg/graph-v1/task_datasets/trainingjob-1')
    assert saved_path == '/job/config.json'


def test_ttng_custom_materialize_validates_bundle_and_early_returns(
        monkeypatch):
    """C1 for custom TT (was test_ttng_custom_train_table_builds_artifacts_via_
    source_mode): the custom-gen ACTIVITY already built +
    persisted the bundle, so materialize_training_table(is_custom_train_table=
    True) just validates the bundle and early-returns. It must NOT build the
    custom artifacts and must NOT ingest here.

    The build-and-persist INTENT now lives in
    test_ttng_custom_gen_activity.py; the "builds even when pre-split stats are
    present" regression is pinned there too (the activity has no stats gate)."""

    monkeypatch.setattr(mtt, 'open_connectors', lambda *_a, **_k: None)
    monkeypatch.setattr(mtt, 'get_job_base_dir', lambda: '/tmp/jobs')
    monkeypatch.setattr(mtt, 'get_job_dir', lambda *_a, **_k: '/tmp/jobs/j1')

    class _Cache:
        table_version_cache = SimpleNamespace(remove=lambda *_a, **_k: None)

    monkeypatch.setattr(mtt, 'get_active_cache', lambda: _Cache())
    graph = SimpleNamespace(set_table_data_versions=lambda *_a, **_k: None)
    monkeypatch.setattr(
        mtt,
        '_create_pquery_with_credentials',
        lambda *_a, **_k: (SimpleNamespace(graph=graph), {}),
    )
    train_table = SimpleNamespace(
        data_source='src',
        get_train_table_stats=lambda after_split=False: {'ok': True},
    )
    task = SimpleNamespace(train_table=train_table, task_type='NODE')
    monkeypatch.setattr(mtt.AutoTrainer, 'parse_pquery', lambda *_a, **_k:
                        (task, None))

    import tempfile
    _root = tempfile.mkdtemp(prefix='c1_gate_')
    logical_path = os.path.join(_root, 'jobs', 'gen-traintable-j1',
                                'train_table')
    train_table_path = _write_current_v3_bundle_at(
        f'{logical_path}.ttng_staging',
        'gen-traintable-j1',
    )

    # Any of these being reached is a C1 violation.
    monkeypatch.setattr(
        mtt, '_build_ttng_custom_train_table_artifacts', lambda **_k:
        (_ for _ in ()).throw(
            AssertionError('C1: materialize must not build custom bundle')))
    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(execute_tt_ingest=lambda **_k: (_ for _ in ()).throw(
            AssertionError('C1: materialize must not ingest'))))

    class _FakeModelPlan:
        pass

    monkeypatch.setattr(mtt, 'ModelPlan', _FakeModelPlan)

    job_input = SimpleNamespace(
        id='trainingjob-j1',
        spec=SimpleNamespace(advanced_options=_FakeModelPlan()),
        graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                 connectors={}),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )

    path_out, validation = mtt.materialize_training_table(
        job_input=job_input,
        settings=SimpleNamespace(),
        train_table_path=train_table_path,
        execute_split=False,
        weight_col_name=None,
        is_custom_train_table=True,
    )

    assert path_out == ''
    assert validation.errors == []


def test_apply_native_split_stats_accepts_multilabel_distribution():
    target_distribution = {
        'count': {
            '1': 2.0,
            '2': 1.0,
        },
        'percentage': {
            '1': 100.0,
            '2': 50.0,
        },
    }

    class _FakeTrainTable:
        task_type = split_mod.TaskType.MULTILABEL_CLASSIFICATION
        saved_to_job_dir = None
        saved_to_cache = None

        def _save_split_train_table_stats_to_job_dir(self, stats):
            self.saved_to_job_dir = stats

        def _update_and_save_stats_cache(self, stats):
            self.saved_to_cache = stats

    training_table = _FakeTrainTable()

    split_mod._apply_native_split_stats(
        training_table,
        manifest_path=None,
        prevalidated_manifest={
            'parsed_split_stats': {
                'train_table_stats': {
                    'num_train_rows': 2,
                    'num_entities': 2,
                    'num_targets': 2,
                    'target_distribution': target_distribution,
                },
            },
        },
    )

    assert training_table._split_train_table_stats is training_table.saved_to_job_dir
    assert training_table.saved_to_job_dir is training_table.saved_to_cache
    assert (training_table.saved_to_job_dir.train_table_stats.
            target_distribution == target_distribution)


def test_apply_native_split_stats_rejects_missing_target_distribution():
    training_table = SimpleNamespace(
        task_type=split_mod.TaskType.MULTILABEL_CLASSIFICATION, )

    with pytest.raises(RuntimeError, match='missing target_distribution'):
        split_mod._apply_native_split_stats(
            training_table,
            manifest_path=None,
            prevalidated_manifest={
                'parsed_split_stats': {
                    'train_table_stats': {
                        'num_train_rows': 2,
                        'num_entities': 2,
                        'num_targets': 2,
                        'target_distribution': None,
                    },
                },
            },
        )


def test_infer_graph_encoders_from_cached_stats_uses_cache_only(monkeypatch):
    infer_called = {'ok': False}

    @dataclass(frozen=True)
    class _ColCfg:
        name: str

    class _FakeCol:
        def __init__(self, name):
            self.name = name
            self.encoder = None
            self.dtype = SimpleNamespace()
            self.stype = SimpleNamespace()

        def base_config(self):
            return _ColCfg(self.name)

    class _FakeTable:
        def __init__(self):
            self.name = 'connector_1.T1'
            self.data_version = 'v1'
            self.connector = object()
            self.cols = [_FakeCol('c1')]

        def fingerprint(self):
            return 'fp1'

    table = _FakeTable()

    class _FakeStat:
        def __init__(self, value):
            self.value = value

        def config(self):
            return {'value': self.value}

    class _FakeCache:
        def load_col_stats_cache(self, _fp, _ver):
            return True

        def get_col_stats(self, _fp, stats_dict, *, data_version=None,
                          approx=False, **_kwargs):
            col_cfg = next(iter(stats_dict.keys()))
            return {col_cfg: {'S': _FakeStat(1 if approx else 2)}}

    class _FakeBackend:
        def cache(self):
            return _FakeCache()

    class _Req:
        def supports(self, _dtype):
            return True

        def supports_stype(self, _stype):
            return True

    class _FakeGraph:
        def __init__(self):
            self.tables = [table]

        def infer_encoders(self, encoder_stats=None, **_kwargs):
            assert encoder_stats is not None
            assert table.name in encoder_stats
            infer_called['ok'] = True

    monkeypatch.setattr(mtt, 'get_active_cache', lambda: _FakeBackend().cache())
    monkeypatch.setattr(mtt.EncoderGenerator, 'REQUIRED_APPROX_STATS', [_Req()])
    monkeypatch.setattr(mtt.EncoderGenerator, 'REQUIRED_EXACT_STATS', [_Req()])
    mtt._infer_graph_encoders_from_cached_stats_or_fail(_FakeGraph())

    assert infer_called['ok'] is True


def test_infer_graph_encoders_from_cached_stats_hard_fails_on_missing(
        monkeypatch):
    class _FakeTable:
        name = 'connector_1.T1'
        data_version = 'v1'
        connector = object()
        cols = [
            SimpleNamespace(name='c1', encoder=None, base_config=lambda: 'cfg')
        ]

        def fingerprint(self):
            return 'fp1'

    class _FakeCache:
        def load_col_stats_cache(self, _fp, _ver):
            return False

    class _FakeBackend:
        def cache(self):
            return _FakeCache()

    class _FakeGraph:
        tables = [_FakeTable()]

    monkeypatch.setattr(mtt, 'get_active_cache', lambda: _FakeBackend().cache())

    with pytest.raises(RuntimeError, match='requires precomputed col stats'):
        mtt._infer_graph_encoders_from_cached_stats_or_fail(_FakeGraph())


# ---------------------------------------------------------------------------
# C1: materialize_training_table TT-NG = validate durable bundle + early-return.
# The gen (or custom-gen) step persists the bundle; the train pod's
# ensure-extension is the sole ingest. materialize_training_table must NOT
# ingest, NOT materialize legacy split inputs -- it only validates + returns.
# ---------------------------------------------------------------------------


def _c1_common_fakes(monkeypatch, *, bundle_present):
    monkeypatch.setattr(mtt, 'open_connectors', lambda *_a, **_k: None)
    monkeypatch.setattr(mtt, 'get_job_base_dir', lambda: '/tmp/jobs')
    monkeypatch.setattr(mtt, 'get_job_dir', lambda *_a, **_k: '/tmp/jobs/j1')

    class _Cache:
        table_version_cache = SimpleNamespace(remove=lambda *_a, **_k: None)

    monkeypatch.setattr(mtt, 'get_active_cache', lambda: _Cache())
    graph = SimpleNamespace(set_table_data_versions=lambda *_a, **_k: None)
    monkeypatch.setattr(
        mtt,
        '_create_pquery_with_credentials',
        lambda *_a, **_k: (SimpleNamespace(graph=graph), {}),
    )
    train_table = SimpleNamespace(
        data_source='src',
        get_train_table_stats=lambda after_split=False: None,
    )
    task = SimpleNamespace(train_table=train_table, task_type='NODE')
    monkeypatch.setattr(mtt.AutoTrainer, 'parse_pquery', lambda *_a, **_k:
                        (task, None))

    class _FakeModelPlan:
        pass

    monkeypatch.setattr(mtt, 'ModelPlan', _FakeModelPlan)

    import tempfile

    # Real bundle on disk so the shared completeness gate
    # (assert_ttng_bundle_complete) can read the spec JSON. Scope under a temp
    # dir so the .ttng_staging bundle is written/verified for real.
    _root = tempfile.mkdtemp(prefix='c1_gate_')
    logical_path = os.path.join(_root, 'jobs', 'gen-traintable-job-j1',
                                'train_table', 'ttng_pq', 'train_table')
    train_table_path = os.path.join(
        f'{logical_path}.ttng_staging',
        'tt.parquet',
    )
    if bundle_present:
        train_table_path = _write_current_v3_bundle_at(
            f'{logical_path}.ttng_staging',
            'gen-traintable-job-j1',
        )

    # Any of these being reached is a C1 violation.
    def _boom_ingest(**_kwargs):
        raise AssertionError('C1: materialize_training_table must not ingest')

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_boom_ingest))
    monkeypatch.setattr(
        mtt, '_build_ttng_custom_train_table_artifacts', lambda **_k:
        (_ for _ in ()).throw(
            AssertionError('C1: must not build custom bundle at materialize')))
    return _FakeModelPlan, train_table_path


def _c1_job_input(model_plan_cls):
    return SimpleNamespace(
        id='trainingjob-j1',
        spec=SimpleNamespace(advanced_options=model_plan_cls()),
        graph_data_compute_input=SimpleNamespace(graph_key='graph1',
                                                 connectors={}),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )


def test_c1_ttng_bundle_present_validates_and_early_returns(monkeypatch):
    model_plan_cls, train_table_path = _c1_common_fakes(monkeypatch,
                                                        bundle_present=True)

    path_out, validation = mtt.materialize_training_table(
        job_input=_c1_job_input(model_plan_cls),
        settings=SimpleNamespace(),
        train_table_path=train_table_path,
        execute_split=True,
        weight_col_name=None,
    )
    # A DB round-trip asserts train_table_materialization_path is not None.
    assert path_out == ''
    assert validation.errors == []


def test_c1_ttng_bundle_present_early_returns_for_custom(monkeypatch):
    # is_custom_train_table=True must also just validate + early-return
    # (the custom-gen activity already persisted the bundle).
    model_plan_cls, train_table_path = _c1_common_fakes(monkeypatch,
                                                        bundle_present=True)

    path_out, validation = mtt.materialize_training_table(
        job_input=_c1_job_input(model_plan_cls),
        settings=SimpleNamespace(),
        train_table_path=train_table_path,
        execute_split=True,
        weight_col_name=None,
        is_custom_train_table=True,
    )
    assert path_out == ''
    assert validation.errors == []


def test_c1_ttng_bundle_missing_raises(monkeypatch):
    model_plan_cls, train_table_path = _c1_common_fakes(monkeypatch,
                                                        bundle_present=False)

    with pytest.raises(mtt.TrainingDataMaterializationError):
        mtt.materialize_training_table(
            job_input=_c1_job_input(model_plan_cls),
            settings=SimpleNamespace(),
            train_table_path=train_table_path,
            execute_split=True,
            weight_col_name=None,
        )


# The TT-NG native-split VAL/TEST single-timestamp guard moved into DG ingest
# (per-stage ts grouping for val_seeds/test_seeds); the Python
# _enforce_single_timestamp_val_test was removed, so its E1 tests are gone too.
# See diskgraph ts_grouping single-ts guard + its Rust unit test.

# ---------------------------------------------------------------------------
# Temporal link prediction with a user-supplied split column.
#
# The whole chain, end to end: the plan value validates, parses back into the
# `ManualSplit` the platform runs on, resolves to a split config, and reaches
# the DiskGraph builder as `ManualSplit`. Before the fix the plan value was
# refused outright, so nothing downstream was reachable.
# ---------------------------------------------------------------------------

TLP_MANUAL_SPLIT = ("ManualSplit(col='__kumo_precomputed_split', "
                    "train_value=0, val_value=1, test_value=2)")


def _temporal_lp_graph():
    class _Graph:
        def __init__(self):
            self._tables = {
                'Customer':
                SimpleNamespace(
                    table_name='Customer',
                    name='Customer',
                    pkey_name='customer_id',
                ),
                'Product':
                SimpleNamespace(
                    table_name='Product',
                    name='Product',
                    pkey_name='product_id',
                    has_time_col=lambda: True,
                    has_end_time_col=lambda: False,
                ),
            }
            self.tables = list(self._tables.values())

        def __getitem__(self, name):
            return self._tables[name]

    return _Graph()


def test_temporal_link_prediction_manual_split_reaches_the_dg_builder(
        monkeypatch, tmp_path):
    from kumo_core.model_plan import ModelPlan
    from kumo_core.training.table import build_split_spec
    from kumo_core.types import TaskType
    from kumoapi.model_plan import RunMode
    from kumoapi.pquery import QueryType

    from kumo.diskgraph._diskgraph_legacy import resolve_split_config
    from kumo.task.split import ManualSplit
    from kumo.task.splits.parsing import get_split

    # 1. The model plan for a temporal-LP query accepts the split value.
    #    (`valid_names` admits the family; the temporal-LP gate admits an
    #    explicitly row-assigned split.)
    plan = ModelPlan.from_default(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    plan.training_table_generation.split = TLP_MANUAL_SPLIT
    plan.validate(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    # 2. and that value parses back into the split object the platform
    #    runs -- the consumer a plan-nameable `ManualSplit` reaches next.
    split = get_split(plan.training_table_generation.split)
    assert isinstance(split, ManualSplit)
    assert split.col == '__kumo_precomputed_split'

    # 3. which resolves to a `ManualSplit` split config, column intact.
    config, family = resolve_split_config(split.config(), None)
    assert family == 'ManualSplit'
    assert config['split_column'] == '__kumo_precomputed_split'
    assert config['manual_train_values'] == ['0']
    assert config['manual_val_values'] == ['1']
    assert config['manual_test_values'] == ['2']

    # 4. and reaches the DiskGraph builder as `ManualSplit`.
    _bundle, calls = _install_custom_tt_bundle_fakes(monkeypatch, tmp_path,
                                                     is_link_pred=True)
    graph = _temporal_lp_graph()
    task = SimpleNamespace(
        graph=graph,
        train_table=SimpleNamespace(
            fkeys=[SimpleNamespace(dst_table='Customer')]),
        task_type=SimpleNamespace(is_link_pred=True,
                                  value='temporal_link_prediction'),
    )
    pquery = SimpleNamespace(
        graph=graph,
        validated_predictive_query=SimpleNamespace(target_stype=SimpleNamespace(
            value='categorical')),
    )

    mtt._build_ttng_custom_train_table_artifacts(
        train_table_path=str(tmp_path / 'tlp_train_table'),
        split_config=config,
        split_family=family,
        task=task,
        pquery=pquery,
        diskgraph_index_path=str(tmp_path / 'index'),
        ttng_tt_id='gen-traintable-tlp',
        time_col_name='TIMESTAMP',
        weight_col_name=None,
    )

    assert calls['build']['split_family_hint'] == 'ManualSplit'
    assert calls['build']['lp_rhs_is_temporal'] is True
    split_spec = calls['build']['split_config']
    assert split_spec.kind == 'ManualSplit'
    assert split_spec.split_column == '__kumo_precomputed_split'
    assert split_spec.manual_train_values == ['0']

    # The spec `build_split_spec` derives independently agrees.
    class _Spec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    assert build_split_spec(config, family, _Spec).kind == 'ManualSplit'
