# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``_maybe_materialize_ttng_task_dataset``.

Covers gate logic, spec construction from activity-context inputs,
and exception wrapping. Does not exercise the surrounding activity
body — that path would require mocking the whole pipeline. The
helper is the testable unit.
"""
from __future__ import annotations

import json
import os
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kumo_core.task_dataset import TaskDatasetMaterializeSpec
from kumo_core.types import TaskType
from kumoapi.encoder import Numerical
from kumoapi.typing import ColStatType, Dtype, Stype

from kumo.diskgraph import DiskgraphHandle
from kumo.diskgraph.service import MaterializeResult
from statedb.utils.errors import TrainingDataMaterializationError

# The activities package's __init__.py re-exports the
# ``materialize_training_task`` function under the same dotted name as
# the submodule, shadowing it for ``patch('...module.attr', ...)``
# resolution. Import the module explicitly so the helper resolves
# unambiguously to the submodule attribute.
_mtt = import_module(
    'rest.backends.ml_backend.activities.materialize_training_task')
_maybe_materialize_ttng_task_dataset = (
    _mtt._maybe_materialize_ttng_task_dataset)

# Lazy imports inside the helper resolve against their source modules
# at call time, so those patches go at the source.
_P_GET_SERVICE = 'kumo.diskgraph.get_service'
_P_COLLECT_ENCODERS = 'kumo.task.task.collect_encoders'

# ---------------------------------------------------------------------------
# Fixture builders — SimpleNamespace over MagicMock so attribute names like
# ``.name`` behave as plain values rather than mock-internal magic.
# ---------------------------------------------------------------------------


def _fkey(name: str, dst_table: str,
          dst_table_name: str | None = None) -> SimpleNamespace:
    # Real ``kumo.graph.ForeignKey`` exposes ``dst_table`` (fully
    # qualified, e.g. ``<connector>.<table>``) and ``dst_table_name``
    # (bare). Most tests don't need to distinguish; default to using
    # the same value for both.
    return SimpleNamespace(name=name, dst_table=dst_table,
                           dst_table_name=dst_table_name or dst_table)


def _table(name: str, fkeys=None,
           table_name: str | None = None) -> SimpleNamespace:
    # Real ``kumo.graph.Table`` exposes two name-like fields: ``name``
    # (fully qualified, e.g. ``<connector>.<table>``) used as the wheel
    # namespace, and ``table_name`` (bare) used as the encoder/mapper
    # bank key. Tests can pass them separately to exercise the
    # FQN-vs-bare distinction; for tests that don't care, both default
    # to ``name`` (matching the historical mock).
    columns = {}

    def col(column_name):
        return columns.setdefault(
            column_name,
            SimpleNamespace(dtype='float', stype='numerical'),
        )

    return SimpleNamespace(name=name, table_name=table_name or name,
                           fkeys=list(fkeys or []), time_col_name=None,
                           has_time_col=lambda: False, col=col)


def _task(
    task_type=TaskType.BINARY_CLASSIFICATION,
    graph_tables=None,
    train_table_fkeys=None,
    train_table_name: str = 'TRAIN_TABLE',
    exclude_cols=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_type=task_type,
        is_temporal=task_type in {
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.FORECASTING,
        },
        graph=SimpleNamespace(tables=list(graph_tables or [])),
        train_table=SimpleNamespace(
            fkeys=list(train_table_fkeys or []),
            table_name=train_table_name,
        ),
        # Real ``Task`` stores leakage exclusions on ``exclude_cols``.
        exclude_cols=exclude_cols or {},
    )


def _job_input(
    job_id: str = 'job-abc',
    graph_key: str = 'gk',
    graph_data_version: str = 'v1',
) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id,
        graph_data_compute_input=SimpleNamespace(graph_key=graph_key),
        graph_version_info=SimpleNamespace(
            graph_data_version=graph_data_version),
    )


def _encoders(per_table: dict) -> SimpleNamespace:
    """Fake EncodersConfig — only ``.encoders`` is read by the helper."""
    resolved = {}
    for table, values in per_table.items():
        resolved[table] = {}
        for column, value in values.items():
            encoder = value if hasattr(value, '_stats') else Numerical()
            encoder._stats = {ColStatType.MEAN: 0.0}
            resolved[table][column] = encoder
    return SimpleNamespace(encoders=resolved, target=None)


def _default_inputs():
    """Two-node-type graph: USERS (fkey -> COMPANIES) and COMPANIES.
    Train table fkey points at ``src1.USERS`` (connector-prefixed).
    """
    companies = _table('COMPANIES', fkeys=[])
    users = _table('USERS', fkeys=[_fkey('company', 'COMPANIES')])
    task_obj = _task(
        graph_tables=[users, companies],
        train_table_fkeys=[_fkey('entity', 'src1.USERS')],
    )
    return task_obj, _job_input(), '/tmp/jobs/job-abc/tt.parquet', '/data/gk/v1'


def _run_helper(task_obj, train_table_path, job_input_obj, dg_index_path,
                encoders_dict):
    """All-gates-on harness. Returns the (handle, spec) tuple the
    helper passed to ``service.materialize_task_dataset``.

    Patches ``JobConfig.load`` / ``save_json`` on the activity module so
    the helper's post-success JobConfig write doesn't touch real files.
    Dedicated assertions on the JobConfig-update behavior live in
    ``test_job_config_updated_with_bundle_dir``.
    """
    service = MagicMock()
    bundle_dir = os.path.join(dg_index_path, 'task_datasets',
                              str(job_input_obj.id))
    service.materialize_task_dataset.return_value = MaterializeResult(
        manifest_path=os.path.join(bundle_dir, 'manifest.json'),
        runtime_manifest_path=os.path.join(bundle_dir, 'runtime_manifest.json'),
        bundle_dir=bundle_dir,
        storage_kind='diskgraph_table',
    )
    fake_jc = SimpleNamespace(diskgraph_task_dataset_bundle_dir=None)
    # tt_id is now sourced from the durable bundle's tt_ingest_spec.json (not
    # path-parsed). For ``.../jobs/job-abc/...`` the spec records 'job-abc'; stub
    # the bundle read so these spec/handle tests don't need a real bundle on disk.
    with patch(_P_GET_SERVICE, return_value=service), \
         patch(_P_COLLECT_ENCODERS, return_value=_encoders(encoders_dict)), \
         patch.object(_mtt, '_ttng_train_table_tt_id', return_value='job-abc'), \
         patch.object(_mtt.fsutil, 'exists', return_value=True), \
         patch.object(_mtt.JobConfig, 'load', return_value=fake_jc), \
         patch.object(_mtt, 'save_json'):
        _maybe_materialize_ttng_task_dataset(
            task=task_obj,
            pquery=MagicMock(),
            train_table_path=train_table_path,
            job_input=job_input_obj,
            diskgraph_index_path=dg_index_path,
            job_config_path='/tmp/job_config.json',
        )
    assert service.materialize_task_dataset.call_count == 1
    args, _ = service.materialize_task_dataset.call_args
    return args  # (handle, spec)


# ---------------------------------------------------------------------------
# Gate-off scenarios — the helper must not call the service.
# ---------------------------------------------------------------------------


def test_noop_when_diskgraph_index_path_none() -> None:
    task_obj, ji, ttp, _ = _default_inputs()
    service = MagicMock()
    with patch(_P_GET_SERVICE, return_value=service):
        _maybe_materialize_ttng_task_dataset(
            task=task_obj,
            pquery=MagicMock(),
            train_table_path=ttp,
            job_input=ji,
            diskgraph_index_path=None,
            job_config_path='/tmp/job_config.json',
        )
    service.materialize_task_dataset.assert_not_called()


# ---------------------------------------------------------------------------
# Success-path: spec + handle construction.
# ---------------------------------------------------------------------------


def test_handle_carries_job_id_and_graph_identity() -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    handle, _ = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {
                'AGE': object()
            },
            'COMPANIES': {}
        },
    )
    assert isinstance(handle, DiskgraphHandle)
    # The display ``graph_key`` ('gk') is normalized to its GraphIDv2
    # ('NA-<fpid>') form so ``_resolve(handle)`` lands on the same on-disk
    # snapshot dir that ``get_diskgraph_index_path_for_job`` (the ingest side) wrote to;
    # otherwise the ingest-binding validator rejects the (NA-keyed) manifest.
    from statedb.config._graph_key_norm import _graph_key_to_v2
    assert handle.graph_key == str(_graph_key_to_v2('gk'))
    assert handle.graph_key.startswith('NA-')
    assert handle.graph_data_version == 'v1'
    assert handle.job_id == 'job-abc'


def test_spec_artifact_root_is_snapshot_relative() -> None:
    """``artifact_root`` must match the convention enforced by
    ``open_task_dataset`` / ``cleanup_task_dataset``: snapshot-scoped,
    not job-scoped. Wrong path here = manifest unreachable."""
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {
                'AGE': object()
            },
            'COMPANIES': {}
        },
    )
    assert isinstance(spec, TaskDatasetMaterializeSpec)
    assert spec.artifact_root == os.path.join(dgp, 'task_datasets', 'job-abc')


def test_spec_tt_manifest_path_is_wheel_side_manifest() -> None:
    """``tt_manifest_path`` resolves to the wheel-side TT manifest at
    ``<diskgraph_index_path>/train_tables/<tt_id>/manifest.json`` — NOT
    the kumo-cache sidecar ``<train_table_path>.manifest.json``.
    ``tt_id`` is derived from the ``.../jobs/<tt_id>/...`` segment of
    ``train_table_path`` (the gen-train-table job id).
    """
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {},
            'COMPANIES': {}
        },
    )
    # _default_inputs ttp = '/tmp/jobs/job-abc/tt.parquet' -> tt_id='job-abc'
    assert spec.tt_manifest_path == os.path.join(dgp, 'train_tables', 'job-abc',
                                                 'manifest.json')


def test_spec_task_identity_for_v1_binary_classification() -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {},
            'COMPANIES': {}
        },
    )
    assert spec.task_family == 'node_prediction'
    assert spec.task_type == 'binary_classification'
    assert spec.is_temporal is False


@pytest.mark.parametrize(
    'task_type, task_type_value',
    [
        (TaskType.MULTICLASS_CLASSIFICATION, 'multiclass_classification'),
        (TaskType.MULTILABEL_CLASSIFICATION, 'multilabel_classification'),
        (TaskType.MULTILABEL_RANKING, 'multilabel_ranking'),
        (TaskType.REGRESSION, 'regression'),
    ],
)
def test_spec_task_identity_for_supported_node_subtypes(
    task_type: TaskType,
    task_type_value: str,
) -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    task_obj.task_type = task_type

    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {},
            'COMPANIES': {}
        },
    )

    assert spec.task_family == 'node_prediction'
    assert spec.task_type == task_type_value
    assert spec.is_temporal is False


def test_spec_uses_task_temporal_property_for_node_prediction() -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    task_obj.task_type = TaskType.MULTILABEL_CLASSIFICATION
    task_obj.is_temporal = True

    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {},
            'COMPANIES': {}
        },
    )

    assert spec.task_family == 'node_prediction'
    assert spec.task_type == 'multilabel_classification'
    assert spec.is_temporal is True


def test_spec_entity_table_strips_connector_prefix() -> None:
    """train_table.fkeys[0].dst_table = 'src1.USERS' -> 'USERS'."""
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {},
            'COMPANIES': {}
        },
    )
    assert spec.entity_table_name == 'USERS'
    assert spec.seed_node_type == 'USERS'


def test_spec_entity_table_no_connector_prefix() -> None:
    """dst_table without ``.`` passes through unchanged."""
    task_obj = _task(
        graph_tables=[_table('USERS', fkeys=[])],
        train_table_fkeys=[_fkey('entity', 'USERS')],
    )
    _, spec = _run_helper(
        task_obj,
        '/tmp/jobs/job-abc/tt.parquet',
        _job_input(),
        '/dg',
        {'USERS': {}},
    )
    assert spec.entity_table_name == 'USERS'


def test_spec_feature_schema_excludes_train_table_entry() -> None:
    """``EncodersConfig.encoders`` contains an entry for the train
    table; ``feature_schema`` must include only graph node types."""
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {
                'AGE': object()
            },
            'COMPANIES': {},
            'TRAIN_TABLE': {
                'TARGET': object()
            },  # must NOT appear
        },
    )
    assert set(spec.feature_schema['node_types']) == {'USERS', 'COMPANIES'}
    assert spec.feature_schema['target_encoder_input'] is None


def test_spec_feature_schema_featureless_node_gets_empty_columns() -> None:
    """Node types absent from ``EncodersConfig.encoders`` get an empty
    ``feature_columns`` list (the runtime adapter fills zero embeddings
    for these). ``mapper_key == node_type`` is the v1 convention."""
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {'USERS': {
            'AGE': object(),
            'NAME': object()
        }},
        # COMPANIES intentionally absent from encoders
    )
    nodes = spec.feature_schema['node_types']
    assert nodes['USERS']['feature_columns'] == ['AGE', 'NAME']
    assert nodes['USERS']['mapper_key'] == 'USERS'
    assert [value['column_name']
            for value in nodes['USERS']['column_specs']] == ['AGE', 'NAME']
    assert nodes['COMPANIES']['feature_columns'] == []
    assert nodes['COMPANIES']['mapper_key'] == 'COMPANIES'


def test_spec_feature_schema_applies_task_exclusions() -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    task_obj.exclude_cols = {'USERS': ['NAME']}

    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {'USERS': {
            'AGE': object(),
            'NAME': object(),
        }},
    )

    assert spec.feature_schema['node_types']['USERS']['feature_columns'] == [
        'AGE'
    ]


def test_spec_feature_schema_excludes_same_table_target() -> None:
    """Static same-table node prediction: the PQL target is a data column on
    the entity table itself. ``Task.exclude_cols`` lists it to prevent leakage;
    the feature_schema must drop it from the entity node's features.

    Regression for the mercari static single-table collapse (target column
    ``price`` leaked into the entity features via the wrong attribute name).
    """
    entity = _table('ITEMS', fkeys=[])
    task_obj = _task(
        task_type=TaskType.REGRESSION,
        graph_tables=[entity],
        train_table_fkeys=[_fkey('entity', 'src1.ITEMS')],
    )
    # The target column lives on the entity table; it must be excluded.
    task_obj.exclude_cols = {'ITEMS': ['PRICE']}

    _, spec = _run_helper(
        task_obj,
        '/tmp/jobs/job-abc/tt.parquet',
        _job_input(),
        '/data/gk/v1',
        {'ITEMS': {
            'NAME': object(),
            'PRICE': object(),
        }},
    )

    assert spec.feature_schema['node_types']['ITEMS']['feature_columns'] == [
        'NAME'
    ]


def test_spec_graph_node_and_edge_types_from_task_graph() -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    _, spec = _run_helper(
        task_obj,
        ttp,
        ji,
        dgp,
        {
            'USERS': {},
            'COMPANIES': {}
        },
    )
    assert spec.graph_node_types == ('USERS', 'COMPANIES')
    # Forward edges only — reverse edges are composed at sampling time
    # in the wheel and don't belong in this schema-level list.
    assert spec.graph_edge_types == (('USERS', 'company', 'COMPANIES'), )


# ---------------------------------------------------------------------------
# Error paths — all are wrapped as TrainingDataMaterializationError.
# ---------------------------------------------------------------------------


def test_unsupported_node_task_raises_wrapped() -> None:
    task_obj = _task(
        task_type=TaskType.INVALID,
        graph_tables=[_table('USERS', fkeys=[])],
        train_table_fkeys=[_fkey('entity', 'USERS')],
    )
    with patch(_P_GET_SERVICE, return_value=MagicMock()), \
         patch(_P_COLLECT_ENCODERS, return_value=_encoders({})):
        with pytest.raises(TrainingDataMaterializationError):
            _maybe_materialize_ttng_task_dataset(
                task=task_obj,
                pquery=MagicMock(),
                train_table_path='/p',
                job_input=_job_input(),
                diskgraph_index_path='/dg',
                job_config_path='/tmp/job_config.json',
            )


def test_empty_train_table_fkeys_raises_wrapped() -> None:
    task_obj = _task(
        graph_tables=[_table('USERS', fkeys=[])],
        train_table_fkeys=[],  # empty
    )
    with patch(_P_GET_SERVICE, return_value=MagicMock()), \
         patch(_P_COLLECT_ENCODERS, return_value=_encoders({})):
        with pytest.raises(TrainingDataMaterializationError):
            _maybe_materialize_ttng_task_dataset(
                task=task_obj,
                pquery=MagicMock(),
                train_table_path='/p',
                job_input=_job_input(),
                diskgraph_index_path='/dg',
                job_config_path='/tmp/job_config.json',
            )


def test_service_failure_wraps_as_training_data_materialization_error() -> None:
    task_obj, ji, ttp, dgp = _default_inputs()
    service = MagicMock()
    service.materialize_task_dataset.side_effect = RuntimeError('boom')
    with patch(_P_GET_SERVICE, return_value=service), \
         patch(_P_COLLECT_ENCODERS, return_value=_encoders({'USERS': {}})):
        with pytest.raises(TrainingDataMaterializationError) as excinfo:
            _maybe_materialize_ttng_task_dataset(
                task=task_obj,
                pquery=MagicMock(),
                train_table_path=ttp,
                job_input=ji,
                diskgraph_index_path=dgp,
                job_config_path='/tmp/job_config.json',
            )
    assert 'TT-NG' in str(excinfo.value)


# ---------------------------------------------------------------------------
# JobConfig handoff — the bundle_dir field is the authoritative TT-NG
# signal that crosses the materialize / train activity boundary.
# ---------------------------------------------------------------------------


def test_job_config_updated_with_bundle_dir(tmp_path) -> None:
    """On success, the helper records the DG-native bundle_dir on the
    on-disk JobConfig at ``job_config_path``. ``bin/kumo-auto`` reads
    that field to dispatch to ``DiskgraphNestedTrainer`` instead of
    the legacy ``AutoTrainer`` path."""
    task_obj, ji, ttp, dgp = _default_inputs()
    expected_bundle = os.path.join(dgp, 'task_datasets', str(ji.id))
    job_config_path = str(tmp_path / 'job_config.json')

    service = MagicMock()
    service.materialize_task_dataset.return_value = MaterializeResult(
        manifest_path=os.path.join(expected_bundle, 'manifest.json'),
        runtime_manifest_path=os.path.join(expected_bundle,
                                           'runtime_manifest.json'),
        bundle_dir=expected_bundle,
        storage_kind='diskgraph_table',
    )
    fake_jc = SimpleNamespace(diskgraph_task_dataset_bundle_dir=None)
    with patch(_P_GET_SERVICE, return_value=service), \
         patch(_P_COLLECT_ENCODERS,
               return_value=_encoders({'USERS': {}})), \
         patch.object(_mtt, '_ttng_train_table_tt_id',
                      return_value='job-abc'), \
         patch.object(_mtt.fsutil, 'exists', return_value=True), \
         patch.object(_mtt.JobConfig, 'load',
                      return_value=fake_jc) as p_load, \
         patch.object(_mtt, 'save_json') as p_save:
        _maybe_materialize_ttng_task_dataset(
            task=task_obj,
            pquery=MagicMock(),
            train_table_path=ttp,
            job_input=ji,
            diskgraph_index_path=dgp,
            job_config_path=job_config_path,
        )

    p_load.assert_called_once_with(job_config_path)
    assert fake_jc.diskgraph_task_dataset_bundle_dir == expected_bundle
    p_save.assert_called_once()
    saved_obj, saved_path = p_save.call_args.args
    assert saved_obj is fake_jc
    assert saved_path == job_config_path


# ---------------------------------------------------------------------------
# Re-ingest the per-job train_tables extension on a torn-down train pod.
#
# The index is keyed by GRAPH (not job), so the gen-traintable pod and the
# training pod resolve to the same index path -- but the per-job
# ``train_tables/<tt_id>/`` extension is NOT part of the dataplane snapshot, so
# the snapshot restore on the train pod rebuilds the base index WITHOUT it.
# ``materialize_task_dataset`` then reads ``spec.tt_manifest_path`` from that
# missing extension -> FileNotFoundError. The fix re-ingests from the durable
# persisted staging bundle when the extension is absent.
# ---------------------------------------------------------------------------

# Materialization already owns the selected immutable bundle, so it delegates
# to the selected-stage self-heal instead of resolving a publication root.
_P_ENSURE_STAGE_INGESTED = (
    'kumo.data.generic.diskgraph_staging.ensure_stage_ingested')
_P_PARSE_TT_COL_STATS = ('kumo.diskgraph.diskgraph_native_tt_artifacts.'
                         'parse_train_table_col_stats_artifact')


def _categorical_train_table_task() -> SimpleNamespace:
    return SimpleNamespace(
        task_type=SimpleNamespace(is_link_pred=False),
        train_table=SimpleNamespace(
            cols=[
                SimpleNamespace(name='ENTITY', dtype=Dtype.string,
                                stype=Stype.categorical),
            ],
            target_col=SimpleNamespace(name='TARGET', dtype=Dtype.stringlist,
                                       stype=Stype.multicategorical),
        ),
    )


def _link_train_table_task() -> SimpleNamespace:
    return SimpleNamespace(
        task_type=SimpleNamespace(is_link_pred=True),
        train_table=SimpleNamespace(cols=[], target_col=None),
    )


def _col_stats_payload(*stat_keys: str) -> dict:
    return {'stats': {key: {'value': []} for key in stat_keys}}


def test_malformed_sidecar_is_non_retryable(tmp_path) -> None:
    sidecar = tmp_path / 'train_table_col_stats.json'
    sidecar.write_text('{"version": 2, "kind": "wrong"}')

    with pytest.raises(TrainingDataMaterializationError) as excinfo:
        _mtt._ttng_validate_train_table_col_stats_sidecar(
            _categorical_train_table_task(), str(sidecar))

    assert excinfo.value.non_retryable is True


def test_real_target_artifact_passes_without_entity_counts(tmp_path) -> None:
    sidecar = tmp_path / 'train_table_col_stats.json'
    sidecar.write_text(
        json.dumps({
            'version': 1,
            'kind': 'tt_train_table_col_stats',
            'stats_format': 'kumo_col_stats_v1',
            'stats': {
                'TARGET1__kumo.multicategorical__'
                'MULTI_CATEGORY_COUNTS__False': {
                    'value': []
                },
            },
        }))

    _mtt._ttng_validate_train_table_col_stats_sidecar(
        _categorical_train_table_task(), str(sidecar))


def test_incomplete_sidecar_fails_before_train_table_reingest() -> None:
    """A durable sidecar gap is terminal before the expensive self-heal."""
    published_sidecar = '/dg/train_tables/tt-id/train_table_col_stats.json'
    durable_sidecar = '/durable/version/train_table_col_stats.json'
    payload = _col_stats_payload(
        'ENTITY1__kumo.categorical__CATEGORY_COUNTS__False')

    with patch.object(
            _mtt, '_ttng_train_table_col_stats_sidecar_path',
            return_value=published_sidecar), \
         patch.object(_mtt.fsutil, 'exists',
                      side_effect=lambda path: path == durable_sidecar), \
         patch('kumo.data.generic.diskgraph_train_table.'
               'resolve_durable_ttng_bundle',
               return_value='/durable/version'), \
         patch(_P_PARSE_TT_COL_STATS, return_value=payload), \
         patch.object(_mtt, '_ttng_ensure_train_table_extension') as ensure:
        with pytest.raises(TrainingDataMaterializationError) as excinfo:
            _mtt._ttng_prepare_train_table_col_stats(
                _categorical_train_table_task(),
                '/durable/version/tt.parquet', '/dg', 'tt-id')

    assert excinfo.value.non_retryable is True
    assert 'TARGET' in str(excinfo.value)
    assert 'MULTI_CATEGORY_COUNTS' in str(excinfo.value)
    ensure.assert_not_called()


def test_sidecar_without_entity_counts_is_validated_before_reingest() -> None:
    published_sidecar = '/dg/train_tables/tt-id/train_table_col_stats.json'
    durable_sidecar = '/durable/version/train_table_col_stats.json'
    payload = _col_stats_payload(
        'TARGET1__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False', )
    events: list[str] = []
    extension_ready = False

    def _exists(path):
        return path == durable_sidecar or (path == published_sidecar
                                           and extension_ready)

    def _parse(_path):
        events.append('validate')
        return payload

    def _ensure(*_args):
        nonlocal extension_ready
        events.append('ensure')
        extension_ready = True

    with patch.object(
            _mtt, '_ttng_train_table_col_stats_sidecar_path',
            return_value=published_sidecar), \
         patch.object(_mtt.fsutil, 'exists', side_effect=_exists), \
         patch('kumo.data.generic.diskgraph_train_table.'
               'resolve_durable_ttng_bundle',
               return_value='/durable/version'), \
         patch(_P_PARSE_TT_COL_STATS, side_effect=_parse), \
         patch.object(_mtt, '_ttng_ensure_train_table_extension',
                      side_effect=_ensure):
        result = _mtt._ttng_prepare_train_table_col_stats(
            _categorical_train_table_task(), '/durable/version/tt.parquet',
            '/dg', 'tt-id')

    assert result == published_sidecar
    assert events == ['validate', 'ensure', 'validate']


def test_link_task_accepts_absent_col_stats_sidecar() -> None:
    published_sidecar = '/dg/train_tables/tt-id/train_table_col_stats.json'

    with patch.object(
            _mtt, '_ttng_train_table_col_stats_sidecar_path',
            return_value=published_sidecar), \
         patch.object(_mtt.fsutil, 'exists', return_value=False), \
         patch.object(_mtt, '_ttng_validate_train_table_col_stats_sidecar'
                      ) as validate, \
         patch.object(_mtt, '_ttng_ensure_train_table_extension') as ensure:
        result = _mtt._ttng_prepare_train_table_col_stats(
            _link_train_table_task(), '/bundle/tt.parquet', '/dg', 'tt-id')

    assert result is None
    ensure.assert_called_once_with('/bundle/tt.parquet', '/dg', 'tt-id')
    validate.assert_not_called()


def test_link_task_validates_historical_present_sidecar() -> None:
    published_sidecar = '/dg/train_tables/tt-id/train_table_col_stats.json'
    task = _link_train_table_task()

    with patch.object(
            _mtt, '_ttng_train_table_col_stats_sidecar_path',
            return_value=published_sidecar), \
         patch.object(_mtt.fsutil, 'exists', return_value=True), \
         patch.object(_mtt, '_ttng_validate_train_table_col_stats_sidecar'
                      ) as validate, \
         patch.object(_mtt, '_ttng_ensure_train_table_extension') as ensure:
        result = _mtt._ttng_prepare_train_table_col_stats(
            task, '/bundle/tt.parquet', '/dg', 'tt-id')

    assert result == published_sidecar
    ensure.assert_called_once_with('/bundle/tt.parquet', '/dg', 'tt-id')
    assert validate.call_count == 2
    validate.assert_called_with(task, published_sidecar)


def test_link_task_malformed_present_sidecar_fails_closed(tmp_path) -> None:
    sidecar = tmp_path / 'train_table_col_stats.json'
    sidecar.write_text('{"version": 2, "kind": "wrong"}')

    with patch.object(
            _mtt, '_ttng_train_table_col_stats_sidecar_path',
            return_value=str(sidecar)), \
         patch.object(_mtt, '_ttng_ensure_train_table_extension') as ensure:
        with pytest.raises(TrainingDataMaterializationError) as excinfo:
            _mtt._ttng_prepare_train_table_col_stats(_link_train_table_task(),
                                                     '/bundle/tt.parquet',
                                                     '/dg', 'tt-id')

    assert excinfo.value.non_retryable is True
    ensure.assert_not_called()


def _spy_ensure_stage_ingested(monkeypatch, sink: dict, *,
                               raises: Exception | None = None):
    """Patch the shared self-heal at its source. Records the kwargs
    ``_ttng_ensure_train_table_extension`` delegated with into
    ``sink['ensure_kwargs']`` (and increments ``sink['ensure_calls']``); if
    ``raises`` is given, the spy raises it to simulate a missing/incomplete
    durable bundle."""
    sink['ensure_calls'] = 0

    def _ensure(**kwargs):
        sink['ensure_calls'] += 1
        sink['ensure_kwargs'] = kwargs
        if raises is not None:
            raise raises
        # Real helper returns the published extension path.
        return os.path.join(kwargs['index_path'], 'train_tables',
                            kwargs['target_tt_id'])

    monkeypatch.setattr(_P_ENSURE_STAGE_INGESTED, _ensure)


def test_ensure_extension_delegates_to_selected_bundle_self_heal(
        monkeypatch) -> None:
    """``_ttng_ensure_train_table_extension`` delegates unconditionally to the
    shared selected-bundle self-heal, which owns completeness validation and
    drop-and-heal, rather than a direct ``execute_tt_ingest`` call."""
    sink: dict = {}
    _spy_ensure_stage_ingested(monkeypatch, sink)
    dg_index = '/data/gk/v1'
    ttp = '/tmp/jobs/job-abc/tt.parquet'
    tt_id = 'job-abc'  # spec tt_id for _default_inputs staging bundle
    _mtt._ttng_ensure_train_table_extension(ttp, dg_index, tt_id)

    assert sink['ensure_calls'] == 1
    assert sink['ensure_kwargs'] == {
        'index_path': dg_index,
        'target_tt_id': tt_id,
        'staging_dir': os.path.dirname(ttp),
        'predict_stage_name': None,
    }


def test_ensure_extension_raises_when_no_staging(monkeypatch) -> None:
    """Extension missing AND selected bundle missing -> selected self-heal
    raises, which is wrapped into a non-retryable TrainingDataMaterializationError
    (NOT a silent fall-through to gen scratch)."""
    sink: dict = {}
    # Durable bundle missing/incomplete -> the shared self-heal raises.
    _spy_ensure_stage_ingested(
        monkeypatch, sink,
        raises=RuntimeError('durable train staging is missing or incomplete'))
    monkeypatch.setattr(_mtt.fsutil, 'exists', lambda _path: False)
    with pytest.raises(TrainingDataMaterializationError) as excinfo:
        _mtt._ttng_ensure_train_table_extension('/tmp/jobs/job-abc/tt.parquet',
                                                '/data/gk/v1', 'TT_ID')
    # Error is non-retryable, names the actionable recovery (persist the
    # complete staging bundle), and chains the underlying cause.
    assert excinfo.value.non_retryable is True
    assert 'staging' in str(excinfo.value).lower()
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    # The selected-stage self-heal was attempted; there is no silent fallback.
    assert sink['ensure_calls'] == 1


def test_ensure_extension_reingest_uses_local_copy_not_durable_s3(
        monkeypatch) -> None:
    """Regression: the old code called execute_tt_ingest directly on the durable
    S3 staging dir, which DG's std::fs cannot read ("cannot read
    <bundle>/tt_ingest_spec.json: os error 2"). The fix routes through
    ensure_stage_ingested (copy-local-first). Assert no direct ingest
    against the durable path."""
    sink: dict = {}
    _spy_ensure_stage_ingested(monkeypatch, sink)
    # Trip-wire: if materialize ever calls execute_tt_ingest directly (the old
    # durable-S3 path), fail loudly.
    direct_calls: list = []
    monkeypatch.setitem(
        __import__('sys').modules, 'diskgraph',
        SimpleNamespace(execute_tt_ingest=lambda **kw: direct_calls.append(kw)))
    dg_index = '/data/gk/v1'
    ttp = '/tmp/jobs/job-abc/tt.parquet'
    tt_id = 'job-abc'
    ext_manifest = os.path.join(dg_index, 'train_tables', tt_id,
                                'manifest.json')
    monkeypatch.setattr(_mtt.fsutil, 'exists', lambda p: p != ext_manifest)
    monkeypatch.setattr(_mtt.fsutil, 'rm', lambda p, **kw: None)

    _mtt._ttng_ensure_train_table_extension(ttp, dg_index, tt_id)

    # No direct execute_tt_ingest against the durable path.
    assert direct_calls == []
    # Delegated to the self-heal with the durable bundle as the source.
    assert sink['ensure_calls'] == 1
    assert sink['ensure_kwargs']['staging_dir'] == os.path.dirname(ttp)


def test_materialize_reingests_extension_before_reading_manifest(
        monkeypatch, tmp_path) -> None:
    """End-to-end through ``_maybe_materialize_ttng_task_dataset``: when the
    extension manifest is missing, the helper delegates to the shared
    selected-stage self-heal (so the downstream service read does not
    FileNotFoundError) and still calls ``materialize_task_dataset`` exactly
    once."""
    sink: dict = {}
    _spy_ensure_stage_ingested(monkeypatch, sink)
    task_obj, ji, ttp, dgp = _default_inputs()
    tt_id = 'job-abc'  # spec tt_id for _default_inputs staging bundle
    ext_manifest = os.path.join(dgp, 'train_tables', tt_id, 'manifest.json')

    def _exists(path):
        # Only the extension manifest is missing; everything else (incl. the
        # durable staging parquet) exists.
        if path == ext_manifest:
            return False
        return True

    service = MagicMock()
    bundle_dir = os.path.join(dgp, 'task_datasets', str(ji.id))
    service.materialize_task_dataset.return_value = MaterializeResult(
        manifest_path=os.path.join(bundle_dir, 'manifest.json'),
        runtime_manifest_path=os.path.join(bundle_dir, 'runtime_manifest.json'),
        bundle_dir=bundle_dir,
        storage_kind='diskgraph_table',
    )
    fake_jc = SimpleNamespace(diskgraph_task_dataset_bundle_dir=None)
    with patch(_P_GET_SERVICE, return_value=service), \
         patch(_P_COLLECT_ENCODERS, return_value=_encoders({'USERS': {}})), \
         patch.object(_mtt, '_ttng_train_table_tt_id', return_value=tt_id), \
         patch.object(_mtt.fsutil, 'exists', _exists), \
         patch.object(_mtt.fsutil, 'rm'), \
         patch.object(_mtt.JobConfig, 'load', return_value=fake_jc), \
         patch.object(_mtt, 'save_json'):
        _maybe_materialize_ttng_task_dataset(
            task=task_obj,
            pquery=MagicMock(),
            train_table_path=ttp,
            job_input=ji,
            diskgraph_index_path=dgp,
            job_config_path=str(tmp_path / 'job_config.json'),
        )

    # Self-heal fired against the durable staging bundle via the shared,
    # cross-pod-safe helper...
    assert sink['ensure_calls'] == 1
    assert sink['ensure_kwargs'] == {
        'index_path': dgp,
        'target_tt_id': tt_id,
        'staging_dir': os.path.dirname(ttp),
        'predict_stage_name': None,
    }
    # ...and the service was still invoked.
    assert service.materialize_task_dataset.call_count == 1
