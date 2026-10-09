# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DiskgraphService task-dataset methods.

TT-NG only.

Covers materialize_task_dataset, open_task_dataset, and
cleanup_task_dataset.

``materialize_task_dataset`` reads the durable ingest bundle that
``execute_tt_ingest`` publishes into the extension as the SINGLE authority:
``diskgraph_manifest.json`` -> ``task_dataset`` and ``stats.json`` ->
``parsed_stats``. The gen ``manifest.json`` is never read here. These tests
write that durable bundle on disk (via ``_setup_tt_manifest``); they exercise
orchestration (bundle → thin/flat sidecars → open round-trip), not the
artifact validator, so we skip the full TT parquet / split.bin / sha256
plumbing. The autouse ``get_validated_artifact_bundle`` patch remains for the
separate ``validate_manifest`` path.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
from kumo_core.task_dataset import TaskDatasetMaterializeSpec

from kumo.diskgraph._backend import DiskgraphHandle
from kumo.diskgraph.service import DiskgraphService, MaterializeResult

_PATCH_TARGET = (
    'kumo.diskgraph.diskgraph_native_tt_artifacts.get_validated_artifact_bundle')


@pytest.fixture(autouse=True)
def patch_artifact_bundle_validator():
    """Replace ``get_validated_artifact_bundle`` with a pass-through
    JSON reader of the manifest path. Orchestration tests don't need
    real TT artifacts on disk; they only need the validator to return
    a manifest-shaped dict for service code to consume."""
    def _read(manifest_path: str) -> dict:
        with open(manifest_path) as f:
            return json.load(f)

    with patch(_PATCH_TARGET, side_effect=_read):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MinimalConfig:
    """Stand-in for DiskgraphConfig in tests."""
    data_root: str


def _make_service(tmp_path: str) -> DiskgraphService:
    config = _MinimalConfig(data_root=tmp_path)  # type: ignore[arg-type]
    return DiskgraphService(config)


def _make_handle(
    graph_key: str = 'test_graph',
    graph_data_version: str = 'v1',
    job_id: Optional[str] = 'job-abc-123',
) -> DiskgraphHandle:
    return DiskgraphHandle(
        graph_key=graph_key,
        graph_data_version=graph_data_version,
        job_id=job_id,
    )


def _make_spec(artifact_root: str, **overrides) -> TaskDatasetMaterializeSpec:
    defaults = dict(
        tt_manifest_path='/tmp/tt/manifest.json',
        job_id='job-abc-123',
        artifact_root=artifact_root,
        task_family='node_prediction',
        task_type='binary_classification',
        is_temporal=False,
        seed_node_type='USERS',
        train_table_name='TT_USERS',
        entity_table_name='USERS',
        feature_schema={
            'node_types': {
                'USERS': {
                    'feature_columns': ['AGE'],
                    'mapper_key':
                    'USERS',
                    'time_column':
                    None,
                    'column_specs': [{
                        'column_name': 'AGE',
                        'dtype': 'float',
                        'stype': 'numerical',
                        'encoder': {
                            'name': 'Numerical'
                        },
                        'stats': {
                            'MEAN': 0.0
                        },
                    }],
                },
            },
            'target_encoder_input': None,
        },
        graph_node_types=('USERS', ),
        graph_edge_types=(),
    )
    defaults.update(overrides)
    return TaskDatasetMaterializeSpec(**defaults)


def _setup_tt_manifest(
    tmp_path,
    stage_counts: Optional[dict[str, int]] = None,
    provided_fields: Optional[list[str]] = None,
    task_dataset: Optional[dict] = None,
    *,
    omit_task_dataset: bool = False,
    omit_ingest_manifest: bool = False,
    omit_durable_stats: bool = False,
    graph_key: str = 'test_graph',
    graph_data_version: str = 'v1',
    tt_id: str = 'tt-test',
) -> str:
    """Write the durable TT-NG ingest bundle into ``<tmp_path>`` and return a
    path whose dirname is that bundle dir (plugs into ``spec.tt_manifest_path``).

    CONTRACT: post-ingest, ``materialize_task_dataset`` reads the durable
    bundle that ``execute_tt_ingest`` publishes into the extension as the
    SINGLE authority:
      - ``diskgraph_manifest.json`` -> ``task_dataset``
      - ``stats.json``              -> ``parsed_stats``
    The gen ``tt_manifest.json`` (``spec.tt_manifest_path``) is consumed
    solely by ``execute_tt_ingest`` and is NOT read here. We still write a
    gen file so the path resolves, but it is irrelevant to materialization.

    ``stage_counts`` defaults to a representative non-trivial split and is
    written into ``task_dataset.stages[<stage>].row_count`` (the sole source
    now that split.bin / top-level stage_counts are gone). ``task_dataset``
    defaults to the canonical wheel-produced block.

    Kwargs to simulate broken durable bundles:
      - ``omit_task_dataset``: write ``diskgraph_manifest.json`` WITHOUT a
        ``task_dataset`` block.
      - ``omit_ingest_manifest``: don't write ``diskgraph_manifest.json`` at all.
      - ``omit_durable_stats``: don't write ``stats.json``.

    ``provided_fields`` is preserved as a kwarg for symmetry with the
    has_time / has_weight tests, but ``materialize_task_dataset`` now
    derives those from ``task_dataset.columns``; we wire timestamp /
    weight column names accordingly when provided_fields is set.
    """
    if stage_counts is None:
        stage_counts = {'train': 100, 'val': 20, 'test': 20}
    total = sum(stage_counts.values())

    graph_index_path = os.path.join(str(tmp_path), graph_key,
                                    graph_data_version)
    tt_extension_path = os.path.join(graph_index_path, 'train_tables', tt_id)
    os.makedirs(tt_extension_path, exist_ok=True)

    if task_dataset is None:
        task_dataset = _default_task_dataset(
            stage_counts, graph_index_path=graph_index_path,
            tt_extension_path=tt_extension_path,
            provided_fields=provided_fields)

    # Gen manifest -- irrelevant to materialize, written only so the path
    # resolves to the bundle dir.
    manifest_path = os.path.join(tt_extension_path, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({'version': 1, 'row_count': total}, f)

    # Durable ingest manifest (the authoritative task_dataset source).
    if not omit_ingest_manifest:
        dg_manifest: dict = {
            'schema_version':
            'diskgraph.tt_execute_result.v1',
            'tt_id':
            tt_id,
            'graph_index_path':
            graph_index_path,
            'tt_extension_path':
            tt_extension_path,
            'diskgraph_manifest_path':
            os.path.join(tt_extension_path, 'diskgraph_manifest.json'),
        }
        if not omit_task_dataset:
            dg_manifest['task_dataset'] = task_dataset
        with open(os.path.join(tt_extension_path, 'diskgraph_manifest.json'),
                  'w') as f:
            json.dump(dg_manifest, f)

    # Durable parsed-stats sidecar.
    if not omit_durable_stats:
        _write_durable_stats(manifest_path)

    with open(os.path.join(tt_extension_path, 'train_table_col_stats.json'),
              'w') as f:
        json.dump(
            {
                'version': 1,
                'kind': 'tt_train_table_col_stats',
                'stats_format': 'kumo_col_stats_v1',
                'stats': {
                    'TARGET1__kumo.categorical__CATEGORY_COUNTS__False': {
                        'value': [[0, 100], [1, 40]],
                    },
                },
            }, f)

    return manifest_path


def _default_task_dataset(
    stage_counts: dict[str, int],
    *,
    graph_index_path: str,
    tt_extension_path: str,
    provided_fields: Optional[list[str]] = None,
) -> dict:
    """Build a fake ``producer_result['task_dataset']`` block matching the
    shape ``execute_tt_plan`` returns in the TT-NG wheel."""
    has_time = bool(provided_fields and 'time' in provided_fields)
    has_weight = bool(provided_fields and 'weight' in provided_fields)
    return {
        'schema_version': 'diskgraph.task_dataset.v1',
        'graph_index_path': graph_index_path,
        'tt_extension_path': tt_extension_path,
        'stages': {
            stage: {
                'table_name': f'{stage}_seeds',
                'index_path': tt_extension_path,
                'row_count': count,
            }
            for stage, count in stage_counts.items()
        },
        'columns': {
            'primary_entity': 'ENTITY1',
            'primary_target': 'TARGET1',
            'timestamp': 'TIMESTAMP' if has_time else None,
            'weight': 'WEIGHT' if has_weight else None,
        },
    }


# ---------------------------------------------------------------------------
# Orchestration tests
# ---------------------------------------------------------------------------


def test_materialize_returns_result_with_manifest(tmp_path) -> None:
    """materialize_task_dataset returns a MaterializeResult pointing
    at a real, readable manifest.json."""
    service = _make_service(str(tmp_path))
    handle = _make_handle()
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(handle, spec)

    assert isinstance(result, MaterializeResult)
    assert result.storage_kind == 'diskgraph_table'
    assert os.path.isfile(result.manifest_path)
    assert result.bundle_dir == os.path.dirname(result.manifest_path)


def test_materialize_writes_all_expected_files(tmp_path) -> None:
    """The manifest dir contains 7 sidecar files plus the flat runtime manifest."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(_make_handle(), spec)

    expected = {
        'manifest.json',
        'metadata.json',
        'feature_schema.json',
        'column_stats.json',
        'graph_runtime_metadata.json',
        'split_info.json',
        'stats.json',
        'checksums.json',
        'runtime_manifest.json',
    }
    actual = set(os.listdir(result.bundle_dir))
    assert expected == actual
    assert result.runtime_manifest_path == os.path.join(
        result.bundle_dir, 'runtime_manifest.json')


def _write_durable_stats(manifest_path: str, **overrides) -> dict:
    """Write a durable, self-describing ``stats.json`` sibling next to the
    manifest, mirroring what ``execute_tt_ingest`` publishes into the
    extension (``copy_tt_artifacts_to_extension``)."""
    payload = {
        'version':
        2,
        'kind':
        'tt_stats',
        'num_total_rows':
        140,
        'num_train_rows':
        100,
        'num_entities':
        50,
        'num_targets':
        7,
    }
    payload.update(overrides)
    stats_path = os.path.join(os.path.dirname(manifest_path), 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(payload, f)
    return payload


def test_materialize_parsed_stats_from_durable_sidecar_not_gen(
        tmp_path) -> None:
    """CONTRACT: ``parsed_stats`` comes from the durable ``stats.json``
    sidecar that ``execute_tt_ingest`` publishes into the extension. The gen
    ``manifest.json`` is never validated -- point ``spec.tt_manifest_path`` at
    a path with NO real gen file and assert materialize still succeeds, with
    the bundle stats.json mirroring the durable sidecar."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    # The gen file is irrelevant; clobber it with garbage to prove it is never
    # read at materialize.
    with open(tt_manifest_path, 'w') as f:
        f.write('this is not json :: gen manifest must never be parsed here')
    durable = _write_durable_stats(tt_manifest_path)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(os.path.join(result.bundle_dir, 'stats.json')) as f:
        written_stats = json.load(f)
    assert written_stats == durable


def test_materialize_derives_graph_index_id_and_path(tmp_path) -> None:
    """graph_index_id is logical ``{gk}/{gv}``; graph_index_path is
    the absolute on-disk snapshot dir resolved from the handle.  The
    trainer-side loader keys off ``graph_index_path`` — locking both
    fields here so a rename breaks fast at this layer."""
    service = _make_service(str(tmp_path))
    handle = _make_handle(graph_key='my_graph', graph_data_version='v42')
    tt_manifest_path = _setup_tt_manifest(tmp_path, graph_key='my_graph',
                                          graph_data_version='v42')
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(handle, spec)

    with open(os.path.join(result.bundle_dir,
                           'graph_runtime_metadata.json')) as f:
        grm = json.load(f)
    assert grm['graph_index_id'] == 'my_graph/v42'
    assert grm['graph_index_path'] == os.path.join(str(tmp_path), 'my_graph',
                                                   'v42')


def test_materialize_derives_stage_counts(tmp_path) -> None:
    """stage_counts in split_info.json come from the TT manifest's
    ``task_dataset.stages[<stage>].row_count`` (dropping the synthetic
    ``unused`` bucket)."""
    service = _make_service(str(tmp_path))
    counts = {'train': 500, 'val': 100, 'test': 100}
    tt_manifest_path = _setup_tt_manifest(tmp_path, stage_counts=counts)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(os.path.join(result.bundle_dir, 'split_info.json')) as f:
        si = json.load(f)
    assert si['stage_counts'] == counts


def test_materialize_stage_table_refs_from_task_dataset(tmp_path) -> None:
    """``stage_table_refs`` come verbatim from the wheel's task_dataset
    block — names + per-stage index_path are what ``execute_tt_plan``
    produced, not a kumo-side guess. The storage_kind switches from the
    legacy placeholder marker to ``'diskgraph_table'`` so downstream
    consumers know the refs point at real DG-ingested tables."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    spec = _make_spec(
        os.path.join(str(tmp_path), 'bundle'),
        train_table_name='TT_ORDERS',
        tt_manifest_path=tt_manifest_path,
    )

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(result.manifest_path) as f:
        manifest = json.load(f)

    refs = manifest['ingested_stage_tables']
    assert set(refs.keys()) == {'train', 'val', 'test'}
    for stage, ref in refs.items():
        assert ref['table_name'] == f'{stage}_seeds'
        assert ref['index_path'] == os.path.join(str(tmp_path), 'test_graph',
                                                 'v1', 'train_tables',
                                                 'tt-test')
        assert ref['storage_kind'] == 'diskgraph_table'


def test_materialize_runtime_manifest_points_at_wheel_stage_tables(
        tmp_path) -> None:
    """The flat ``runtime_manifest.json`` (consumed by the wheel iterator)
    must point each stage at the wheel-produced ``tt_extension_path`` +
    ``table_name``. This was the bug that motivated de-placeholdering:
    the placeholder wrote ``train_table__{stage}`` against a kumo-guessed
    index_path, which the wheel couldn't open."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(result.runtime_manifest_path) as f:
        rm = json.load(f)
    stages = rm['stages']
    assert set(stages.keys()) == {'train', 'val', 'test'}
    for stage, entry in stages.items():
        assert entry['table_name'] == f'{stage}_seeds'
        assert entry['index_path'] == os.path.join(str(tmp_path), 'test_graph',
                                                   'v1', 'train_tables',
                                                   'tt-test')


def test_materialize_reads_authoritative_ingest_manifest_not_gen(
        tmp_path) -> None:
    """INVARIANT: post-ingest, the durable ``diskgraph_manifest.json``
    (written by ``execute_tt_ingest``) is the SOLE task_dataset authority.
    The gen ``manifest.json`` is never read at materialize -- even when it
    carries a DIFFERENT (or garbage) ``task_dataset``, the ingest manifest
    must win.

    Setup: write a sibling ``diskgraph_manifest.json`` naming ``INGEST_*``
    tables, and clobber the gen ``manifest.json`` with disagreeing ``GEN_*``
    JSON. The materialized runtime manifest must reflect the INGEST names; if
    service code (incorrectly) consumed the gen manifest, the assertion below
    would see ``GEN_*``."""
    service = _make_service(str(tmp_path))

    # Durable ingest manifest (authoritative) with the REAL task_dataset.
    ingest_counts = {'train': 100, 'val': 20, 'test': 20}
    graph_index_path = os.path.join(str(tmp_path), 'test_graph', 'v1')
    ingest_extension_path = os.path.join(graph_index_path, 'train_tables',
                                         'tt-test')
    ingest_td = _default_task_dataset(ingest_counts,
                                      graph_index_path=graph_index_path,
                                      tt_extension_path=ingest_extension_path)
    for stage in ingest_td['stages']:
        ingest_td['stages'][stage]['table_name'] = f'INGEST_{stage}'
    gen_manifest_path = _setup_tt_manifest(tmp_path, stage_counts=ingest_counts,
                                           task_dataset=ingest_td)

    # Gen manifest (transient) WITH a disagreeing task_dataset block -- must
    # never be read, so it being inconsistent (or garbage) cannot affect us.
    gen_td = _default_task_dataset({
        'train': 1,
        'val': 1,
        'test': 1
    }, graph_index_path=graph_index_path,
                                   tt_extension_path=ingest_extension_path)
    for stage in gen_td['stages']:
        gen_td['stages'][stage]['table_name'] = f'GEN_{stage}'
    with open(gen_manifest_path, 'w') as f:
        json.dump({'version': 1, 'task_dataset': gen_td}, f)

    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=gen_manifest_path)
    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(result.runtime_manifest_path) as f:
        rm = json.load(f)
    for stage, entry in rm['stages'].items():
        assert entry['table_name'] == f'INGEST_{stage}', (
            'service consumed the transient gen manifest instead of the '
            'authoritative diskgraph_manifest.json')
        assert entry['index_path'] == ingest_extension_path

    with open(os.path.join(result.bundle_dir, 'split_info.json')) as f:
        si = json.load(f)
    assert si['stage_counts'] == ingest_counts


def test_materialize_rejects_ingest_manifest_for_other_graph(tmp_path) -> None:
    """A published extension is immutable for one resolved snapshot only."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    dg_manifest_path = os.path.join(os.path.dirname(tt_manifest_path),
                                    'diskgraph_manifest.json')
    with open(dg_manifest_path) as f:
        dg_manifest = json.load(f)
    dg_manifest['graph_index_path'] = os.path.join(str(tmp_path), 'other', 'v2')
    with open(dg_manifest_path, 'w') as f:
        json.dump(dg_manifest, f)

    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)
    with pytest.raises(RuntimeError, match='graph_index_path'):
        service.materialize_task_dataset(_make_handle(), spec)


def test_materialize_rejects_stage_from_other_extension(tmp_path) -> None:
    """Each stage must be under the same published extension as the manifest."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    dg_manifest_path = os.path.join(os.path.dirname(tt_manifest_path),
                                    'diskgraph_manifest.json')
    with open(dg_manifest_path) as f:
        dg_manifest = json.load(f)
    dg_manifest['task_dataset']['stages']['train']['index_path'] = (
        os.path.join(str(tmp_path), 'test_graph', 'v1', 'train_tables',
                     'wrong-extension'))
    with open(dg_manifest_path, 'w') as f:
        json.dump(dg_manifest, f)

    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)
    with pytest.raises(RuntimeError, match='index_path'):
        service.materialize_task_dataset(_make_handle(), spec)


def _predict_ingest_result(tmp_path, *, row_count: int) -> dict:
    graph_index_path = os.path.join(str(tmp_path), 'test_graph', 'v1')
    tt_id = 'pred_job-abc-123'
    tt_extension_path = os.path.join(graph_index_path, 'train_tables', tt_id)
    return {
        'schema_version':
        'diskgraph.tt_execute_result.v1',
        'tt_id':
        tt_id,
        'graph_index_path':
        graph_index_path,
        'tt_extension_path':
        tt_extension_path,
        'diskgraph_manifest_path':
        os.path.join(tt_extension_path, 'diskgraph_manifest.json'),
        'task_dataset': {
            'schema_version': 'diskgraph.task_dataset.v1',
            'graph_index_path': graph_index_path,
            'tt_extension_path': tt_extension_path,
            'stages': {
                'predict': {
                    'table_name': 'predict_seeds',
                    'index_path': tt_extension_path,
                    'row_count': row_count,
                },
            },
        },
    }


def test_materialize_prediction_staging_dir_ingest_only_no_second_gen(
        tmp_path, monkeypatch) -> None:
    """One-gen predict (LP): with ``predict_staging_dir`` set, materialize ONLY
    runs ``execute_tt_ingest`` on the KEPT staging dir produced by the generator
    activity — it must NOT call ``execute_tt_gen`` again. The bundle's predict
    stage points at the wheel-returned ``tt_extension_path``."""
    service = _make_service(str(tmp_path))
    spec = _make_spec(os.path.join(str(tmp_path),
                                   'bundle'), task_family='link_prediction',
                      task_type='static_link_prediction', rhs_node_type='ITEMS')

    staging_dir = '/dg_index/train_tables/.pred_job-abc-123.staging'
    calls = {}

    def _fake_gen(plan_json, index_path, tt_id):
        # Must NOT be reached on the ingest-only (one-gen) path.
        calls['gen'] = (plan_json, index_path, tt_id)
        raise AssertionError(
            'execute_tt_gen must not run in materialize on the one-gen path')

    def _fake_ingest(*, staging_dir, index_path, tt_id, predict_stage_name):
        calls['ingest'] = dict(staging_dir=staging_dir, index_path=index_path,
                               tt_id=tt_id,
                               predict_stage_name=predict_stage_name)
        return _predict_ingest_result(tmp_path, row_count=7)

    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(execute_tt_gen=_fake_gen,
                        execute_tt_ingest=_fake_ingest))

    result = service.materialize_prediction_task_dataset(
        _make_handle(),
        spec,
        predict_staging_dir=staging_dir,
        predict_seed_time_ns=9_000_000_000,
    )

    # No second gen; ingest ran on the KEPT staging dir with the predict stage.
    assert 'gen' not in calls
    assert calls['ingest']['staging_dir'] == staging_dir
    assert calls['ingest']['tt_id'] == 'pred_job-abc-123'
    assert calls['ingest']['predict_stage_name'] == 'predict'

    # Bundle's predict stage points at the wheel extension path (NOT a kumo
    # pred_tables guess); #29140 contract preserved.
    with open(result.runtime_manifest_path) as f:
        rm = json.load(f)
    assert rm['stages']['predict']['index_path'] == os.path.join(
        str(tmp_path), 'test_graph', 'v1', 'train_tables', 'pred_job-abc-123')
    assert rm['stages']['predict']['table_name'] == 'predict_seeds'
    with open(result.manifest_path) as f:
        manifest = json.load(f)
    assert 'predict' in manifest['ingested_stage_tables']

    # The prediction anchor is emitted into split_info.json's stage_seed_times
    # under the 'predict' stage so the datamodule's predict_time resolves to a
    # non-None RHS NodeRange seed time.
    bundle_dir = os.path.dirname(result.manifest_path)
    with open(os.path.join(bundle_dir, 'split_info.json')) as f:
        split_info = json.load(f)
    assert split_info['stage_seed_times']['predict'] == [
        9_000_000_000, 9_000_000_000
    ]


def test_materialize_prediction_no_anchor_omits_seed_times(
        tmp_path, monkeypatch) -> None:
    """Without ``predict_seed_time_ns`` (node prediction / non-LP) the bundle
    carries no predict seed-time range — byte-identical to before the fix."""
    service = _make_service(str(tmp_path))
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      task_family='node_prediction')

    def _fake_ingest(*, staging_dir, index_path, tt_id, predict_stage_name):
        return _predict_ingest_result(tmp_path, row_count=3)

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))

    result = service.materialize_prediction_task_dataset(
        _make_handle(), spec,
        predict_staging_dir='/dg_index/train_tables/.pred_job-abc-123.staging')

    bundle_dir = os.path.dirname(result.manifest_path)
    with open(os.path.join(bundle_dir, 'split_info.json')) as f:
        split_info = json.load(f)
    assert 'stage_seed_times' not in split_info


def test_materialize_prediction_rejects_ingest_for_other_graph(
        tmp_path, monkeypatch) -> None:
    """Prediction must reject an ingest result bound to another snapshot."""
    service = _make_service(str(tmp_path))
    spec = _make_spec(os.path.join(str(tmp_path),
                                   'bundle'), task_family='link_prediction',
                      task_type='static_link_prediction')
    bad_result = _predict_ingest_result(tmp_path, row_count=3)
    bad_result['graph_index_path'] = os.path.join(str(tmp_path), 'other', 'v2')

    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(execute_tt_ingest=lambda **_: bad_result))

    with pytest.raises(RuntimeError, match='graph_index_path'):
        service.materialize_prediction_task_dataset(
            _make_handle(),
            spec,
            predict_staging_dir='/kept-staging',
        )


def test_materialize_raises_when_task_dataset_missing(tmp_path) -> None:
    """When the durable ``diskgraph_manifest.json`` is present but lacks the
    ``task_dataset`` block, materialize fails loudly. Silently falling back
    would produce a runtime manifest pointing at non-existent stage tables."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path, omit_task_dataset=True)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    with pytest.raises(RuntimeError, match='missing ``task_dataset``'):
        service.materialize_task_dataset(_make_handle(), spec)


def test_materialize_raises_when_ingest_manifest_missing(tmp_path) -> None:
    """When the durable ``diskgraph_manifest.json`` is absent, materialize
    raises an actionable "ingest manifest missing" error rather than falling
    back to the gen scratch (removed contract)."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path, omit_ingest_manifest=True)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    with pytest.raises(RuntimeError, match='ingest manifest missing'):
        service.materialize_task_dataset(_make_handle(), spec)


def test_materialize_raises_when_durable_stats_missing(tmp_path) -> None:
    """When the durable ``stats.json`` sidecar is absent (ingest manifest
    present), materialize raises an actionable "ingest stats sidecar missing"
    error -- the old gen-bundle fallback for parsed_stats is removed."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path, omit_durable_stats=True)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    with pytest.raises(RuntimeError, match='ingest stats sidecar missing'):
        service.materialize_task_dataset(_make_handle(), spec)


def test_materialize_writes_bundle_stats_sidecar(tmp_path) -> None:
    """``stats.json`` in the bundle is the durable parsed-stats payload from
    the ingest extension's ``stats.json`` sidecar, forwarded verbatim. The
    TT-NG trainer does not read this sidecar, but downstream kumo callers do."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    parsed_stats = _write_durable_stats(tt_manifest_path, num_targets=2)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(os.path.join(result.bundle_dir, 'stats.json')) as f:
        stats = json.load(f)
    assert stats == parsed_stats


def test_materialize_has_time_false_by_default(tmp_path) -> None:
    """Without time/weight column names in task_dataset.columns, both
    flags are False. has_time / has_weight derive from the wheel's
    task_dataset.columns (timestamp/weight non-null iff present)."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(os.path.join(result.bundle_dir, 'metadata.json')) as f:
        meta = json.load(f)
    assert meta['has_time'] is False
    assert meta['has_weight'] is False
    assert meta['train_table_columns']['time_column'] is None
    assert meta['train_table_columns']['weight_column'] is None


def test_materialize_has_time_from_task_dataset_columns(tmp_path) -> None:
    """When ``task_dataset.columns.timestamp`` / ``weight`` are non-null,
    has_time / has_weight are True. ``provided_fields`` is a legacy alias
    on the fixture that maps to these column names."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path,
                                          provided_fields=['time', 'weight'])
    spec = _make_spec(
        os.path.join(str(tmp_path), 'bundle'),
        is_temporal=True,
        tt_manifest_path=tt_manifest_path,
    )

    result = service.materialize_task_dataset(_make_handle(), spec)

    with open(os.path.join(result.bundle_dir, 'metadata.json')) as f:
        meta = json.load(f)
    assert meta['has_time'] is True
    assert meta['has_weight'] is True
    assert meta['train_table_columns']['time_column'] == 'TIMESTAMP'
    assert meta['train_table_columns']['weight_column'] == 'WEIGHT'


def test_materialize_idempotent(tmp_path) -> None:
    """Running materialize twice on the same artifact_root succeeds."""
    service = _make_service(str(tmp_path))
    tt_manifest_path = _setup_tt_manifest(tmp_path)
    spec = _make_spec(os.path.join(str(tmp_path), 'bundle'),
                      tt_manifest_path=tt_manifest_path)

    result1 = service.materialize_task_dataset(_make_handle(), spec)
    result2 = service.materialize_task_dataset(_make_handle(), spec)

    assert result1.manifest_path == result2.manifest_path
    assert os.path.isfile(result2.manifest_path)


def test_hosted_target_matches_portable(tmp_path) -> None:
    from kumo_core.automl.runtime import (
        apply_encoder_overrides,
        collect_resolved_encoders,
    )
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import (
        DiskgraphTaskManifest,
        materialize_task_dataset,
    )
    from kumoapi.encoder import EncoderType, MaxLogNumerical, Numerical
    from kumoapi.typing import ColStatType
    from pydantic import TypeAdapter

    tt_manifest_path = _setup_tt_manifest(tmp_path)
    extension = os.path.dirname(tt_manifest_path)
    stats = {
        'COUNT': 140,
        'NUM_NA': 0,
        'MEAN': 4.2,
        'MEDIAN': 4.2,
        'STD': 1.0,
        'MIN': 3.0,
        'MAX': 5.0,
        'KURTOSIS': 0.0,
        'QUANTILES': [3.5, 4.2, 4.8],
    }
    with open(os.path.join(extension, 'train_table_col_stats.json'), 'w') as f:
        json.dump(
            {
                'version': 1,
                'kind': 'tt_train_table_col_stats',
                'stats_format': 'kumo_col_stats_v1',
                'stats': {
                    f'TARGET1__kumo.numerical__{name}__False': {
                        'value': value
                    }
                    for name, value in stats.items()
                },
            }, f)
    node_types = {
        'USERS': {
            'feature_columns': [],
            'mapper_key': 'USERS',
            'time_column': None,
            'column_specs': [],
        },
        'TT_USERS': {
            'feature_columns': [],
            'mapper_key': 'TT_USERS',
            'time_column': None,
            'column_specs': [],
        },
    }
    # As hosted serializes it: only the default encoder's stats.
    encoder_config = TypeAdapter(EncoderType).dump_python(
        Numerical(), mode='json')
    encoder_config.pop('_stats', None)
    hosted_spec = _make_spec(
        os.path.join(str(tmp_path),
                     'hosted'), tt_manifest_path=tt_manifest_path,
        task_family='forecasting', task_type='forecasting', feature_schema={
            'node_types': node_types,
            'target_encoder_input': {
                'column_name': 'TARGET',
                'dtype': 'float',
                'stype': 'numerical',
                'encoder': encoder_config,
                'stats': {
                    'MEAN': 4.2
                },
            },
        })
    portable_spec = _make_spec(
        os.path.join(str(tmp_path),
                     'portable'), tt_manifest_path=tt_manifest_path,
        task_family='forecasting', task_type='forecasting', feature_schema={
            'node_types': node_types,
            'target_encoder_input': None,
        })

    hosted = _make_service(str(tmp_path)).materialize_task_dataset(
        _make_handle(), hosted_spec)
    portable = materialize_task_dataset(
        graph_index_id='test_graph/v1',
        graph_index_path=os.path.join(str(tmp_path), 'test_graph', 'v1'),
        spec=portable_spec,
    )

    hosted_manifest = DiskgraphTaskManifest.load(hosted.manifest_path)
    portable_manifest = DiskgraphTaskManifest.load(portable.manifest_path)
    assert (hosted_manifest.feature_schema['target_encoder_input'] ==
            portable_manifest.feature_schema['target_encoder_input'])
    task = ResolvedNumericalTask.from_manifest(hosted_manifest, name='forecast')
    encoders = collect_resolved_encoders(task)
    apply_encoder_overrides(
        task, encoders,
        {'train_table.TARGET': MaxLogNumerical(na_strategy='mean')})
    assert encoders.target.encoder._stats[ColStatType.MIN] == 3.0
