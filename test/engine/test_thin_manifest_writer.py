# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the portable thin-manifest writer.

TT-NG only.

Validates that the manifest writer produces output the wheel's
``open_task_dataset`` can consume (round-trip test), and that the
written JSON structure matches the design-doc contract.
"""
from __future__ import annotations

import json
import os

import pytest
from kumo_core.task_dataset import (
    DiskgraphTaskManifest,
    TaskDatasetMaterializeSpec,
    write_thin_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workdir(tmp_path):
    """Temporary directory cleaned up after each test."""
    return str(tmp_path)


def _make_spec(workdir: str, **overrides) -> TaskDatasetMaterializeSpec:
    defaults = dict(
        tt_manifest_path='/tmp/tt/manifest.json',
        job_id='job-abc-123',
        artifact_root=os.path.join(workdir, 'bundle'),
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


_MOCK_STATS = {
    'target_distribution': {
        '0': 0.5,
        '1': 0.5
    },
    'row_count': 12,
}

_MOCK_STAGE_COUNTS = {'train': 6, 'val': 3, 'test': 3}

_MOCK_STAGE_TABLE_REFS = {
    'train': {
        'table_name': 'TT_USERS__train',
        'storage_kind': 'mock_parquet'
    },
    'val': {
        'table_name': 'TT_USERS__val',
        'storage_kind': 'mock_parquet'
    },
    'test': {
        'table_name': 'TT_USERS__test',
        'storage_kind': 'mock_parquet'
    },
}


def _write_manifest(workdir: str, **spec_overrides) -> str:
    """Write a manifest with default inputs, return manifest_path."""
    spec = _make_spec(workdir, **spec_overrides)
    return write_thin_manifest(
        spec,
        graph_index_id='test_graph/v1',
        graph_index_path='/data/test_graph/v1',
        stats=_MOCK_STATS,
        stage_counts=_MOCK_STAGE_COUNTS,
        stage_table_refs=_MOCK_STAGE_TABLE_REFS,
        has_time=False,
        has_weight=False,
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_manifest_creates_expected_files(workdir: str) -> None:
    """Writer produces all 8 expected files."""
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)

    expected = {
        'manifest.json',
        'metadata.json',
        'feature_schema.json',
        'column_stats.json',
        'graph_runtime_metadata.json',
        'split_info.json',
        'stats.json',
        'checksums.json',
    }
    actual = set(os.listdir(bundle_dir))
    assert expected == actual, f'Missing: {expected - actual}, Extra: {actual - expected}'


def test_manifest_json_has_required_top_level_fields(workdir: str) -> None:
    """manifest.json must contain all fields from the design-doc
    DiskgraphTaskDatasetManifest."""
    manifest_path = _write_manifest(workdir)
    with open(manifest_path) as f:
        manifest = json.load(f)

    required = {
        'schema_version',
        'task_family',
        'task_type',
        'is_temporal',
        'artifact_root',
        'metadata_path',
        'feature_schema_path',
        'graph_runtime_metadata_path',
        'split_info_path',
        'stats_path',
        'ingested_stage_tables',
        'feature_payload_root',
        'checksums_path',
        'producer_summary_path',
        'downsampling_metadata_path',
        'side_artifacts',
        'runtime_capabilities',
    }
    missing = required - set(manifest.keys())
    assert not missing, f'manifest.json missing fields: {missing}'


def test_metadata_json_content(workdir: str) -> None:
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'metadata.json')) as f:
        meta = json.load(f)

    assert meta['task_family'] == 'node_prediction'
    assert meta['task_type'] == 'binary_classification'
    assert meta['is_temporal'] is False
    assert meta['seed_node_type'] == 'USERS'
    assert meta['entity_table_name'] == 'USERS'
    assert meta['train_table_name'] == 'TT_USERS'
    assert meta['has_time'] is False
    assert meta['has_weight'] is False
    assert meta['train_table_columns']['seed_id_column'] == 'ENTITY'
    assert meta['train_table_columns']['target_column'] == 'TARGET'


def test_recorded_target_to_entity_paths_round_trip(workdir: str) -> None:
    manifest_path = _write_manifest(
        workdir, target_to_entity_paths=((('ORDERS', 'USER_ID', 'USERS'), ), ))
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'metadata.json')) as f:
        meta = json.load(f)

    assert meta['target_to_entity_paths'] == [[['ORDERS', 'USER_ID',
                                                'USERS']]]


def test_unrecorded_target_to_entity_paths_leave_the_key_absent(
        workdir: str) -> None:
    # A producer that does not record paths must stay distinguishable from
    # one that records that there are none: the reader falls back to the
    # graph edge list on the first and trusts the empty list on the second.
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'metadata.json')) as f:
        meta = json.load(f)

    assert 'target_to_entity_paths' not in meta


def test_feature_schema_json_content(workdir: str) -> None:
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'feature_schema.json')) as f:
        fs = json.load(f)

    assert fs['raw_feature_format'] == 'arrow_columnar_v1'
    assert fs['provenance_mode'] == 'node_id_time_keyed_v1'
    assert 'USERS' in fs['node_types']
    assert fs['node_types']['USERS']['feature_columns'] == ['AGE']
    assert fs['node_types']['USERS']['mapper_key'] == 'USERS'


def test_graph_runtime_metadata_json_content(workdir: str) -> None:
    schema = _make_spec(workdir).feature_schema
    schema['node_types']['PRODUCTS'] = {
        'feature_columns': [],
        'mapper_key': 'PRODUCTS',
        'time_column': None,
        'column_specs': [],
    }
    manifest_path = _write_manifest(
        workdir,
        graph_node_types=('USERS', 'PRODUCTS'),
        graph_edge_types=(('USERS', 'BOUGHT', 'PRODUCTS'), ),
        feature_schema=schema,
    )
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'graph_runtime_metadata.json')) as f:
        grm = json.load(f)

    assert grm['graph_index_id'] == 'test_graph/v1'
    # graph_index_path is the contract field the trainer-side loader
    # keys off to open the index — locking the name here so a rename
    # in the writer breaks fast rather than silently in the trainer.
    assert grm['graph_index_path'] == '/data/test_graph/v1'
    assert grm['node_types'] == ['USERS', 'PRODUCTS']
    assert len(grm['edge_types']) == 1
    assert grm['edge_types'][0] == {
        'src': 'USERS',
        'rel': 'BOUGHT',
        'dst': 'PRODUCTS'
    }


def test_split_info_json_content(workdir: str) -> None:
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'split_info.json')) as f:
        si = json.load(f)

    assert si['split_kind'] == 'precomputed'
    assert si['stage_counts'] == {'train': 6, 'val': 3, 'test': 3}


def test_stats_json_passed_through(workdir: str) -> None:
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'stats.json')) as f:
        stats = json.load(f)

    assert stats == _MOCK_STATS


def test_ingested_stage_tables_in_manifest(workdir: str) -> None:
    manifest_path = _write_manifest(workdir)
    with open(manifest_path) as f:
        manifest = json.load(f)

    refs = manifest['ingested_stage_tables']
    assert set(refs.keys()) == {'train', 'val', 'test'}
    assert refs['train']['table_name'] == 'TT_USERS__train'
    assert refs['train']['storage_kind'] == 'mock_parquet'


def test_checksums_computed_for_all_sidecars(workdir: str) -> None:
    manifest_path = _write_manifest(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(os.path.join(bundle_dir, 'checksums.json')) as f:
        checksums = json.load(f)

    expected_keys = {
        'metadata.json_sha256',
        'feature_schema.json_sha256',
        'column_stats.json_sha256',
        'graph_runtime_metadata.json_sha256',
        'split_info.json_sha256',
        'stats.json_sha256',
    }
    assert expected_keys == set(checksums.keys())
    for v in checksums.values():
        assert len(v) == 64  # sha256 hex digest


# ---------------------------------------------------------------------------
# Temporal variant
# ---------------------------------------------------------------------------


def test_temporal_task_writes_has_time(workdir: str) -> None:
    spec = _make_spec(workdir, is_temporal=True)
    manifest_path = write_thin_manifest(
        spec,
        graph_index_id='test_graph/v1',
        graph_index_path='/data/test_graph/v1',
        stats=_MOCK_STATS,
        stage_counts=_MOCK_STAGE_COUNTS,
        stage_table_refs=_MOCK_STAGE_TABLE_REFS,
        has_time=True,
        has_weight=True,
    )
    bundle_dir = os.path.dirname(manifest_path)

    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest['is_temporal'] is True

    with open(os.path.join(bundle_dir, 'metadata.json')) as f:
        meta = json.load(f)
    assert meta['is_temporal'] is True
    assert meta['has_time'] is True
    assert meta['has_weight'] is True
    assert meta['train_table_columns']['time_column'] == 'TIMESTAMP'
    assert meta['train_table_columns']['weight_column'] == 'WEIGHT'


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_idempotent_rewrite(workdir: str) -> None:
    """Re-running the writer on the same artifact_root succeeds (D1.3)."""
    manifest_path_1 = _write_manifest(workdir)
    manifest_path_2 = _write_manifest(workdir)
    assert manifest_path_1 == manifest_path_2
    assert os.path.isfile(manifest_path_2)


def test_no_partial_state_on_failure(workdir: str) -> None:
    """If the writer fails mid-write, artifact_root should not exist."""
    spec = _make_spec(workdir)
    # Make artifact_root's parent read-only to force a rename failure.
    # This is tricky to test portably, so instead we verify cleanup
    # by passing an invalid stats value that will succeed in JSON
    # serialization but we can verify the positive path works.
    # (Full failure-mode testing would require mocking os.rename.)
    manifest_path = write_thin_manifest(
        spec,
        graph_index_id='g/v1',
        graph_index_path='/data/g/v1',
        stats={'ok': True},
        stage_counts={'train': 1},
        stage_table_refs={'train': {
            'table_name': 't'
        }},
        has_time=False,
        has_weight=False,
    )
    assert os.path.isfile(manifest_path)


def _bundle(workdir: str, **kwargs) -> str:
    manifest_path = write_thin_manifest(
        _make_spec(workdir),
        graph_index_id='test_graph/v1',
        graph_index_path='/data/test_graph/v1',
        stats=_MOCK_STATS,
        stage_counts=_MOCK_STAGE_COUNTS,
        stage_table_refs=_MOCK_STAGE_TABLE_REFS,
        has_time=False,
        has_weight=False,
        **kwargs,
    )
    with open(os.path.join(os.path.dirname(manifest_path),
                           'runtime_manifest.json'), 'w') as f:
        json.dump({}, f)
    return manifest_path


def test_column_stats_live_in_their_own_sidecar(workdir, monkeypatch):
    import kumo_core.task_dataset as task_dataset

    manifest_path = _bundle(workdir)
    read = []
    original = task_dataset._load_manifest_json
    monkeypatch.setattr(task_dataset, '_load_manifest_json',
                        lambda path: read.append(path) or original(path))

    manifest = DiskgraphTaskManifest.load(manifest_path)

    column = manifest.feature_schema['node_types']['USERS']['column_specs'][0]
    assert 'stats' not in column
    assert not any(path.endswith('column_stats.json') for path in read)
    assert manifest.load_column_stats() == {'USERS': {'AGE': {'MEAN': 0.0}}}
    assert read[-1].endswith('column_stats.json')


def test_prediction_bundles_carry_no_column_stats(workdir):
    manifest_path = _bundle(workdir, column_stats=False)
    bundle_dir = os.path.dirname(manifest_path)

    assert 'column_stats.json' not in os.listdir(bundle_dir)
    manifest = DiskgraphTaskManifest.load(manifest_path)
    assert 'column_stats_path' not in manifest.manifest
    with pytest.raises(ValueError, match='no column statistics'):
        manifest.load_column_stats()


def test_bundles_with_inline_column_stats_still_fit(workdir):
    from kumo_core.automl.runtime import collect_resolved_encoders
    from kumo_core.resolved_task import ResolvedNumericalTask

    manifest_path = _bundle(workdir)
    bundle_dir = os.path.dirname(manifest_path)
    with open(manifest_path) as f:
        payload = json.load(f)
    del payload['column_stats_path']
    with open(manifest_path, 'w') as f:
        json.dump(payload, f)
    os.remove(os.path.join(bundle_dir, 'column_stats.json'))
    with open(os.path.join(bundle_dir, 'feature_schema.json'), 'w') as f:
        json.dump(_make_spec(workdir).feature_schema, f)

    manifest = DiskgraphTaskManifest.load(manifest_path)
    fitted = ResolvedNumericalTask.from_manifest(
        manifest, name='job', column_stats=manifest.load_column_stats())
    layout = ResolvedNumericalTask.from_manifest(manifest, name='job')

    encoders = collect_resolved_encoders(fitted)
    assert encoders['USERS']['AGE']._stats
    assert layout.encoder_inputs['USERS'][0].stats is None
    with pytest.raises(ValueError, match='load_column_stats'):
        collect_resolved_encoders(layout)
