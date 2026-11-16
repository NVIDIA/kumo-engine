# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the portable flat-manifest writer.

TT-NG only.

Validates that the flat runtime manifest matches the wheel iterator's
expected shape (PR #251): single file, ``kind: diskgraph_task_dataset``,
raw TT-gen column names (``ENTITY1``/``TARGET1``), per-stage
``table_name``/``index_path``/``row_count``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from kumo_core.task_dataset import (
    TaskDatasetMaterializeSpec,
    write_flat_manifest,
)


@pytest.fixture()
def bundle_dir(tmp_path):
    d = tmp_path / 'bundle'
    d.mkdir()
    return str(d)


def _make_spec(**overrides) -> TaskDatasetMaterializeSpec:
    defaults = dict(
        tt_manifest_path='/idx/train_tables/tt_abc/manifest.json',
        job_id='job-abc-123',
        artifact_root='/bundles/job-abc-123',
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


_STAGE_COUNTS = {'train': 6, 'val': 3, 'test': 3}
_STAGE_TABLE_REFS = {
    'train': {
        'table_name': 'TT_USERS__train'
    },
    'val': {
        'table_name': 'TT_USERS__val'
    },
    'test': {
        'table_name': 'TT_USERS__test'
    },
}


def _write(bundle_dir: str, **overrides) -> str:
    spec = _make_spec(**overrides.pop('spec_overrides', {}))
    return write_flat_manifest(
        spec,
        bundle_dir=bundle_dir,
        graph_index_path=overrides.pop('graph_index_path', '/idx'),
        stage_index_path=overrides.pop('stage_index_path',
                                       '/idx/train_tables/tt_abc'),
        stage_counts=overrides.pop('stage_counts', _STAGE_COUNTS),
        stage_table_refs=overrides.pop('stage_table_refs', _STAGE_TABLE_REFS),
        has_time=overrides.pop('has_time', False),
        has_weight=overrides.pop('has_weight', False),
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_writes_runtime_manifest_at_expected_path(bundle_dir: str) -> None:
    path = _write(bundle_dir)
    assert path == os.path.join(bundle_dir, 'runtime_manifest.json')
    assert os.path.isfile(path)


def test_no_stray_staging_files_left_behind(bundle_dir: str) -> None:
    """Atomic write must clean up its temp file."""
    _write(bundle_dir)
    entries = os.listdir(bundle_dir)
    assert entries == ['runtime_manifest.json'
                       ], (f'unexpected files in bundle dir: {entries}')


def test_top_level_shape(bundle_dir: str) -> None:
    path = _write(bundle_dir)
    with open(path) as f:
        m = json.load(f)
    required = {
        'schema_version',
        'kind',
        'task',
        'table',
        'graph',
        'stages',
    }
    assert required <= set(m.keys()), f'missing: {required - set(m.keys())}'
    assert m['kind'] == 'diskgraph_task_dataset'


# ---------------------------------------------------------------------------
# Field contents
# ---------------------------------------------------------------------------


def test_task_and_table_fields(bundle_dir: str) -> None:
    path = _write(bundle_dir)
    with open(path) as f:
        m = json.load(f)

    assert m['task']['task_type'] == 'binary_classification'
    assert m['task']['is_temporal'] is False
    assert m['table']['train_table_name'] == 'TT_USERS'
    assert m['table']['seed_entity_table'] == 'USERS'


def test_columns_use_raw_tt_gen_names_not_bridge_names(bundle_dir: str) -> None:
    """Wheel iterator reads per-stage parquets pre-bridge; columns
    are TT-gen's raw ``ENTITY1`` / ``TARGET1``, not the sidecar's
    post-bridge ``ENTITY`` / ``TARGET``.
    """
    path = _write(bundle_dir)
    with open(path) as f:
        m = json.load(f)
    cols = m['table']['columns']
    assert cols['primary_entity'] == 'ENTITY1'
    assert cols['primary_target'] == 'TARGET1'
    assert cols['timestamp'] is None  # has_time=False
    assert cols['weight'] is None  # has_weight=False


def test_has_time_and_has_weight_set_columns(bundle_dir: str) -> None:
    path = _write(bundle_dir, has_time=True, has_weight=True)
    with open(path) as f:
        m = json.load(f)
    assert m['table']['columns']['timestamp'] == 'TIMESTAMP'
    assert m['table']['columns']['weight'] == 'WEIGHT'


def test_graph_block(bundle_dir: str) -> None:
    schema = _make_spec().feature_schema
    schema['node_types']['PRODUCTS'] = {
        'feature_columns': [],
        'mapper_key': 'PRODUCTS',
        'time_column': None,
        'column_specs': [],
    }
    path = _write(
        bundle_dir,
        spec_overrides=dict(
            graph_node_types=('USERS', 'PRODUCTS'),
            graph_edge_types=(('USERS', 'BOUGHT', 'PRODUCTS'), ),
            feature_schema=schema,
        ),
        graph_index_path='/data/g/v1',
    )
    with open(path) as f:
        m = json.load(f)

    assert m['graph']['index_path'] == '/data/g/v1'
    assert m['graph']['node_types'] == ['USERS', 'PRODUCTS']
    # Edge types are serialized as 3-element lists (JSON-native tuple form),
    # not as {src, rel, dst} dicts — those are sidecar-shape.
    assert m['graph']['edge_types'] == [['USERS', 'BOUGHT', 'PRODUCTS']]


def test_stages_dict_per_stage(bundle_dir: str) -> None:
    path = _write(bundle_dir, stage_index_path='/idx/train_tables/tt_abc')
    with open(path) as f:
        m = json.load(f)

    assert set(m['stages'].keys()) == {'train', 'val', 'test'}
    train = m['stages']['train']
    assert train['table_name'] == 'TT_USERS__train'
    assert train['index_path'] == '/idx/train_tables/tt_abc'
    assert train['row_count'] == 6
    # All stages share the same TT-extension index for v1.
    assert m['stages']['val']['index_path'] == '/idx/train_tables/tt_abc'
    assert m['stages']['test']['index_path'] == '/idx/train_tables/tt_abc'


def test_runtime_manifest_omits_encoder_statistics(bundle_dir: str) -> None:
    before = Path(_write(bundle_dir)).read_bytes()
    schema = _make_spec().feature_schema
    schema['node_types']['USERS']['column_specs'][0]['stats'][
        'CATEGORY_COUNTS'] = [[str(i), i] for i in range(10_000)]
    path = _write(bundle_dir, spec_overrides={'feature_schema': schema})
    after = Path(path).read_bytes()

    assert before == after
    assert 'features' not in json.loads(after)


# ---------------------------------------------------------------------------
# Atomicity / re-run
# ---------------------------------------------------------------------------


def test_overwrites_on_rewrite(bundle_dir: str) -> None:
    path_1 = _write(bundle_dir)
    path_2 = _write(bundle_dir, stage_counts={'train': 99})
    assert path_1 == path_2
    with open(path_2) as f:
        m = json.load(f)
    assert m['stages']['train']['row_count'] == 99
    assert 'val' not in m['stages']  # second write replaces, not merges
