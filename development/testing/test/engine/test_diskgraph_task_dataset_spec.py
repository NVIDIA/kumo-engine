# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the core-owned task-dataset materialization specification.

TT-NG only.

Spec field set is locked against the design-doc contracts and the
wheel's manifest reader. Tests validate:
  * every spec field maps to a real manifest consumer
  * spec round-trips through JSON (Temporal activity boundary)
  * frozen + hashable invariants
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from kumo_core.encoders.config import TargetEncoderConfig
from kumo_core.task_dataset import TaskDatasetMaterializeSpec
from kumoapi.encoder import Index, Numerical
from kumoapi.typing import ColStatType

from io_utils.json import from_json_dict, to_json_dict
from kumo.diskgraph._diskgraph_legacy import build_feature_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(**overrides) -> TaskDatasetMaterializeSpec:
    """Build a valid spec with sensible defaults.  Override any field."""
    defaults = dict(
        tt_manifest_path='/tmp/tt/manifest.json',
        job_id='job-abc-123',
        artifact_root='/tmp/bundles/job-abc-123',
        task_family='node_prediction',
        task_type='binary_classification',
        is_temporal=False,
        seed_node_type='USERS',
        train_table_name='TT_USERS',
        entity_table_name='USERS',
        feature_schema={
            'node_types': {
                'USERS': {
                    'feature_columns': ['AGE', 'INCOME'],
                    'mapper_key':
                    'USERS',
                    'time_column':
                    None,
                    'column_specs': [{
                        'column_name': name,
                        'dtype': 'float',
                        'stype': 'numerical',
                        'encoder': {
                            'name': 'Numerical'
                        },
                        'stats': {
                            'MEAN': 0.0
                        },
                    } for name in ('AGE', 'INCOME')],
                },
            },
            'target_encoder_input': None,
        },
        graph_node_types=('USERS', ),
        graph_edge_types=(('USERS', 'FRIENDS', 'USERS'), ),
    )
    defaults.update(overrides)
    return TaskDatasetMaterializeSpec(**defaults)


# ---------------------------------------------------------------------------
# Drift guard: every spec field maps to a real consumer
# ---------------------------------------------------------------------------

# The expected field set — adding or removing fields from the spec
# should require updating this test, forcing an explicit decision.
_EXPECTED_FIELDS = frozenset({
    'tt_manifest_path',
    'job_id',
    'artifact_root',
    'task_family',
    'task_type',
    'is_temporal',
    'seed_node_type',
    'train_table_name',
    'entity_table_name',
    'feature_schema',
    'graph_node_types',
    'graph_edge_types',
    'rhs_node_type',
    # Fail closed when a temporal split leaves a required dataset stage empty.
    'required_stages',
    # The query's target->entity paths. Recorded rather than derived: the
    # graph edge list cannot reproduce them, because for link prediction the
    # path runs from the fact table to the seed entity, which is neither the
    # seed nor the RHS node type the spec already names.
    'target_to_entity_paths',
})


def test_spec_field_set_matches_locked_contract() -> None:
    """Drift guard: spec fields must match the locked contract above.

    If you need to add or remove a field, update ``_EXPECTED_FIELDS``
    and document why in the commit message.
    """
    spec = _make_spec()
    actual = frozenset(spec.__dataclass_fields__.keys())
    assert actual == _EXPECTED_FIELDS, (
        f'Spec fields drifted from locked contract.\n'
        f'  Added:   {actual - _EXPECTED_FIELDS}\n'
        f'  Removed: {_EXPECTED_FIELDS - actual}')


def test_spec_does_not_contain_derivable_fields() -> None:
    """Fields derivable from TT manifest or handle must NOT be on the spec."""
    spec = _make_spec()
    derivable = {
        'has_time',
        'has_weight',
        'graph_index_id',
        'seed_id_column',
        'target_column',
        'raw_feature_format',
        'provenance_mode',
    }
    actual = set(spec.__dataclass_fields__.keys())
    leaked = actual & derivable
    assert not leaked, (f'Derivable fields should not be on the spec: {leaked}')


# ---------------------------------------------------------------------------
# JSON round-trip (Temporal activity boundary)
# ---------------------------------------------------------------------------


def test_spec_round_trips_through_json() -> None:
    """Spec must JSON-serialize/deserialize cleanly since Temporal
    activities pass it as an argument."""
    spec = _make_spec()
    payload = to_json_dict(spec)
    restored = from_json_dict(payload, TaskDatasetMaterializeSpec)
    assert restored == spec


def test_spec_round_trips_with_temporal_task() -> None:
    """Temporal binary classification round-trips correctly."""
    spec = _make_spec(is_temporal=True)
    payload = to_json_dict(spec)
    restored = from_json_dict(payload, TaskDatasetMaterializeSpec)
    assert restored.is_temporal is True
    assert restored == spec


def test_spec_round_trips_with_complex_feature_schema() -> None:
    """Feature schema with multiple node types round-trips."""
    schema = {
        'node_types': {
            table: {
                'feature_columns':
                columns,
                'mapper_key':
                table,
                'time_column': ('TIMESTAMP' if table == 'USERS' else None),
                'column_specs': [{
                    'column_name': name,
                    'dtype': 'float',
                    'stype': 'numerical',
                    'encoder': {
                        'name': 'Numerical'
                    },
                    'stats': {
                        'MEAN': 0.0
                    },
                } for name in columns],
            }
            for table, columns in {
                'USERS': ['AGE', 'INCOME'],
                'PRODUCTS': ['PRICE', 'CATEGORY'],
            }.items()
        },
        'target_encoder_input': None,
    }
    spec = _make_spec(
        feature_schema=schema,
        graph_node_types=('USERS', 'PRODUCTS'),
        graph_edge_types=(('USERS', 'BOUGHT', 'PRODUCTS'), ),
    )
    payload = to_json_dict(spec)
    restored = from_json_dict(payload, TaskDatasetMaterializeSpec)
    assert restored.feature_schema == schema


# ---------------------------------------------------------------------------
# Frozen invariant
# ---------------------------------------------------------------------------


def test_spec_is_frozen() -> None:
    spec = _make_spec()
    with pytest.raises(AttributeError):
        spec.job_id = 'different'  # type: ignore[misc]


def test_monolith_builder_emits_exact_core_feature_schema() -> None:
    age = SimpleNamespace(dtype='int64', stype='numerical')
    target = SimpleNamespace(
        name='TARGET',
        dtype='bool',
        stype='categorical',
    )
    table = SimpleNamespace(
        table_name='USERS',
        name='USERS',
        time_col_name=None,
        has_time_col=lambda: False,
        col=lambda name: {'AGE': age}[name],
    )
    train_table = SimpleNamespace(
        table_name='TRAIN_TABLE',
        target_col=target,
        col=lambda name: None,
    )
    age_encoder = Numerical()
    age_encoder._stats = {ColStatType.MEAN: 42.0}
    target_encoder = Index()
    target_encoder._stats = {
        ColStatType.CATEGORY_COUNTS: [[False, 8], [True, 2]]
    }
    fitted = SimpleNamespace(
        encoders={'USERS': {
            'AGE': age_encoder
        }},
        target=TargetEncoderConfig('TARGET', target_encoder),
    )
    task = SimpleNamespace(
        task_type='binary_classification',
        graph=SimpleNamespace(tables=[table]),
        train_table=train_table,
        exclude_cols={},
        exclude_cols_dict={},
    )

    schema = build_feature_schema(task, fitted)

    assert set(schema) == {'node_types', 'target_encoder_input'}
    users = schema['node_types']['USERS']
    assert users['feature_columns'] == ['AGE']
    assert users['column_specs'][0]['stats'] == {'MEAN': 42.0}
    target_input = schema['target_encoder_input']
    assert target_input['column_name'] == 'TARGET'
    assert target_input['dtype'] == 'bool'
    assert target_input['stype'] == 'categorical'
    assert target_input['encoder']['name'] == 'Index'
    assert target_input['stats'] == {
        'CATEGORY_COUNTS': [[False, 8], [True, 2]],
    }
