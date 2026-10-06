# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the TT-NG batch-prediction inference path.

Verifies the predict task-dataset manifest (a single ``predict`` stage) is
well-formed and consumable by both the wheel iterator's manifest parser and the
trainer's manifest loader. The full serve loop needs a real DiskGraph index +
trained model, so it is covered by an integration run rather than here.
"""
import json
import os

try:
    # Ships only in the tt-ng wheel; CI on master installs the non-NG prod
    # wheel, so the assertion that uses it is guarded below.
    from diskgraph import task_stage_size
except ImportError:
    task_stage_size = None


def test_predict_manifest_predict_stage_is_consumable(tmp_path):
    """A one-stage (``predict``) bundle — as written by
    ``materialize_prediction_task_dataset`` — must parse as a
    ``diskgraph_task_dataset`` for the wheel iterator and load via the
    trainer's manifest reader."""
    from kumo_core.task_dataset import (
        TaskDatasetMaterializeSpec,
        write_flat_manifest,
        write_thin_manifest,
    )

    index_path = str(tmp_path / 'index')
    ext_root = os.path.join(index_path, 'train_tables', 'pred-bp-job-1')
    spec = TaskDatasetMaterializeSpec(
        tt_manifest_path='',  # unused on the BP path
        job_id='bp-job-1',
        artifact_root=os.path.join(index_path, 'task_datasets', 'bp-job-1'),
        task_family='node_prediction',
        task_type='binary_classification',
        is_temporal=False,
        seed_node_type='users',
        train_table_name='tt_train',
        entity_table_name='users',
        feature_schema={
            'node_types': {
                'users': {
                    'feature_columns': ['age'],
                    'mapper_key':
                    'users',
                    'time_column':
                    None,
                    'column_specs': [{
                        'column_name': 'age',
                        'dtype': 'int64',
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
        graph_node_types=('users', ),
        graph_edge_types=(),
    )
    stage_counts = {'predict': 5}
    stage_table_refs = {
        'predict': {
            'table_name': 'predict',
            'index_path': ext_root,
            'storage_kind': 'diskgraph_table',
        }
    }

    manifest_path = write_thin_manifest(
        spec,
        graph_index_id='g/1',
        graph_index_path=index_path,
        stats={},
        stage_counts=stage_counts,
        stage_table_refs=stage_table_refs,
        has_time=True,
        has_weight=False,
    )
    bundle_dir = os.path.dirname(manifest_path)
    runtime_manifest_path = write_flat_manifest(
        spec,
        bundle_dir=bundle_dir,
        graph_index_path=index_path,
        stage_index_path=ext_root,
        stage_counts=stage_counts,
        stage_table_refs=stage_table_refs,
        has_time=True,
        has_weight=False,
    )

    # Flat (runtime) manifest: a diskgraph_task_dataset with a predict stage,
    # raw TT-gen column names on the wheel side.
    with open(runtime_manifest_path) as f:
        flat = json.load(f)
    assert flat['kind'] == 'diskgraph_task_dataset'
    assert flat['stages']['predict']['table_name'] == 'predict'
    assert flat['stages']['predict']['index_path'] == ext_root
    assert flat['stages']['predict']['row_count'] == 5
    assert flat['table']['columns']['primary_entity'] == 'ENTITY1'

    # The wheel iterator's manifest parser accepts the predict stage (tt-ng
    # wheel only).
    if task_stage_size is not None:
        assert task_stage_size(runtime_manifest_path, 'predict') == 5

    # The trainer's manifest loader reads the predict metadata.
    import kumo_core.diskgraph.lightning_data as diskgraph_lightning_data

    tm = diskgraph_lightning_data.DiskgraphTaskManifest.load(manifest_path)
    assert tm.metadata['train_table_name'] == 'tt_train'
    assert tm.metadata['entity_table_name'] == 'users'
    assert tm.stage_counts.get('predict') == 5
