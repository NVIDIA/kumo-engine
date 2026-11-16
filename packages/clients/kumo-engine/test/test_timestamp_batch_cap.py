# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from diskgraph import RuntimeIndex, SplitSpec, execute_tt_ingest, ingest
from diskgraph.tt_artifacts import (
    build_train_table_artifacts_for_existing_table,
)
from kumo_core.diskgraph.lightning_data import (
    DiskgraphTaskBatchIterable,
    _allocate_stage_batches,
    _ShuffleStreamState,
    _stage_bucket_counts,
)
from torch.utils.data import DataLoader

VAL_TIMESTAMP = 3000


@pytest.fixture
def temporal_manifest(tmp_path):
    source = tmp_path / 'users.parquet'
    pq.write_table(pa.table({'id': list(range(12))}), source)
    parent = tmp_path / 'index'
    ingest(
        config_json=json.dumps({
            'dataset_name':
            'temporal_cap',
            'num_shards':
            1,
            'tables': [{
                'name': 'users',
                'primary_key': 'id',
                'columns': [{
                    'name': 'id',
                    'ingested_dtype': 'int64'
                }],
                'source': {
                    'type': 'parquet',
                    'paths': [str(source)]
                }
            }],
        }), output_dir=str(parent), col_stats=False)
    source = tmp_path / 'training.parquet'
    pq.write_table(
        pa.table({
            'ENTITY1':
            list(range(12)) * 4,
            'TARGET1': [[i] for i in range(12)] * 4,
            'TIMESTAMP':
            [t for t in [1000, 2000, 3000, 4000] for _ in range(12)],
        }), source)
    result = build_train_table_artifacts_for_existing_table(
        output_path=str(source),
        index_path=str(parent),
        tt_id='cap',
        split_config=SplitSpec(kind='TimeRangeSplit', ranges=[(0, 2500),
                                                              (2501, 3500),
                                                              (3501, 4500)],
                               is_temporal=True, timestamp_unit='seconds',
                               unit='seconds'),
        input_timestamp_unit='seconds',
        split_stats_config={
            'entity_table': 'users',
            'entity_pk_column': 'id'
        },
        train_table_col_stats_config={'target_stype': 'ID'},
        target_entity_table='users',
        task_type='temporal_link_prediction',
        lp_rhs_is_temporal=True,
    )
    task = execute_tt_ingest(staging_dir=result['staging_dir'],
                             index_path=str(parent),
                             tt_id='cap')['task_dataset']
    manifest = tmp_path / 'runtime.json'
    manifest.write_text(
        json.dumps({
            'kind': 'diskgraph_task_dataset',
            'graph': {
                'index_path': str(parent)
            },
            'stages': task['stages'],
            'task': {
                'is_temporal': True
            },
        }))
    return str(manifest)


def batch_rows(batch):
    task = batch['task']
    return list(
        zip(task['seed_timestamps'].to_pylist(),
            task['tt_stage_node_ids'].to_pylist()))


def make_dataset(manifest, stages, cap, workers, **kwargs):
    return DiskgraphTaskBatchIterable(
        manifest_path=manifest,
        stage=stages,
        sampling_plan={'layers': []},
        batch_size=5,
        shuffle=True,
        drop_last=False,
        base_seed=17,
        shuffle_state=_ShuffleStreamState(),
        group_by_timestamp=True,
        min_steps_per_epoch=cap,
        max_steps_per_epoch=cap,
        dataloader_num_workers=workers,
        runtime_options={
            'entity_node_type': 'users',
            'num_threads': 1
        },
        **kwargs,
    )


def load(dataset, workers):
    options = {'multiprocessing_context': 'fork'} if workers else {}
    loader = DataLoader(dataset, batch_size=None, num_workers=workers,
                        **options)
    return [batch_rows(batch) for batch in loader]


@pytest.mark.parametrize('workers', [0, 2, 5])
@pytest.mark.parametrize('cap', [2, 5, 6])
def test_grouped_loader_consumes_exact_global_prefix(temporal_manifest, workers,
                                                     cap):
    index = RuntimeIndex(temporal_manifest, rnd_seed=17, num_threads=1)
    expected = []
    for epoch in range(2):
        native, _ = index.iter(stage='train', sampling_plan={'layers': []},
                               num_workers=1, worker_id=0, shuffle=True,
                               epoch=epoch, batch_size=5,
                               entity_node_type='users',
                               group_by_timestamp=True)
        expected.extend(batch_rows(batch) for batch in native)
    dataset = make_dataset(temporal_manifest, ('train', ), cap, workers)
    actual = load(dataset, workers)
    assert actual == expected[:cap]
    assert len(actual) == len(dataset) == cap
    rows = [row for batch in actual for row in batch]
    if cap <= 6:
        assert len(rows) == len(set(rows))


@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('cap', [5, 9, 12])
def test_refit_with_grouping_honors_stage_targets(temporal_manifest, workers,
                                                  cap):
    stages = ('train', 'val')
    natural = {
        stage: sum(-(-rows // 5)
                   for _, rows in _stage_bucket_counts(temporal_manifest,
                                                       stage))
        for stage in stages
    }
    targets = _allocate_stage_batches(natural, cap, require_each_stage=True)
    dataset = make_dataset(
        temporal_manifest, stages, cap, workers, stage_batch_targets=targets,
        stage_num_seeds={
            'train': 24,
            'val': 12
        })
    actual = load(dataset, workers)
    assert len(actual) == len(dataset) == sum(targets.values()) == cap
    assert all(len({ts for ts, _ in batch}) == 1 for batch in actual)
    val_batches = [b for b in actual if b[0][0] == VAL_TIMESTAMP]
    assert len(val_batches) == targets['val']
    assert len(actual) - len(val_batches) == targets['train']
