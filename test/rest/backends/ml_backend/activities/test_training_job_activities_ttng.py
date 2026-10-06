# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TT-NG training-postprocess metadata contract tests."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from kumo_core.job_config import JobConfig
from kumoapi.train import TrainingStage

from io_utils.json import save_json
from rest.backends.ml_backend.activities import training_job_activities as _act
from statedb.config.common import DataSnapshot


def test_ttng_postprocess_reads_dataplane_bundle(tmp_path, monkeypatch) -> None:
    """Postprocess reads durable dataplane sidecars, not local pool sidecars."""
    stats = {
        'payload': {
            'split_payload': {
                'artifacts_v3': {
                    'timestamp_stage_assignments': [
                        {
                            'stage': 'train',
                            'timestamp': 1_700_000_000
                        },
                        {
                            'stage': 'train',
                            'timestamp': 1_700_086_400
                        },
                        {
                            'stage': 'val',
                            'timestamp': 1_700_172_800
                        },
                    ],
                },
            },
        },
    }
    split_info = {'stage_counts': {'train': 12, 'val': 3, 'test': 2}}
    snapshot_id = 'graph-1bf8d6a669f6a6caba68a2393f1044e3@v1'
    graph_fpid, graph_data_version = DataSnapshot.GRAPH.parse_id(snapshot_id)
    data_persist_dir = tmp_path / 'persist'
    remote_root = (data_persist_dir / 'diskgraph_job_sidecars' /
                   f'NA-{graph_fpid}' / graph_data_version)
    remote_bundle = remote_root / 'task_datasets' / 'trainingjob-1'
    remote_bundle.mkdir(parents=True)
    (remote_bundle / 'stats.json').write_text(json.dumps(stats))
    (remote_bundle / 'split_info.json').write_text(json.dumps(split_info))
    (remote_root / 'manifests').mkdir()
    (remote_root / 'manifests' / 'trainingjob-1.json').write_text(
        json.dumps({
            'schema_version': 1,
            'snapshot_id': snapshot_id,
            'job_id': 'trainingjob-1',
            'relpaths': ['task_datasets/trainingjob-1'],
            'bundle_relpath': 'task_datasets/trainingjob-1',
        }))

    local_index = tmp_path / 'local-index'
    local_bundle = local_index / 'task_datasets' / 'trainingjob-1'
    local_bundle.mkdir(parents=True)
    (local_bundle / 'stats.json').write_text('{}')
    (local_bundle / 'split_info.json').write_text('{}')
    job_config = JobConfig(
        home_dir=str(tmp_path / 'home'),
        task_name='task',
        diskgraph_index_path=str(local_index),
        diskgraph_task_dataset_bundle_dir=str(local_bundle),
    )
    config_path = tmp_path / 'job.json'
    save_json(job_config, str(config_path))
    monkeypatch.setattr(_act, 'get_persist_dir', lambda: str(data_persist_dir))

    read_stats, read_split_info = _act._load_ttng_published_metadata(
        job_config,
        snapshot_id=snapshot_id,
        job_config_path=str(config_path),
    )

    assert read_stats == stats
    assert read_split_info == split_info
    periods = _act._time_periods_from_ttng_stats(read_stats)
    assert periods[TrainingStage.TRAIN].start == datetime.fromtimestamp(
        1_700_000_000, tz=timezone.utc)
    assert periods[TrainingStage.TRAIN].end == datetime.fromtimestamp(
        1_700_086_400, tz=timezone.utc)
    sizes = _act._split_sizes_from_ttng_split_info(read_split_info)
    assert (sizes.train, sizes.val, sizes.test) == (12, 3, 2)


def test_ttng_postprocess_rejects_missing_durable_sidecar(tmp_path) -> None:
    """Missing bundle metadata is an invariant violation, not an empty split."""
    (tmp_path / 'stats.json').write_text('{}')
    job_config = SimpleNamespace(
        diskgraph_task_dataset_bundle_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match='split_info'):
        _act._load_ttng_published_metadata(job_config)
