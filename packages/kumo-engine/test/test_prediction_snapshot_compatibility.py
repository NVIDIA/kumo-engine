# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from kumoapi.jobs import PredictionArtifactType as ApiType
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan

from kumo.engine.graph import GraphSnapshot
from kumo.engine.pquery.prediction_table import PredictionTable
from kumo.engine.trainer._spec import MANIFEST_FILE, write_training_spec
from kumo.engine.trainer.output_config import PredictionOutputConfig
from kumo.engine.trainer.trainer import Trainer

QUERY = 'PREDICT users.label FOR EACH users.id'
TRAINING_SNAPSHOT_ID = 'graph123@olddata00000000'
UPDATED_SNAPSHOT_ID = 'graph123@newdata00000000'


@dataclass(frozen=True)
class _TaskDataset:
    job_id: str = 'generated-job'
    artifact_root: str = 'generated-artifacts'


def _graph_config(*, alternate: bool = False):
    return {
        'tables': [{
            'table_name': 'users',
            'source_table_name': ('other.schema.users'
                                  if alternate else 'db.schema.users'),
            'pkey': {
                'name': 'id',
                'dtype': 'int64',
                'stype': 'ID',
            },
            'cols': ([{
                'name': 'label',
                'dtype': 'bool',
                'stype': 'categorical',
            }] + ([{
                'name': 'new_feature',
                'dtype': 'float64',
                'stype': 'numerical',
            }] if alternate else [])),
            'fkeys': [],
        }]
    }


def _fake_published_bundle(root: Path) -> Path:
    bundle = root / 'published-model'
    (bundle / 'model').mkdir(parents=True)
    (bundle / 'model' / 'config.json').write_text('{}')
    (bundle / 'metrics.json').write_text(json.dumps({'auc': 0.9}))
    artifacts = []
    for role, relative_path in (
        ('trainer_config', 'model/config.json'),
        ('metrics', 'metrics.json'),
    ):
        artifact = bundle / relative_path
        artifacts.append({
            'role': role,
            'relative_path': relative_path,
            'size_bytes': artifact.stat().st_size,
            'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        })
    (bundle / MANIFEST_FILE).write_text(json.dumps({
        'schema_version': 1,
        'policy': 'vital-training-artifacts-v1',
        'excluded_by_default': [],
        'artifacts': artifacts,
    }))
    write_training_spec(
        bundle,
        predictive_query=QUERY,
        model_plan=ModelPlan(),
        snapshot_id=TRAINING_SNAPSHOT_ID,
        training_table_generation=TrainingTableGenerationPlan(),
    )
    return bundle


def _prediction_table(root: Path) -> PredictionTable:
    return PredictionTable(QUERY, root, root, 0, 'tt-1', (), ())


def _predict_with_stubbed_scorer(bundle: Path, root: Path,
                                 snapshot_id: str, *, alternate: bool = False) -> list:
    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.task_dataset import DiskgraphTaskManifest
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    from kumo_core.train.io import StreamingFrameWriter
    from kumo_core.types.enums import PredictionArtifactType

    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text('{}')
    trainer = Trainer.load(bundle)
    trainer._id = 'fresh-process'
    snapshot = GraphSnapshot(snapshot_id, root / 'index',
                             _graph_config(alternate=alternate))
    table = _prediction_table(root)
    output = PredictionOutputConfig(
        SimpleNamespace(
            sink_backend=SimpleNamespace(write_files=lambda *a, **k: None)),
        ApiType.PREDICTIONS,
        'out',
    )

    def fake_predict(**kwargs) -> None:
        kind = PredictionArtifactType.PREDICTIONS
        worker = kwargs['partition_id']
        writer = StreamingFrameWriter(
            str(Path(kwargs['output_dir']) / kind.output_file_name /
                f'part-{worker:05d}'),
            max_file_size=1 << 20,
            max_buffer_size=1 << 20,
            index=False,
            stats_sample_rows=1000,
        )
        with writer:
            writer.write(
                pd.DataFrame({
                    'True_PROB': [.8],
                    'TARGET_PRED': [True],
                }))
        kwargs['artifact_stats_collector'].add_partition(
            worker, {kind: writer.stats})

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
                return_value=_TaskDataset(),
            ))
        stack.enter_context(
            patch(
                'kumo_core.prediction.materialize_prediction_task',
                return_value=SimpleNamespace(
                    manifest_path='task',
                    bundle_dir='bundle',
                ),
            ))
        stack.enter_context(
            patch(
                'kumo.engine.trainer.trainer._stage_model_bundle',
                return_value=root,
            ))
        stack.enter_context(
            patch.object(
                DiskgraphTaskManifest,
                'load',
                return_value=SimpleNamespace(task_type='binary_classification',
                                             stage_counts={'predict': 1}),
            ))
        stack.enter_context(
            patch.object(ResolvedNumericalTask, 'from_manifest',
                         return_value=None))
        stack.enter_context(
            patch('kumo_core.train.trainer.load_encoders', return_value=None))
        stack.enter_context(
            patch.object(
                DiskgraphNestedTrainer,
                'load',
                return_value=SimpleNamespace(predict=fake_predict),
            ))
        result = trainer.predict(
            table,
            snapshot=snapshot,
            output_config=output,
            binary_classification_threshold=.5,
        )
    return result.frame['TARGET_PRED'].tolist()


def test_fresh_process_loaded_model_predicts_new_data_version(
        tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path / 'bundle')
    child_root = tmp_path / 'child'
    code = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from test_prediction_snapshot_compatibility import (
    UPDATED_SNAPSHOT_ID,
    _predict_with_stubbed_scorer,
)
from pathlib import Path
print(json.dumps(_predict_with_stubbed_scorer(
    Path(sys.argv[2]), Path(sys.argv[3]), UPDATED_SNAPSHOT_ID)))
"""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(str(item) for item in sys.path)
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            code,
            str(Path(__file__).parent),
            str(bundle),
            str(child_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == [True]


def test_loaded_model_accepts_compatible_independently_constructed_graph(
        tmp_path: Path) -> None:
    bundle = _fake_published_bundle(tmp_path / 'bundle')
    assert _predict_with_stubbed_scorer(
        bundle,
        tmp_path / 'different',
        'different-graph@newdata00000000',
        alternate=True,
    ) == [True]
