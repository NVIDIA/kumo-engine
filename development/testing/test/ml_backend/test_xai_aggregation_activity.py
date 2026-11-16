# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from kumo_core.job_config import BestTrainerConfig
from kumo_core.types import TaskType
from kumoapi.explain import Feature, Subgraph, SubgraphFeatureStore
from kumoapi.typing import Stype

from io_utils.json import load_json, save_json
from kumo.config.explain import StrategyMappings
from kumo.home import HomeDirManager
from rest.backends.ml_backend.activities import ml_backend_activities as act


class _TrainingJobInput:
    def __init__(self, job_id: str, task_name: str):
        self.id = job_id
        self.spec = SimpleNamespace(
            predictive_query_id=SimpleNamespace(name=task_name),
            task_type=TaskType.BINARY_CLASSIFICATION,
        )


def _write_training_xai_fixture(tmp_path):
    job_id = 'trainingjob-xai-aggr'
    task_name = 'task'
    home = HomeDirManager(str(tmp_path / job_id))
    best_trainer_dir = (tmp_path / job_id / 'models' / task_name / 'automl' /
                        'version_0')
    subgraph_dir = best_trainer_dir / 'subgraph_collector' / 'test'
    subgraph_json_dir = subgraph_dir / 'subgraph_json'
    subgraph_json_dir.mkdir(parents=True)

    save_json(
        BestTrainerConfig(best_trainer=str(best_trainer_dir)),
        home.get_best_trainer_config_path(task_name),
    )
    save_json(
        StrategyMappings(mappings={'random': ['0', '1']}),
        str(subgraph_dir / 'strategy_mappings.json'),
    )
    for entity_id, age in [('0', 20), ('1', 40)]:
        feature_store = SubgraphFeatureStore(
            features={
                0: {
                    'age':
                    Feature(
                        table_name='users',
                        name='age',
                        value=age,
                        stype=Stype.numerical,
                    )
                }
            })
        save_json(Subgraph(feature_store=feature_store),
                  str(subgraph_json_dir / f'{entity_id}.json'))

    return job_id, task_name, subgraph_dir


def _aggr_feature_path(subgraph_dir):
    return (subgraph_dir / 'global_aggr_feature' / 'random' / 'users' / 'age' /
            'mean' / 'feature.json')


def test_xai_aggregation_writes_global_features_without_rfm(
        monkeypatch, tmp_path):
    job_id, task_name, subgraph_dir = _write_training_xai_fixture(tmp_path)
    monkeypatch.setattr(act, 'TrainingJobInput', _TrainingJobInput)
    monkeypatch.setattr(act, 'get_job_base_dir', lambda: str(tmp_path))

    act.xai_aggregation_activity(_TrainingJobInput(job_id, task_name),
                                 SimpleNamespace())

    feature_path = _aggr_feature_path(subgraph_dir)

    assert feature_path.exists()
    feature = load_json(str(feature_path), Feature)
    assert feature.table_name == 'users'
    assert feature.name == 'age'
    assert feature.value == 30
