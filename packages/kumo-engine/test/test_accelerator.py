# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan

from kumo.engine import FileConnector, Graph, PredictiveQuery, Table, Trainer
from kumo.engine._scratch import get_scratch_dir


@pytest.mark.parametrize('accelerator', ['cuda', 'tpu', '', None, 1])
def test_invalid_accelerator_fails_at_construction(accelerator):
    with pytest.raises(ValueError, match='accelerator must be one of'):
        Trainer(ModelPlan(), accelerator=accelerator)


def test_accelerator_defaults_to_auto_and_is_instance_local(monkeypatch):
    monkeypatch.setenv('KUMO_ACCELERATOR', 'mps')
    automatic = Trainer(ModelPlan())
    cpu = Trainer(ModelPlan(), accelerator='cpu')
    assert automatic._accelerator == 'auto'
    assert cpu._accelerator == 'cpu'
    assert os.environ['KUMO_ACCELERATOR'] == 'mps'


def test_cpu_fit_refit_reload_and_predict_override_environment(
        tmp_path, monkeypatch):
    from kumo_core.train.trainer import Trainer as CoreTrainer

    import kumo.engine as kumo

    devices = []
    original = CoreTrainer.create_pl_trainer

    def create_pl_trainer(self):
        original(self)
        devices.append(self.pl_trainer.strategy.root_device.type)

    monkeypatch.setattr(CoreTrainer, 'create_pl_trainer', create_pl_trainer)
    monkeypatch.setenv('KUMO_ACCELERATOR', 'mps')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        data = tmp_path / 'data'
        data.mkdir()
        pd.DataFrame({
            'id': range(100),
            'label': [bool(i % 2) if i < 80 else None for i in range(100)],
            'value': [i / 100 for i in range(100)],
        }).to_parquet(data / 'users.parquet')
        users = Table.from_source_table(
            FileConnector(root_dir=str(data))['users'], primary_key='id')
        graph = Graph({'users': users})
        snapshot = graph.snapshot()
        query = PredictiveQuery(graph, 'PREDICT users.label FOR EACH users.id')
        plan = ModelPlan()
        plan.training_job.num_experiments = 1
        plan.training_job.refit_trainval = True
        plan.training_job.disable_explain = True
        plan.training_job.disable_compilation = True
        plan.optimization.max_epochs = 1
        plan.optimization.min_steps_per_epoch = 2
        plan.optimization.max_steps_per_epoch = 2
        plan.model_architecture.channels = [8]
        trainer = Trainer(plan, accelerator='cpu')
        result = trainer.fit(query.generate_training_table(snapshot),
                             snapshot=snapshot)
        assert len(devices) >= 2
        fit_count = len(devices)
        restored = Trainer.load(result.bundle_dir, accelerator='cpu')
        prediction = restored.predict(
            query.generate_prediction_table(snapshot), snapshot=snapshot,
            binary_classification_threshold=0.5)
        assert len(prediction.frame) == 20
        assert len(devices) > fit_count
        assert set(devices) == {'cpu'}
    finally:
        kumo.configure(scratch_dir=previous)
