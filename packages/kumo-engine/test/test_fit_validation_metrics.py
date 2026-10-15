# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pandas as pd
import pytest
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


@pytest.mark.parametrize("invalid", ["metric", "split"])
def test_invalid_model_plan_fails_before_task_materialization(
        tmp_path, monkeypatch, invalid):
    from copy import deepcopy
    from dataclasses import replace
    from unittest.mock import Mock

    from kumo_core.model_plan import TrainingTableGenerationPlan

    import kumo.engine.trainer.trainer as trainer_module

    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        data = tmp_path / 'data'
        data.mkdir()
        pd.DataFrame({'ID': range(100), 'Y': np.arange(100) / 100}).to_parquet(
            data / 'ents.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        graph = kumo.Graph({'ents': kumo.Table.from_source_table(
            connector['ents'], primary_key='ID')})
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph, 'PREDICT ents.Y FOR EACH ents.ID')
        table = query.generate_training_table(snapshot)
        before = set(snapshot.index_path.rglob('*'))
        plan = ModelPlan()
        if invalid == "metric":
            plan.training_job.metrics = ['auroc']
            plan.training_job.tune_metric = 'auroc'
            error = "Unsupported metric 'auroc'"
        else:
            table = replace(
                table, generation_plan=TrainingTableGenerationPlan(
                    split="TemporalSplit(ratio=[0.8, 0.1, 0.1])"))
            error = "temporal.*doesn't have a 'Create Date'"
        original = deepcopy(plan)
        trainer = kumo.Trainer(plan)
        materialize = Mock(side_effect=AssertionError('materialization began'))
        monkeypatch.setattr(trainer_module,
                            'build_generated_training_task_dataset', materialize)

        with pytest.raises(ValueError, match=error):
            trainer.fit(table, snapshot=snapshot)

        materialize.assert_not_called()
        assert set(snapshot.index_path.rglob('*')) == before
        assert not list(trainer.scratch_dir.glob('runs/*-fit-*'))
        assert plan == original
    finally:
        kumo.configure(scratch_dir=previous)


def test_fit_warns_about_random_split_before_task_materialization(
        tmp_path, monkeypatch, caplog):
    from unittest.mock import Mock

    from kumoapi.model_plan import TrainingTableGenerationPlan

    import kumo.engine.trainer.trainer as trainer_module

    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        data = tmp_path / 'data'
        data.mkdir()
        pd.DataFrame({
            'ID': range(100),
            'Y': np.arange(100) / 100,
            'TS': pd.date_range('2024-01-01', periods=100),
        }).to_parquet(data / 'ents.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        graph = kumo.Graph({
            'ents':
            kumo.Table.from_source_table(connector['ents'], primary_key='ID',
                                         time_column='TS')
        })
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph, 'PREDICT ents.Y FOR EACH ents.ID')
        table = query.generate_training_table(
            snapshot,
            TrainingTableGenerationPlan(
                split='RandomSplit(ratio=[0.8, 0.1, 0.1])'))
        caplog.clear()
        materialize = Mock(side_effect=RuntimeError('stop after validation'))
        monkeypatch.setattr(trainer_module,
                            'build_generated_training_task_dataset',
                            materialize)

        with pytest.raises(RuntimeError, match='stop after validation'):
            kumo.Trainer(ModelPlan()).fit(table, snapshot=snapshot)

        materialize.assert_called_once()
        assert 'Possible Data Leakage: Random Split' in caplog.text
    finally:
        kumo.configure(scratch_dir=previous)


@pytest.mark.parametrize('weight_column, declared', [
    ('importance', True), ('WEIGHT', False)
])
def test_model_plan_uses_normalized_weight_schema(tmp_path, monkeypatch,
                                                  weight_column, declared):
    import json
    from pathlib import Path
    from unittest.mock import Mock

    import kumo_core.training.workflow as workflow
    from kumoapi.train import TrainingTableSpec

    import kumo.engine.trainer.trainer as trainer_module

    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        data = tmp_path / 'data'
        data.mkdir()
        pd.DataFrame({'ID': range(100), 'Y': np.arange(100) / 100}).to_parquet(
            data / 'ents.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        graph = kumo.Graph({'ents': kumo.Table.from_source_table(
            connector['ents'], primary_key='ID')})
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph, 'PREDICT ents.Y FOR EACH ents.ID')
        pd.DataFrame({
            'ENTITY1': range(100),
            'TARGET1': np.arange(100) / 100,
            weight_column: np.arange(100) / 100 + 1,
            'SPLIT': [0] * 70 + [1] * 15 + [2] * 15,
        }).to_parquet(data / 'custom.parquet')
        table = kumo.TrainingTable.from_source_table(
            connector['custom'], TrainingTableSpec(
                weight_col=weight_column if declared else None,
                split_col='SPLIT'), pq=query, snapshot=snapshot)
        plan = ModelPlan()
        plan.training_job.num_experiments = 1
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        plan.optimization.weight_mode = ['weighted_loss']
        trainer = kumo.Trainer(plan)
        resolve = Mock(wraps=workflow.resolve_model_plan)
        monkeypatch.setattr(workflow, 'resolve_model_plan', resolve)

        if declared:
            result = trainer.fit(table, snapshot=snapshot)
            assert [call.kwargs['has_train_table_weight_col']
                    for call in resolve.call_args_list] == [True, True]
            resolved = json.loads(
                Path(result.artifact_paths['resolved_model_plan']).read_text())
            assert resolved['optimization']['weight_mode'] == ['weighted_loss']
        else:
            before = set(snapshot.index_path.rglob('*'))
            materialize = Mock(side_effect=AssertionError('materialization began'))
            monkeypatch.setattr(trainer_module,
                                'build_generated_training_task_dataset',
                                materialize)
            with pytest.raises(ValueError, match='weight_mode.*not supported'):
                trainer.fit(table, snapshot=snapshot)
            assert resolve.call_args.kwargs['has_train_table_weight_col'] is False
            materialize.assert_not_called()
            assert set(snapshot.index_path.rglob('*')) == before
            assert not list(trainer.scratch_dir.glob('runs/*-fit-*'))
    finally:
        kumo.configure(scratch_dir=previous)


def test_fit_result_keeps_validation_metrics_through_reload(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        rng = np.random.default_rng(0)
        (tmp_path / 'data').mkdir()
        pd.DataFrame({
            'ID': np.arange(400),
            'Y': rng.random(400),
            'F1': rng.normal(size=400),
        }).to_parquet(tmp_path / 'data' / 'ents.parquet')
        connector = kumo.FileConnector(root_dir=str(tmp_path / 'data'))
        ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                            primary_key='ID').infer_metadata()
        graph = kumo.Graph(tables={'ents': ents}, edges=[])
        snapshot = graph.snapshot()
        pquery = kumo.PredictiveQuery(graph=graph,
                                      query='PREDICT ents.Y FOR EACH ents.ID')
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        trainer = kumo.Trainer(model_plan=plan)
        result = trainer.fit(pquery.generate_training_table(snapshot),
                             snapshot=snapshot)
    finally:
        kumo.configure(scratch_dir=previous)

    for prefix in ('train_', 'val_', 'test_'):
        assert any(name.startswith(prefix) for name in result.metrics)
    assert all(math.isfinite(value) for value in result.metrics.values())

    reloaded = kumo.Trainer.load(result.bundle_dir)
    assert reloaded._training_result.metrics == result.metrics
