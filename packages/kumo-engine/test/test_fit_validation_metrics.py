# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pandas as pd
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


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
