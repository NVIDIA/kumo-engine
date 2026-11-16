# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


def test_skewed_regression_target_gets_max_log_encoder(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        rng = np.random.default_rng(0)
        (tmp_path / 'data').mkdir()
        pd.DataFrame({
            'ID': np.arange(400),
            'Y': np.where(rng.random(400) < 0.2, 50.0, 1.0) + rng.random(400),
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
        trainer.fit(pquery.generate_training_table(snapshot),
                    snapshot=snapshot)
        [path] = trainer.scratch_dir.glob(
            'runs/*-fit-*/task_dataset/feature_schema.json')
    finally:
        kumo.configure(scratch_dir=previous)
    target = json.loads(path.read_text())['target_encoder_input']

    assert target['encoder']['name'] == 'MaxLogNumerical'
    assert {'MEAN', 'MEDIAN', 'MIN', 'MAX', 'STD'} <= set(target['stats'])
