# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from kumo_core.automl.optuna_searcher import TPESearcher
from kumo_core.automl.searcher import searched_params
from kumo_core.model_plan import ModelPlan


def test_searched_params_reports_only_varied_fields_as_json():
    space = ModelPlan()
    space.training_job.tune_metric = 'mae'
    space.optimization.base_lr = [0.1, 0.01]
    space.optimization.batch_size = [512]
    space.optimization.optimizer = ['Adam', 'AdamW']
    space.neighbor_sampling.num_neighbors = [[16, 8], [8, 4]]
    space.model_architecture.channels = [64, 128]

    cfg = TPESearcher(space, seed=0).suggest()
    params = searched_params(space, cfg)

    assert set(params) == {
        'optimization.base_lr', 'optimization.optimizer',
        'model_architecture.channels', 'neighbor_sampling.num_neighbors'
    }
    assert params['optimization.base_lr'] == cfg.optimization.base_lr[0]
    assert params['neighbor_sampling.num_neighbors'] == str(
        cfg.neighbor_sampling.num_neighbors[0])
    json.dumps(params)
