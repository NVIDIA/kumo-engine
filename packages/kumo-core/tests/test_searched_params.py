# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from kumo_core.automl.optuna_searcher import TPESearcher
from kumo_core.model_plan import ModelPlan


def test_searcher_records_only_varied_fields_as_json():
    space = ModelPlan()
    space.training_job.tune_metric = 'mae'
    space.optimization.base_lr = [0.1, 0.01]
    space.optimization.batch_size = [512]
    space.optimization.optimizer = ['Adam', 'AdamW']
    space.neighbor_sampling.num_neighbors = [[16, 8], [8, 4]]
    space.model_architecture.channels = [64, 128]
    searcher = TPESearcher(space, seed=0)

    def chosen(cfg):
        return {
            'optimization.base_lr': cfg.optimization.base_lr[0],
            'optimization.optimizer': cfg.optimization.optimizer[0],
            'model_architecture.channels': cfg.model_architecture.channels[0],
            'neighbor_sampling.num_neighbors':
            str(cfg.neighbor_sampling.num_neighbors[0]),
        }

    cfg = searcher.suggest()
    assert searcher.sampled_params == chosen(cfg)
    json.dumps(searcher.sampled_params)

    searcher.max_attempts = 0
    cfg = searcher.suggest()
    assert searcher.sampled_params == chosen(cfg)
    assert (searcher.sampled_params
            is searcher.fallback_sampler.sampled_params)
