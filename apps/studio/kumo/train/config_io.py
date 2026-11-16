# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monolith adapters for portable training job configuration records."""

from __future__ import annotations

from kumo_core.job_config import JobConfig, PredictionConfig, TrainDirOverrides
from kumo_core.model_plan import ModelPlan

from async_utils.sync import blocking_func
from io_utils.json import load_json
from kumo.home import HomeDirManager
from kumo.task.task import Task


@blocking_func()
def load_job_task(job_config: JobConfig | PredictionConfig) -> Task:
    config_dir = (job_config.home_dir if isinstance(job_config, JobConfig) else
                  job_config.train_dir)
    return Task.load(job_config.task_name, config_dir)


def load_job_automl_config(job_config: JobConfig) -> ModelPlan:
    assert job_config.automl_cfg_path is not None
    return load_json(job_config.automl_cfg_path, ModelPlan)


def get_pred_job_model_dir(prediction_config: PredictionConfig) -> str:
    model_dir = prediction_config.model_dir
    overrides_path = HomeDirManager(
        prediction_config.train_dir).train_dir_overrides_file
    overrides = load_json(
        overrides_path,
        TrainDirOverrides,
        default=None,
        raise_on_missing=False,
    )
    return overrides.model_dir if overrides is not None else model_dir


__all__ = [
    'get_pred_job_model_dir',
    'load_job_automl_config',
    'load_job_task',
]
