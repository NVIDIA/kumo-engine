# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import shutil

import pytest
from kumo_core.job_config import (
    JobConfig,
    PredictionConfig,
    _load_k8s_job_config,
)
from pydantic import TypeAdapter, ValidationError

from io_utils import kumo_dataclass
from io_utils.json import save_json

ROOT_DIR = osp.dirname(osp.realpath(__file__))
AUTO_TRAIN_TESTDIR = osp.join(ROOT_DIR, 'test_results')


@kumo_dataclass
class JobConfigV1:
    home_dir: str
    task_name: str
    automl_cfg_path: str | None = None
    data_backend_persist_dir: str | None = None


@kumo_dataclass
class PredictionConfigV1:
    home_dir: str
    train_dir: str
    task_name: str
    model_dir: str
    return_embeddings: bool
    data_dir: str | None = None
    top_k: int | None = None
    binary_threshold: float | None = 0.5


def module_teardown_function():
    yield
    shutil.rmtree(AUTO_TRAIN_TESTDIR, ignore_errors=True)


def test_load_k8s_job_config():
    job_cfg_ignore_params = [
        'remote_graphengine_address',
        'tables_to_load',
        'nodes_to_load',
        'edges_to_load',
        'disable_explain',
        'use_core_v2_api',
        'graph_backend',
        # Removed from JobConfig (data persist dir now derives from
        # deploy_env); legacy job JSONs may still carry it, so it is ignored.
        'data_backend_persist_dir',
    ]
    # Test load job v1 to v2 config
    job_json_dicts_v1 = dict(
        home_dir="",
        task_name="",
        training_job_id="",
        automl_cfg_path="",
        data_backend_persist_dir="",
    )
    job_json_dicts_v2 = _load_k8s_job_config(
        job_json_dicts_v1, JobConfig, json_ignore_params=job_cfg_ignore_params)
    job_config_v2 = TypeAdapter(JobConfig).validate_python(job_json_dicts_v2)
    job_config_v2_dict = job_config_v2.__dict__
    job_config_v2_dict.pop('__pydantic_initialised__', None)

    # Keep this compatibility check resilient to additive JobConfig fields.
    assert {
        'home_dir',
        'task_name',
        'training_job_id',
        'automl_cfg_path',
    }.issubset(job_config_v2_dict.keys())
    # Test load job v2 to v1 config
    job_json_dicts_v2 = dict(
        home_dir="",
        task_name="",
        automl_cfg_path="",
        data_backend_persist_dir="",
        remote_graphengine_address="",
        tables_to_load="",
        nodes_to_load="",
        edges_to_load="",
        disable_explain=False,
    )
    job_json_dicts_v1 = _load_k8s_job_config(
        job_json_dicts_v2, JobConfigV1,
        json_ignore_params=job_cfg_ignore_params)
    assert len(job_json_dicts_v1) == 4


def test_load_prediction_config():
    json_ignore_params = ['output_dir']
    config_raise_params = ['output_dir']
    # Test loading prediction v1 to v2 config
    # As we require json dicts to have output_dir, error will be raised here
    pred_json_dicts_v1 = dict(home_dir="", train_dir="", task_name="",
                              model_dir="", return_embeddings=False)
    try:
        _load_k8s_job_config(pred_json_dicts_v1, PredictionConfig,
                             json_ignore_params=json_ignore_params,
                             config_raise_params=config_raise_params)
        assert False
    except ValueError:
        assert True

    # Test loading prediction v2 to v1 config
    predict_dicts_v2 = dict(home_dir="", train_dir="", task_name="",
                            model_dir="", output_dir="",
                            return_embeddings=False)
    predict_dicts_v1 = _load_k8s_job_config(
        predict_dicts_v2, PredictionConfigV1,
        json_ignore_params=json_ignore_params,
        config_raise_params=config_raise_params)
    prediction_config_v1 = TypeAdapter(PredictionConfigV1).validate_python(
        predict_dicts_v1)
    prediction_config_v1_dict = prediction_config_v1.__dict__
    prediction_config_v1_dict.pop('__pydantic_initialised__', None)
    assert len(prediction_config_v1_dict) == 8


@pytest.mark.parametrize('return_predictions', [False, True])
def test_prediction_config_return_predictions_roundtrip(tmp_path,
                                                        return_predictions):
    config = PredictionConfig(
        home_dir='/h',
        train_dir='/t',
        task_name='task',
        model_dir='/m',
        output_dir='/o',
        return_embeddings=True,
        return_predictions=return_predictions,
        pred_table_path='/pred',
    )
    path = str(tmp_path / 'prediction_config.json')
    save_json(config, path)

    loaded = PredictionConfig.load(path)

    assert loaded.return_predictions is return_predictions


def test_prediction_config_requires_authoritative_prediction_output_flag():
    with pytest.raises(ValidationError, match='return_predictions'):
        TypeAdapter(PredictionConfig).validate_python(
            dict(
                home_dir='/h',
                train_dir='/t',
                task_name='task',
                model_dir='/m',
                output_dir='/o',
                return_embeddings=True,
                pred_table_path='/pred',
            ))
