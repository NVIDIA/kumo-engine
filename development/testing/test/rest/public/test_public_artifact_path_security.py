# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kumoapi.jobs as api
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rest.deps.authentication import get_ctx
from rest.public.routers import gen_pred_table, gen_train_table, training
from rest.public.routers.gen_pred_table import GeneratePredictionTableRouter
from rest.public.routers.gen_train_table import GenerateTrainTableRouter
from rest.public.routers.training import TrainingRouterV1
from rest.utils import public as public_utils
from statedb.config.common import (
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    TrainingJobID,
)

_INTERNAL_PATH = '/internal/control-plane/secret.parquet'
_IMMUTABLE_PATH = '/internal/control-plane/_bundles/bundle/tt.parquet'
_SAFE_URL = 'https://signed.example.test/artifact'


def _done_job():
    return SimpleNamespace(
        job_status_report=SimpleNamespace(status=api.JobStatus.DONE),
        tags={},
    )


@pytest.fixture
def artifact_path_client(monkeypatch: pytest.MonkeyPatch):
    presign = AsyncMock(return_value=[_SAFE_URL])
    presign_func = SimpleNamespace(coroutine=presign)
    monkeypatch.setattr(gen_train_table, 'presign_parquet_table_data',
                        presign_func)
    monkeypatch.setattr(gen_pred_table, 'presign_parquet_table_data',
                        presign_func)
    train_router = GenerateTrainTableRouter.__new__(GenerateTrainTableRouter)
    train_router.get = AsyncMock(return_value=_done_job())
    train_router._ml_job_backend = SimpleNamespace(
        get_train_table_result=AsyncMock(return_value=SimpleNamespace(
            train_table_path=_IMMUTABLE_PATH,
        )))

    pred_router = GeneratePredictionTableRouter.__new__(
        GeneratePredictionTableRouter)
    pred_router.get = AsyncMock(return_value=_done_job())
    pred_router._ml_job_backend = SimpleNamespace(
        get_pred_table_result=AsyncMock(return_value=SimpleNamespace(
            pred_table_path=_INTERNAL_PATH)))

    training_router = TrainingRouterV1.__new__(TrainingRouterV1)
    training_router.get = AsyncMock(return_value=_done_job())
    training_router._ml_job_backend = SimpleNamespace(
        get_job_home_dir=AsyncMock(return_value='/internal/job'))

    best_dir = SimpleNamespace(coroutine=AsyncMock(
        return_value='/internal/best'))
    monkeypatch.setattr(
        training, 'HomeDirManager',
        lambda _path: SimpleNamespace(get_best_trainer_dir=best_dir))
    monkeypatch.setattr(training.fsutil, 'exists_async',
                        AsyncMock(return_value=True))
    download = AsyncMock(return_value=_SAFE_URL)
    helper = SimpleNamespace(generate_download_url=SimpleNamespace(
        coroutine=download))
    monkeypatch.setattr(training.fsutil, 'get_presigned_helper',
                        lambda _path: helper)

    app = FastAPI()
    app.dependency_overrides[get_ctx] = lambda: MagicMock()
    app.get('/train/{id}')(train_router.get_table_data)
    app.get('/pred/{id}')(pred_router.get_table_data)
    app.get('/holdout/{training_job_id}')(training_router.get_holdout_df)
    with TestClient(app) as client:
        yield client, app, presign, download


@pytest.mark.parametrize('route_kind', ['train-table', 'pred-table', 'holdout'])
def test_public_artifact_routes_never_expose_internal_paths(
        artifact_path_client, route_kind) -> None:
    client, app, presign, download = artifact_path_client
    cases = {
        'train-table': (
            f'/train/{GenerateTrainTableJobID.make_random_id()}',
            '/train/{id}',
            [_SAFE_URL],
        ),
        'pred-table': (
            f'/pred/{GeneratePredictionTableJobID.make_random_id()}',
            '/pred/{id}',
            [_SAFE_URL],
        ),
        'holdout': (
            f'/holdout/{TrainingJobID.make_random_id()}',
            '/holdout/{training_job_id}',
            _SAFE_URL,
        ),
    }
    route, schema_path, response_body = cases[route_kind]
    response = client.get(route, params={
        'presigned': 'false',
        'raw_path': 'true'
    })
    assert response.status_code == 200, response.text
    assert response.json() == response_body
    assert _INTERNAL_PATH not in response.text
    parameter_names = {
        item['name']
        for item in app.openapi()['paths'][schema_path]['get']['parameters']
    }
    assert not parameter_names & {'presigned', 'raw_path'}
    if route_kind == 'holdout':
        download.assert_awaited_once_with(
            os.path.join('/internal/best', 'test_df.parquet'))
        presign.assert_not_awaited()
    else:
        presign.assert_awaited_once()
        download.assert_not_awaited()
    if route_kind == 'train-table':
        presign.assert_awaited_once_with(_IMMUTABLE_PATH)


def test_resolved_native_train_table_is_presigned_as_one_file(monkeypatch):
    download = MagicMock(return_value=_SAFE_URL)
    helper = SimpleNamespace(generate_download_url=download)
    isdir = MagicMock(return_value=False)
    monkeypatch.setattr(public_utils.fsutil, 'get_presigned_helper',
                        lambda _path: helper)
    monkeypatch.setattr(public_utils.fsutil, 'isdir', isdir)
    monkeypatch.setattr(
        public_utils.fsutil,
        'listdir',
        lambda *_args, **_kwargs: pytest.fail(
            'single immutable parquet was treated as a directory'),
    )

    assert public_utils.presign_parquet_table_data(_IMMUTABLE_PATH) == [
        _SAFE_URL
    ]
    isdir.assert_called_once_with(_IMMUTABLE_PATH)
    download.assert_called_once_with(_IMMUTABLE_PATH)


@pytest.mark.asyncio
async def test_custom_validation_receives_physical_immutable_table(
        monkeypatch: pytest.MonkeyPatch) -> None:
    router = GenerateTrainTableRouter.__new__(GenerateTrainTableRouter)
    router.get = AsyncMock(return_value=SimpleNamespace(
        job_status_report=SimpleNamespace(status=api.JobStatus.DONE),
        project_id=None,
    ))
    router._ml_job_backend = SimpleNamespace(
        get_train_table_result=AsyncMock(return_value=SimpleNamespace(
            train_table_path=_IMMUTABLE_PATH,
        )), )
    validation = api.ValidationResponse()
    router._driver = SimpleNamespace(
        validate_custom_train_table=AsyncMock(return_value=validation), )

    result = await router.validate_custom_train_table(
        GenerateTrainTableJobID.make_random_id(),
        MagicMock(),
        api.TrainingTableSpec(weight_col='WEIGHT', split_col='SPLIT'),
        False,
        MagicMock(),
    )

    assert result is validation
    call_args = router._driver.validate_custom_train_table.await_args.args
    assert call_args[2] == _IMMUTABLE_PATH
    assert call_args[-1] == 'SPLIT'
