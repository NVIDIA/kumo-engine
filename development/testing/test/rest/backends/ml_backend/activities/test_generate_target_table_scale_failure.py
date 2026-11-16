# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kumo.validation import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from statedb.utils.errors import TrainingTableGenerationError


class _RequestsBoundaryError(RuntimeError):
    __module__ = 'requests.exceptions'
    _kumo_restricted_log_message = True


class _FakeTrainingJobInput:
    def __init__(self) -> None:
        self.id = "trainingjob-11111111111111111111111111111111"
        self.generate_train_table_job_id = self.id
        self.spec = SimpleNamespace(predictive_query=object(), split=None)
        self.graph_data_compute_input = SimpleNamespace(connectors={})


class _FakeValidator:
    def pre_generation(self) -> ValidationResponse:
        return ValidationResponse()

    def post_generation(self, *_args) -> ValidationResponse:
        return ValidationResponse()


class _FakeGraph:
    data_sources = []

    def set_table_data_versions(self, _job_id: str) -> None:
        pass

    def config(self) -> SimpleNamespace:
        return SimpleNamespace()


class _FakePredictiveQuery:
    instance = None

    @classmethod
    def from_config(cls, _config):
        return cls.instance


def _configure_activity(
    monkeypatch,
    tmp_path,
    *,
    scale_response: ValidationResponse | None,
    producer_result=None,
    producer_error: Exception | None = None,
):
    from rest.backends.ml_backend.activities import \
        generate_target_table as mod
    materialize_training_task = import_module(
        "rest.backends.ml_backend.activities.materialize_training_task")

    table_generator = SimpleNamespace(
        validate_pre_train_table_generate=lambda **_kwargs: ValidationResponse(),
        validate_scale=lambda: scale_response,
    )
    predictive_query = _FakePredictiveQuery()
    predictive_query.graph = _FakeGraph()
    predictive_query.table_generator = table_generator
    predictive_query.query_type = None
    predictive_query.save = lambda **_kwargs: None
    predictive_query.get_task_type = lambda: None
    predictive_query.get_train_table_stats = lambda **_kwargs: {}

    def generate_train_table(**_kwargs):
        if producer_error is not None:
            raise producer_error
        return producer_result

    predictive_query.generate_train_table = generate_train_table
    _FakePredictiveQuery.instance = predictive_query

    monkeypatch.setattr(mod, "TrainingJobInput", _FakeTrainingJobInput)
    monkeypatch.setattr(mod, "PredictiveQueryV2", _FakePredictiveQuery)
    monkeypatch.setattr(mod, "GenerateTrainingTableValidator",
                        lambda **_kwargs: _FakeValidator())
    monkeypatch.setattr(mod, "hydrate_predictive_query_for_activity",
                        lambda *_args: (object(), {}))
    monkeypatch.setattr(mod, "open_connectors", lambda *_args: None)
    monkeypatch.setattr(mod, "save_job_storage_path_to_state_db",
                        lambda *_args: None)
    monkeypatch.setattr(mod, "get_active_cache",
                        lambda: SimpleNamespace(
                            load_all_table_caches=lambda *_args: None,
                            copy_pquery_table_stats_cache=lambda *_args: None,
                        ))
    monkeypatch.setattr(
        mod,
        "kumo_timing",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(mod.ValidationResponse, "save", lambda *_args: None)
    monkeypatch.setattr(mod.json, "save_json", lambda *_args: None)
    monkeypatch.setattr(
        materialize_training_task,
        "_mirror_ttng_split_stats_to_job_dir",
        lambda **_kwargs: None,
    )
    return mod, _FakeTrainingJobInput(), str(tmp_path / "job")


def test_large_table_producer_failure_returns_sanitized_not_ok_response(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    secret = "provider-token-must-not-escape"
    scale_warning = ValidationWarning("Large table", "Reduce its scale")
    mod, job_input, home_dir = _configure_activity(
        monkeypatch,
        tmp_path,
        scale_response=ValidationResponse(warnings=[scale_warning]),
        producer_error=RuntimeError(secret),
    )

    with caplog.at_level(logging.ERROR):
        path, response, returned_home = mod._generate_target_table(
            job_input,
            MagicMock(),
            home_dir=home_dir,
            diskgraph_index_path="/diskgraph/index",
        )

    assert path == ""
    assert returned_home == ""
    assert not response.ok
    assert response.warnings == [scale_warning]
    assert response.errors == [
        ValidationError(
            "Training table generation failed",
            "Kumo encountered an error while generating the training table.",
        )
    ]
    assert secret not in response.message()
    assert secret in caplog.text


def test_scale_warning_restricted_failure_requires_debug_for_raw_detail(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    canary = 'authorization-header-must-require-debug'
    scale_warning = ValidationWarning("Large table", "Reduce its scale")
    mod, job_input, home_dir = _configure_activity(
        monkeypatch,
        tmp_path,
        scale_response=ValidationResponse(warnings=[scale_warning]),
        producer_error=_RequestsBoundaryError(canary),
    )

    with caplog.at_level(logging.DEBUG, logger=mod.log.name):
        _, response, _ = mod._generate_target_table(
            job_input,
            MagicMock(),
            home_dir=home_dir,
            diskgraph_index_path="/diskgraph/index",
        )

    error_messages = '\n'.join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR)
    debug_records = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert canary not in error_messages
    assert debug_records
    assert canary in caplog.text
    assert canary not in response.message()


def test_no_scale_warning_producer_failure_keeps_public_error_contract(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    secret = "provider-token-must-not-escape"
    mod, job_input, home_dir = _configure_activity(
        monkeypatch,
        tmp_path,
        scale_response=None,
        producer_error=RuntimeError(secret),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(TrainingTableGenerationError) as exc_info:
            mod._generate_target_table(
                job_input,
                MagicMock(),
                home_dir=home_dir,
                diskgraph_index_path="/diskgraph/index",
            )

    assert "Kumo encountered an error while generating the training table" in (
        str(exc_info.value))
    assert secret not in str(exc_info.value)
    assert secret in caplog.text


def test_no_scale_warning_restricted_failure_requires_debug_for_raw_detail(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    canary = 'authorization-header-must-require-debug'
    mod, job_input, home_dir = _configure_activity(
        monkeypatch,
        tmp_path,
        scale_response=None,
        producer_error=_RequestsBoundaryError(canary),
    )

    with caplog.at_level(logging.DEBUG, logger=mod.log.name):
        with pytest.raises(TrainingTableGenerationError) as exc_info:
            mod._generate_target_table(
                job_input,
                MagicMock(),
                home_dir=home_dir,
                diskgraph_index_path="/diskgraph/index",
            )

    error_messages = '\n'.join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR)
    debug_records = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert canary not in error_messages
    assert debug_records
    assert canary in caplog.text
    assert canary not in str(exc_info.value)


def test_large_table_success_is_unchanged(monkeypatch, tmp_path) -> None:
    scale_warning = ValidationWarning("Large table", "Reduce its scale")
    success = ValidationResponse()
    mod, job_input, home_dir = _configure_activity(
        monkeypatch,
        tmp_path,
        scale_response=ValidationResponse(warnings=[scale_warning]),
        producer_result=("/durable/train-table", None, success),
    )

    path, response, returned_home = mod._generate_target_table(
        job_input,
        MagicMock(),
        home_dir=home_dir,
        diskgraph_index_path="/diskgraph/index",
    )

    assert path == "/durable/train-table"
    assert response.ok
    assert returned_home == home_dir


@pytest.mark.asyncio
async def test_not_ok_gen_tt_response_never_dispatches_empty_materialization(
    monkeypatch,
) -> None:
    from rest.backends.ml_backend.workflows import \
        generate_table_workflows as mod

    failure = ValidationResponse(errors=[
        ValidationError(
            "Training table generation failed",
            "Kumo encountered an error while generating the training table.",
        )
    ])
    workflow = mod.GenerateTrainingTableWorkflow()
    workflow._standalone_mode = False
    workflow.job_id = "gen-traintable-job-11111111111111111111111111111111"
    workflow._job_input = SimpleNamespace(
        id="trainingjob-11111111111111111111111111111111",
        generate_train_table_job_id=workflow.job_id,
        graph_data_compute_input=SimpleNamespace(
            graph_key="NA-22222222222222222222222222222222"),
        graph_version_info=SimpleNamespace(
            graph_data_version="33333333333333333333333333333333",
            table_data_versions={},
        ),
        spec=SimpleNamespace(
            predictive_query=SimpleNamespace(graph=SimpleNamespace(tables=[])),
            # A job that generates carries no supplied table.
            custom_train_table=None),
    )
    workflow._ml_job_settings = MagicMock()

    monkeypatch.setattr(mod, "DataBackendContext", lambda *_args: object())
    monkeypatch.setattr(mod, "deploy_env", object)
    monkeypatch.setattr(mod.workflow, "execute_activity", AsyncMock())
    monkeypatch.setattr(
        mod,
        "run_pooled_activity",
        AsyncMock(return_value=("", failure, "")),
    )
    materialize = AsyncMock()
    monkeypatch.setattr(workflow, "_materialize_training_table", materialize)
    monkeypatch.setattr(workflow, "_materialize_and_split_table", materialize)

    result = await workflow._generate_table()

    assert result.train_table_path == ""
    assert not result.validation.ok
    materialize.assert_not_awaited()
