# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types.schema import FPID

from kumo.validation.validation import ValidationResponse
from rest.driver import validators as validators_module
from rest.driver.validators import (
    BatchPredictionValidator,
    GraphValidator,
    TableValidator,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GraphIDv2,
    ProjectID,
    TableIDv2,
    TableName,
)
from statedb.utils.errors import (
    BatchPredictionDoesNotExistError,
    TableDoesNotExistError,
)

_PROJECT_ID = ProjectID("project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture(scope="session")
def setup_rest_test_session() -> None:
    return None


def _table_id(fpid: FPID) -> TableIDv2:
    return TableIDv2.public(
        table_name=TableName("customers"),
        fpid=fpid,
        pquery_id=None,
    )


def _table_id_by_alias() -> TableIDv2:
    return TableIDv2("NA.customers-NA")


def _validator(table_service) -> TableValidator:
    return TableValidator(
        data_backend_ctx=MagicMock(),
        job_base_dir="/tmp/jobs",
        state_services=SimpleNamespace(tables=table_service),
        project_id=_PROJECT_ID,
    )


def _batch_prediction_validator(ml_job_service) -> BatchPredictionValidator:
    return BatchPredictionValidator(
        data_backend_ctx=MagicMock(),
        job_base_dir="/tmp/jobs",
        ml_job_backend=MagicMock(),
        ctx=MagicMock(),
        state_services=SimpleNamespace(ml_jobs=ml_job_service),
    )


def _graph_validator(state_services) -> GraphValidator:
    return GraphValidator(
        data_backend_ctx=MagicMock(),
        job_base_dir="/tmp/jobs",
        state_services=state_services,
        project_id=_PROJECT_ID,
    )


@pytest.mark.asyncio
async def test_table_validator_get_warnings_checks_v2_ids_with_state_services():
    fpid = FPID("a" * 32)
    table_id = _table_id(fpid)
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={fpid: object()}))
    validator = _validator(table_service)
    validator.run_batch_validators = AsyncMock(return_value=[[]])

    response = await validator.get_warnings([table_id])

    table_service.get_table_entries.assert_awaited_once_with(
        _PROJECT_ID, [fpid])
    validator.run_batch_validators.assert_awaited_once_with([table_id])
    assert response[table_id].errors == []
    assert response[table_id].warnings == []
    assert response[table_id].info_items == []


@pytest.mark.asyncio
async def test_table_validator_checks_alias_id_with_state_services():
    fpid = FPID("c" * 32)
    table_id = _table_id_by_alias()
    table_entry = SimpleNamespace(id=fpid)
    table_service = SimpleNamespace(
        get_table_entry_by_alias=AsyncMock(
            side_effect=AssertionError("scalar alias lookup used")),
        get_table_entries_by_aliases=AsyncMock(
            return_value={
                TableName("customers"): table_entry,
            }),
        get_table_entries=AsyncMock(return_value={fpid: table_entry}),
    )
    validator = _validator(table_service)
    validator.run_batch_validators = AsyncMock(return_value=[[]])

    response = await validator.get_warnings([table_id])

    table_service.get_table_entries_by_aliases.assert_awaited_once()
    aliases = table_service.get_table_entries_by_aliases.await_args.args[1]
    assert table_service.get_table_entries_by_aliases.await_args.args[0] == (
        _PROJECT_ID)
    assert list(aliases) == [TableName("customers")]
    table_service.get_table_entry_by_alias.assert_not_awaited()
    table_service.get_table_entries.assert_awaited_once_with(
        _PROJECT_ID, [fpid])
    validator.run_batch_validators.assert_awaited_once_with([table_id])
    assert response[table_id].errors == []


@pytest.mark.asyncio
async def test_table_validator_state_services_missing_v2_id_raises():
    fpid = FPID("b" * 32)
    table_id = _table_id(fpid)
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={}))
    validator = _validator(table_service)
    validator.run_batch_validators = AsyncMock()

    with pytest.raises(TableDoesNotExistError):
        await validator.get_warnings([table_id])

    table_service.get_table_entries.assert_awaited_once_with(
        _PROJECT_ID, [fpid])
    validator.run_batch_validators.assert_not_called()


@pytest.mark.asyncio
async def test_graph_validator_uses_batched_graph_logic(monkeypatch):
    graph_ids = [
        GraphIDv2.compose(None, FPID("a" * 32)),
        GraphIDv2.compose(None, FPID("b" * 32)),
    ]
    state_services = SimpleNamespace()
    graph_logic_calls = []

    class FakeGraphLogic:
        def __init__(self, data_backend_ctx, *, state_services=None,
                     project_id=None):
            self.data_backend_ctx = data_backend_ctx
            self.state_services = state_services
            self.project_id = project_id

        async def get_many_data_compute_inputs(self, ids):
            graph_logic_calls.append(
                (self.state_services, self.project_id, list(ids)))
            return [
                SimpleNamespace(graph_key="graph-a", tables={}),
                SimpleNamespace(graph_key="graph-b", tables={}),
            ]

    get_many_graph_validations = SimpleNamespace(coroutine=AsyncMock(
        return_value={
            "graph-a": ValidationResponse(),
            "graph-b": ValidationResponse(),
        }))

    monkeypatch.setattr(validators_module, "GraphLogic", FakeGraphLogic)
    monkeypatch.setattr(
        validators_module.DataBackend,
        "get_many_graph_validations",
        get_many_graph_validations,
    )

    result = await _graph_validator(state_services).graph_has_errors_warnings(
        graph_ids)

    assert result == [[], []]
    assert graph_logic_calls == [(state_services, _PROJECT_ID, graph_ids)]
    get_many_graph_validations.coroutine.assert_awaited_once_with(
        ["graph-a", "graph-b"])


@pytest.mark.asyncio
async def test_batch_prediction_validator_checks_ids_with_state_services():
    bp_id = BatchPredictionJobID("bp-1")
    ml_job_service = SimpleNamespace(get_batch_prediction_rows=AsyncMock(
        return_value=[SimpleNamespace(id=bp_id)]))
    validator = _batch_prediction_validator(ml_job_service)
    validator.run_batch_validators = AsyncMock(return_value=[[]])

    response = await validator.get_warnings([bp_id])

    ml_job_service.get_batch_prediction_rows.assert_awaited_once_with(
        [bp_id], project_ids=None)
    validator.run_batch_validators.assert_awaited_once_with([bp_id])
    assert response[bp_id].errors == []
    assert response[bp_id].warnings == []
    assert response[bp_id].info_items == []


@pytest.mark.asyncio
async def test_batch_prediction_validator_state_services_missing_id_raises():
    bp_id = BatchPredictionJobID("bp-1")
    ml_job_service = SimpleNamespace(get_batch_prediction_rows=AsyncMock(
        return_value=[]))
    validator = _batch_prediction_validator(ml_job_service)
    validator.run_batch_validators = AsyncMock()

    with pytest.raises(BatchPredictionDoesNotExistError):
        await validator.get_warnings([bp_id])

    ml_job_service.get_batch_prediction_rows.assert_awaited_once_with(
        [bp_id], project_ids=None)
    validator.run_batch_validators.assert_not_called()
