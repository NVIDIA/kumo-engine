# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types.schema import FPID

from rest.config.graph_v2 import GraphSnapshotRequest
from rest.deps.context import _current_user_ctx
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.driver.graph_driver import GraphDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.table_driver import TableDriver
from statedb.config.common import (
    BatchPredictionJobID,
    PredictiveQueryID,
    ProjectID,
)
from statedb.utils.errors import PredictiveQueryDoesNotExistError

_PROJECT_A = ProjectID('project_' + 'a' * 32)
_PROJECT_B = ProjectID('project_' + 'b' * 32)
_FPID_A = FPID('a' * 32)
_FPID_B = FPID('b' * 32)


class _ProjectDenied(RuntimeError):
    pass


@pytest.fixture
def principal():
    value = MagicMock(name='member-principal')
    token = _current_user_ctx.set(value)
    try:
        yield value
    finally:
        _current_user_ctx.reset(token)


def _denied_projects():
    return SimpleNamespace(get_project=AsyncMock(side_effect=_ProjectDenied))


@pytest.mark.asyncio
async def test_table_id_alias_and_list_deny_before_repository_or_cache(
        principal):
    tables = SimpleNamespace(
        get_table_entry=AsyncMock(),
        get_table_entry_by_alias=AsyncMock(),
        list_tables=AsyncMock(),
    )
    driver = TableDriver(MagicMock(), MagicMock())
    projects = _denied_projects()
    driver._configure_state_services(
        SimpleNamespace(projects=projects, tables=tables))

    with pytest.raises(_ProjectDenied):
        await driver.get_table(_FPID_A, str(_PROJECT_B))
    with pytest.raises(_ProjectDenied):
        await driver.get_table_by_name('secret-table', str(_PROJECT_B))
    with pytest.raises(_ProjectDenied):
        await driver.list_tables(str(_PROJECT_B))

    tables.get_table_entry.assert_not_awaited()
    tables.get_table_entry_by_alias.assert_not_awaited()
    tables.list_tables.assert_not_awaited()
    assert projects.get_project.await_count == 3
    assert all(call.args == (_PROJECT_B, principal)
               for call in projects.get_project.await_args_list)


@pytest.mark.asyncio
async def test_graph_id_alias_and_list_deny_before_repository(principal):
    graphs = SimpleNamespace(
        get_graph_entry=AsyncMock(),
        get_graph_entry_by_alias=AsyncMock(),
        list_named_graphs=AsyncMock(),
    )
    driver = GraphDriver(MagicMock(), MagicMock(), MagicMock())
    projects = _denied_projects()
    driver._configure_state_services(
        SimpleNamespace(projects=projects, graphs=graphs))

    with pytest.raises(_ProjectDenied):
        await driver.get_graph_v2(_FPID_A, _PROJECT_B)
    with pytest.raises(_ProjectDenied):
        await driver.get_graph_v2_by_alias('secret-graph', _PROJECT_B)
    with pytest.raises(_ProjectDenied):
        await driver.list_graph_names_v2(_PROJECT_B)

    graphs.get_graph_entry.assert_not_awaited()
    graphs.get_graph_entry_by_alias.assert_not_awaited()
    graphs.list_named_graphs.assert_not_awaited()
    assert projects.get_project.await_count == 3


@pytest.mark.asyncio
async def test_snapshot_denies_before_snapshot_cache_lookup(principal):
    driver = DataSnapshotApiDriver(MagicMock())
    projects = _denied_projects()
    driver._configure_state_services(SimpleNamespace(projects=projects))
    driver._get_graph_snapshot_info = AsyncMock()

    with pytest.raises(_ProjectDenied):
        await driver.get_graph_snapshot(
            GraphSnapshotRequest(graph_snapshot_id='graph-snapshot-secret'),
            _PROJECT_B,
        )

    driver._get_graph_snapshot_info.assert_not_awaited()
    projects.get_project.assert_awaited_once_with(_PROJECT_B, principal)


def _pquery_driver(state_services):
    driver = PQueryApiDriver(
        ml_job_backend=SimpleNamespace(),
        data_backend_ctx=MagicMock(),
        allow_local_batch_pred_paths=True,
        data_scale_limits=MagicMock(),
        pquery_settings=MagicMock(),
        table_driver=MagicMock(),
    )
    driver._configure_state_services(state_services)
    return driver


@pytest.mark.asyncio
async def test_pquery_get_list_and_get_many_deny_before_global_indexes(
        principal):
    pqueries = SimpleNamespace(
        get_definition_by_id=AsyncMock(),
        list_aliases=AsyncMock(),
        get_definitions=AsyncMock(),
    )
    ml_jobs = SimpleNamespace(get_job_if_exists_generic=AsyncMock())
    projects = _denied_projects()
    driver = _pquery_driver(
        SimpleNamespace(projects=projects, pqueries=pqueries, ml_jobs=ml_jobs,
                        graphs=SimpleNamespace()))
    pquery_id = PredictiveQueryID.public(_FPID_A)

    with pytest.raises(_ProjectDenied):
        await driver.get_predictive_query_v2(pquery_id, _PROJECT_B)
    with pytest.raises(_ProjectDenied):
        await driver.list_predictive_queries_v2(_PROJECT_B)
    with pytest.raises(_ProjectDenied):
        await driver.get_many_predictive_queries_v2(
            {BatchPredictionJobID.make_random_id(): pquery_id},
            _PROJECT_B,
        )

    pqueries.get_definition_by_id.assert_not_awaited()
    pqueries.list_aliases.assert_not_awaited()
    pqueries.get_definitions.assert_not_awaited()
    ml_jobs.get_job_if_exists_generic.assert_not_awaited()
    assert projects.get_project.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('job_project,job_fpid', [
    (_PROJECT_B, _FPID_A),
    (_PROJECT_A, _FPID_B),
])
async def test_get_many_rejects_cross_project_or_forged_job_mapping_before_defs(
        principal, job_project, job_fpid):
    job_id = BatchPredictionJobID.make_random_id()
    claimed = PredictiveQueryID.public(_FPID_A)
    projects = SimpleNamespace(get_project=AsyncMock())
    pqueries = SimpleNamespace(get_definitions=AsyncMock())
    ml_jobs = SimpleNamespace(get_job_if_exists_generic=AsyncMock(
        return_value=SimpleNamespace(
            project_id=job_project,
            config=SimpleNamespace(
                predictive_query_id=PredictiveQueryID.public(job_fpid)),
        )))
    driver = _pquery_driver(
        SimpleNamespace(projects=projects, pqueries=pqueries, ml_jobs=ml_jobs,
                        graphs=SimpleNamespace()))

    with pytest.raises(PredictiveQueryDoesNotExistError):
        await driver.get_many_predictive_queries_v2(
            {job_id: claimed},
            _PROJECT_A,
        )

    projects.get_project.assert_awaited_once_with(_PROJECT_A, principal)
    ml_jobs.get_job_if_exists_generic.assert_awaited_once_with(job_id)
    pqueries.get_definitions.assert_not_awaited()
