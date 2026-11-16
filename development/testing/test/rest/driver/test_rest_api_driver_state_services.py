# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kumo_core.types.schema import FPID
from kumoapi.train import TrainingStage
from kumoapi.typing import Dtype, Stype

from kumo.config.explain import StrategyMappings
from rest.config.file_type import FileType
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.predictive_query import PQueryDefinition
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDefinition,
)
from rest.driver.batch_prediction_job_logic import BatchPredictionJobLogic
from rest.driver.pquery_logic import PQueryLogic
from rest.driver.rest_api_driver import RestApiDriver
from statedb.config.common import (
    BatchPredictionJobID,
    ConnectorID,
    PredictiveQueryID,
    ProjectID,
    TableName,
    TrainingJobID,
)
from statedb.config.timestamp import get_current_timestamps
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS

_PROJECT_ID = ProjectID("project_" + "1" * 32)
_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


class _ConfigurableDriver:
    def __init__(self) -> None:
        self.configured_with = None

    def _configure_state_services(self, state_services) -> None:
        self.configured_with = state_services


def _table_definition() -> TableDefinition:
    return TableDefinition(
        source_table=S3SourceTable(
            source={
                'mode': 'relative',
                'path': "customers"
            },
            connector_id=_CONNECTOR_ID,
            file_type=FileType.CSV,  # type: ignore[arg-type]
            data_source_type=DataSourceType.S3,
        ),
        cols=[Column(name="customer_id", stype=Stype.ID, dtype=Dtype.int64)],
        pkey="customer_id",
    )


def _rest_api_driver_with_children(
    *,
    table_driver=None,
    graph_driver=None,
):
    driver = RestApiDriver.__new__(RestApiDriver)
    driver._storage_dao = None
    driver._data_backend_ctx = MagicMock()
    driver._connector_api_driver = _ConfigurableDriver()
    driver._table_driver = table_driver or _ConfigurableDriver()
    driver._graph_driver = graph_driver or _ConfigurableDriver()
    driver._pquery_api_driver = _ConfigurableDriver()
    driver._data_snapshot_api_driver = _ConfigurableDriver()
    driver._artifact_export_api_driver = _ConfigurableDriver()
    return driver


def _artifact_state_services():
    return SimpleNamespace(
        projects=SimpleNamespace(get_project=AsyncMock(
            return_value=SimpleNamespace(id=_PROJECT_ID))), )


def _job():
    return SimpleNamespace(project_id=_PROJECT_ID)


def test_rest_api_driver_configures_state_services_for_child_drivers():
    state_services = _artifact_state_services()
    driver = RestApiDriver.__new__(RestApiDriver)
    connector_driver = _ConfigurableDriver()
    table_driver = _ConfigurableDriver()
    pquery_driver = _ConfigurableDriver()
    snapshot_driver = _ConfigurableDriver()
    artifact_driver = _ConfigurableDriver()
    graph_driver = _ConfigurableDriver()
    driver._connector_api_driver = connector_driver
    driver._table_driver = table_driver
    driver._graph_driver = graph_driver
    driver._pquery_api_driver = pquery_driver
    driver._data_snapshot_api_driver = snapshot_driver
    driver._artifact_export_api_driver = artifact_driver

    driver._configure_state_services(state_services)

    assert connector_driver.configured_with is state_services
    assert table_driver.configured_with is state_services
    assert graph_driver.configured_with is state_services
    assert pquery_driver.configured_with is state_services
    assert snapshot_driver.configured_with is state_services
    assert artifact_driver.configured_with is state_services


@pytest.mark.asyncio
async def test_get_bp_available_xai_entities_uses_state_services_without_legacy_storage(
):
    expected_response = StrategyMappings(mappings={"top": ["entity-1"]})
    batch_prediction_job_id = BatchPredictionJobID("bp-job")
    state_services = _artifact_state_services()
    driver = _rest_api_driver_with_children()
    driver._ml_job_backend = SimpleNamespace(
        get_batch_prediction_job=AsyncMock(return_value=_job()),
        get_job_home_dir=AsyncMock(return_value="/tmp/bp_job"),
    )
    driver._configure_state_services(state_services)
    captured = {}

    async def get_available_xai_entities(self, home_mgr):
        captured["state_services"] = self._state_services
        captured["home_dir"] = home_mgr.home_dir
        return expected_response

    with patch.object(
            BatchPredictionJobLogic,
            "get_available_xai_entities",
            get_available_xai_entities,
    ):
        result = await driver.get_bp_available_xai_entities(
            MagicMock(), batch_prediction_job_id)

    assert result is expected_response
    assert captured == {
        "state_services": state_services,
        "home_dir": "/tmp/bp_job",
    }
    driver._ml_job_backend.get_job_home_dir.assert_awaited_once_with(
        batch_prediction_job_id)


@pytest.mark.asyncio
async def test_get_available_xai_entities_uses_state_services_without_legacy_storage(
):
    expected_response = StrategyMappings(mappings={"top": ["entity-1"]})
    training_job_id = TrainingJobID("training-job")
    state_services = _artifact_state_services()
    driver = _rest_api_driver_with_children()
    driver._ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=_job()),
        get_job_home_dir=AsyncMock(return_value="/tmp/training_job"),
    )
    driver._configure_state_services(state_services)
    captured = {}

    async def get_available_xai_entities(self, home_mgr, stage):
        captured["state_services"] = self._state_services
        captured["home_dir"] = home_mgr.home_dir
        captured["stage"] = stage
        return expected_response

    with patch.object(
            PQueryLogic,
            "get_available_xai_entities",
            get_available_xai_entities,
    ):
        result = await driver.get_available_xai_entities(
            MagicMock(), None, training_job_id)

    assert result is expected_response
    assert captured == {
        "state_services": state_services,
        "home_dir": "/tmp/training_job",
        "stage": TrainingStage.TEST,
    }
    driver._ml_job_backend.get_job_home_dir.assert_awaited_once_with(
        training_job_id)


@pytest.mark.asyncio
async def test_get_entity_output_uses_state_services_without_legacy_storage():
    expected_response = object()
    training_job_id = TrainingJobID("training-job")
    state_services = _artifact_state_services()
    driver = _rest_api_driver_with_children()
    driver._ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=_job()),
        get_job_home_dir=AsyncMock(return_value="/tmp/training_job"),
    )
    driver._configure_state_services(state_services)
    captured = {}

    async def get_entity_output(self, home_mgr, entity_id, stage):
        captured["state_services"] = self._state_services
        captured["home_dir"] = home_mgr.home_dir
        captured["entity_id"] = entity_id
        captured["stage"] = stage
        return expected_response

    with patch.object(PQueryLogic, "get_entity_output", get_entity_output):
        result = await driver.get_entity_output(
            MagicMock(),
            None,
            "customer-1",
            job_id=training_job_id,
        )

    assert result is expected_response
    assert captured == {
        "state_services": state_services,
        "home_dir": "/tmp/training_job",
        "entity_id": "customer-1",
        "stage": TrainingStage.TEST,
    }
    driver._ml_job_backend.get_job_home_dir.assert_awaited_once_with(
        training_job_id)


@pytest.mark.asyncio
async def test_get_regression_histogram_uses_state_services_without_legacy_storage(
):
    expected_response = object()
    training_job_id = TrainingJobID("training-job")
    state_services = _artifact_state_services()
    driver = _rest_api_driver_with_children()
    driver._configure_state_services(state_services)
    driver._ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=_job()),
        get_job_home_dir=AsyncMock(return_value="/tmp/training_job"),
    )
    captured = {}

    async def get_regression_histogram(self, home_mgr):
        captured["state_services"] = self._state_services
        captured["home_dir"] = home_mgr.home_dir
        return expected_response

    with patch.object(
            PQueryLogic,
            "get_regression_histogram",
            get_regression_histogram,
    ):
        result = await driver.get_regression_histogram(MagicMock(), None,
                                                       training_job_id)

    assert result is expected_response
    assert captured == {
        "state_services": state_services,
        "home_dir": "/tmp/training_job",
    }


@pytest.mark.asyncio
async def test_dump_as_fixtures_v2_lists_all_pqueries_from_state_services_without_legacy_storage(
):
    project_id = _PROJECT_ID
    principal = MagicMock()
    table_definition = _table_definition()
    graph_fpid = FPID("1" * 32)
    table_name = TableName("customers")
    graph_entry = GraphDBEntryV2(
        id=graph_fpid,
        project_id=project_id,
        table_name_to_dbid={table_name: table_definition.fpid},
        col_groups=[],
        timestamps=get_current_timestamps(),
        alias="customers_graph",
    )
    pquery_defs = [
        PQueryDefinition(
            timestamps=get_current_timestamps(),
            pquery_name="pq_one",
            query_str=(
                "PREDICT COUNT(customers.*) FOR EACH customers.customer_id"),
            graph_fpid=graph_fpid,
            list_distinct_target_is_array=True,
        ),
        PQueryDefinition(
            timestamps=get_current_timestamps(),
            pquery_name="pq_two",
            query_str=(
                "PREDICT COUNT(customers.*) FOR EACH customers.customer_id"),
            graph_fpid=graph_fpid,
            lp_bp_layout="source-target",
        ),
    ]
    pquery_service = SimpleNamespace(
        list_definitions=AsyncMock(return_value=pquery_defs),
        get_definition_by_id=AsyncMock(side_effect=AssertionError(
            "explicit pquery lookup should not run")),
    )
    graph_service = SimpleNamespace(
        get_graph_entry=AsyncMock(return_value=graph_entry),
        get_graph_entry_by_alias=AsyncMock(return_value=None),
    )
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={
            table_definition.fpid: SimpleNamespace(config=table_definition),
        }))
    connector_service = SimpleNamespace(
        connector_ids_for_graph=MagicMock(
            return_value={table_definition.get_connector_id()}),
        execution_ref_for_project=AsyncMock(),
    )
    state_services = SimpleNamespace(
        pqueries=pquery_service,
        graphs=graph_service,
        tables=table_service,
        connectors=connector_service,
        projects=SimpleNamespace(get_project=AsyncMock()),
    )
    driver = _rest_api_driver_with_children()
    driver._configure_state_services(state_services)

    fixture = await driver.dump_as_fixtures_v2(
        None,
        project_id,
        principal,
    )

    pquery_service.list_definitions.assert_awaited_once_with()
    pquery_service.get_definition_by_id.assert_not_awaited()
    graph_service.get_graph_entry.assert_awaited_once_with(
        project_id, graph_fpid)
    table_service.get_table_entries.assert_awaited()
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        principal,
        project_id,
        table_definition.get_connector_id(),
        CONNECTOR_GRAPH_ACTIONS,
    )
    assert fixture.tables == [table_definition]
    assert len(fixture.graphs) == 1
    assert fixture.graphs[0].tables == {table_name: table_definition}
    assert [query.pquery_name for query in fixture.queries] == [
        "pq_one",
        "pq_two",
    ]
    assert fixture.queries[0].list_distinct_target_is_array is True
    assert fixture.queries[1].lp_bp_layout == "source-target"


@pytest.mark.asyncio
async def test_dump_as_fixtures_v2_explicit_ids_bulk_loads_pqueries_from_state_services(
):
    project_id = _PROJECT_ID
    principal = MagicMock()
    table_definition = _table_definition()
    graph_fpid = FPID("1" * 32)
    table_name = TableName("customers")
    graph_entry = GraphDBEntryV2(
        id=graph_fpid,
        project_id=project_id,
        table_name_to_dbid={table_name: table_definition.fpid},
        col_groups=[],
        timestamps=get_current_timestamps(),
        alias="customers_graph",
    )
    pquery_def = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="pq_one",
        query_str="PREDICT COUNT(customers.*) FOR EACH customers.customer_id",
        graph_fpid=graph_fpid,
    )
    pquery_id = PredictiveQueryID("pq_one")
    pquery_service = SimpleNamespace(
        list_definitions=AsyncMock(
            side_effect=AssertionError("list definitions should not run")),
        get_definition_by_id=AsyncMock(
            side_effect=AssertionError("scalar pquery lookup used")),
        get_definitions_by_ids=AsyncMock(return_value={
            pquery_id: pquery_def,
        }),
    )
    graph_service = SimpleNamespace(
        get_graph_entry=AsyncMock(return_value=graph_entry),
        get_graph_entry_by_alias=AsyncMock(return_value=None),
    )
    table_service = SimpleNamespace(get_table_entries=AsyncMock(
        return_value={
            table_definition.fpid: SimpleNamespace(config=table_definition),
        }))
    connector_service = SimpleNamespace(
        connector_ids_for_graph=MagicMock(
            return_value={table_definition.get_connector_id()}),
        execution_ref_for_project=AsyncMock(),
    )
    state_services = SimpleNamespace(
        pqueries=pquery_service,
        graphs=graph_service,
        tables=table_service,
        connectors=connector_service,
        projects=SimpleNamespace(get_project=AsyncMock()),
    )
    driver = _rest_api_driver_with_children()
    driver._configure_state_services(state_services)

    fixture = await driver.dump_as_fixtures_v2(
        [pquery_id],
        project_id,
        principal,
    )

    pquery_service.get_definitions_by_ids.assert_awaited_once_with([pquery_id])
    pquery_service.get_definition_by_id.assert_not_awaited()
    connector_service.execution_ref_for_project.assert_awaited_once_with(
        principal,
        project_id,
        table_definition.get_connector_id(),
        CONNECTOR_GRAPH_ACTIONS,
    )
    assert [query.pquery_name for query in fixture.queries] == ["pq_one"]


@pytest.mark.asyncio
async def test_dump_as_fixtures_v2_explicit_ids_errors_when_bulk_lookup_misses(
):
    pquery_id = PredictiveQueryID("pq_one")
    pquery_service = SimpleNamespace(
        list_definitions=AsyncMock(
            side_effect=AssertionError("list definitions should not run")),
        get_definition_by_id=AsyncMock(
            side_effect=AssertionError("scalar pquery lookup used")),
        get_definitions_by_ids=AsyncMock(return_value={}),
    )
    driver = _rest_api_driver_with_children()
    driver._configure_state_services(
        SimpleNamespace(
            pqueries=pquery_service,
            projects=SimpleNamespace(get_project=AsyncMock()),
        ))

    with pytest.raises(ValueError,
                       match=f"Predictive query id {pquery_id} is not found"):
        await driver.dump_as_fixtures_v2(
            [pquery_id],
            _PROJECT_ID,
            MagicMock(),
        )
    pquery_service.get_definitions_by_ids.assert_awaited_once_with([pquery_id])
    pquery_service.list_definitions.assert_not_awaited()
    pquery_service.get_definition_by_id.assert_not_awaited()
