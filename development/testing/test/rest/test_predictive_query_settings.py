# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from typing import Callable, Generator

import pytest
from fastapi.testclient import TestClient
from kumoapi import jobs as jobs_api
from kumoapi.pquery import PQueryResource

from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from rest.app import AppBuilder
from rest.config.graph_v2 import GraphResourceResponse
from rest.config.predictive_query_v2 import PQueryValidationRequest
from rest.config.status import IDResponse
from rest.settings import PredictiveQuerySettings
from rest.testing.config import TestFunctionCleanupOptions
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import GraphID


@pytest.fixture
def hm_graph_id(
    client: TestClient,
    storage_dao,
    graph_definition_v2,
    default_project_id: str,
) -> Generator[GraphID, None, None]:
    response = post_and_expect_success(
        client,
        f"/v1/graphs/?project_id={default_project_id}",
        graph_definition_v2,
        IDResponse,
    )
    graph_id = get_and_expect_success(
        client,
        f"/v1/graphs/{response.id}",
        {
            'project_id': default_project_id
        },
        GraphResourceResponse,
    ).id
    yield graph_id
    # Reset StateDB upon teardown of test module.
    asyncio.run(storage_dao.reset_storage())


@pytest.fixture(scope="function")
def test_cleanup_options() -> TestFunctionCleanupOptions:
    return TestFunctionCleanupOptions(should_reset_rest_db=False)


@pytest.fixture
def apply_predictive_query_setting(deploy_env_config: DeployEnvConfig,
                                   client: TestClient, monkeypatch):
    app_builder: AppBuilder = getattr(client, 'app_builder')
    pquery_driver = app_builder._driver.pquery_driver
    # Snapshot so settings mutated by one test don't leak into the next
    # (the driver is shared across the test session).
    original_settings = pquery_driver._pquery_settings

    def _apply(key: str, value: bool):
        monkeypatch.setenv(key, str(value))

        predictive_query_settings = PredictiveQuerySettings.parse_from_env(
            deploy_env_config)
        predictive_query_settings.__dict__.update({key: value})
        pquery_driver._pquery_settings = predictive_query_settings

    yield _apply
    pquery_driver._pquery_settings = original_settings


@dataclass(frozen=True)
class ProjectGraphDefinition:
    graph: object
    project_id: str


@pytest.fixture
def hm_graph_definition(client: TestClient, hm_graph_id: GraphID,
                        default_project_id: str) -> ProjectGraphDefinition:
    """The full graph definition for the hm fixture graph.

    The v1 ``/validate`` route validates against a graph *definition* (the
    graph object itself), not a ``graph_id``, so we fetch it once here.
    """
    response = get_and_expect_success(
        client,
        f"/v1/graphs/{hm_graph_id}",
        {'project_id': default_project_id},
        GraphResourceResponse,
    )
    return ProjectGraphDefinition(response.graph, default_project_id)


def _query_errors(client: TestClient, graph_definition: ProjectGraphDefinition,
                  query_yaml: str) -> list:
    """Validate a query against the v1 stack, returning its error messages.

    Replaces the legacy ``POST /v0/predictive_queries/`` create-and-assert
    pattern: ``/v1/predictive_queries/validate`` is a read-only check that
    surfaces setting violations (e.g. disabled task types) as ``ERROR`` items
    rather than as an HTTP error. Only ``ERROR``-type items are returned
    (``INFO``/warnings are ignored, mirroring v0 create which only failed on
    errors).
    """
    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={graph_definition.project_id}',
        PQueryValidationRequest(
            pquery_resource=PQueryResource(
                name='pq',
                query_string=query_yaml,
                graph=graph_definition.graph,
                desc='',
            ),
            model_plan='',
        ),
        jobs_api.ErrorDetails,
    )
    return [
        item.description for item in response.items
        if item.type == jobs_api.ErrorType.ERROR
    ]


@pytest.mark.parametrize("disable_temporal_link_prediction", [True, False])
def test_temporal_link_pred(
    client: TestClient,
    hm_graph_definition,
    apply_predictive_query_setting: Callable[[str, bool], None],
    disable_temporal_link_prediction: bool,
):
    query_yaml = """
PREDICT LIST_DISTINCT(transactions.article_id, 0, 180, days) RANK TOP 1
FOR EACH customers.customer_id WHERE SUM(transactions.price, -90, 0, days) > 1
    """

    apply_predictive_query_setting('DISABLE_TEMPORAL_LINK_PREDICTION',
                                   disable_temporal_link_prediction)

    errors = _query_errors(client, hm_graph_definition, query_yaml)
    if disable_temporal_link_prediction:
        assert any("not supported" in e for e in errors), errors
    else:
        assert errors == [], errors


@pytest.mark.parametrize("disable_static_node_prediction", [True, False])
def test_static_node_pred(
    client: TestClient,
    hm_graph_definition,
    apply_predictive_query_setting: Callable[[str, bool], None],
    disable_static_node_prediction: bool,
):
    query_yaml = """
    PREDICT customers.club_member_status
    FOR EACH customers.customer_id
    """

    apply_predictive_query_setting('DISABLE_STATIC_NODE_PREDICTION',
                                   disable_static_node_prediction)

    errors = _query_errors(client, hm_graph_definition, query_yaml)
    if disable_static_node_prediction:
        assert any("not supported" in e for e in errors), errors
    else:
        assert errors == [], errors


@pytest.mark.parametrize("disable_first_last", [True, False])
def test_first_last(
    client: TestClient,
    hm_graph_definition,
    apply_predictive_query_setting: Callable[[str, bool], None],
    disable_first_last: bool,
):
    first_on_non_fkey_yaml = """
    PREDICT FIRST(transactions.price, 0, 90, days)
    FOR EACH customers.customer_id
    """
    last_on_non_fkey_yaml = """
    PREDICT LAST(transactions.price, 0, 90, days)
    FOR EACH customers.customer_id
    """

    apply_predictive_query_setting('DISABLE_FIRST_LAST', disable_first_last)

    first_errors = _query_errors(client, hm_graph_definition,
                                 first_on_non_fkey_yaml)
    last_errors = _query_errors(client, hm_graph_definition,
                                last_on_non_fkey_yaml)
    if disable_first_last:
        assert any("not supported" in e for e in first_errors), first_errors
        assert any("not supported" in e for e in last_errors), last_errors
    else:
        assert first_errors == [], first_errors
        assert last_errors == [], last_errors


def test_pqlv2(
    client: TestClient,
    hm_graph_definition,
):
    query_yaml = """
    PREDICT FIRST(transactions.price, 0, 90, days)
    FOR EACH customers.customer_id
    """

    assert _query_errors(client, hm_graph_definition, query_yaml) == []


@pytest.mark.parametrize("disable_first_last_on_fkey", [True, False])
def test_first_last_on_fkey(
    client: TestClient,
    hm_graph_definition,
    apply_predictive_query_setting: Callable[[str, bool], None],
    disable_first_last_on_fkey: bool,
):
    first_on_fkey_yaml = """
    PREDICT FIRST(transactions.customer_id, 0, 30, days)
    FOR EACH articles.article_id
    """
    first_on_non_fkey_yaml = """
    PREDICT FIRST(transactions.price, 0, 90, days)
    FOR EACH customers.customer_id
    """
    last_on_fkey_yaml = """
    PREDICT LAST(transactions.article_id, 0, 30, days)
    FOR EACH customers.customer_id
    """
    last_on_non_fkey_yaml = """
    PREDICT LAST(transactions.price, 0, 90, days)
    FOR EACH customers.customer_id
    """

    apply_predictive_query_setting('DISABLE_FIRST_LAST_ON_FKEY',
                                   disable_first_last_on_fkey)

    if disable_first_last_on_fkey:
        assert any("not supported" in e for e in _query_errors(
            client, hm_graph_definition, first_on_fkey_yaml))
        assert _query_errors(client, hm_graph_definition,
                             first_on_non_fkey_yaml) == []
        assert any("not supported" in e for e in _query_errors(
            client, hm_graph_definition, last_on_fkey_yaml))
        assert _query_errors(client, hm_graph_definition,
                             last_on_non_fkey_yaml) == []
    # TODO(jinu): Uncomment this once we support first/last on fkey
    # correctly.
    """
    else:
        assert _query_errors(
            client, hm_graph_definition, first_on_fkey_yaml) == []
        assert _query_errors(
            client, hm_graph_definition, first_on_non_fkey_yaml) == []
        assert _query_errors(
            client, hm_graph_definition, last_on_fkey_yaml) == []
        assert _query_errors(
            client, hm_graph_definition, last_on_non_fkey_yaml) == []
    """


@pytest.mark.parametrize("disable_temporal_node_regression", [True, False])
def test_temporal_node_regression(
    client: TestClient,
    hm_graph_definition,
    apply_predictive_query_setting: Callable[[str, bool], None],
    disable_temporal_node_regression: bool,
):
    query_yaml = """
    PREDICT SUM(transactions.price, 0, 90, days)
    FOR EACH customers.customer_id
    """

    apply_predictive_query_setting('DISABLE_TEMPORAL_NODE_REGRESSION',
                                   disable_temporal_node_regression)

    errors = _query_errors(client, hm_graph_definition, query_yaml)
    if disable_temporal_node_regression:
        assert any("not supported" in e for e in errors), errors
    else:
        assert errors == [], errors


@pytest.mark.parametrize("disable", [True, False])
def test_multilabel_classification(
    client: TestClient,
    hm_graph_definition,
    apply_predictive_query_setting: Callable[[str, bool], None],
    disable: bool,
):
    query_yaml = """
    PREDICT LIST_DISTINCT(transactions.t_dat, 0, 30) CLASSIFY
    FOR EACH customers.customer_id
    """

    apply_predictive_query_setting('DISABLE_MULTILABEL_CLASSIFICATION', disable)

    errors = _query_errors(client, hm_graph_definition, query_yaml)
    if disable:
        assert any("not supported" in e for e in errors), errors
    else:
        assert errors == [], errors
