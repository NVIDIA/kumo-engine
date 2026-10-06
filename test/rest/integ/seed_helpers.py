# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared DG-native seed helpers for the REST integ suite.

These factor the working setup flow out of
``test/rest/integ/test_diskgraph_e2e.py::_setup_and_train`` so that the other
rest/integ e2e tests can drive the SAME real, DiskGraph-backed pipeline:
connector create -> v2 predictive-query register -> (graph refresh via the
training-table flow OR a standalone training job).

Two entry points:

* :func:`seed_connectors_and_pquery` -- POSTs every connector in the fixture
  and registers the requested v2 predictive query (and thus its v2 graph).
  This is the "ingest / graph-refresh only" half: it does not launch a
  training job. Callers that only need DG-ingested column statistics (e.g.
  PK-warning validation) drive a graph/table refresh themselves after this.

* :func:`seed_and_train` -- everything :func:`seed_connectors_and_pquery`
  does, plus fetching a default model plan (when the fixture query carries
  none), clamping the e2e training budget, and POSTing a training job.
  Returns ``(pquery_id, training_job_id)``.

Both run the real ``GraphRefreshWorkflow`` -> DiskGraph ingest -> index path
through Temporal; they require the diskgraph pool/Temporal fixtures the
calling module already activates (``_pool_subprocesses`` etc.).
"""
from __future__ import annotations

from test.rest.integ.fixture_connector_args import provision_fixture_connectors
from test.rest.integ.training_budget import clamp_e2e_training_budget
from typing import Optional, Tuple

from fastapi.testclient import TestClient
from kumoapi.model_plan import RunMode, SuggestModelPlanRequest

from kumo.config.model_plan_yaml import model_plan_from_yaml
from kumo.testing.rest import post_and_expect_success
from rest.config.api_command import FixtureDefinition
from rest.config.predictive_query_v2 import ModelPlanResponse
from rest.config.status import EmptyResponse
from rest.config.training_job_v2 import TrainingJobRequest
from rest.routers.fixture_utils import fixture_to_pquery_resources_v2
from statedb.config.common import (
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)


async def seed_connectors_and_pquery(
    client: TestClient,
    fixture: FixtureDefinition,
    pquery_name: str,
    *,
    provision_connectors: bool = True,
    project_id: str | None = None,
):
    """POST every connector in ``fixture`` and register the v2 predictive
    query named ``pquery_name`` (which also materializes its v2 graph in the
    DB).

    This is the ingest/graph-refresh half of the seed: it does NOT launch a
    training job. Returns ``(pquery_id, pquery_resource, project_id)`` so
    callers can reach the registered query and its graph while preserving
    the project that owns every downstream operation.
    """
    if provision_connectors:
        typed_project_id, _ = await provision_fixture_connectors(
            client, fixture, project_id)
    else:
        if project_id is None:
            raise ValueError('project_id is required when connectors exist')
        typed_project_id = ProjectID(project_id)
    project_id = str(typed_project_id)

    pquery_resources = await fixture_to_pquery_resources_v2.coroutine(fixture)
    target = next((r for r in pquery_resources if r.name == pquery_name), None)
    assert target is not None, f"Query {pquery_name} not found in fixture"

    # Register the predictive query so its v2 graph exists before downstream
    # graph and job operations.
    post_and_expect_success(
        client,
        '/v1/predictive_queries/?dialect=PQLANG_V2',
        target,
        EmptyResponse,
        project_id=project_id,
    )
    return PredictiveQueryID(target.name), target, project_id


async def seed_and_train(
    client: TestClient,
    fixture: FixtureDefinition,
    pquery_name: str,
    run_mode: RunMode = RunMode.DEBUG,
    manual_seed: Optional[int] = 42,
    *,
    provision_connectors: bool = True,
    project_id: str | None = None,
) -> Tuple[PredictiveQueryID, TrainingJobID]:
    """Create connectors/pquery and kick off a real DiskGraph training job.

    Factored from ``test_diskgraph_e2e._setup_and_train``. When the fixture
    query carries no model plan, a default plan is fetched from
    ``/v1/predictive_queries/model_plan`` and the e2e training budget
    is clamped. Returns ``(pquery_id, training_job_id)``.
    """
    pquery_id, target, project_id = await seed_connectors_and_pquery(
        client,
        fixture,
        pquery_name,
        provision_connectors=provision_connectors,
        project_id=project_id,
    )

    query_args = next(
        (q for q in fixture.queries if q.pquery_name == pquery_name), None)
    assert query_args is not None

    if len(query_args.model_planner_yaml) == 0:
        # V2 graphs are keyed as GraphIDv2 (NA-<fpid>), not graph_<fpid[:8]>.
        graph_id_v2 = str(GraphIDv2.compose(fpid=target.graph.fpid))
        model_plan_response = post_and_expect_success(
            client,
            '/v1/predictive_queries/model_plan',
            SuggestModelPlanRequest(
                query_string=target.query_string,
                graph_id=graph_id_v2,
                run_mode=run_mode,
                graph_definition=target.graph,
            ),
            ModelPlanResponse,
            project_id=project_id,
        )
        query_args.model_planner_yaml = (
            model_plan_response.suggested_model_plan_yaml)

    model_plan_obj = model_plan_from_yaml(query_args.model_planner_yaml)
    clamp_e2e_training_budget(model_plan_obj)
    if manual_seed is not None:
        model_plan_obj.training_job.manual_seed = manual_seed

    training_job_id = post_and_expect_success(
        client,
        '/v0/training_jobs/train',
        TrainingJobRequest(
            predictive_query=target.query_string,
            graph_definition=target.graph,
            training_table_gen_plan=model_plan_obj.training_table_generation,
            model_plan=model_plan_obj,
            project_id=ProjectID(project_id),
        ),
        TrainingJobID,
    )

    return pquery_id, TrainingJobID(training_job_id)
