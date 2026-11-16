# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import dataclasses
from typing import Dict, Tuple, Union

import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
from fastapi.testclient import TestClient
from kumo_core.types import TaskType
from kumoapi import jobs as jobs_api
from kumoapi.graph import GraphDefinition as apiGraphDefinition
from kumoapi.model_plan import RunMode

from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query_v2 import (
    ModelPlanResponse,
    PQueryResponse,
    PQueryValidationRequest,
)
from rest.config.status import IDResponse
from rest.config.training_job import TrainingJobResponse
from rest.config.training_job_v2 import TrainingJobRequest
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    PredictiveQueryID,
    TrainingJobID,
)


def test_create_and_get_pquery_v2(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    )
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    valid_request = pquery_api.PQueryResource(
        name='pq',
        query_string="PREDICT COUNT(transactions.*, 0, 20) FOR EACH "
        "customers.customer_id",
        graph=graph_definition_internal,
        desc="Sample predictive query",
    )

    response = post_and_expect_success(
        client,
        f"/v1/predictive_queries/?project_id={default_project_id}",
        valid_request,
        IDResponse,
    )

    assert response is not None
    assert response.id is not None
    _, fpid = PredictiveQueryID.parse(response.id)
    assert fpid is not None

    response = get_and_expect_success(
        client,
        f'/v1/predictive_queries/{response.id}',
        {'project_id': default_project_id},
        PQueryResponse,
    )

    assert response is not None
    assert response.name == 'pq'
    assert response.query_string == valid_request.query_string


def test_create_pquery_v2_accepts_display_and_empty_names(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    for name in ['display - name', '']:
        request = pquery_api.PQueryResource(
            name=name,
            query_string="PREDICT COUNT(transactions.*, 0, 20) FOR EACH "
            "customers.customer_id",
            graph=graph_definition_internal,
            desc="Sample predictive query",
        )

        response = post_and_expect_success(
            client,
            f"/v1/predictive_queries/?project_id={default_project_id}",
            request,
            IDResponse,
        )

        assert response is not None
        assert response.id is not None
        _, fpid = PredictiveQueryID.parse(response.id)
        assert fpid is not None


def test_private_create_pquery_accepts_internal_metadata(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    )
    payload = dataclasses.asdict(
        pquery_api.PQueryResource(
            name='private_metadata_query',
            query_string='PREDICT COUNT(transactions.*, 0, 20) FOR EACH '
            'customers.customer_id',
            graph=graph_definition_v2,
        ))
    payload.update({
        'graph_display_name': 'Private graph display name',
        'advanced_options_yaml': None,
        'runtime_options': {
            'list_distinct_target_is_array': False,
            'lp_bp_layout': 'private-layout',
        },
    })

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/?project_id={default_project_id}',
        payload,
        IDResponse,
    )

    assert response.id is not None


def test_private_create_pquery_rejects_unknown_fields(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    payload = dataclasses.asdict(
        pquery_api.PQueryResource(
            name='unknown_field_query',
            query_string='PREDICT COUNT(transactions.*, 0, 20) FOR EACH '
            'customers.customer_id',
            graph=graph_definition_v2,
        ))
    payload['unknown_internal_field'] = True

    response = client.post(
        f'/v1/predictive_queries/?project_id={default_project_id}',
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()[0]['loc'] == ['body', 'unknown_internal_field']


def test_get_default_model_plan(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))
    # Testing model plan generation for regression
    response = post_and_expect_success(
        client=client,
        path=(f'/v1/predictive_queries/model_plan?project_id='
              f'{default_project_id}'),
        payload=plan_api.SuggestModelPlanRequest(
            query_string='PREDICT COUNT(transactions.*, 0, 20) FOR EACH '
            'customers.customer_id',
            graph_id="",
            run_mode=RunMode.FAST,
            graph_definition=graph_definition_internal,
        ),
        response_type=ModelPlanResponse,
    )

    assert response is not None
    assert response.suggested_model_plan is not None
    assert response.suggested_model_plan.model_plan is not None
    assert (response.suggested_model_plan.model_plan.training_job.metrics
            is not None)
    assert all(
        metric in response.suggested_model_plan.model_plan.training_job.metrics
        for metric in ['mae', 'mse', 'rmse'])

    # Testing model plan generation for binary classification
    response = post_and_expect_success(
        client=client,
        path=(f'/v1/predictive_queries/model_plan?project_id='
              f'{default_project_id}'),
        payload=plan_api.SuggestModelPlanRequest(
            query_string='PREDICT COUNT(transactions.*, 0, 20) = 0 FOR EACH '
            'customers.customer_id',
            graph_id="",
            run_mode=RunMode.FAST,
            graph_definition=graph_definition_internal,
        ),
        response_type=ModelPlanResponse,
    )

    assert response is not None
    assert response.suggested_model_plan is not None
    assert response.suggested_model_plan.model_plan is not None
    assert (response.suggested_model_plan.model_plan.training_job.metrics
            is not None)
    assert all(
        metric in response.suggested_model_plan.model_plan.training_job.metrics
        for metric in ['auroc', 'ap', 'auprc'])

    # Make sure hidden fields are not included in the yaml returned
    assert response.suggested_model_plan_yaml is not None
    assert 'manual_seed' not in response.suggested_model_plan_yaml
    assert 'disable_compilation' not in response.suggested_model_plan_yaml

    # Testing model plan generation for link prediction
    response = post_and_expect_success(
        client=client,
        path=(f'/v1/predictive_queries/model_plan?project_id='
              f'{default_project_id}'),
        payload=plan_api.SuggestModelPlanRequest(
            query_string=(
                'PREDICT LIST_DISTINCT(transactions.article_id, 0, 30)'
                'RANK TOP 10'
                'FOR EACH customers.customer_id'),
            graph_id="",
            run_mode=RunMode.FAST,
            graph_definition=graph_definition_internal,
        ),
        response_type=ModelPlanResponse,
    )
    assert response is not None
    assert response.suggested_model_plan is not None
    assert response.suggested_model_plan.model_plan is not None
    assert (response.suggested_model_plan.model_plan.training_job.metrics
            is not None)
    expected = [
        'f1@1', 'f1@10', 'f1@100', 'map@1', 'map@10', 'map@100', 'precision@1',
        'precision@10', 'precision@100', 'recall@1', 'recall@10', 'recall@100'
    ]
    assert collections.Counter(
        response.suggested_model_plan.model_plan.training_job.metrics
    ) == collections.Counter(expected)


def test_validate_predictive_query(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    )

    # Test when model plan is none and query is correct
    valid_request = PQueryValidationRequest(
        pquery_resource=pquery_api.PQueryResource(
            name='pq',
            query_string="PREDICT COUNT(transactions.*, 0, 20) FOR EACH "
            "customers.customer_id",
            graph=graph_definition_v2,
            desc="Sample predictive query",
        ),
        model_plan='',
    )

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={default_project_id}',
        valid_request,
        jobs_api.ErrorDetails,
    )

    assert response is not None
    assert response.items is not None
    assert len(response.items) == 0

    # The public validate route must accept an unsaved query (no name yet).
    # Previously this raised 500 because the driver tried to coerce the
    # empty name into a PredictiveQueryID even though uniqueness was
    # explicitly not being checked.
    unnamed_request = PQueryValidationRequest(
        pquery_resource=pquery_api.PQueryResource(
            name='',
            query_string="PREDICT COUNT(transactions.*, 0, 20) FOR EACH "
            "customers.customer_id",
            graph=graph_definition_v2,
            desc="Sample predictive query",
        ),
        model_plan='',
    )

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={default_project_id}',
        unnamed_request,
        jobs_api.ErrorDetails,
    )

    assert response is not None
    assert response.items is not None
    assert len(response.items) == 0

    # Display/template names containing characters that are illegal in a
    # PredictiveQueryID (e.g. spaces, dashes) must also be accepted by the
    # validate route as long as uniqueness is not being checked. This locks
    # in the second bug class: coercing a user-facing name into a
    # PredictiveQueryID raised even when `check_unique_id` was False.
    dash_named_request = PQueryValidationRequest(
        pquery_resource=pquery_api.PQueryResource(
            name='display - name',
            query_string="PREDICT COUNT(transactions.*, 0, 20) FOR EACH "
            "customers.customer_id",
            graph=graph_definition_v2,
            desc="Sample predictive query",
        ),
        model_plan='',
    )

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={default_project_id}',
        dash_named_request,
        jobs_api.ErrorDetails,
    )

    assert response is not None
    assert response.items is not None
    assert len(response.items) == 0

    # Test when model plan is none and
    # query has future looking error
    query_string = ("PREDICT MIN(transactions.sales_channel_id, 0, 2, days) "
                    "< 1 FOR EACH customers.customer_id WHERE "
                    "COUNT(transactions.*, -4, 0, days) > 0 AND "
                    "COUNT(transactions.price, 0, 2, days) > 0.1")
    valid_request = PQueryValidationRequest(
        pquery_resource=pquery_api.PQueryResource(
            name='pq',
            query_string=query_string,
            graph=graph_definition_v2,
            desc="Sample predictive query",
        ),
        model_plan='',
    )

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={default_project_id}',
        valid_request,
        jobs_api.ErrorDetails,
    )

    assert "Future-looking time range not allowed." in str(response.items)

    # Test when model plan is none and query is incorrect
    valid_request = PQueryValidationRequest(
        pquery_resource=pquery_api.PQueryResource(
            name='pq',
            query_string="PREDICT COUNT(DOES_NOT_EXISTS.*, 0, 20) FOR EACH "
            "customers.customer_id",
            graph=graph_definition_v2,
            desc="Sample predictive query",
        ),
        model_plan='',
    )

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={default_project_id}',
        valid_request,
        jobs_api.ErrorDetails,
    )

    assert response is not None
    assert response.items is not None
    assert len(response.items) == 1
    assert ('Unknown table name' in response.items[0].description
            or 'is not found in the Graph' in response.items[0].description)
    assert 'DOES_NOT_EXIST' in response.items[0].description

    # Test when model plan is not none and incorrect but query is correct
    valid_request = PQueryValidationRequest(
        pquery_resource=pquery_api.PQueryResource(
            name='pq',
            query_string="PREDICT COUNT(transactions.*, 0, 20) FOR EACH "
            "customers.customer_id",
            graph=graph_definition_v2,
            desc="Sample predictive query",
        ),
        model_plan=(
            'training_table_generation:\n'
            '  split: IncorrectSplitName(ratio=[0.8, 0.1, 0.1])\n'
            '  train_start_offset: inferred\n'
            '  train_end_offset: inferred\n'
            '  timeframe_step: inferred\n'
            '  forecast_length: 1\n'
            '  lag_timesteps: 0\n'
            '  year_over_year: false\n'
            'training_job:\n'
            '  num_experiments: 4\n'
            '  metrics:\n'
            '  - mae\n'
            '  - mse\n'
            '  - rmse\n'
            '  tune_metric: mae\n'
            '  refit_trainval: false\n'
            '  refit_full: false\n'
            'column_processing:\n'
            '  encoder_overrides:\n'
            'neighbor_sampling:\n'
            '  num_neighbors:\n'
            '  - hop1:\n'
            '      default: 12\n'
            '      customers.customer_id->transactions.customer_id: inferred\n'
            '    hop2:\n'
            '      default: 12\n'
            'optimization:\n'
            '  max_epochs: 8\n'
            '  min_steps_per_epoch: 30\n'
            '  max_steps_per_epoch: 1000\n'
            '  max_val_steps: 1000\n'
            '  max_test_steps: 2000\n'
            '  loss:\n'
            '  - huber\n'
            '  base_lr:\n'
            '  - 0.0001\n'
            '  - 0.0005\n'
            '  - 0.001\n'
            '  - 0.005\n'
            '  - 0.01\n'
            '  - 0.05\n'
            '  weight_decay:\n'
            '  - 0.0\n'
            '  - 5.0e-08\n'
            '  - 5.0e-07\n'
            '  - 5.0e-06\n'
            '  batch_size:\n'
            '  - 512\n'
            '  - 1024\n'
            '  early_stopping:\n'
            '  - min_delta: 0.0\n'
            '    patience: 3\n'
            '  lr_scheduler:\n'
            '  - name: cosine_with_warmup_restarts\n'
            '    interval: step\n'
            '    kwargs:\n'
            '      warmup_ratio_or_steps: 0.1\n'
            '  - name: constant_with_warmup\n'
            '    interval: step\n'
            '    kwargs:\n'
            '      warmup_ratio_or_steps: 0.1\n'
            '  - name: linear_with_warmup\n'
            '    interval: step\n'
            '    kwargs:\n'
            '      warmup_ratio_or_steps: 0.1\n'
            '  - name: cosine_with_warmup\n'
            '    interval: step\n'
            '    kwargs:\n'
            '      warmup_ratio_or_steps: 0.1\n'
            'model_architecture:\n'
            '  channels:\n'
            '  - 64\n'
            '  - 128\n'
            '  - 256\n'
            '  num_pre_message_passing_layers:\n'
            '  - 0\n'
            '  - 1\n'
            '  - 2\n'
            '  num_post_message_passing_layers:\n'
            '  - 1\n'
            '  - 2\n'
            '  aggregation:\n'
            '  - [sum, mean, max]\n'
            '  - [sum, mean, min, max, std]\n'
            '  activation:\n'
            '  - relu\n'
            '  - leaky_relu\n'
            '  - elu\n'
            '  - gelu\n'
            '  normalization:\n'
            '  - layer_norm\n'
            '  use_seq_id:\n'
            '  - false\n'
            '  prediction_time_encodings:\n'
            '  - false\n'
            '  past_encoder:\n'
            '  - decomposed\n'
            '  handle_new_entities: true'),
    )

    response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/validate?project_id={default_project_id}',
        valid_request,
        jobs_api.ErrorDetails,
    )

    assert response is not None
    assert response.items is not None
    assert len(response.items) == 1
    assert response.items[0].description == (
        'Invalid model plan: Check its field names, values, and expression '
        'syntax.')
    assert 'IncorrectSplitName' not in response.items[0].description


def test_get_many_pqueries_v2(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    )

    pquery_str = (
        'PREDICT COUNT(transactions.*, 0, 30) FOR EACH customers.customer_id')

    payload = plan_api.SuggestModelPlanRequest(
        query_string=pquery_str,
        graph_id=graph_id.id,
        run_mode=plan_api.RunMode.FAST,
        graph_definition=graph_definition_internal,
    )

    model_plan = post_and_expect_success(
        client=client,
        path=(f'/v1/predictive_queries/model_plan?project_id='
              f'{default_project_id}'),
        payload=payload,
        response_type=ModelPlanResponse,
    )

    request = TrainingJobRequest(
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    # create a training job
    response = post_and_expect_success(
        client,
        '/v0/training_jobs/train',
        request,
        TrainingJobID,
    )
    assert response is not None

    training_job_response = get_and_expect_success(
        client,
        f'/v0/training_jobs/{response}',
        {'project_id': default_project_id},
        TrainingJobResponse,
    )
    assert training_job_response.id is not None
    assert training_job_response.tags['pquery_id'] is not None

    request_many_pqs = {
        training_job_response.id: training_job_response.tags['pquery_id']
    }

    many_predictive_queries_response = post_and_expect_success(
        client,
        f'/v1/predictive_queries/get_many?project_id={default_project_id}',
        request_many_pqs,
        Dict[Union[TrainingJobID, BatchPredictionJobID,
                   GeneratePredictionTableJobID, GenerateTrainTableJobID],
             Tuple[str, TaskType]],
    )

    assert len(many_predictive_queries_response[training_job_response.id]) == 2
    assert many_predictive_queries_response[
        training_job_response.id][0] == pquery_str
    assert many_predictive_queries_response[
        training_job_response.id][1] == 'regression'
