# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for custom training-table workflow helpers and dispatch."""
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from pool.activities import SnapshotActivities
from pool.dataclasses import SnapshotActivity
from rest.backends.ml_backend.validation import ValidationResponse
from rest.backends.ml_backend.workflows import \
    custom_train_table_workflow as mod
from rest.backends.ml_backend.workflows.custom_train_table_workflow import (
    _resolve_entity_time_col,
)


def _table(name, time_col=None):
    return SimpleNamespace(
        table_name=name,
        time_col=(SimpleNamespace(
            name=time_col) if time_col is not None else None),
    )


@pytest.mark.parametrize(
    ('entity_table', 'tables', 'expected'),
    [
        (
            'users',
            [_table('orders', 'order_ts'),
             _table('users', 'created_at')],
            'created_at',
        ),
        ('users', [_table('users')], None),
        ('missing', [_table('users', 'created_at')], None),
    ],
)
def test_resolve_entity_time_col(entity_table, tables, expected):
    config = SimpleNamespace(
        validated_predictive_query=SimpleNamespace(entity_table=entity_table),
        graph=SimpleNamespace(tables=tables),
    )
    assert _resolve_entity_time_col(config) == expected


def test_resolve_entity_time_col_handles_incomplete_config():
    config = SimpleNamespace(validated_predictive_query=SimpleNamespace(
        entity_table='users'))
    assert _resolve_entity_time_col(config) is None


def _job_input():
    predictive_query = SimpleNamespace(
        query_type='TEMPORAL',
        list_distinct_target_is_array=False,
        validated_predictive_query=SimpleNamespace(entity_table='users'),
        graph=SimpleNamespace(tables=[]),
    )
    return SimpleNamespace(
        id='trainingjob-1',
        spec=SimpleNamespace(
            task_type='REGRESSION',
            predictive_query=predictive_query,
        ),
        weight_col_name='WEIGHT',
        graph_data_compute_input=SimpleNamespace(
            graph_key='NA-11111111111111111111111111111111'),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('data_source_type', ['s3', 'databricks'])
async def test_custom_bundle_dispatch_and_materialization(
    monkeypatch,
    data_source_type,
):
    """Databricks takes the identical route to every other source: ingest the
    custom table, validate it against the generated one, then build the bundle
    from the staged path. No source spec is handed to the bundle activity."""
    validation = ValidationResponse()
    custom_path = 's3://bucket/custom_tt'
    original_path = 's3://bucket/original_tt'
    connector_spec = SimpleNamespace(connector_id='databricks')
    source = SimpleNamespace(
        data_source_type=data_source_type,
        table='weighted_training',
    )
    custom_spec = SimpleNamespace(
        source_table=source,
        connector_spec=connector_spec,
        # Carried to the validation activity: the split column is the
        # specification's, per table, on both backends.
        split_col_name='SPLIT',
    )
    execute_activity = AsyncMock(side_effect=[
        (custom_path, validation),
        validation,
        ('materialized', validation),
    ])
    published_path = (
        's3://bucket/custom_tt.ttng_staging/_bundles/bundle-1/tt.parquet')
    run_pooled = AsyncMock(return_value=published_path)
    monkeypatch.setattr(mod.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(mod, 'run_pooled_activity', run_pooled)
    monkeypatch.setattr(mod, 'deploy_env', SimpleNamespace)
    job_input = _job_input()

    result = await mod.CustomTrainTableWorkflow().run(
        cast(Any, custom_spec),
        original_path,
        custom_path,
        cast(Any, job_input),
        cast(Any, SimpleNamespace()),
    )

    assert result.train_table_materialization_path == 'materialized'
    pooled = run_pooled.await_args.kwargs
    assert pooled['activity'] == SnapshotActivity.GEN_CUSTOM_TT
    assert (pooled['activity_fn']
            is SnapshotActivities.build_ttng_custom_train_table_bundle)
    assert pooled['args'][0] == ('graph-11111111111111111111111111111111@v1')
    materialize = execute_activity.await_args_list[-1]
    assert materialize.args[0] is mod.materialize_training_table
    assert materialize.kwargs['args'][0] is job_input
    assert materialize.kwargs['args'][2] == published_path
    assert materialize.kwargs['args'][-1] is True
    assert result.train_table_path == published_path

    assert execute_activity.await_count == 3
    ingest = execute_activity.await_args_list[0]
    assert ingest.args[0] is mod.ingest_custom_train_table
    assert ingest.kwargs['args'][:2] == [custom_spec, custom_path]
    validate = execute_activity.await_args_list[1]
    assert validate.args[0] is mod.validate_weighted_train_table
    assert validate.kwargs['args'][:3] == [original_path, custom_path, 'WEIGHT']
    assert validate.kwargs['args'][-1] == 'SPLIT'
    assert pooled['args'][3:] == [custom_path, 'WEIGHT']


@pytest.mark.asyncio
async def test_a_registered_table_reuses_its_own_copy_without_reingesting(
    monkeypatch,
):
    """A table `register_custom_table` already copied into Kumo storage
    (`GenerateTrainTableResult.registered`) must not be re-ingested from the
    caller's original source at train time -- that source may have been
    deleted or rewritten since registration. `already_registered=True` skips
    the ingest and validate activities entirely and bundles straight from
    the registered copy.
    """
    validation = ValidationResponse()
    custom_path = 's3://bucket/custom_tt'
    registered_copy_path = 's3://bucket/registered_tt'
    connector_spec = SimpleNamespace(connector_id='databricks')
    source = SimpleNamespace(data_source_type='s3', table='weighted_training')
    custom_spec = SimpleNamespace(source_table=source,
                                  connector_spec=connector_spec)
    execute_activity = AsyncMock(side_effect=[('materialized', validation)])
    published_path = (
        's3://bucket/registered_tt.ttng_staging/_bundles/bundle-1/tt.parquet')
    run_pooled = AsyncMock(return_value=published_path)
    monkeypatch.setattr(mod.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(mod, 'run_pooled_activity', run_pooled)
    monkeypatch.setattr(mod, 'deploy_env', SimpleNamespace)
    job_input = _job_input()

    result = await mod.CustomTrainTableWorkflow().run(
        cast(Any, custom_spec),
        registered_copy_path,
        custom_path,
        cast(Any, job_input),
        cast(Any, SimpleNamespace()),
        True,
    )

    # Neither ingest nor validate ran: the only activity call is the final
    # materialize, and the bundle was built from the registered copy path
    # rather than from anything an ingest activity would have produced.
    assert execute_activity.await_count == 1
    materialize = execute_activity.await_args_list[0]
    assert materialize.args[0] is mod.materialize_training_table
    assert materialize.kwargs['args'][2] == published_path
    pooled = run_pooled.await_args.kwargs
    assert pooled['args'][3:] == [registered_copy_path, 'WEIGHT']
    assert result.train_table_path == published_path
    assert result.train_table_materialization_path == 'materialized'
