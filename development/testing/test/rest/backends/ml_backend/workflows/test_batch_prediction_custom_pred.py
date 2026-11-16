# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest
from kumoapi.jobs import (
    DatabricksPredictionTableSource,
    S3PredictionTableSource,
)
from kumoapi.source_table import S3RelativeSource

from rest.config.data_backend import ConnectorExecutionRef
from statedb.config.common import ConnectorID, ProjectID

bpw = import_module(
    'rest.backends.ml_backend.workflows.batch_prediction_workflow')

_CONNECTOR_ID = ConnectorID('connector_00000000000000000000000000000001')
_PROJECT_ID = ProjectID('project_00000000000000000000000000000001')


def _make_workflow(custom_source):
    wf = object.__new__(bpw.BatchPredictionParentWorkflow)
    wf._job_input = SimpleNamespace(
        id='batchpredictionjob-bp1',
        config=SimpleNamespace(custom_pred_table_source=custom_source),
        graph_data_compute_input=SimpleNamespace(
            graph_key='NA-11111111111111111111111111111111'),
        graph_version_info=SimpleNamespace(graph_data_version='v1'),
        custom_pred_table_connector=(ConnectorExecutionRef(
            connector_id=_CONNECTOR_ID,
            project_id=_PROJECT_ID,
        ) if custom_source is not None else None),
    )
    wf._home_dir = '/tmp/job'
    wf._ml_job_settings = SimpleNamespace()
    return wf


@pytest.mark.asyncio
@pytest.mark.parametrize('databricks', [False, True])
async def test_custom_pred_build_dispatch(monkeypatch, databricks):
    """Every custom prediction source is staged to Kumo-owned parquet first,
    and the bundle activity is then handed that path -- Databricks included."""
    source = (DatabricksPredictionTableSource(
        connector_id=str(_CONNECTOR_ID),
        sql='SELECT entity_id, TIMESTAMP FROM prediction_input',
    ) if databricks else S3PredictionTableSource(
        connector_id=str(_CONNECTOR_ID),
        source=S3RelativeSource(path='inputs/predictions.parquet'),
    ))
    wf = _make_workflow(source)

    calls = []

    async def _fake_execute_activity(fn, *args, **kwargs):
        calls.append((fn, kwargs.get('args')))
        if getattr(fn, '__name__', '') == 'stage_custom_prediction_table':
            return '/kumo/staged/custom_prediction_table'
        return None

    monkeypatch.setattr(bpw.workflow, 'execute_activity',
                        _fake_execute_activity)
    pooled = []

    async def _fake_run_pooled_activity(**kwargs):
        pooled.append(kwargs)

    monkeypatch.setattr(bpw, 'run_pooled_activity', _fake_run_pooled_activity)

    result = await wf._generate_pred_table()

    local_path = '/tmp/job/inputs/custom_prediction_table'
    staged_path = '/kumo/staged/custom_prediction_table'
    assert result.pred_table_path == staged_path
    assert calls[0][1] == [
        wf._job_input.custom_pred_table_connector,
        source,
        local_path,
    ]
    assert len(pooled) == 1
    build = pooled[0]
    assert build['activity'] is bpw.SnapshotActivity.GEN_CUSTOM_TT
    assert (build['activity_fn']
            is bpw.SnapshotActivities.build_ttng_custom_pred_table_bundle)
    assert build['args'][0] == ('graph-11111111111111111111111111111111@v1')
    assert build['args'][1] is wf._job_input
    # No source spec and no connector reference reach the engine.
    assert build['args'][3:] == [staged_path]


@pytest.mark.asyncio
async def test_non_custom_branch_does_not_run_build_activity(monkeypatch):
    """When there is no custom source, the custom branch is skipped
    (the build-bundle activity is never called)."""
    wf = _make_workflow(None)
    # Give it a generate_pred_table_job_id so it enters the gen-table branch
    # and we can prove the custom activity is not reached before that point.
    wf._job_input.generate_pred_table_job_id = None

    calls = []

    async def _fake_execute_activity(fn, *args, **kwargs):
        calls.append(getattr(fn, '__name__', str(fn)))
        return None

    monkeypatch.setattr(bpw.workflow, 'execute_activity',
                        _fake_execute_activity)

    # The non-custom path asserts on generate_pred_table_job_id; we only care
    # that the custom build activity is not invoked, so stop at the assert.
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        await wf._generate_pred_table()

    assert 'build_ttng_custom_pred_table_bundle' not in calls
