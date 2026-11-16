# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LAYER 1 TDD unit tests: BatchPredictionParentWorkflow.run terminalizes a
RAISED failure through the canonical _exit_cleanup chokepoint (so a safe
public message and an internal debug trace reach StateDB), and does NOT
terminalize cancellation as FAILED.

These drive BatchPredictionParentWorkflow.run directly with a stubbed _run,
patching the Temporal-facing helpers, rather than spinning up a full Temporal
test env.  Mirrors test_base_training_terminalization.py (PR #29713) for the
batch-prediction parent workflow.
"""
from unittest import mock

import pytest
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.exceptions import TimeoutType

from common_utils.error import PublicException
from rest.backends.ml_backend.validation import (
    ValidationError,
    ValidationResponse,
)
from rest.backends.ml_backend.workflows import batch_prediction_workflow as bpw
from rest.backends.ml_backend.workflows.batch_prediction_workflow import (
    BatchPredictionParentWorkflow,
)
from rest.config.error_handling import root_cause_message


class _RaisingWorkflow(BatchPredictionParentWorkflow):
    """Concrete workflow whose _run raises a chosen exception.

    ``run_until_canceled`` is overridden to just await the coroutine: the real
    implementation calls Temporal's ``workflow.wait`` which needs the workflow
    sandbox. We only exercise ``run``'s except/finally terminalization
    contract, so awaiting the coro directly (which re-raises what _run raised)
    is exactly the interception point under test.
    """
    def __init__(self, exc: BaseException):
        self._exc = exc

    async def _run(self, job_input, ml_job_settings):
        raise self._exc

    async def run_until_canceled(self, coro):
        return await coro


def _job_input():
    return mock.Mock(id='batchpredictionjob-bp1')


@pytest.mark.asyncio
async def test_run_terminalizes_raised_failure_without_public_internal_text():
    """A non-public failure gets a safe message and is then re-raised."""
    boom = RuntimeError('predict blew up: KeyError pred_seeds')
    wf = _RaisingWorkflow(boom)
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(bpw, 'dequeue_job',
                              new=mock.AsyncMock()) as dequeue, \
            mock.patch.object(bpw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(bpw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(RuntimeError, match='predict blew up'):
            await wf.run(job_input, mock.Mock())

    # _exit_cleanup was called once, at the canonical chokepoint.
    exit_cleanup.assert_awaited_once()
    args, kwargs = exit_cleanup.call_args
    validation_response, job_id, home_dir = args
    assert job_id == job_input.id
    assert home_dir == '/base/job'
    assert not validation_response.ok
    assert 'internal error' in validation_response.error_message()
    assert 'contact Kumo support' in validation_response.error_message()
    assert 'predict blew up' not in validation_response.error_message()
    assert 'pred_seeds' not in validation_response.error_message()
    # The caught exception still supplies the employee-only debug stack.
    assert kwargs['exception'] is boom
    # finally still ran.
    dequeue.assert_awaited_once_with(job_input.id)


def _child_workflow_failure_chain() -> ApplicationError:
    """A nested temporalio failure chain like the one a failed child workflow
    produces: ChildWorkflowError(wrapper) -> ActivityError(wrapper) ->
    ApplicationError(real cause)."""
    real = ApplicationError('KeyError: bp_abc/pred_seeds', type='KeyError',
                            non_retryable=True)
    activity = ApplicationError('Activity task failed', type='ActivityError')
    activity.__cause__ = real
    child = ApplicationError('Child Workflow execution failed',
                             type='ChildWorkflowError')
    child.__cause__ = activity
    return child


def test_root_cause_message_unwraps_child_workflow_error():
    """The deepest temporalio failure message is surfaced, not the generic
    ChildWorkflowError wrapper text."""
    child = _child_workflow_failure_chain()
    assert 'Child Workflow execution failed' in str(child)
    assert root_cause_message(child) == 'KeyError: bp_abc/pred_seeds'


def test_root_cause_message_falls_back_to_str_for_plain_exception():
    """A plain Python exception (no temporalio layer) falls back to str."""
    assert root_cause_message(
        RuntimeError('predict blew up')) == 'predict blew up'


@pytest.mark.asyncio
async def test_exit_cleanup_populates_stacktrace_from_exception():
    """_exit_cleanup threads the caught exception into
    BatchPredictionJobOutput.failed so failure_info.stacktrace carries the
    real internal trace; it used to pass only the ValidationResponse. The
    public API deliberately exposes only the curated failure_info.message."""
    wf = _RaisingWorkflow(RuntimeError('unused'))
    validation_response = ValidationResponse()
    validation_response.errors.append(
        ValidationError(
            title='Batch prediction failed',
            message='An internal error occurred. Please contact Kumo support.',
        ))

    saved = []

    async def save_output(job_output):
        saved.append(job_output)

    try:
        raise RuntimeError('predict blew up')
    except RuntimeError as boom:
        with mock.patch.object(bpw, 'save_validation_response',
                               new=mock.AsyncMock()), \
                mock.patch.object(wf, '_save_output', new=save_output):
            await wf._exit_cleanup(validation_response,
                                   'batchpredictionjob-bp1', '/base/job',
                                   exception=boom)

    (job_output, ) = saved
    assert job_output.failure_info is not None
    assert 'Batch prediction failed' in job_output.failure_info.message
    assert 'predict blew up' not in job_output.failure_info.message
    # The debug stacktrace is now populated from the exception.
    assert 'RuntimeError' in job_output.failure_info.stacktrace
    assert 'predict blew up' in job_output.failure_info.stacktrace


@pytest.mark.asyncio
async def test_run_redacts_non_public_child_workflow_root_cause():
    """A raw child-workflow cause remains debug-only."""
    wf = _RaisingWorkflow(_child_workflow_failure_chain())
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(bpw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(bpw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(bpw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(ApplicationError):
            await wf.run(job_input, mock.Mock())

    exit_cleanup.assert_awaited_once()
    validation_response, _job_id, _home_dir = exit_cleanup.call_args[0]
    assert 'bp_abc/pred_seeds' not in validation_response.error_message()
    assert 'Child Workflow execution failed' not in (
        validation_response.error_message())
    assert 'internal error' in validation_response.error_message()
    assert root_cause_message(
        exit_cleanup.call_args.kwargs['exception']
    ) == 'KeyError: bp_abc/pred_seeds'


@pytest.mark.asyncio
async def test_run_does_not_terminalize_cancellation():
    """A cancellation is re-raised WITHOUT terminalizing as FAILED
    (cancellation has its own path)."""
    wf = _RaisingWorkflow(CancelledError('cancelled'))
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(bpw, 'dequeue_job',
                              new=mock.AsyncMock()) as dequeue, \
            mock.patch.object(bpw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(bpw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(CancelledError):
            await wf.run(job_input, mock.Mock())

    # Cancellation must NOT be terminalized as FAILED.
    exit_cleanup.assert_not_awaited()
    # finally still ran (dequeue safety-net).
    dequeue.assert_awaited_once_with(job_input.id)


@pytest.mark.asyncio
async def test_run_terminalize_failure_does_not_swallow_original():
    """If terminalization itself fails, the ORIGINAL exception still propagates
    (best-effort guard)."""
    boom = RuntimeError('original failure')
    wf = _RaisingWorkflow(boom)
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(
        side_effect=RuntimeError('terminalize failed'))

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(bpw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(bpw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(bpw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(RuntimeError, match='original failure'):
            await wf.run(job_input, mock.Mock())


def _incident_placement_timeout_chain() -> ApplicationError:
    """The failure chain of the 2026-07-16 acme placement-outage incident
    (kumo-ai/kumo#30110), as it would arrive at the batch-prediction parent:
    wrapper stand-ins over REAL Temporal TimeoutErrors
    (SCHEDULE_TO_CLOSE -> HEARTBEAT), zero PublicException anywhere."""
    heartbeat = TemporalTimeoutError('activity Heartbeat timeout',
                                     type=TimeoutType.HEARTBEAT,
                                     last_heartbeat_details=[])
    s2c = TemporalTimeoutError('activity ScheduleToClose timeout',
                               type=TimeoutType.SCHEDULE_TO_CLOSE,
                               last_heartbeat_details=[])
    s2c.__cause__ = heartbeat
    activity = ApplicationError('Activity task timed out', type='ActivityError')
    activity.__cause__ = s2c
    child = ApplicationError('Child Workflow execution failed',
                             type='ChildWorkflowError')
    child.__cause__ = activity
    return child


@pytest.mark.asyncio
async def test_run_timeout_only_chain_composes_actionable_message():
    """A timeout-only failure chain (the kumo-ai/kumo#30110 incident shape) is
    terminalized with the curated, actionable timeout translation - NOT the
    bare raw Temporal string 'activity Heartbeat timeout' as the whole user
    message."""
    wf = _RaisingWorkflow(_incident_placement_timeout_chain())
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(bpw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(bpw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(bpw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(ApplicationError):
            await wf.run(job_input, mock.Mock())

    exit_cleanup.assert_awaited_once()
    validation_response, _job_id, _home_dir = exit_cleanup.call_args[0]
    message = validation_response.error_message()
    # The raw Temporal internals no longer reach the user...
    assert 'activity Heartbeat timeout' not in message
    assert 'activity ScheduleToClose timeout' not in message
    # ...the curated heartbeat translation does (actionable guidance).
    assert 'stopped reporting progress' in message
    assert 'retry' in message.lower()


@pytest.mark.asyncio
async def test_run_attached_public_exception_wins_over_timeout_text():
    """A PublicException attached anywhere on the chain (e.g. the
    PoolPlacementUnavailableError ApplicationError of PR kumo-ai/kumo#29874)
    is surfaced as the user message, beating both the raw internals and the
    generic timeout translation."""
    public_message = ('The platform compute pool could not place your job. '
                      'This is a platform issue, not a problem with your '
                      'data. Please contact Kumo support.')
    chain = _incident_placement_timeout_chain()
    activity_leg = chain.__cause__
    placement = ApplicationError(
        'diskgraph placement did not complete',
        PublicException(message=public_message).to_dict(),
        type='PoolPlacementUnavailableError',
        non_retryable=True,
    )
    placement.__cause__ = activity_leg
    chain.__cause__ = placement

    wf = _RaisingWorkflow(chain)
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(bpw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(bpw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(bpw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(ApplicationError):
            await wf.run(job_input, mock.Mock())

    exit_cleanup.assert_awaited_once()
    validation_response, _job_id, _home_dir = exit_cleanup.call_args[0]
    message = validation_response.error_message()
    # The attached public message wins outright.
    assert public_message in message
    assert 'activity Heartbeat timeout' not in message
    assert 'stopped reporting progress' not in message
