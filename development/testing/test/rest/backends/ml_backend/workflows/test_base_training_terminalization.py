# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseTrainingWorkflow.run terminalizes a raised failure through _exit_cleanup
(so a safe public message and an internal debug trace reach StateDB), and does
NOT terminalize cancellation as FAILED.

Drives run() directly with a stubbed _run rather than a full Temporal test env.
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
from rest.backends.ml_backend.workflows import base_training_workflow as btw
from rest.backends.ml_backend.workflows.base_training_workflow import (
    BaseTrainingWorkflow,
)
from rest.config.error_handling import root_cause_message
from statedb.config.common import TrainingJobID


class _RaisingWorkflow(BaseTrainingWorkflow):
    """Concrete workflow whose _run raises a chosen exception.

    ``run_until_canceled`` is overridden to just await the coro (the real one
    needs the Temporal workflow sandbox); we only exercise run()'s
    terminalization contract.
    """
    def __init__(self, exc: BaseException):
        super().__init__()
        self._exc = exc

    async def _run(self, job_input, ml_job_settings):
        raise self._exc

    async def run_until_canceled(self, coro):
        return await coro


def _job_input():
    return mock.Mock(id='train-job-123')


@pytest.mark.asyncio
async def test_run_terminalizes_raised_failure_without_public_internal_text():
    """A non-public failure gets a safe message and is then re-raised."""
    boom = RuntimeError('materialize blew up: KeyError train_seeds')
    wf = _RaisingWorkflow(boom)
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()) as dequeue, \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(RuntimeError, match='materialize blew up'):
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
    assert 'materialize blew up' not in validation_response.error_message()
    assert 'train_seeds' not in validation_response.error_message()
    # The caught exception still supplies the employee-only debug stack.
    assert kwargs['exception'] is boom
    # finally still ran.
    dequeue.assert_awaited_once_with(job_input.id)


def _child_workflow_failure_chain() -> ApplicationError:
    """A nested temporalio failure chain like the one a failed child workflow
    produces: ChildWorkflowError(wrapper) -> ActivityError(wrapper) ->
    ApplicationError(real cause)."""
    real = ApplicationError('KeyError: pred_abc/train_seeds', type='KeyError',
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
    assert root_cause_message(child) == 'KeyError: pred_abc/train_seeds'


def test_root_cause_message_falls_back_to_str_for_plain_exception():
    """A plain Python exception (no temporalio layer) falls back to str."""
    assert root_cause_message(
        RuntimeError('materialize blew up')) == 'materialize blew up'


@pytest.mark.asyncio
async def test_exit_cleanup_populates_stacktrace_from_exception():
    """_exit_cleanup threads the caught exception into
    TrainingJobOutput.failed so failure_info.stacktrace carries the real
    internal trace. The public API deliberately exposes only the curated
    failure_info.message."""
    wf = _RaisingWorkflow(RuntimeError('unused'))
    validation_response = ValidationResponse()
    validation_response.errors.append(
        ValidationError(
            title='Training failed',
            message='An internal error occurred. Please contact Kumo support.',
        ))

    saved = []

    async def save_output(job_output):
        saved.append(job_output)

    try:
        raise RuntimeError('materialize blew up')
    except RuntimeError as boom:
        with mock.patch.object(btw, 'save_validation_response',
                               new=mock.AsyncMock()), \
                mock.patch.object(wf, '_save_output', new=save_output):
            await wf._exit_cleanup(validation_response,
                                   TrainingJobID('train-job-123'), '/base/job',
                                   exception=boom)

    (job_output, ) = saved
    assert job_output.failure_info is not None
    assert 'Training failed' in job_output.failure_info.message
    assert 'materialize blew up' not in job_output.failure_info.message
    # The debug stacktrace is now populated from the exception.
    assert 'RuntimeError' in job_output.failure_info.stacktrace
    assert 'materialize blew up' in job_output.failure_info.stacktrace


@pytest.mark.asyncio
async def test_exit_cleanup_without_exception_keeps_empty_stacktrace():
    """Validation-failure early-returns (no exception in hand) keep their
    existing behavior: message from the ValidationResponse, empty
    stacktrace."""
    wf = _RaisingWorkflow(RuntimeError('unused'))
    validation_response = ValidationResponse()
    validation_response.errors.append(
        ValidationError(title='Bad input', message='column missing'))

    saved = []

    async def save_output(job_output):
        saved.append(job_output)

    with mock.patch.object(btw, 'save_validation_response',
                           new=mock.AsyncMock()), \
            mock.patch.object(wf, '_save_output', new=save_output):
        await wf._exit_cleanup(validation_response,
                               TrainingJobID('train-job-123'), '/base/job')

    (job_output, ) = saved
    assert job_output.failure_info is not None
    assert 'Bad input' in job_output.failure_info.message
    assert job_output.failure_info.stacktrace == ''


@pytest.mark.asyncio
async def test_run_redacts_non_public_child_workflow_root_cause():
    """A raw child-workflow cause remains debug-only."""
    wf = _RaisingWorkflow(_child_workflow_failure_chain())
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(ApplicationError):
            await wf.run(job_input, mock.Mock())

    exit_cleanup.assert_awaited_once()
    validation_response, _job_id, _home_dir = exit_cleanup.call_args[0]
    assert 'pred_abc/train_seeds' not in validation_response.error_message()
    assert 'Child Workflow execution failed' not in (
        validation_response.error_message())
    assert 'internal error' in validation_response.error_message()
    assert root_cause_message(
        exit_cleanup.call_args.kwargs['exception']
    ) == 'KeyError: pred_abc/train_seeds'


@pytest.mark.asyncio
async def test_run_does_not_terminalize_cancellation():
    """A cancellation is re-raised WITHOUT terminalizing as FAILED (cancellation
    has its own path)."""
    wf = _RaisingWorkflow(CancelledError('cancelled'))
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()) as dequeue, \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
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
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(RuntimeError, match='original failure'):
            await wf.run(job_input, mock.Mock())


def _incident_placement_timeout_chain() -> ApplicationError:
    """The failure chain of the 2026-07-16 acme placement-outage incident
    (kumo-ai/kumo#30110): ChildWorkflowError(GraphRefresh) ->
    ChildWorkflowError(dg-ingest) -> ActivityError(place_snapshot_activity) ->
    TimeoutError(SCHEDULE_TO_CLOSE) -> TimeoutError(HEARTBEAT), with zero
    PublicException anywhere. Wrapper layers use ApplicationError stand-ins
    (like _child_workflow_failure_chain); the timeouts are REAL Temporal
    TimeoutErrors, which is what the public-error machinery keys on."""
    heartbeat = TemporalTimeoutError('activity Heartbeat timeout',
                                     type=TimeoutType.HEARTBEAT,
                                     last_heartbeat_details=[])
    s2c = TemporalTimeoutError('activity ScheduleToClose timeout',
                               type=TimeoutType.SCHEDULE_TO_CLOSE,
                               last_heartbeat_details=[])
    s2c.__cause__ = heartbeat
    activity = ApplicationError('Activity task timed out', type='ActivityError')
    activity.__cause__ = s2c
    dg_ingest = ApplicationError('Child Workflow execution failed',
                                 type='ChildWorkflowError')
    dg_ingest.__cause__ = activity
    graph_refresh = ApplicationError('Child Workflow execution failed',
                                     type='ChildWorkflowError')
    graph_refresh.__cause__ = dg_ingest
    return graph_refresh


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
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
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
    # dg-ingest's ActivityError leg, as re-raised by _place_snapshot with the
    # curated PublicException carried in ApplicationError details.
    dg_ingest_leg = chain.__cause__
    assert dg_ingest_leg is not None
    activity_leg = dg_ingest_leg.__cause__
    placement = ApplicationError(
        'diskgraph placement did not complete',
        PublicException(message=public_message).to_dict(),
        type='PoolPlacementUnavailableError',
        non_retryable=True,
    )
    placement.__cause__ = activity_leg
    dg_ingest_leg.__cause__ = placement

    wf = _RaisingWorkflow(chain)
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(ApplicationError):
            await wf.run(job_input, mock.Mock())

    exit_cleanup.assert_awaited_once()
    validation_response, _job_id, _home_dir = exit_cleanup.call_args[0]
    message = validation_response.error_message()
    # The attached public message wins outright.
    assert public_message in message
    assert 'activity Heartbeat timeout' not in message
    assert 'stopped reporting progress' not in message


@pytest.mark.asyncio
async def test_run_direct_public_exception_message_reaches_user():
    """A PublicException raised directly by _run surfaces its own public
    message."""
    wf = _RaisingWorkflow(
        PublicException(message='Column xyz is missing from table abc.'))
    job_input = _job_input()

    exit_cleanup = mock.AsyncMock(return_value=ValidationResponse())

    with mock.patch.object(wf, '_exit_cleanup', exit_cleanup), \
            mock.patch.object(btw, 'dequeue_job',
                              new=mock.AsyncMock()), \
            mock.patch.object(btw, 'get_job_base_dir', return_value='/base'), \
            mock.patch.object(btw, 'get_job_dir', return_value='/base/job'):
        with pytest.raises(PublicException):
            await wf.run(job_input, mock.Mock())

    exit_cleanup.assert_awaited_once()
    validation_response, _job_id, _home_dir = exit_cleanup.call_args[0]
    assert ('Column xyz is missing from table abc.'
            in validation_response.error_message())
