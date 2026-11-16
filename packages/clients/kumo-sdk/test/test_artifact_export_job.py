# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest import mock

import pytest
from kumoapi.common import JobStatus

from kumoai_studio.artifact_export.job import (
    ArtifactExportResult,
    _poll_export,
)


def test_poll_export_done_returns_result_without_sleeping():
    with mock.patch(
            'kumoai_studio.artifact_export.job.get_export_status',
            return_value=JobStatus.DONE,
    ) as get_status, mock.patch(
            'kumoai_studio.artifact_export.job.asyncio.sleep',
            new_callable=mock.AsyncMock,
    ) as sleep:
        result = asyncio.run(_poll_export('export-job', 'project-id'))

    assert isinstance(result, ArtifactExportResult)
    assert result.job_id == 'export-job'
    assert result.kumo_project_id == 'project-id'
    get_status.assert_called_once_with('export-job', 'project-id')
    sleep.assert_not_awaited()


@pytest.mark.parametrize('status', [JobStatus.FAILED, JobStatus.CANCELLED])
def test_poll_export_normal_terminal_failures_do_not_sleep(status):
    with mock.patch(
            'kumoai_studio.artifact_export.job.get_export_status',
            return_value=status,
    ) as get_status, mock.patch(
            'kumoai_studio.artifact_export.job.asyncio.sleep',
            new_callable=mock.AsyncMock,
    ) as sleep, pytest.raises(RuntimeError, match=f'job status {status}'):
        asyncio.run(_poll_export('export-job', 'project-id'))

    get_status.assert_called_once_with('export-job', 'project-id')
    sleep.assert_not_awaited()


def test_poll_export_treats_missing_history_as_terminal_failure():
    with mock.patch(
            'kumoai_studio.artifact_export.job.get_export_status',
            side_effect=[JobStatus.RUNNING, JobStatus.MISSING],
    ) as get_status, mock.patch(
            'kumoai_studio.artifact_export.job.asyncio.sleep',
            new_callable=mock.AsyncMock,
    ) as sleep, pytest.raises(
            RuntimeError,
            match=(r'Export job export-job is missing.*'
                   r'history may have expired'),
    ):
        asyncio.run(_poll_export('export-job', 'project-id'))

    assert get_status.call_args_list == [
        mock.call('export-job', 'project-id'),
        mock.call('export-job', 'project-id'),
    ]
    sleep.assert_awaited_once_with(10)
