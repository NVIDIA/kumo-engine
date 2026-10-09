# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pool.errors helpers (no-space detection across the Temporal
serialization boundary)."""
from temporalio.exceptions import ActivityError, ApplicationError

from pool.errors import NO_SPACE_ERROR_TYPE, is_no_space_activity_error


def _activity_error_with_type(app_type: str) -> Exception:
    """ActivityError -> ApplicationError(type=app_type) chain, exactly as a
    workflow sees an activity failure."""
    try:
        raise ApplicationError('disk full', type=app_type)
    except ApplicationError as app_err:
        try:
            raise ActivityError(
                message='activity failed',
                scheduled_event_id=1,
                started_event_id=1,
                identity='x',
                activity_type='ensure_snapshot',
                activity_id='a',
                retry_state=None,
            ) from app_err
        except ActivityError as activity_err:
            return activity_err


def test_detects_no_space_application_error():
    err = _activity_error_with_type(NO_SPACE_ERROR_TYPE)
    assert is_no_space_activity_error(err) is True


def test_ignores_other_application_error_types():
    err = _activity_error_with_type('NonRetryableDiskgraphError')
    assert is_no_space_activity_error(err) is False


def test_ignores_error_without_cause():
    assert is_no_space_activity_error(RuntimeError('boom')) is False
