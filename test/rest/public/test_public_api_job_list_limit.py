# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for finding 020: the public job-list endpoints must reject a
non-positive ``limit`` at the FastAPI boundary with a clean 422 instead of
silently mis-slicing (HTTP 200 wrong/empty) or 500-ing on a negative SQL LIMIT.

Before the fix the four ``list`` handlers typed ``limit: Optional[int] = None``
with no lower bound, so ``limit=-1`` passed validation and reached
``job_entries[-limit:]`` (front-truncation -> silent 200) on three endpoints and
a SQL ``LIMIT -1`` (Postgres error -> raw 500) on ``prediction_jobs``. The fix
constrains the param to ``Annotated[Optional[int], Query(ge=1)]`` so the
validation fires uniformly before any handler body runs.

These tests exercise the assembled public app (``public_api_client``), so they
prove the real FastAPI query-param constraint, which a direct handler call would
bypass. The negative/zero cases fail against the pre-fix source (200 or 500, no
``limit`` validation error) and pass against the fix.
"""
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID

# The four public job-list collection routes (mounted under /v1). Every one
# accepts the same ``limit`` pagination param and must validate it identically.
_JOB_LIST_PATHS = [
    "/v1/training_jobs",
    "/v1/gentraintable_jobs",
    "/v1/genpredtable_jobs",
    "/v1/prediction_jobs",
]


def _limit_validation_errors(response) -> list:
    """Return the 422 validation errors scoped to the ``limit`` query param.

    Empty when the response is not a 422 or carries no ``limit``-scoped error.
    """
    if response.status_code != HTTPStatus.UNPROCESSABLE_ENTITY:
        return []
    detail = response.json().get("detail", [])
    if not isinstance(detail, list):
        return []
    return [d for d in detail if d.get("loc") == ["query", "limit"]]


@pytest.mark.parametrize("path", _JOB_LIST_PATHS)
@pytest.mark.parametrize("bad_limit", [-1, -5, 0])
def test_job_list_rejects_non_positive_limit_with_422(
        public_api_client: TestClient, path: str, bad_limit: int) -> None:
    response = public_api_client.get(
        path, params={
            "project_id": DEFAULT_PROJECT_ID,
            "limit": bad_limit
        })

    # Fixed contract: a clean 422 (not a silent 200 slice, not a raw 500).
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
        f"{path}?limit={bad_limit} expected 422, got "
        f"{response.status_code}: {response.text}")

    limit_errors = _limit_validation_errors(response)
    assert len(limit_errors) == 1, (
        f"{path}?limit={bad_limit} expected exactly one limit-scoped "
        f"validation error, got: {response.text}")
    assert "greater than or equal to 1" in limit_errors[0]["msg"], (
        f"{path}?limit={bad_limit} 422 message should name the >=1 "
        f"constraint, got: {limit_errors[0]}")


@pytest.mark.parametrize("path", _JOB_LIST_PATHS)
def test_job_list_accepts_positive_limit(public_api_client: TestClient,
                                         path: str) -> None:
    # limit=1 is the ge=1 boundary; larger values exercise the same validation
    # branch. The request must clear the constraint and succeed end to end.
    response = public_api_client.get(
        path, params={
            "project_id": DEFAULT_PROJECT_ID,
            "limit": 1
        })

    assert response.status_code == HTTPStatus.OK, (
        f"{path}?limit=1 expected 200, got "
        f"{response.status_code}: {response.text}")


@pytest.mark.parametrize("path", _JOB_LIST_PATHS)
def test_job_list_accepts_omitted_limit(public_api_client: TestClient,
                                        path: str) -> None:
    # Omitting ``limit`` entirely stays valid (default None -> return all).
    response = public_api_client.get(path,
                                     params={"project_id": DEFAULT_PROJECT_ID})

    assert response.status_code == HTTPStatus.OK, (
        f"{path} with no limit expected 200, got "
        f"{response.status_code}: {response.text}")
