# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import os
from typing import Generator
from unittest.mock import MagicMock

import pytest
import requests_mock
from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID

from kumoai_studio import global_state, init
from kumoai_studio.client.endpoints import Endpoint, HTTPMethod

# Not mock:// due to https://stackoverflow.com/a/76056002
MOCK_URL = "http://kumo.ai"


def pytest_addoption(parser):
    parser.addoption('--runintegration', action='store_true', default=False,
                     help="run integration tests")


def pytest_collection_modifyitems(config, items):
    # check if you got an option like --key=snowflake
    if not config.getoption("--runintegration"):
        skip_integ = pytest.mark.skip(reason="integration test")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integ)
    else:
        skip_integ = pytest.mark.skip(reason="no integration test")
        for item in items:
            if ("integration" not in item.keywords):
                item.add_marker(skip_integ)


@pytest.fixture(scope="class")
def mock_api() -> Generator[requests_mock.Mocker, None, None]:
    with requests_mock.Mocker() as m:
        yield m


@pytest.fixture(scope="class")
def setup_mock_client(mock_api):
    # Create the client:
    mock_api.get(f"{MOCK_URL}/v1/authentication", status_code=204)
    mock_api.get(
        f"{MOCK_URL}/v1/groups", json=[{
            'id': DEFAULT_GROUP_ID,
            'slug': 'default',
            'display_name': 'Default',
        }])
    mock_api.get(
        f"{MOCK_URL}/v1/projects/{DEFAULT_PROJECT_ID}", json={
            'id': DEFAULT_PROJECT_ID,
            'slug': 'default',
            'group_id': DEFAULT_GROUP_ID,
            'display_name': 'Default',
        })
    init(url=MOCK_URL, api_key="DISABLED")

    # Run the test:
    yield

    # Cleanup, do not re-use clients across unit tests:
    global_state.clear()


@pytest.fixture(scope="class")
def setup_integration_client():
    # Create the client:
    init(
        url=os.environ["KUMO_API_ENDPOINT"],
        api_key=os.environ["KUMO_API_KEY"],
    )

    # Run the test:
    yield

    # Cleanup, do not re-use clients across unit tests:
    global_state.clear()


@pytest.fixture
def protocol_members():
    """Return the public member names a Protocol declares.

    ``__protocol_attrs__`` only exists on Python 3.12+, and the SDK supports
    3.10, so read the class dict instead: a Protocol's declared members are
    exactly its non-dunder class attributes.
    """
    def members(protocol) -> list:
        return sorted(name for name in vars(protocol)
                      if not name.startswith('_'))

    return members


@pytest.fixture
def mock_backend(monkeypatch) -> Generator[MagicMock, None, None]:
    """Install a mock execution backend for the duration of one test.

    Use this to assert on which backend operations a relational object
    performs, without standing up an HTTP mock. The areas are reached as
    ``mock_backend.graph``, ``.table``, ``.pquery``, ``.train_table``,
    ``.training``, ``.prediction_table``, and ``.prediction``.
    """
    backend = MagicMock()
    monkeypatch.setattr('kumoai_studio.backend._backend', backend)
    yield backend


@pytest.fixture
def disable_artifact_export_polling(
        monkeypatch) -> Generator[None, None, None]:
    """Keep request-only export tests from scheduling background pollers."""
    pending_futures = []

    def create_pending_future(coroutine):
        coroutine.close()
        future = concurrent.futures.Future()
        pending_futures.append(future)
        return future

    monkeypatch.setattr('kumoai_studio.artifact_export.job.create_future',
                        create_pending_future)
    yield

    for future in pending_futures:
        future.cancel()


def get_mock_method(mock_api, endpoint: Endpoint):
    method_map = {
        HTTPMethod.GET: mock_api.get,
        HTTPMethod.POST: mock_api.post,
        HTTPMethod.PATCH: mock_api.patch,
        HTTPMethod.DELETE: mock_api.delete,
    }
    if endpoint.method not in method_map:
        raise ValueError(f"Unsupported HTTP method: {endpoint.method}")

    return method_map[endpoint.method]
