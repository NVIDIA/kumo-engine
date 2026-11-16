# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient
from pytest import LogCaptureFixture

from kumo.testing.rest import post_and_expect_success
from rest.config.status import EmptyResponse
from rest.routers.log import LogErrorArgs, LogLevel


def test_log_ui_error(client: TestClient, caplog: LogCaptureFixture):
    testMessage = "test message"
    args = LogErrorArgs(message=testMessage, level=LogLevel.ERROR,
                        stacktrace=None)
    post_and_expect_success(client, "/v0/log/log_ui_error", args,
                            EmptyResponse)
    assert testMessage in caplog.text
