# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from examples.flows.setup import s3 as s3_example


def test_s3_example_uses_explicit_identity_when_configured(
        monkeypatch) -> None:
    connector = Mock()
    factory = Mock(return_value=connector)
    monkeypatch.setattr(s3_example.kumo, "S3Connector", factory)

    assert s3_example._create_any_uri_connector("identity-123") is connector
    factory.assert_called_once_with(identity="identity-123")


def test_s3_example_no_argument_flow_requests_ambient_identity(
        monkeypatch) -> None:
    connector = Mock()
    factory = Mock(return_value=connector)
    monkeypatch.setattr(s3_example.kumo, "S3Connector", factory)

    assert s3_example._create_any_uri_connector(None) is connector
    factory.assert_called_once_with()
