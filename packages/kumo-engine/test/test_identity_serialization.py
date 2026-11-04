# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity registration follows connectors into a new execution process."""

import pickle

import pytest
from kumo.engine import SnowflakeConnector, SnowflakeIdentity
from kumo_core.sources.principals import PrincipalStore


@pytest.mark.parametrize("already_registered", [False, True])
@pytest.mark.parametrize("password_auth", [False, True])
def test_unpickled_connector_resolves_identity(
    monkeypatch: pytest.MonkeyPatch, already_registered: bool, password_auth: bool
) -> None:
    """Pickle restores source identity registration without duplicating it."""
    sender = PrincipalStore()
    monkeypatch.setattr("kumo.engine.connector.identity.store", sender)
    identity = (
        SnowflakeIdentity(
            account="test-account", username="test-user", password="test-only-password"
        )
        if password_auth
        else SnowflakeIdentity(account="test-account")
    )
    connector = SnowflakeConnector(
        identity=identity, warehouse="WH", database="DB", schema_name="SC"
    )
    payload = pickle.dumps(connector)
    receiver = sender if already_registered else PrincipalStore()
    monkeypatch.setattr("kumo.engine.connector.identity.store", receiver)

    restored = pickle.loads(payload)

    assert receiver.names() == [identity.slug]
    assert receiver.resolve(restored.identity) == identity.slug
    assert receiver.get(identity.slug) == sender.get(identity.slug)
    assert restored.identity_id == connector.identity_id
