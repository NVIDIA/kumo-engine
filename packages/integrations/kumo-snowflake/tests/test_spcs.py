# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from kumo_core.sources.principals import Principal
from kumo_snowflake import session, spcs
from kumo_snowflake.spcs import (
    ACCOUNT_ENV,
    HOST_ENV,
    spcs_identity_create_args,
)
from kumoapi.data_source import SnowflakeAuthenticationKind


@pytest.fixture
def container(monkeypatch, tmp_path):
    """A mounted token and the two variables the runtime exports beside it.

    The path is redirected rather than made overridable on the function: what
    a caller may pass is the whole question here, and a test that reached for
    an argument nobody else has would not be testing the same code.
    """
    token = tmp_path / "token"
    token.write_text("ey.mounted.token")
    monkeypatch.setattr(spcs, "SPCS_TOKEN_PATH", str(token))
    monkeypatch.setenv(ACCOUNT_ENV, "xy12345")
    monkeypatch.setenv(HOST_ENV, "snowflake-host.internal")
    return token


def test_the_container_environment_supplies_every_field(container):
    args = spcs_identity_create_args("prod-sf")

    assert args.display_name == "prod-sf"
    assert args.config.account == "xy12345"
    assert args.config.host == "snowflake-host.internal"
    # No secret: the token is a file, so there is nothing to store, and the
    # schema refuses credentials for this kind.
    assert args.credentials is None
    assert args.config.username is None
    assert (args.config.authentication
            is SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE)


def test_the_mounted_path_is_left_for_the_reader_to_resolve(container):
    """Storing it would freeze today's mount point into every identity
    created, when the reader resolves it on each connect."""
    args = spcs_identity_create_args("prod-sf")

    assert args.config.token_path is None


def test_an_oauth_container_session_keeps_the_server_session_alive(
        container, monkeypatch):
    """Long local work must not leave the final Snowflake write unauthenticated."""
    monkeypatch.setattr(session, "DEFAULT_TOKEN_PATH", str(container))
    args = spcs_identity_create_args("prod-sf")

    settings = session._authentication(
        Principal(name="prod-sf", config=args.config, credentials=None))

    assert settings["client_session_keep_alive"] is True


def test_a_host_the_runtime_does_not_export_is_left_unset(
        container, monkeypatch):
    """Absent, the driver derives the endpoint from the account."""
    monkeypatch.delenv(HOST_ENV, raising=False)

    assert spcs_identity_create_args("prod-sf").config.host is None


def test_only_the_name_and_description_are_the_callers(container):
    """Everything describing the principal comes from the runtime, so there is
    nothing to pass that could make this describe a different one."""
    args = spcs_identity_create_args("prod-sf", description="read replica")

    assert args.description == "read replica"
    with pytest.raises(TypeError):
        spcs_identity_create_args("prod-sf", account="somewhere-else")


def test_no_mounted_token_means_this_is_not_a_container(monkeypatch, tmp_path):
    """The likeliest way to arrive here is calling this on a laptop, so the
    message names that rather than the missing file."""
    monkeypatch.setattr(spcs, "SPCS_TOKEN_PATH", str(tmp_path / "absent"))
    monkeypatch.setenv(ACCOUNT_ENV, "xy12345")

    with pytest.raises(ValueError, match="not a snowpark container services"):
        spcs_identity_create_args("prod-sf")


def test_a_token_without_an_account_is_reported_separately(
        container, monkeypatch):
    """A runtime sets both. Reporting this as "not a container" would send a
    caller looking for the wrong thing."""
    monkeypatch.delenv(ACCOUNT_ENV, raising=False)

    with pytest.raises(ValueError, match="is mounted but"):
        spcs_identity_create_args("prod-sf")
