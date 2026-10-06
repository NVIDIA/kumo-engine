# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What ``kumoai_studio.test_credentials`` runs for an S3 identity.

The point of these is that a probe *probes*: resolving credentials proves
nothing about them, because a mistyped secret key resolves perfectly well and
only fails when something signs with it. So the assertions are about the call
that goes to AWS, and about the two cases where deliberately none does.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeSession
from kumo_core.sources.errors import ReconnectRequiredError
from kumo_core.sources.principals import Principal
from kumo_s3 import backend
from kumoapi.data_source import (
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AmbientAuthentication,
    S3ConnectionConfig,
    S3IdentityConfig,
)
from pydantic import SecretStr

ENDPOINT = 'https://minio.internal:9000'


class ProbedSession(FakeSession):
    """A boto3 session that records the clients built from it, and answers
    ``GetCallerIdentity`` with :obj:`answer` -- an exception to raise, or the
    response to return.
    """
    def __init__(self, *, answer: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.clients: list = []
        self._answer = answer

    def client(self, service: str) -> Any:
        self.clients.append(service)

        def get_caller_identity() -> Any:
            if isinstance(self._answer, Exception):
                raise self._answer
            return self._answer or {'Arn': 'arn:aws:iam::1:user/kumo'}

        return SimpleNamespace(get_caller_identity=get_caller_identity)


def _client_error(code: str) -> Exception:
    from botocore.exceptions import ClientError
    return ClientError({'Error': {
        'Code': code,
        'Message': code
    }}, 'GetCallerIdentity')


def _principal(config: S3IdentityConfig, name: str = 'aws') -> Principal:
    return Principal(
        name=name, config=config, credentials=S3AccessKeyCredentials(
            access_key_id=SecretStr('AKIA'),
            secret_access_key=SecretStr('secret')))


def _access_key(*, endpoint_url: str | None = None) -> S3IdentityConfig:
    connection = (None if endpoint_url is None else S3ConnectionConfig(
        endpoint_url=endpoint_url, region='us'))
    return S3IdentityConfig(authentication=S3AccessKeyAuthentication(),
                            connection=connection)


@pytest.fixture
def session(monkeypatch):
    """Whatever the test hands back from the connect step."""
    def use(built: Any) -> Any:
        monkeypatch.setattr(backend, '_session_for', lambda principal: built)
        return built

    return use


def test_a_probe_authenticates_rather_than_shape_checking(session):
    """The regression this exists for: `get_credentials` returning something
    callable says only that boto3 built a session, which it does for any
    material at all.
    """
    probed = session(ProbedSession())

    backend.probe(_principal(_access_key()))

    assert probed.clients == ['sts']


def test_a_secret_key_aws_rejects_is_reported_as_a_lapsed_credential(session):
    """A typo'd secret key resolves, signs, and is refused by AWS alone.
    Reported as reconnect-required so the caller is sent to re-enter it rather
    than to widen a policy that was never the problem.
    """
    session(ProbedSession(answer=_client_error('SignatureDoesNotMatch')))

    with pytest.raises(ReconnectRequiredError, match='SignatureDoesNotMatch'):
        backend.probe(_principal(_access_key()))


@pytest.mark.parametrize('code', ['InvalidClientTokenId', 'ExpiredToken'])
def test_the_other_answers_that_mean_the_material_is_stale(session, code):
    session(ProbedSession(answer=_client_error(code)))

    with pytest.raises(ReconnectRequiredError, match=code):
        backend.probe(_principal(_access_key()))


def test_an_unrelated_client_error_is_not_dressed_up_as_a_reconnect(session):
    """`translated` still classifies it; what must not happen is a service
    outage being reported as "your credentials expired".
    """
    from kumo_core.sources.errors import TransientSourceError
    session(ProbedSession(answer=_client_error('ServiceUnavailable')))

    with pytest.raises(TransientSourceError) as raised:
        backend.probe(_principal(_access_key()))
    assert not isinstance(raised.value, ReconnectRequiredError)


def test_a_session_that_resolves_nothing_is_refused_before_sts(session):
    probed = session(ProbedSession(credentials=None))

    with pytest.raises(ReconnectRequiredError, match="'aws'"):
        backend.probe(_principal(_access_key()))
    assert probed.clients == []


def test_a_half_resolved_credential_is_refused_too(session):
    partial = SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(
        access_key='key', secret_key='', token=None))
    probed = session(ProbedSession(credentials=partial))

    with pytest.raises(ReconnectRequiredError):
        backend.probe(_principal(_access_key()))
    assert probed.clients == []


def test_a_non_aws_endpoint_is_not_put_to_aws_sts(session):
    """`endpoint_url` names an S3-compatible service whose keys AWS has never
    issued; calling STS would fail every MinIO identity there is.
    """
    probed = session(ProbedSession())

    backend.probe(_principal(_access_key(endpoint_url=ENDPOINT)))

    assert probed.clients == []


def test_an_ambient_identity_is_authenticated_like_any_other(session):
    """It resolves the host's own chain, and a host that resolves nothing
    usable should hear so here rather than at first read.
    """
    probed = session(ProbedSession())

    backend.probe(
        Principal(
            name='ambient',
            config=S3IdentityConfig(authentication=S3AmbientAuthentication()),
            credentials=None))

    assert probed.clients == ['sts']


def test_a_ready_filesystem_is_taken_as_already_connected(session, monkeypatch):
    """A caller that bound its own `S3FileSystem` has no boto3 session to put
    to STS, and the filesystem may not address AWS at all.
    """
    from pyarrow.fs import S3FileSystem
    session(S3FileSystem(anonymous=True))

    backend.probe(_principal(_access_key()))
