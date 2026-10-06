# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

import kumo_s3
from botocore.config import Config
from kumo_core.sources.errors import ReconnectRequiredError
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumo_core.sources.object_store_backend import ObjectStoreSourceBackend
from kumo_core.sources.object_store_sessions import (
    ObjectStorePaths,
    ObjectStoreSessions,
)
from kumo_core.sources.principals import IdentityRef, Principal
from kumo_core.sources.types import ReadMode
from kumo_s3.session import anonymous_filesystem, arrow_connection_options
from kumoapi.data_source import (
    DataSourceType,
    S3AccessKeyAuthentication,
    S3AccessKeyCredentials,
    S3AmbientAuthentication,
    S3AssumeRoleAuthentication,
    S3AssumeRoleCredentials,
    S3ConnectionConfig,
    S3ConnectorResourceConfig,
    S3IdentityConfig,
    S3RootedConnectorTarget,
)
from kumoapi.s3 import (
    normalize_s3_destination_relative_path,
    normalize_s3_destination_uri,
    normalize_s3_source_uri,
)
from kumoapi.source_table import S3RelativeSource, S3SourceTable, S3URISource
from pyarrow.fs import S3FileSystem
from pydantic import SecretStr

#: What an S3 identity resolves to: a ready filesystem, or a
#: ``boto3.session.Session`` whose resolved credentials are read off it.
S3Session = Union[S3FileSystem, Any]


class S3Sessions(ObjectStoreSessions):
    r"""Resolving a connector and a table name to a reader over a live bucket.

    The source backend and the sink backend both subclass this: a read and a
    write of one connector therefore execute the same :meth:`_filesystem` and
    cannot resolve to two different principals.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an identity ID or the name it was created
            under. Unset is the common case, and the one the ambient
            credential chain serves.
        storage_namespace: The storage system this process reaches, recorded
            in every graph coordinate a table contributes. Defaults to
            ordinary AWS S3; name an S3-compatible service explicitly, or its
            buckets share graph IDs with AWS buckets of the same name.
    """

    namespace = kumo_s3.AWS_NAMESPACE
    rooted = S3RootedConnectorTarget
    resource_config = S3ConnectorResourceConfig
    paths = ObjectStorePaths(
        scheme='s3',
        provider='S3',
        source_uri=normalize_s3_source_uri,
        destination_uri=normalize_s3_destination_uri,
        destination_path=normalize_s3_destination_relative_path,
    )

    def _resolved_session(
            self, connector: Any, bucket: str
    ) -> tuple[S3Session, bool, Optional[S3ConnectionConfig]]:
        principal = self._principal(connector)
        with kumo_s3.translated(f'Signing for {bucket}', table=bucket):
            session = (_session_for(principal)
                       if principal is not None else kumo_s3.ambient_session())
        connection = (principal.config.connection if principal is not None
                      and isinstance(principal.config, S3IdentityConfig) else
                      None)
        return session, principal is None or _is_ambient(
            principal.config), connection

    def _source_access(self, connector: Any, bucket: str,
                       region: Optional[str]) -> ObjectStoreAccess:
        session, allow_anonymous, connection = self._resolved_session(
            connector, bucket)
        if isinstance(session, S3FileSystem):
            return ObjectStoreAccess(session)
        _require_credentials(session)
        with kumo_s3.translated(f'Signing for {bucket}', table=bucket):
            return kumo_s3.source_access(
                bucket, session=session,
                default_region=(connection.region or region)
                if connection else region, allow_anonymous=allow_anonymous,
                endpoint_url=connection.endpoint_url if connection else None,
                options=arrow_connection_options(connection.endpoint_url,
                                                 connection.addressing_style)
                if connection else None, client_config=Config(
                    signature_version='s3v4', s3={
                        'addressing_style': connection.addressing_style
                    }) if connection else None)

    def _filesystem(self, connector: Any, bucket: str,
                    region: Optional[str]) -> S3FileSystem:
        r"""A ready filesystem is taken as it is: it may address a service
        that is not AWS at all, and probing its bucket name against AWS would
        answer for somebody else's bucket.
        """

        session, allow_anonymous, connection = self._resolved_session(
            connector, bucket)
        if isinstance(session, S3FileSystem):
            return session
        with kumo_s3.translated(f'Signing for {bucket}', table=bucket):
            _require_credentials(session)
            # Resolve through the package so applications/tests replacing its
            # filesystem factory affect both source and sink sessions.
            resolved = kumo_s3.filesystem(
                bucket, session=session,
                default_region=(connection.region or region)
                if connection else region,
                endpoint_url=connection.endpoint_url if connection else None,
                options=arrow_connection_options(connection.endpoint_url,
                                                 connection.addressing_style)
                if connection else None)
        if resolved is not None:
            return resolved
        if not allow_anonymous:
            # An identity that names its own material and resolved none of it
            # must not quietly become somebody else: pyarrow's chain would
            # sign as whatever ambient role this host carries, and a read that
            # succeeded would have read as the wrong principal.
            raise ReconnectRequiredError(
                f"The selected identity resolved no usable AWS "
                f"credentials, so {bucket} cannot be signed for. Re-create it "
                f"with current credentials, or leave the connector's identity "
                f"unset to sign with the ambient chain.", table=bucket)
        # No credential resolved: pyarrow's own chain, which is what serves a
        # public bucket, and what the platform falls back to for one. An
        # ambient identity reaches here too -- it is the unset case named out
        # loud, and resolves the same way.
        if connection is not None:
            return anonymous_filesystem(
                bucket, region=connection.region or region
                or session.region_name,
                options=arrow_connection_options(connection.endpoint_url,
                                                 connection.addressing_style))
        from pyarrow.fs import FileSystem
        filesystem, _ = FileSystem.from_uri(f's3://{bucket}')
        return filesystem


class S3SourceBackend(S3Sessions, ObjectStoreSourceBackend):
    r"""Resolves source tables by reading Amazon S3 in this process.

    Installed as ``pip install 'kumoai-studio[s3]'``. With no identity named
    the reader signs with the ambient AWS credential chain, which is what
    makes an instance profile or ``aws configure`` work with no further
    setup.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an :class:`~kumoai_studio.identity.Identity`, an
            identity ID, or the name it was created under.
        storage_namespace: The storage system this process reaches, recorded
            in every graph coordinate an S3 table contributes. Defaults to
            ordinary AWS S3; name an S3-compatible service explicitly, or its
            buckets share graph IDs with AWS buckets of the same name.
        mode: A :class:`~kumo_core.sources.types.ReadMode` to force. An object
            store offers only ``streamed``, so this exists to assert that.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumo_s3 import S3SourceBackend  # doctest: +SKIP
        >>> kumo.set_backend(kumo.LocalBackend(  # doctest: +SKIP
        ...     sources={kumo.DataSourceType.S3: S3SourceBackend()}))
        >>> events = kumo.S3Connector(  # doctest: +SKIP
        ...     name='events', root_dir='s3://company-data/events/')
    """

    source_type = DataSourceType.S3

    relative_source = S3RelativeSource
    uri_source = S3URISource
    source_table = S3SourceTable

    def __init__(
        self,
        identity: Optional[IdentityRef] = None,
        storage_namespace: Optional[str] = None,
        mode: Optional[ReadMode] = None,
    ) -> None:
        super().__init__(identity, storage_namespace)
        self.mode = mode

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(identity={self._identity!r}, '
                f'storage_namespace={self.storage_namespace!r})')


def _session_for(principal: Principal) -> S3Session:
    r"""Builds the AWS session an S3 identity's configuration describes.

    Ambient authentication holds nothing, so the host's own credential chain
    is what a caller gets -- the same thing an unset identity resolves to,
    named explicitly. An access key holds its material. Assume-role exchanges
    a role for temporary credentials through STS, which is the one kind that
    costs a call.
    """
    config = principal.config
    if not isinstance(config, S3IdentityConfig):
        raise ValueError(
            f"Identity {principal.name!r} is a {config.type.value} "
            f"identity, so it cannot sign for S3.")
    authentication = config.authentication
    credentials = principal.credentials

    if isinstance(authentication, S3AccessKeyAuthentication):
        if not isinstance(credentials, S3AccessKeyCredentials):
            raise ValueError(
                f"Identity {principal.name!r} authenticates with an access "
                f"key but holds {type(credentials).__name__} credentials.")
        import boto3
        return boto3.Session(
            aws_access_key_id=credentials.access_key_id.get_secret_value(),
            aws_secret_access_key=(
                credentials.secret_access_key.get_secret_value()),
            aws_session_token=_optional(credentials.session_token),
            region_name=config.connection.region if config.connection else None,
        )

    if isinstance(authentication, S3AssumeRoleAuthentication):
        import boto3
        external_id = (_optional(credentials.external_id) if isinstance(
            credentials, S3AssumeRoleCredentials) else None)
        arguments = {
            'RoleArn': authentication.role_arn,
            'RoleSessionName': 'kumo-local-backend',
        }
        if external_id is not None:
            arguments['ExternalId'] = external_id
        assumed = boto3.Session().client('sts').assume_role(**arguments)
        issued = assumed['Credentials']
        return boto3.Session(
            aws_access_key_id=issued['AccessKeyId'],
            aws_secret_access_key=issued['SecretAccessKey'],
            aws_session_token=issued['SessionToken'],
            region_name=config.connection.region if config.connection else None,
        )

    if config.connection is not None and config.connection.region is not None:
        import boto3
        return boto3.Session(region_name=config.connection.region)
    return kumo_s3.ambient_session()


def _optional(value: Optional[SecretStr]) -> Optional[str]:
    return None if value is None else value.get_secret_value()


def _is_ambient(config: Any) -> bool:
    r"""Whether an identity carries no material of its own, and so resolves
    to exactly what an unset identity does.
    """
    return isinstance(config, S3IdentityConfig) and isinstance(
        config.authentication, S3AmbientAuthentication)


def _require_credentials(session: S3Session) -> None:
    r"""Checked here rather than deep inside a read."""
    if not callable(getattr(session, 'get_credentials', None)):
        raise TypeError(
            f"An S3 session must be a 'pyarrow.fs.S3FileSystem' or a "
            f"'boto3.session.Session'; got {type(session).__name__!r}. "
            f"Leaving the identity unset is also supported: the reader then "
            f"signs with the ambient AWS credential chain.")


#: STS's answers that mean the material itself is unusable, rather than STS
#: being unreachable or the caller's account being in a state it can do
#: nothing about. These are the ones a caller fixes by reconnecting, so they
#: are reported as :class:`~kumo_core.sources.errors.ReconnectRequiredError`
#: and everything else is left to :func:`kumo_s3.translated`.
_LAPSED_STS_CODES = frozenset({
    'AuthFailure',
    'ExpiredToken',
    'InvalidAccessKeyId',
    'InvalidClientTokenId',
    'SignatureDoesNotMatch',
    'UnrecognizedClientException',
})


def _authenticate(principal: Principal, session: S3Session) -> None:
    r"""One round trip to AWS proving :obj:`session`'s credentials sign.

    ``sts:GetCallerIdentity`` is the call AWS grants to every principal
    unconditionally -- it cannot be denied by policy -- so it separates "these
    credentials are not valid" from "this identity is not allowed to read that
    bucket", and a scoped identity that may touch exactly one prefix still
    passes it. A bucket operation would conflate the two and fail an identity
    that works.

    Skipped for an identity pinned to a non-AWS endpoint: ``endpoint_url``
    names an *S3* service (MinIO, Ceph, an S3-compatible appliance), whose
    credentials AWS's STS has never heard of, and calling it would fail every
    such identity. Their credentials are still resolved here, and are put to
    the service itself on first read -- the honest bound, stated in
    :func:`probe` rather than left for a caller to discover.
    """
    from botocore.exceptions import ClientError

    credentials = session.get_credentials()
    frozen = None if credentials is None else \
        credentials.get_frozen_credentials()
    if frozen is None or not frozen.access_key or not frozen.secret_key:
        raise ReconnectRequiredError(
            f"Identity {principal.name!r} resolved no usable AWS "
            f"credentials. Re-create it with material the AWS credential "
            f"chain can resolve, or leave the identity unset to sign with "
            f"the ambient chain.")

    connection = getattr(principal.config, 'connection', None)
    if connection is not None and connection.endpoint_url:
        return

    try:
        session.client('sts').get_caller_identity()
    except ClientError as error:
        code = error.response.get('Error', {}).get('Code', '')
        if code in _LAPSED_STS_CODES:
            raise ReconnectRequiredError(
                f"Identity {principal.name!r} did not authenticate with AWS "
                f"({code}). Re-create it with current credentials.") from None
        raise


def probe(principal: Principal) -> None:
    r"""Authenticates :obj:`principal`'s credentials against AWS, and drops
    what it built.

    Resolving the credentials is not enough on its own: a mistyped secret key
    resolves perfectly well and only fails when something signs with it, so
    the material is put to STS here -- the same thing Snowflake's ``SELECT
    1``, BigQuery's ``refresh`` and Databricks' ``authenticate`` do for their
    identities. What this does *not* answer is whether the identity may read
    any particular bucket; that is a grant on the bucket, and a connector
    naming one reports on it at first read.

    One identity is not put to AWS: one pinned to a non-AWS ``endpoint_url``,
    whose keys AWS never issued. Its credentials are resolved and checked for
    shape, and the service that issued them is what validates them, at first
    read.

    S3 opens a connection per call and caches nothing here, so there is no
    ``close`` beside this: a plugin that defines none says exactly that, and
    the identity area reads it that way.
    """
    from pyarrow.fs import S3FileSystem

    session = _session_for(principal)
    if isinstance(session, S3FileSystem):
        return
    _require_credentials(session)
    with kumo_s3.translated(f'Testing identity {principal.name!r}'):
        _authenticate(principal, session)
