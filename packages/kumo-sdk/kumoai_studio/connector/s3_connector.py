# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional
from urllib.parse import urlsplit

from kumoapi.data_source import (
    ConnectorResponse,
    DataSourceType,
    S3AnyURIConnectorTarget,
    S3ConnectorResourceConfig,
    S3RootedConnectorTarget,
)
from kumoapi.source_table import S3RelativeSource, S3URISource
from typing_extensions import Self, override

from kumoai_studio.connector.file_connector import (
    FileURI,
    ObjectStoreConnector,
)
from kumoai_studio.identity import Identity


class S3URI(FileURI):
    """Strict S3 URI parser."""

    _schemes = ('s3', )

    @override
    def validate(self) -> Self:
        super().validate()
        path = urlsplit(self.uri).path
        if path.startswith('//') or re.search(r'%(?:2f|5c)', path,
                                              flags=re.IGNORECASE):
            raise ValueError(
                'S3 paths cannot contain ambiguous or encoded separators.')
        return self

    @classmethod
    def decoded_prefix(cls, path: str) -> str:
        """Decode one canonical root prefix without collapsing separators."""
        if path.startswith('//') or re.search(r'%(?:2f|5c)', path,
                                              flags=re.IGNORECASE):
            raise ValueError(
                'S3 prefixes cannot contain ambiguous or encoded separators.')
        decoded = cls._decode_path(path)
        if decoded.startswith('/'):
            decoded = decoded[1:]
        if decoded.endswith('/'):
            decoded = decoded[:-1]
        if decoded.startswith('/') or decoded.endswith('/'):
            raise ValueError('S3 prefix has an ambiguous separator.')
        if (cls._unsafe_decoded_character(decoded)
                or (decoded and any(part in {'', '.', '..'}
                                    for part in decoded.split('/')))):
            raise ValueError('S3 prefix contains an empty, dot, or unsafe '
                             'path component.')
        return decoded

    @classmethod
    def normalize_relative_table_name(cls, name: str) -> str:
        """Return a decoded, canonical rooted S3 source coordinate."""
        return S3RelativeSource(path=name).path


class S3Connector(ObjectStoreConnector):
    r"""Connect one Kumo project to Amazon S3 or an S3-compatible service.

    Two target modes are available:

    * Set ``root_dir`` to constrain the connector to one bucket and optional
      prefix. Table names are arbitrary-depth relative paths; a same-root full
      ``s3://`` URI is accepted for convenience.
    * Leave ``root_dir=None`` for rootless mode. Every read and write uses a
      canonical full ``s3://`` URI; discovery remains unavailable.

    ``identity=None`` selects Kumo's canonical ambient AWS workload identity.
    On the Kumo platform this is not an anonymous or local AWS
    credential-chain fallback: a deployment may disable ambient S3 or restrict
    its workload role, in which case the server returns an authorization
    error. Under
    :class:`~kumo_s3.backend.S3SourceBackend` there is no
    server and no workload role, so ``identity=None`` does mean the local AWS
    credential chain -- an instance profile, ``aws configure``, or the
    environment. Pass an immutable S3 identity ID (or
    :class:`~kumoai_studio.identity.Identity`) to use an administrator-managed
    assume-role or access-key identity instead.

    Custom endpoint, signing-region, addressing-style and credential settings
    belong to the identity and are honoured by both hosted and local backends.
    Hosted deployments also support administrator-managed CA settings. For an
    S3-compatible service, name its storage namespace on the local backend --
    ``S3SourceBackend(storage_namespace='minio:data.example.com')`` -- or its
    buckets share graph IDs with AWS buckets of the same name.

    .. code-block:: python

        import os

        import kumoai_studio

        # Rooted: paths can be nested to any depth below this prefix.
        events = kumoai_studio.S3Connector(
            name="events",
            root_dir="s3://company-data/events/",
            region="us-west-2",
        )
        july = events.table("archive/2026/07/events.parquet")
        july = events.table_from_uri(
            "s3://company-data/events/archive/2026/07/events.parquet"
        )

        # Any-URI: one connector can address unrelated buckets and prefixes.
        s3 = kumoai_studio.S3Connector()
        users = s3.table("s3://customers-prod/deep/v4/users.parquet")
        orders = s3.table("s3://analytics-lake/exports/2026/07/orders/")
        output = kumoai_studio.artifact_export.OutputConfig(
            output_types={"predictions"}, output_connector=s3,
            output_table_name="s3://model-results/churn/2026/07/output",
        )

        # Scope operations without changing kumoai_studio's global active project.
        isolated = kumoai_studio.S3Connector(project_id="project_...")

        # MinIO: create the access-key identity as an administrator, then bind
        # a project connector to its credential-free response.
        minio_identity = kumoai_studio.create_identity(
            kumoai_studio.CreateIdentityArgs(
                display_name="Private MinIO",
                group_ids=["analytics"],
                config=kumoai_studio.S3IdentityConfig(
                    authentication=kumoai_studio.S3AccessKeyAuthentication(),
                    connection=kumoai_studio.S3ConnectionConfig(
                        endpoint_url="https://minio.example.com:9000",
                        region="us-east-1",
                        addressing_style=kumoai_studio.S3AddressingStyle.PATH,
                    ),
                ),
                credentials=kumoai_studio.s3_access_key_credentials(
                    access_key_id=os.environ["MINIO_ACCESS_KEY"],
                    secret_access_key=os.environ["MINIO_SECRET_KEY"],
                ),
            )
        )
        minio = kumoai_studio.S3Connector(
            name="events",
            root_dir="s3://customer-data/events",
            identity=minio_identity,
        )

    Args:
        name: Project-local connector display name. Defaults to ``"s3"``.
        root_dir: Optional S3 bucket/prefix root. ``None`` selects any-URI
            mode. A rooted connector accepts paths at any depth below the
            root; an any-URI connector requires full ``s3://`` URIs.
        identity: Immutable S3 identity ID or credential-free
            :class:`~kumoai_studio.identity.Identity`. ``None`` requests the canonical
            ambient workload identity in both target modes.
        region: Optional bucket region for rooted mode. It is invalid in
            any-URI mode because each operation can address a different
            bucket.
        description: Safe human-readable connector description.
        project_id: Exact Kumo project to use. ``None`` snapshots the active
            project when the connector is constructed. Supplying this argument
            never mutates the global active project.

    Raises:
        :class:`ValueError`: If the target, identity, or project is invalid.
        :class:`~kumoai_studio.exceptions.HTTPException`: If access is denied,
            ambient S3 is disabled, or the name conflicts with a different
            connector in the selected project.

    Note:
        All constructor arguments are keyword-only. Positional construction is
        rejected so target and identity fields cannot be confused.

        Repeating construction with the same project, name, target, and
        resolved identity is idempotent.
    """

    _uri_class = S3URI
    root_dir: Optional[str]

    def __init__(
        self,
        *,
        name: str = 's3',
        root_dir: Optional[str] = None,
        identity: str | Identity | None = None,
        region: Optional[str] = None,
        description: str = '',
        project_id: Optional[str] = None,
        _response: Optional[ConnectorResponse] = None,
    ) -> None:
        if root_dir is None:
            if region is not None:
                raise ValueError(
                    'region is only supported when root_dir is set')
            target = S3AnyURIConnectorTarget()
        else:
            parsed = urlsplit(S3URI(root_dir).validate().uri)
            target = S3RootedConnectorTarget(
                bucket=parsed.netloc,
                prefix=S3URI.decoded_prefix(parsed.path),
                region=region,
            )
        config = S3ConnectorResourceConfig(target=target)
        self.target = target
        self.any_uri = isinstance(target, S3AnyURIConnectorTarget)
        self.bucket = (target.bucket if isinstance(
            target, S3RootedConnectorTarget) else None)
        self.prefix = (target.prefix if isinstance(
            target, S3RootedConnectorTarget) else None)
        self.region = (target.region if isinstance(
            target, S3RootedConnectorTarget) else None)
        self.root_dir = self._root_dir(config)
        self._bind_connector(name, identity, config, description, _response,
                             project_id=project_id)

    @staticmethod
    def _root_dir(config: S3ConnectorResourceConfig) -> Optional[str]:
        target = config.target
        if isinstance(target, S3AnyURIConnectorTarget):
            return None
        return FileURI.canonical_uri('s3', target.bucket, target.prefix)

    @override
    def _normalize_table_reference(self, name: str) -> str:
        if self.any_uri:
            return S3URISource(uri=name).uri
        parsed = urlsplit(name)
        if parsed.scheme or parsed.netloc:
            assert self.root_dir is not None
            name = S3URI(name).relative_table_name(self.root_dir)
        return S3RelativeSource(path=name).path

    @classmethod
    def get_by_name(
        cls,
        name: str = 's3',
        project_id: Optional[str] = None,
    ) -> Self:
        r"""Return an S3 connector from one project.

        Args:
            name: Project-local display name. Defaults to ``"s3"``.
            project_id: Exact Kumo project, or ``None`` for the active project.

        Returns:
            The matching rooted or any-URI :class:`S3Connector`.

        Raises:
            :class:`ValueError`: If the connector is absent or is not S3.
        """
        response = cls._response_by_name(name, project_id=project_id)
        config = response.config
        if not isinstance(config, S3ConnectorResourceConfig):
            raise ValueError(f"Connector {name!r} is not S3.")
        target = config.target
        return cls(
            name=response.name,
            identity=response.identity_id,
            root_dir=cls._root_dir(config),
            region=(target.region
                    if isinstance(target, S3RootedConnectorTarget) else None),
            description=response.description,
            project_id=response.project_id,
            _response=response,
        )

    @override
    @property
    def source_type(self) -> DataSourceType:
        """Return the S3 data-source type."""
        return DataSourceType.S3
