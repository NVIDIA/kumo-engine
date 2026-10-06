# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from urllib.parse import urlsplit

from kumoapi.data_source import (
    ConnectorResponse,
    DataSourceType,
    GCSAnyURIConnectorTarget,
    GCSConnectorResourceConfig,
    GCSRootedConnectorTarget,
)
from kumoapi.source_table import GCSRelativeSource, GCSURISource
from typing_extensions import Self, override

from kumoai_studio.connector.file_connector import (
    FileURI,
    ObjectStoreConnector,
)
from kumoai_studio.identity import Identity


class GCSURI(FileURI):
    """Strict Google Cloud Storage URI parser."""

    _schemes = ('gs', )


class GCSConnector(ObjectStoreConnector):
    r"""Connect one Kumo project to Google Cloud Storage.

    Set ``root_dir`` to restrict reads and writes to one bucket/prefix. Leave
    it unset for rootless mode, where every operation uses an exact
    ``gs://bucket/path``. Discovery remains unavailable. Requester-pays
    billing applies in both modes.

    The project-owned connector stores the bucket, normalized prefix, and
    optional requester-pays billing project. Authentication uses an
    administrator-managed GCS identity whose safe
    ``config.authentication.mode`` is ``"workload_identity"`` (preferred) or
    ``"service_account_key"``. Workload mode impersonates the configured
    service account without a stored customer key. Key mode keeps the private
    key and key ID encrypted and server-side. Neither mode exposes credentials
    or short-lived tokens to the SDK, and this constructor accepts none.

    If ``identity`` is omitted, rooted mode selects the one compatible identity
    that can access the target. Rootless mode has no target to probe: one
    compatible identity is selected, while multiple require an explicit
    choice. Exact URI access is checked when each operation runs.

    .. code-block:: python

        import kumoai_studio

        training = kumoai_studio.GCSConnector(
            name="training-data",
            root_dir="gs://ml-data/training/",
            requester_pays_project="billing-project",
        )

        identity = kumoai_studio.get_identity("Production GCP")
        print(identity.config.authentication.mode)  # safe metadata

        training = kumoai_studio.GCSConnector(
            name="training-data",
            root_dir="gs://ml-data/training/",
            identity=identity,
            requester_pays_project="billing-project",
        )

        source_table = training.table("2026/07/examples.parquet")
        source_table = training.table_from_uri(
            "gs://ml-data/training/2026/07/examples.parquet"
        )

        all_gcs = kumoai_studio.GCSConnector(
            name="all-gcs",
            identity=identity,
            requester_pays_project="billing-project",
        )
        source_table = all_gcs.table(
            "gs://another-approved-bucket/2026/07/examples.parquet"
        )

    Args:
        name: Project-local connector display name. Defaults to ``"gcs"``.
        root_dir: Optional GCS URI containing the bucket and table prefix, for
            example ``"gs://ml-data/training/"``. Every table path must
            remain beneath this immutable project-owned root. ``None``
            selects rootless full-URI mode.
        identity: A credential-free :class:`~kumoai_studio.identity.Identity` or an
            immutable identity ID. ``None`` delegates unambiguous selection to
            the server. A string is never interpreted as a display name. Both
            Google authentication modes use this same argument; key material,
            tokens, and workload-source credentials are never SDK arguments.
        requester_pays_project: Google Cloud project to bill when the bucket
            has requester pays enabled.
        description: Safe human-readable connector description.
        project_id: Exact Kumo project to use. ``None`` snapshots the active
            project without mutating global SDK state.

    Raises:
        :class:`ValueError`: If the target configuration is invalid or an
            explicit identity is empty or has the wrong provider type.
        :class:`~kumoai_studio.exceptions.HTTPException`: If target validation or
            identity resolution fails, or the name conflicts in this project.

    Note:
        Repeating construction with the same name and target is idempotent.
        An existing connector created with automatic resolution keeps its
        persisted identity.
    """

    _uri_class = GCSURI

    def __init__(
        self,
        name: str = 'gcs',
        root_dir: Optional[str] = None,
        identity: str | Identity | None = None,
        requester_pays_project: Optional[str] = None,
        description: str = '',
        project_id: Optional[str] = None,
        _response: Optional[ConnectorResponse] = None,
    ) -> None:
        if root_dir is None:
            target = GCSAnyURIConnectorTarget()
        else:
            parsed = urlsplit(GCSURI(root_dir).validate().uri)
            target = GCSRootedConnectorTarget(
                bucket=parsed.netloc,
                prefix=GCSURI.decoded_prefix(parsed.path),
            )
        config = GCSConnectorResourceConfig(
            target=target,
            requester_pays_project=requester_pays_project,
        )
        self.target = target
        self.any_uri = isinstance(target, GCSAnyURIConnectorTarget)
        self.bucket = (target.bucket if isinstance(
            target, GCSRootedConnectorTarget) else None)
        self.prefix = (target.prefix if isinstance(
            target, GCSRootedConnectorTarget) else None)
        self.root_dir = self._root_dir(config)
        self.requester_pays_project = config.requester_pays_project
        self._bind_connector(name, identity, config, description, _response,
                             project_id=project_id)

    @staticmethod
    def _root_dir(config: GCSConnectorResourceConfig) -> Optional[str]:
        target = config.target
        if isinstance(target, GCSAnyURIConnectorTarget):
            return None
        return FileURI.canonical_uri('gs', target.bucket, target.prefix)

    @override
    def _normalize_table_reference(self, name: str) -> str:
        if self.any_uri:
            return GCSURISource(uri=name).uri
        parsed = urlsplit(name)
        if parsed.scheme or parsed.netloc:
            assert self.root_dir is not None
            name = GCSURI(name).relative_table_name(self.root_dir)
        return GCSRelativeSource(path=name).path

    @classmethod
    def get_by_name(
        cls,
        name: str = 'gcs',
        project_id: Optional[str] = None,
    ) -> Self:
        r"""Return an existing GCS connector from one exact project.

        Args:
            name: Project-local connector display name. Defaults to ``"gcs"``.
            project_id: Exact Kumo project, or ``None`` for the active project.

        Returns:
            The matching :class:`GCSConnector` with safe target metadata.

        Raises:
            :class:`ValueError`: If the name is missing or belongs to another
                connector type. Connectors in other projects are not searched.
        """
        response = cls._response_by_name(name, project_id=project_id)
        config = response.config
        if not isinstance(config, GCSConnectorResourceConfig):
            raise ValueError(f"Connector {name!r} is not GCS.")
        return cls(
            name=response.name,
            identity=response.identity_id,
            root_dir=cls._root_dir(config),
            requester_pays_project=config.requester_pays_project,
            description=response.description,
            project_id=response.project_id,
            _response=response,
        )

    @override
    @property
    def source_type(self) -> DataSourceType:
        """Return the GCS data-source type."""
        return DataSourceType.GCS
