# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from urllib.parse import urlsplit

from kumoapi.data_source import (
    ADLSAnyURIConnectorTarget,
    ADLSConnectorResourceConfig,
    ADLSRootedConnectorTarget,
    ConnectorResponse,
    DataSourceType,
)
from kumoapi.source_table import ADLSRelativeSource, ADLSURISource
from typing_extensions import Self, override

from kumoai_studio.connector.file_connector import (
    FileURI,
    ObjectStoreConnector,
)
from kumoai_studio.identity import Identity


class ADLSURI(FileURI):
    """Strict ADLS Gen2 URI parser."""

    _schemes = ('abfss', )


class ADLSConnector(ObjectStoreConnector):
    r"""Connect one Kumo project to Azure Data Lake Storage Gen2.

    Set ``root_dir`` to restrict reads and writes to one filesystem/prefix.
    Leave it unset for rootless mode, where every operation uses an exact
    ``abfss://filesystem@account.endpoint/path``. Discovery remains
    unavailable. Public, US Government, and China endpoints are preserved and
    validated from each URI.

    The project-owned connector stores the storage account, filesystem,
    normalized prefix, and endpoint suffix parsed from ``root_dir``.
    Authentication uses an administrator-managed ADLS identity representing
    an Azure application. Workload tokens never enter the SDK.

    If ``identity`` is omitted, rooted mode selects the one compatible identity
    that can access the target. Rootless mode has no target to probe: one
    compatible identity is selected, while multiple require an explicit
    choice. Exact URI access is checked when each operation runs.

    .. code-block:: python

        import kumoai_studio

        orders = kumoai_studio.ADLSConnector(
            name="curated-orders",
            root_dir=(
                "abfss://curated@companylake.dfs.core.windows.net/orders/"
            ),
        )

        orders = kumoai_studio.ADLSConnector(
            name="curated-orders",
            root_dir=(
                "abfss://curated@companylake.dfs.core.windows.net/orders/"
            ),
            identity=kumoai_studio.get_identity("Production Azure"),
        )

        source_table = orders.table("2026/07/orders.parquet")
        source_table = orders.table_from_uri(
            "abfss://curated@companylake.dfs.core.windows.net/"
            "orders/2026/07/orders.parquet"
        )

        all_adls = kumoai_studio.ADLSConnector(
            name="all-adls",
            identity=kumoai_studio.get_identity("Production Azure"),
        )
        source_table = all_adls.table(
            "abfss://curated@anotherlake.dfs.core.windows.net/"
            "orders/2026/07/orders.parquet"
        )

    Args:
        name: Project-local connector display name. Defaults to ``"adls"``.
        root_dir: Optional fully qualified ``abfss://`` URI in the form
            ``abfss://<filesystem>@<account>.<endpoint>/<prefix>``. Every
            table path must remain beneath this immutable project-owned root.
            ``None`` selects rootless full-URI mode.
        identity: A credential-free :class:`~kumoai_studio.identity.Identity` or an
            immutable identity ID. ``None`` delegates unambiguous selection to
            the server. A string is never interpreted as a display name.
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

    _uri_class = ADLSURI

    def __init__(
        self,
        name: str = 'adls',
        root_dir: Optional[str] = None,
        identity: str | Identity | None = None,
        description: str = '',
        project_id: Optional[str] = None,
        _response: Optional[ConnectorResponse] = None,
    ) -> None:
        if root_dir is None:
            target = ADLSAnyURIConnectorTarget()
        else:
            parsed = urlsplit(ADLSURI(root_dir).validate().uri)
            try:
                filesystem, host = parsed.netloc.split('@', 1)
                storage_account, endpoint_suffix = host.split('.', 1)
            except ValueError as error:
                raise ValueError(
                    "ADLS root_dir must be "
                    "abfss://<filesystem>@<account>.<endpoint>/<prefix>."
                ) from error
            target = ADLSRootedConnectorTarget(
                storage_account=storage_account,
                filesystem=filesystem,
                prefix=ADLSURI.decoded_prefix(parsed.path),
                endpoint_suffix=endpoint_suffix,
            )
        config = ADLSConnectorResourceConfig(target=target)
        self.target = target
        self.any_uri = isinstance(target, ADLSAnyURIConnectorTarget)
        self.storage_account = (target.storage_account if isinstance(
            target, ADLSRootedConnectorTarget) else None)
        self.filesystem = (target.filesystem if isinstance(
            target, ADLSRootedConnectorTarget) else None)
        self.prefix = (target.prefix if isinstance(
            target, ADLSRootedConnectorTarget) else None)
        self.endpoint_suffix = (target.endpoint_suffix if isinstance(
            target, ADLSRootedConnectorTarget) else None)
        self.root_dir = self._root_dir(config)
        self._bind_connector(name, identity, config, description, _response,
                             project_id=project_id)

    @staticmethod
    def _root_dir(config: ADLSConnectorResourceConfig) -> Optional[str]:
        target = config.target
        if isinstance(target, ADLSAnyURIConnectorTarget):
            return None
        authority = (f'{target.filesystem}@{target.storage_account}.'
                     f'{target.endpoint_suffix}')
        return FileURI.canonical_uri('abfss', authority, target.prefix)

    @override
    def _normalize_table_reference(self, name: str) -> str:
        if self.any_uri:
            return ADLSURISource(uri=name).uri
        parsed = urlsplit(name)
        if parsed.scheme or parsed.netloc:
            assert self.root_dir is not None
            name = ADLSURI(name).relative_table_name(self.root_dir)
        return ADLSRelativeSource(path=name).path

    @classmethod
    def get_by_name(
        cls,
        name: str = 'adls',
        project_id: Optional[str] = None,
    ) -> Self:
        r"""Return an existing ADLS connector from one exact project.

        Args:
            name: Project-local connector display name. Defaults to
                ``"adls"``.
            project_id: Exact Kumo project, or ``None`` for the active project.

        Returns:
            The matching :class:`ADLSConnector` with safe target metadata.

        Raises:
            :class:`ValueError`: If the name is missing or belongs to another
                connector type. Connectors in other projects are not searched.
        """
        response = cls._response_by_name(name, project_id=project_id)
        config = response.config
        if not isinstance(config, ADLSConnectorResourceConfig):
            raise ValueError(f"Connector {name!r} is not ADLS.")
        return cls(
            name=response.name,
            identity=response.identity_id,
            root_dir=cls._root_dir(config),
            description=response.description,
            project_id=response.project_id,
            _response=response,
        )

    @override
    @property
    def source_type(self) -> DataSourceType:
        """Return the ADLS data-source type."""
        return DataSourceType.ADLS
