# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from kumoapi.data_source import (
    ConnectorResponse,
    DatabricksConnectorResourceConfig,
    DataSourceType,
)
from typing_extensions import Self, override

from kumoai_studio.connector.identity_backed import IdentityBackedConnector
from kumoai_studio.identity import Identity


class DatabricksConnector(IdentityBackedConnector):
    r"""Connect one Kumo project to a Databricks catalog and schema.

    The Databricks workspace and OAuth principal belong to an
    administrator-managed identity. This project-owned connector stores only
    the catalog, schema, and compute selection; no client secret or access
    token is accepted by or returned to the SDK.

    Specify exactly one of ``warehouse_id`` and ``cluster_id``. If
    ``identity`` is omitted, the server validates the target with compatible
    identities available to the project's group and succeeds only when one
    identity can access it. Select an identity explicitly if multiple
    principals work.

    .. code-block:: python

        import kumoai_studio

        features = kumoai_studio.DatabricksConnector(
            name="features",
            catalog="main",
            schema_name="features",
            warehouse_id="4b9b953939869799",
        )

        features = kumoai_studio.DatabricksConnector(
            name="features",
            catalog="main",
            schema_name="features",
            warehouse_id="4b9b953939869799",
            identity=kumoai_studio.get_identity("Production Databricks"),
        )

        source_table = features["customer_features"]

    Args:
        name: Project-local connector display name.
        catalog: Databricks Unity Catalog name.
        schema_name: Schema within ``catalog``. A concrete schema is required.
        identity: A credential-free :class:`~kumoai_studio.identity.Identity` or an
            immutable identity ID. ``None`` delegates unambiguous selection to
            the server. A string is never interpreted as a display name.
        warehouse_id: Databricks SQL warehouse ID. Mutually exclusive with
            ``cluster_id``.
        cluster_id: Databricks all-purpose cluster ID. Mutually exclusive with
            ``warehouse_id``.
        description: Safe human-readable connector description.
        project_id: Exact Kumo project to use. ``None`` snapshots the active
            project without mutating global SDK state.

    Raises:
        :class:`ValueError`: If the target configuration is invalid, including
            when zero or two compute IDs are supplied, or an explicit identity
            is empty or has the wrong provider type.
        :class:`~kumoai_studio.exceptions.HTTPException`: If the compute or target is
            invalid, identity resolution fails, or the name conflicts in this
            project.

    Note:
        Repeating construction with the same name and target is idempotent.
        An existing connector created with automatic resolution keeps its
        persisted identity.
    """
    def __init__(
        self,
        name: str,
        catalog: str,
        schema_name: str,
        identity: str | Identity | None = None,
        warehouse_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        description: str = '',
        project_id: Optional[str] = None,
        _response: Optional[ConnectorResponse] = None,
    ) -> None:
        config = DatabricksConnectorResourceConfig(
            catalog=catalog,
            schema_name=schema_name,
            warehouse_id=warehouse_id,
            cluster_id=cluster_id,
        )
        self.catalog = config.catalog
        self.schema_name = config.schema_name
        self.warehouse_id = config.warehouse_id
        self.cluster_id = config.cluster_id
        self._bind_connector(name, identity, config, description, _response,
                             project_id=project_id)

    @classmethod
    def get_by_name(cls, name: str, project_id: Optional[str] = None) -> Self:
        r"""Return an existing Databricks connector from one exact project.

        Args:
            name: Project-local connector display name.
            project_id: Exact Kumo project, or ``None`` for the active project.

        Returns:
            The matching :class:`DatabricksConnector` with safe target
            metadata and immutable identity ID.

        Raises:
            :class:`ValueError`: If the name is missing or belongs to another
                connector type. Connectors in other projects are not searched.
        """
        response = cls._response_by_name(name, project_id=project_id)
        config = response.config
        if not isinstance(config, DatabricksConnectorResourceConfig):
            raise ValueError(f"Connector {name!r} is not Databricks.")
        return cls(
            name=response.name,
            identity=response.identity_id,
            catalog=config.catalog,
            schema_name=config.schema_name,
            warehouse_id=config.warehouse_id,
            cluster_id=config.cluster_id,
            description=response.description,
            project_id=response.project_id,
            _response=response,
        )

    @override
    @property
    def source_type(self) -> DataSourceType:
        """Return the Databricks data-source type."""
        return DataSourceType.DATABRICKS

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(name={self.name!r}, '
                f'catalog={self.catalog!r}, schema_name={self.schema_name!r})')
