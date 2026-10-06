# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from kumoapi.data_source import (
    BigQueryConnectorResourceConfig,
    ConnectorResponse,
    DataSourceType,
)
from typing_extensions import Self, override

from kumoai_studio.connector.identity_backed import IdentityBackedConnector
from kumoai_studio.identity import Identity


class BigQueryConnector(IdentityBackedConnector):
    r"""Connect one Kumo project to a Google BigQuery dataset.

    Authentication belongs to an administrator-managed BigQuery identity.
    Its safe ``config.authentication.mode`` is either
    ``"workload_identity"`` (preferred) or ``"service_account_key"``.
    Workload mode impersonates the configured service account without storing
    a customer key. Key mode keeps the private key and key ID encrypted and
    server-side; the SDK sees only safe principal metadata and status.

    This project-owned connector stores only the data project, dataset,
    location, optional billing project, and identity reference. Its
    constructor is intentionally credential-free.

    If ``identity`` is omitted, the server validates the target with compatible
    identities available to the active project's group and succeeds only when
    exactly one can access it. Select an identity explicitly when multiple
    service accounts have access.

    .. code-block:: python

        import kumoai_studio

        events = kumoai_studio.BigQueryConnector(
            name="events",
            project_id="data-project",
            dataset_id="events",
            location="US",
            billing_project_id="billing-project",
        )

        identity = kumoai_studio.get_identity("Production BigQuery")
        print(identity.config.authentication.mode)  # safe metadata

        events = kumoai_studio.BigQueryConnector(
            name="events",
            project_id="data-project",
            dataset_id="events",
            location="US",
            identity=identity,
        )

        source_table = events["page_views"]

    Args:
        name: Project-local connector display name.
        project_id: Google Cloud project containing the dataset.
        dataset_id: BigQuery dataset ID.
        identity: A credential-free :class:`~kumoai_studio.identity.Identity` or an
            immutable identity ID. ``None`` delegates unambiguous selection to
            the server. A string is never interpreted as a display name. Both
            Google authentication modes use this same argument; private keys,
            tokens, and workload-source credentials are never SDK arguments.
        location: BigQuery dataset location, such as ``"US"`` or
            ``"europe-west1"``.
        billing_project_id: Optional Google Cloud project to bill for queries.
        description: Safe human-readable connector description.
        kumo_project_id: Exact Kumo project to use. This distinct name avoids
            ambiguity with BigQuery's ``project_id``. ``None`` snapshots the
            active Kumo project without mutating global SDK state.

    Raises:
        :class:`ValueError`: If the target configuration is invalid or an
            explicit identity is empty or has the wrong provider type.
        :class:`~kumoai_studio.exceptions.HTTPException`: If the target cannot be
            accessed, identity resolution fails, or the name conflicts in this
            project.

    Note:
        Repeating construction with the same name and target is idempotent.
        An existing connector created with automatic resolution keeps its
        persisted identity.
    """
    def __init__(
        self,
        name: str,
        project_id: str,
        dataset_id: str,
        identity: str | Identity | None = None,
        location: str = 'US',
        billing_project_id: Optional[str] = None,
        description: str = '',
        kumo_project_id: Optional[str] = None,
        _response: Optional[ConnectorResponse] = None,
    ) -> None:
        config = BigQueryConnectorResourceConfig(
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
            billing_project_id=billing_project_id,
        )
        self.project_id = config.project_id
        self.dataset_id = config.dataset_id
        self.location = config.location
        self.billing_project_id = config.billing_project_id
        self._bind_connector(name, identity, config, description, _response,
                             project_id=kumo_project_id)

    @classmethod
    def get_by_name(cls, name: str,
                    kumo_project_id: Optional[str] = None) -> Self:
        r"""Return an existing BigQuery connector from one Kumo project.

        Args:
            name: Project-local connector display name.
            kumo_project_id: Exact Kumo project, or ``None`` for the active
                Kumo project.

        Returns:
            The matching :class:`BigQueryConnector` with safe target metadata
            and immutable identity ID.

        Raises:
            :class:`ValueError`: If the name is missing or belongs to another
                connector type. Connectors in other projects are not searched.
        """
        response = cls._response_by_name(name, project_id=kumo_project_id)
        config = response.config
        if not isinstance(config, BigQueryConnectorResourceConfig):
            raise ValueError(f"Connector {name!r} is not BigQuery.")
        return cls(
            name=response.name,
            identity=response.identity_id,
            project_id=config.project_id,
            dataset_id=config.dataset_id,
            location=config.location,
            billing_project_id=config.billing_project_id,
            description=response.description,
            kumo_project_id=response.project_id,
            _response=response,
        )

    @override
    @property
    def source_type(self) -> DataSourceType:
        """Return the BigQuery data-source type."""
        return DataSourceType.BIGQUERY

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(name={self.name!r}, '
                f'project_id={self.project_id!r}, '
                f'dataset_id={self.dataset_id!r})')
