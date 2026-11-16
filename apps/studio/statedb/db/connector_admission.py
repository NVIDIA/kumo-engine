# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from statedb.config.common import ConnectorID, ProjectID
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.group_relations import lock_group_relations
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateTx
from statedb.rest.connector_db import ExternalConnectorDBEntry
from statedb.rest.group import GroupStatus, KumoGroupDBEntry
from statedb.rest.identity import IdentityDBEntry, IdentityGroupEntry
from statedb.rest.project import ProjectDBEntry, ProjectStatus

_CONNECTOR_ADMISSION_DELETE_LOCK_PREFIX = 'statedb:connector_admission_delete'


def _connector_admission_delete_lock_key(
    project_id: ProjectID | str,
    connector_id: ConnectorID | str,
) -> int:
    value = f'{_CONNECTOR_ADMISSION_DELETE_LOCK_PREFIX}:{project_id}:{connector_id}'
    digest = hashlib.blake2b(value.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big', signed=True)


async def lock_connector_admission_and_delete(
    tx: StateTx,
    project_id: ProjectID | str,
    connector_id: ConnectorID | str,
) -> None:
    """Serialize reference admission with connector tombstoning.

    Callers acquire this transaction-scoped lock before validating and
    persisting a connector-backed resource that must remain live (currently
    tables), and before checking those resources to tombstone the connector.
    It does not protect in-flight workflows, which must re-resolve credentials
    and fail after tombstoning.
    """
    if tx.backend == 'postgres':
        await tx.fetchrow(
            'SELECT pg_advisory_xact_lock($1)',
            _connector_admission_delete_lock_key(project_id, connector_id),
        )


async def require_connectors_for_admission(
    tx: StateTx,
    project_id: ProjectID,
    connector_ids: Iterable[ConnectorID],
) -> None:
    """Lock and recheck every connector before persisting a live reference.

    All connector-use locks are acquired in sorted order before the owning
    group's relation lock. Under that lock the project/group, connector,
    external identity, binding, and type compatibility are reloaded together.
    """
    ids = sorted({ConnectorID(str(value)) for value in connector_ids}, key=str)
    if not ids:
        return
    for connector_id in ids:
        await lock_connector_admission_and_delete(tx, project_id, connector_id)

    projects = JsonRepository('projects', ProjectDBEntry)
    groups = JsonRepository('groups', KumoGroupDBEntry)
    identities = JsonRepository('identities', IdentityDBEntry)
    bindings = JsonRepository('identity_groups', IdentityGroupEntry)
    connectors = StateDBConnectorRepository()

    project_row = await projects.get(tx, str(project_id))
    if (project_row is None
            or project_row.payload.status != ProjectStatus.ACTIVE):
        raise ValueError(f"Active project '{project_id}' not found")
    group_id = project_row.payload.group_id
    await lock_group_relations(tx, [group_id])

    project_row = await projects.get(tx, str(project_id))
    group_row = await groups.get(tx, str(group_id))
    if (project_row is None
            or project_row.payload.status != ProjectStatus.ACTIVE
            or project_row.payload.group_id != group_id or group_row is None
            or group_row.payload.status != GroupStatus.ACTIVE):
        raise ValueError(f"Active project '{project_id}' not found")

    for connector_id in ids:
        connector = await connectors.get(tx, connector_id)
        if (connector is None or connector.project_id != project_id
                or connector.tombstoned_at is not None):
            raise ValueError(
                f"Connector '{connector_id}' is not active in project "
                f"'{project_id}'")
        if not isinstance(connector, ExternalConnectorDBEntry):
            continue
        identity_row = await identities.get(tx, str(connector.identity_id))
        binding = await bindings.find_one_by_fields(
            tx, {
                'identity_id': str(connector.identity_id),
                'group_id': str(group_id),
            })
        if (identity_row is None or binding is None
                or identity_row.payload.config.type.value !=
                connector.config.type.value):
            raise ValueError(
                f"Connector '{connector_id}' identity is not active for "
                f"project '{project_id}'")
