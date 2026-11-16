# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rest.config.connector_db import ConnectorDBEntry
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.runtime import StateDB


async def seed_connector(
    state_db: StateDB,
    connector: ConnectorDBEntry,
) -> ConnectorDBEntry:
    """Insert a connector directly for tests that need fixed internal IDs."""
    async with state_db.write_tx() as tx:
        await StateDBConnectorRepository().create(tx, connector)
    return connector
