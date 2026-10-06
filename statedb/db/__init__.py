# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified State DB runtime primitives for REST and ML metadata."""

from importlib import import_module

from statedb.db.bootstrap import (
    BootstrapResult,
    StateDBBootstrapBusyError,
    StateDBSchemaMismatchError,
    bootstrap_postgres,
    bootstrap_sqlite,
)
from statedb.db.config import (
    default_sqlite_path,
    open_state_db_from_deploy_config,
    open_state_db_from_postgres_config,
    resolve_state_db_postgres_dsn,
)
from statedb.db.runtime import StateDB, StateDBSchemaStatus, StateTx
from statedb.db.schema import (
    SCHEMA_ROW_ID,
    SCHEMA_VERSION,
    STATE_DB_TABLE_NAMES,
    STATE_DB_TABLES,
    TableSpec,
    schema_identity,
)

_LAZY_EXPORTS = {
    "MLJobService": ("statedb.db.ml_jobs", "MLJobService"),
    "StateServices": ("statedb.db.services", "StateServices"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BootstrapResult",
    "SCHEMA_ROW_ID",
    "SCHEMA_VERSION",
    "STATE_DB_TABLES",
    "STATE_DB_TABLE_NAMES",
    "StateDB",
    "StateDBBootstrapBusyError",
    "StateDBSchemaStatus",
    "StateDBSchemaMismatchError",
    "StateServices",
    "StateTx",
    "TableSpec",
    "MLJobService",
    "bootstrap_postgres",
    "bootstrap_sqlite",
    "default_sqlite_path",
    "open_state_db_from_deploy_config",
    "open_state_db_from_postgres_config",
    "resolve_state_db_postgres_dsn",
    "schema_identity",
]
