# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal

BackendName = Literal["postgres", "sqlite"]

SCHEMA_ROW_ID = "current"
SCHEMA_VERSION = 17

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(identifier: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid State DB identifier: {identifier!r}")


@dataclass(frozen=True)
class IndexSpec:
    name: str
    postgres_expression: str
    sqlite_expression: str
    unique: bool = False
    postgres_where: str | None = None
    sqlite_where: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.name)

    def ddl(self, table_name: str, backend: BackendName) -> str:
        validate_identifier(table_name)
        expression = (self.postgres_expression
                      if backend == "postgres" else self.sqlite_expression)
        where = (self.postgres_where
                 if backend == "postgres" else self.sqlite_where)
        unique = "UNIQUE " if self.unique else ""
        ddl = f"CREATE {unique}INDEX {self.name} ON {table_name} ({expression})"
        if where:
            ddl += f" WHERE {where}"
        return ddl

    def identity(self, backend: BackendName) -> dict[str, object]:
        expression = (self.postgres_expression
                      if backend == "postgres" else self.sqlite_expression)
        where = (self.postgres_where
                 if backend == "postgres" else self.sqlite_where)
        return {
            "name": self.name,
            "expression": expression,
            "unique": self.unique,
            "where": where,
        }


@dataclass(frozen=True)
class TableSpec:
    name: str
    purpose: str
    indexes: tuple[IndexSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.name)

    def identity(self, backend: BackendName) -> dict[str, object]:
        return {
            "name": self.name,
            "indexes": [index.identity(backend) for index in self.indexes],
        }


def _json_field(name: str) -> tuple[str, str]:
    return f"(payload->>'{name}')", f"json_extract(payload, '$.{name}')"


def _json_path(path: tuple[str, ...]) -> tuple[str, str]:
    if not path:
        raise ValueError("JSON index path cannot be empty")
    for path_field in path:
        validate_identifier(path_field)
    return f"(payload#>>'{{{','.join(path)}}}')", (
        f"json_extract(payload, '$.{'.'.join(path)}')")


def _index(
    name: str,
    field_name: str,
    *,
    unique: bool = False,
    where_field: str | None = None,
) -> IndexSpec:
    postgres_expression, sqlite_expression = _json_field(field_name)
    postgres_where = sqlite_where = None
    if where_field is not None:
        postgres_where = f"payload->>'{where_field}' IS NOT NULL"
        sqlite_where = f"json_extract(payload, '$.{where_field}') IS NOT NULL"
    return IndexSpec(
        name=name,
        postgres_expression=postgres_expression,
        sqlite_expression=sqlite_expression,
        unique=unique,
        postgres_where=postgres_where,
        sqlite_where=sqlite_where,
    )


def _path_index(
    name: str,
    path: tuple[str, ...],
) -> IndexSpec:
    postgres_expression, sqlite_expression = _json_path(path)
    return IndexSpec(
        name=name,
        postgres_expression=postgres_expression,
        sqlite_expression=sqlite_expression,
    )


def _compound_index(
    name: str,
    fields: tuple[str, ...],
    *,
    unique: bool = False,
    where_nonempty_field: str | None = None,
) -> IndexSpec:
    postgres_expression = ", ".join(_json_field(field)[0] for field in fields)
    sqlite_expression = ", ".join(_json_field(field)[1] for field in fields)
    postgres_where = sqlite_where = None
    if where_nonempty_field is not None:
        validate_identifier(where_nonempty_field)
        postgres_field, sqlite_field = _json_field(where_nonempty_field)
        postgres_where = f"COALESCE({postgres_field}, '') <> ''"
        sqlite_where = f"COALESCE({sqlite_field}, '') <> ''"
    return IndexSpec(
        name=name,
        postgres_expression=postgres_expression,
        sqlite_expression=sqlite_expression,
        unique=unique,
        postgres_where=postgres_where,
        sqlite_where=sqlite_where,
    )


STATE_DB_TABLES: tuple[TableSpec, ...] = (
    TableSpec("statedb_schema", "schema identity and bootstrap metadata"),
    TableSpec("projects", "project records",
              (_compound_index("projects_status_group_private_idx",
                               ("status", "group_id", "is_private")),
               _compound_index("projects_status_created_by_idx",
                               ("status", "created_by")),
               _index("projects_slug_unique_idx", "slug", unique=True),
               _index("projects_display_name_unique_idx",
                      "normalized_display_name", unique=True))),
    TableSpec("groups", "Kumo group records",
              (_index("groups_status_idx", "status"),
               _index("groups_slug_unique_idx", "slug", unique=True),
               _index("groups_display_name_unique_idx",
                      "normalized_display_name", unique=True))),
    TableSpec("group_members", "manual Kumo group memberships",
              (_index("group_members_group_id_idx", "group_id"),
               _index("group_members_principal_id_idx", "principal_id"),
               _compound_index("group_members_unique_idx",
                               ("group_id", "principal_type", "principal_id"),
                               unique=True))),
    TableSpec(
        "external_group_mappings", "identity-provider group mappings",
        (_index("external_group_mappings_group_id_idx", "group_id"),
         _compound_index("external_group_mappings_provider_hash_unique_idx",
                         ("identity_provider_id", "external_group_id_hash"),
                         unique=True))),
    TableSpec("group_membership_rules",
              "native dynamic Kumo group membership rules",
              (_index("group_membership_rules_group_id_idx", "group_id"),
               _compound_index("group_membership_rules_type_unique_idx",
                               ("group_id", "type"), unique=True))),
    TableSpec(
        "connectors",
        "connector records",
        (_index("connectors_kind_idx",
                "kind"), _index("connectors_project_id_idx", "project_id"),
         IndexSpec(
             name="connectors_live_identity_idx",
             postgres_expression="(payload->>'identity_id')",
             sqlite_expression="json_extract(payload, '$.identity_id')",
             postgres_where="payload->>'tombstoned_at' IS NULL",
             sqlite_where="json_extract(payload, '$.tombstoned_at') IS NULL",
         ),
         _index("connectors_tombstoned_at_idx", "tombstoned_at",
                where_field="tombstoned_at"),
         _path_index("connectors_config_type_idx", ("config", "type")),
         IndexSpec(
             name="connectors_project_name_unique_idx",
             postgres_expression=("(payload->>'project_id'), "
                                  "(payload->>'normalized_name')"),
             sqlite_expression=("json_extract(payload, '$.project_id'), "
                                "json_extract(payload, '$.normalized_name')"),
             unique=True,
             postgres_where="payload->>'tombstoned_at' IS NULL",
             sqlite_where="json_extract(payload, '$.tombstoned_at') IS NULL",
         ), _path_index("connectors_config_root_dir_idx",
                        ("config", "root_dir"))),
    ),
    TableSpec(
        "uploaded_files",
        "completed project-owned uploaded files",
        (_index("uploaded_files_project_id_idx", "project_id"),
         _index("uploaded_files_connector_id_idx", "connector_id"),
         _compound_index("uploaded_files_project_name_unique_idx",
                         ("project_id", "normalized_source_table_name"),
                         unique=True)),
    ),
    TableSpec(
        "connector_authorizations",
        "hashed workflow-scoped connector authorization capabilities",
        (_index("connector_authorizations_token_hash_unique_idx", "token_hash",
                unique=True),
         _compound_index("connector_authorizations_resource_idx",
                         ("resource_kind", "resource_id")),
         _index("connector_authorizations_project_id_idx", "project_id"),
         _index("connector_authorizations_expires_at_idx", "expires_at"),
         _index("connector_authorizations_revoked_at_idx", "revoked_at",
                where_field="revoked_at")),
    ),
    TableSpec(
        "connector_authorization_uses",
        "activity uses of workflow-scoped connector authorizations",
        (_compound_index("connector_authorization_uses_retry_unique_idx",
                         ("authorization_id", "workflow_run_id", "activity_id",
                          "action"), unique=True),
         _index("connector_authorization_uses_authorization_id_idx",
                "authorization_id")),
    ),
    TableSpec(
        "artifact_export_jobs",
        "immutable project-owned artifact export requests",
        (),
    ),
    TableSpec(
        "identities",
        "external principals and encrypted identity credentials",
        (IndexSpec(
             name="identities_identity_key_unique_idx",
             postgres_expression="(payload->>'identity_key')",
             sqlite_expression="json_extract(payload, '$.identity_key')",
             unique=True,
             postgres_where="payload->>'tombstoned_at' IS NULL",
             sqlite_where="json_extract(payload, '$.tombstoned_at') IS NULL",
         ),
         IndexSpec(
             name="identities_slug_unique_idx",
             postgres_expression="(payload->>'slug')",
             sqlite_expression="json_extract(payload, '$.slug')",
             unique=True,
             postgres_where="payload->>'tombstoned_at' IS NULL",
             sqlite_where="json_extract(payload, '$.tombstoned_at') IS NULL",
         ),
         IndexSpec(
             name="identities_display_name_unique_idx",
             postgres_expression="(payload->>'normalized_display_name')",
             sqlite_expression=(
                 "json_extract(payload, '$.normalized_display_name')"),
             unique=True,
             postgres_where="payload->>'tombstoned_at' IS NULL",
             sqlite_where="json_extract(payload, '$.tombstoned_at') IS NULL",
         ),
         _path_index("identities_config_type_idx", ("config", "type"))),
    ),
    TableSpec(
        "identity_groups",
        "many-to-many identity to Kumo group bindings",
        (_index("identity_groups_identity_id_idx", "identity_id"),
         _index("identity_groups_group_id_idx", "group_id"),
         _compound_index("identity_groups_pair_unique_idx",
                         ("identity_id", "group_id"), unique=True)),
    ),
    TableSpec(
        "tables",
        "table records",
        (_index("tables_project_id_idx", "project_id"),
         _compound_index("tables_project_alias_unique_idx",
                         ("project_id", "normalized_alias"), unique=True,
                         where_nonempty_field="normalized_alias"),
         _path_index("tables_source_connector_id_idx",
                     ("config", "source_table", "connector_id")),
         _path_index("tables_source_bigquery_connector_id_idx",
                     ("config", "source_table", "bigquery_connector_id")),
         _path_index("tables_source_snowflake_connector_id_idx",
                     ("config", "source_table", "snowflake_connector_id")),
         _path_index("tables_source_databricks_connector_id_idx",
                     ("config", "source_table", "databricks_connector_id"))),
    ),
    TableSpec("graphs", "graph records",
              (_index("graphs_project_id_idx", "project_id"),
               _compound_index("graphs_project_alias_unique_idx",
                               ("project_id", "normalized_alias"), unique=True,
                               where_nonempty_field="normalized_alias"))),
    TableSpec(
        "predictive_queries",
        "predictive query records",
        (_index("predictive_queries_graph_id_idx",
                "graph_id"), _index("predictive_queries_name_idx", "name")),
    ),
    TableSpec("pquery_definitions", "frozen predictive query definitions"),
    TableSpec("pquery_aliases", "predictive query name to definition aliases"),
    TableSpec(
        "api_keys",
        "API key metadata records",
        (_compound_index("api_keys_type_owner_idx", ("type", "owner_user_id")),
         _compound_index("api_keys_verification_lookup_idx",
                         ("name", "owner_user_hash")),
         _compound_index("api_keys_owner_name_unique_idx",
                         ("owner_user_id", "name"), unique=True)),
    ),
    TableSpec("api_key_secrets", "API key secret verification records",
              (_index("api_key_secrets_revoked_idx", "revoked"), )),
    TableSpec(
        "training_jobs",
        "training job metadata",
        (_index("training_jobs_status_idx",
                "status"), _index("training_jobs_project_id_idx", "project_id"),
         _index("training_jobs_pquery_id_idx", "pquery_id")),
    ),
    TableSpec(
        "batch_prediction_jobs",
        "batch prediction job metadata",
        (_index("batch_prediction_jobs_status_idx", "status"),
         _index("batch_prediction_jobs_project_id_idx", "project_id"),
         _index("batch_prediction_jobs_pquery_id_idx", "pquery_id"),
         _index("batch_prediction_jobs_workflow_id_idx",
                "batch_prediction_workflow_id",
                where_field="batch_prediction_workflow_id")),
    ),
    TableSpec(
        "generate_train_table_jobs",
        "generate train table job metadata",
        (_index("generate_train_table_jobs_status_idx", "status"),
         _index("generate_train_table_jobs_project_id_idx", "project_id"),
         _index("generate_train_table_jobs_pquery_id_idx", "pquery_id")),
    ),
    TableSpec(
        "generate_pred_table_jobs",
        "generate prediction table job metadata",
        (_index("generate_pred_table_jobs_status_idx", "status"),
         _index("generate_pred_table_jobs_project_id_idx", "project_id"),
         _index("generate_pred_table_jobs_pquery_id_idx", "pquery_id")),
    ),
    TableSpec(
        "job_tags",
        "normalized job tag rows",
        (_compound_index("job_tags_job_lookup_idx", ("job_type", "job_id")),
         _compound_index("job_tags_key_value_idx", ("key", "value"))),
    ),
    TableSpec("direct_connect_config", "Direct Connect OIDC/password config"),
    TableSpec(
        "direct_connect_users",
        "Direct Connect users",
        (_index("direct_connect_users_email_idx", "normalized_email"), ),
    ),
    TableSpec(
        "direct_connect_identity_links",
        "Direct Connect IdP identity links",
        (_index("direct_connect_identity_links_user_id_idx", "user_id",
                unique=True), ),
    ),
    TableSpec(
        "direct_connect_sessions",
        "Direct Connect UI session records",
        (_index("direct_connect_sessions_user_id_idx", "user_id"),
         _index("direct_connect_sessions_expires_time_idx", "expires_time")),
    ),
    TableSpec("direct_connect_password_credentials",
              "Direct Connect password verifier records"),
    TableSpec(
        "direct_connect_password_reset_tokens",
        "Direct Connect password reset token records",
        (_index("direct_connect_password_reset_tokens_user_id_idx", "user_id"),
         _index("direct_connect_password_reset_tokens_expires_time_idx",
                "expires_time"),
         _index("direct_connect_password_reset_tokens_used_idx", "used")),
    ),
    TableSpec(
        "direct_connect_password_login_attempts",
        "Direct Connect password login rate-limit records",
        (_index("direct_connect_password_login_attempts_locked_until_idx",
                "locked_until"),
         _index("direct_connect_password_login_attempts_expires_time_idx",
                "expires_time"),
         _index("direct_connect_password_login_attempts_first_failed_time_idx",
                "first_failed_time")),
    ),
)

STATE_DB_TABLE_NAMES = tuple(table.name for table in STATE_DB_TABLES)


def _schema_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def legacy_schema_identity(
    tables: tuple[TableSpec, ...] = STATE_DB_TABLES, ) -> str:
    """Return the pre-backend-specific schema hash for metadata upgrades."""
    payload = {
        "format":
        "kumo-state-db-schema-v1",
        "tables": [{
            "name":
            table.name,
            "purpose":
            table.purpose,
            "indexes": [{
                "name": index.name,
                "postgres_expression": index.postgres_expression,
                "sqlite_expression": index.sqlite_expression,
                "unique": index.unique,
                "postgres_where": index.postgres_where,
                "sqlite_where": index.sqlite_where,
            } for index in table.indexes],
        } for table in tables],
    }
    return _schema_hash(payload)


def schema_identity(
    backend: BackendName,
    tables: tuple[TableSpec, ...] = STATE_DB_TABLES,
) -> str:
    """Hash the physical schema relevant to one database backend."""
    payload = {
        "format": "kumo-state-db-schema-v2",
        "backend": backend,
        "tables": [table.identity(backend) for table in tables],
    }
    return _schema_hash(payload)
