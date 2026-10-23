# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The `Connector` base class.

A `Connector` is a plain, directly-referenced object -- never a string id
looked up in a registry.
The provider connectors
(`SnowflakeConnector`, `S3Connector`, `GCSConnector`, `BigQueryConnector`,
`DatabricksConnector`) wrap an implementation of
`kumoapi.plugin.SourceBackend` and, where relevant, `ArtifactSink` /
`ModelPublishingSink` / `ModelFetchingSink` / `ModelListingSink`, taken
directly from the existing first-party plugin packages (`kumo-snowflake`,
`kumo-s3`, `kumo-gcs`, `kumo-bigquery`, `kumo-databricks`).

Those mirror the *argument lists* of the equivalent cloud SDK classes
(`kumoai/connector/{snowflake,s3,gcs,bigquery,databricks}_connector.py`), but
drop everything that exists only to round-trip a server-side connector
registry: `name` (a project-scoped display name for `get_by_name` lookups),
`description`, `project_id`/`kumo_project_id`, and `_response`. In particular,
construction here never calls anything resembling the cloud SDK's
`IdentityBackedConnector._bind_connector` -> `get_backend().connector.
create_if_not_exist(...)` -- there is no `Backend`, so there is nothing to
round-trip through. Construction is pure Python: build the
`kumoapi.data_source.*ConnectorResourceConfig` and hold it.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Union

from kumoapi.data_source import ConnectorResourceConfig, DataSourceType
from kumoapi.plugin import ArtifactSink, SourceBackend

from kumo.engine._local_schema import FileConnectorConfig, LocalSourceTableType
from kumo.engine._scratch import fingerprint
from kumo.engine.connector.identity import Identity


class Connector(ABC):
    r"""Base class for all local-mode connectors.

    Instances are immutable and are threaded directly through `Table`
    construction and through output/export configs -- never referenced by a
    string id.

    Args:
        identity: Credentials to authenticate as -- an `Identity` object
            (e.g. `SnowflakeIdentity`, `S3Identity`), passed directly rather
            than by name/id. `None` defers to whatever the underlying plugin
            package treats as its ambient default (e.g. the local AWS/GCP
            credential chain for S3/GCS).
    """
    def __init__(self, identity: Optional[Identity] = None):
        self.identity = identity
        self.identity_id = identity.slug if identity is not None else None

    @property
    @abstractmethod
    def source_type(self) -> Union[DataSourceType, Literal['LOCAL_FILE']]:
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> Union[ConnectorResourceConfig, FileConnectorConfig]:
        r"""What this connector points at, in its provider's own terms --
        the provider's `kumoapi.data_source.*ConnectorResourceConfig`, or
        a local-only `FileConnectorConfig`, held directly rather than
        round-tripped through a server.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_backend(self) -> SourceBackend:
        r"""The `kumoapi.plugin.SourceBackend` implementation this connector
        reads through. Used by `Graph.snapshot()` to read table data during
        materialization.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sink_backend(self) -> ArtifactSink:
        r"""The `kumoapi.plugin.ArtifactSink` implementation this connector
        writes through -- `write_table`, and, for a connector whose sink also
        satisfies `ModelPublishingSink`/`ModelFetchingSink`/`ModelListingSink`,
        publishing to, loading from, and listing a model registry
        (`kumo.engine.trainer.output_config.ModelOutputConfig`, `Trainer.fit(
        export_config=...)`/`Trainer.load_from_connector(...)`).

        A connector with no sink plugin overrides this to raise rather than
        omitting it, so the failure names what is missing instead of surfacing
        as a plain `AttributeError` from `connector.sink_backend.write_table(...)`.
        """
        raise NotImplementedError

    @abstractmethod
    def source_table_for(self, table_name: str) -> LocalSourceTableType:
        r"""The validated hosted or local-only source coordinate naming
        `table_name` within this connector -- what `Table.definition.
        source_table` is built from (`kumo.engine._infer.resolve_table_columns`).
        """
        raise NotImplementedError

    @cached_property
    def id(self) -> str:
        r"""A content-fingerprint id derived from `source_type` + `config` --
        no server-issued id, no connector registry. Two connectors pointed at
        the same target converge on the same id, the same way `Graph.id`
        does.
        """
        return fingerprint(
            repr(self.source_type).encode(),
            repr(self.config).encode(),
        )

    def validate_scratch_dir(self, scratch_dir: Path) -> None:
        """Hook for connectors with local sources to reject scratch overlap."""

    def table(self, table_name: str):
        r"""A reference to `table_name` within this connector -- a
        `kumo.engine.connector.source_table.SourceTable`. Unannotated (not
        `object`/`Any`) rather than imported at module level, the same way
        `source_backend`/`sink_backend` on the concrete connector subclasses
        are: `kumo.engine.connector.source_table` itself imports `Connector` from
        this module, so importing it back here at module level would cycle.
        """
        from kumo.engine.connector.source_table import SourceTable

        return SourceTable(table_name, self)

    def __getitem__(self, table_name: str):
        return self.table(table_name)

    def sql(self, query: str):
        r"""A `SQLSourceTable` whose rows come from the read-only `SELECT`
        `query`, run in this connector's warehouse. Mirrors the cloud SDK's
        `Connector.sql`. Only Snowflake, Databricks and BigQuery run SQL.
        """
        from kumo.engine.connector.source_table import SQLSourceTable

        return SQLSourceTable(query, self)

    def table_names(self) -> List[str]:
        r"""Source table names available through this connector. Mirrors
        the cloud SDK's `Connector.table_names()` -- convenience over
        `self.source_backend.list_tables(self).table_names`, which is what
        this calls.
        """
        return self.source_backend.list_tables(self).table_names

    def has_table(self, table_name: str) -> bool:
        r"""Whether `table_name` is visible through this connector. Mirrors
        the cloud SDK's `Connector.has_table(name)` -- convenience over
        `table_name in self.table_names()`.
        """
        return table_name in self.table_names()

    def __contains__(self, table_name: str) -> bool:
        return self.has_table(table_name)
