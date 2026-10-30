# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Writing a produced artifact back to a Databricks table.

The write direction of :mod:`~kumo_databricks.backend`, sharing its
:class:`~kumo_databricks.session._DatabricksSessions` so that the
principal a write runs as is the one a read runs as.

It is short for the same reason the reader is: the load itself belongs to
:mod:`~kumo_databricks.write`, which the hosted platform runs as
well -- ``rest/backends/ml_backend/artifact_export.py``'s
``write_artifact_to_databricks`` reaches it through
``DatabricksConnector.write_artifacts_to_table``. What this module adds is the
two things only a *local* caller knows: that the artifact is one frame in
memory rather than a directory a producer filled, and which connection its
identity opened.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd
from databricks.sql.client import Connection
from kumo_core.object_store_sink import FilesystemSink
from kumo_databricks.registry import DatabricksModelRegistryMixin
from kumo_databricks.session import _target_config
from kumo_databricks.volume import is_volume_path, volume_filesystem
from kumo_databricks.write import RunVolume
from kumoapi.data_source import DataSourceType
from kumoapi.jobs import WriteMode
from kumoapi.plugin import (
    ModelRegistryLocation,
    PublishedModel,
    SourceConnector,
)
from pyarrow.fs import FileSystem


class DatabricksSinkBackend(DatabricksModelRegistryMixin):
    r"""Writes a produced artifact to a Databricks Delta table.

    Registered by default beside
    :class:`~kumo_databricks.backend.DatabricksSourceBackend`, so a
    run that read its graph from a catalog exports its predictions back to one
    with no further setup. Installed as
    ``pip install 'kumoai-studio[databricks]'``.

    A destination names its schema and its table in one string, joined by
    ``__kumo__``, because that is the only field a Databricks output has --
    :func:`kumoai_studio.databricks.to_db_table_name` builds it from the
    ``(schema, table)`` tuple an :class:`~kumoai_studio.artifact_export.OutputConfig`
    takes. The schema must be the connector's own, which
    :func:`kumo_databricks.output.within_schema` checks and the
    hosted connector checks through as well.

    Every write stages through the connector's managed volume,
    ``<catalog>.<schema>._kumo_staging``, created on first use. That is not a
    local choice: ``COPY INTO`` is how a Delta table is loaded, and hosted
    states the alternatives -- S3, ADLS, an arbitrary volume, inline cloud
    credentials -- are unsupported on purpose. The volume is emptied of what
    this export put there once the table has landed.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. As with the reader, leaving it unset is the common
            case, since a bound
            :class:`~kumoai_studio.connector.DatabricksConnector` records the
            identity it was created with. That identity needs ``CREATE TABLE``
            on the connector's schema and ``READ``/``WRITE VOLUME`` on its
            staging volume, which a read-only role used for ingest will not
            have.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumo_databricks import (  # doctest: +SKIP
        ...     DatabricksSinkBackend)
        >>> kumo.set_backend(kumo.LocalBackend(  # doctest: +SKIP
        ...     storage=paths,
        ...     sinks={kumo.DataSourceType.DATABRICKS:
        ...            DatabricksSinkBackend(identity='prod-dbx')},
        ... ))
    """

    source_type = DataSourceType.DATABRICKS

    _default_hint = ('LocalBackend(sinks={DataSourceType.DATABRICKS: '
                     'DatabricksSinkBackend(identity=identity)})')

    def write_table(
        self,
        connector: SourceConnector,
        table_name: str,
        frame: pd.DataFrame,
        write_mode: WriteMode = WriteMode.OVERWRITE,
        project_id: Optional[str] = None,
        *,
        export_id: Optional[str] = None,
    ) -> str:
        r"""Writes :obj:`frame` into the connector's catalog and schema.

        Args:
            connector: The Databricks target the destination belongs to.
            table_name: ``<schema>__kumo__<table>``. The schema must be the
                connector's; the catalog is always the connector's, and is not
                named here.
            frame: What to write.
            write_mode: ``OVERWRITE`` replaces the table, which is what makes
                a re-run idempotent. ``APPEND`` adds to it, matching columns
                by name, and creates it where it does not exist.
            project_id: Ignored; a local runtime has no projects.
            export_id: Accepted for the sink protocol and ignored. It names a
                partial result a rerun could adopt, and this write leaves
                none: the local Parquet lives in a
                :class:`tempfile.TemporaryDirectory`, the managed-volume stage
                is emptied by ``load``'s own cleanup whether it succeeds or
                fails, and the table itself is replaced in one statement.
                Where the platform does key a stage on an export id, it is for
                a *distributed* producer that uploaded its own files and hands
                ``load`` a ``staged_files`` manifest -- a path a local export
                never takes.

        Returns:
            The backtick-quoted, catalog-qualified name of the table written.

        Raises:
            ValueError: If the destination names no schema and table, if it
                names a schema other than the connector's, or if the staging
                volume is neither reachable nor creatable.
        """
        if self.holds_files(connector, table_name):
            return FilesystemSink(self).write_table(
                connector, table_name, frame, write_mode, export_id=export_id)
        del project_id, export_id
        from kumo_databricks import output, sql, write

        config = _target_config(connector)
        schema, table = output.within_schema(table_name, config.schema_name)
        with tempfile.TemporaryDirectory(prefix='kumo-databricks-') as scratch:
            artifact = os.path.join(scratch, 'part-00000000.parquet')
            write.to_parquet(frame, artifact)
            # The scratch directory is what `PUT` is allowed to read, and the
            # driver fixes that when the connection is built rather than when
            # the command runs -- so the file has to exist first.
            with self._connected(connector, staging_from=scratch) as connection:
                write.load(
                    execute=lambda statement: _run(connection, statement),
                    run=_volume_runner(connection),
                    workers=lambda files, commands: 1,
                    catalog=config.catalog,
                    schema=schema,
                    table=table,
                    artifact_path=artifact,
                    resolve_local=lambda path: [Path(path)],
                    replace=write_mode == WriteMode.OVERWRITE,
                )
        return sql.qualified_name(config.catalog, schema, table)

    def write_files(self, connector, table_name, files,
                    write_mode=WriteMode.OVERWRITE, project_id=None, *,
                    export_id=None, extra_columns=None):
        """Copy Parquet to a volume path, or COPY it into a table."""
        if self.holds_files(connector, table_name):
            return FilesystemSink(self).write_files(
                connector, table_name, files, write_mode,
                export_id=export_id, extra_columns=extra_columns)
        del project_id, export_id
        from functools import partial

        from kumo_core.artifact_export import prepared_artifact_file
        from kumo_core.artifact_staging import selected_local_parquet_files
        from kumo_databricks import output, sql, write

        selected = selected_local_parquet_files(files)
        config = _target_config(connector)
        schema, table = output.within_schema(table_name, config.schema_name)
        # The Databricks driver accepts the local path in the server's response,
        # so the connection must permit only files this export prepared. Retain
        # the existing COPY-compatible timestamp precision while rewriting at
        # most one selected file at a time beneath one private export root.
        with tempfile.TemporaryDirectory(
                prefix='kumo-databricks-export-') as scratch:
            prepare = partial(
                prepared_artifact_file,
                extra_columns=extra_columns,
                parquet_options=write.PARQUET_OPTIONS,
                scratch_root=scratch,
            )
            with self._connected(connector, staging_from=scratch) as connection:
                write.load(
                    execute=lambda statement: _run(connection, statement),
                    run=_volume_runner(connection),
                    workers=lambda files, commands: 1, catalog=config.catalog,
                    schema=schema, table=table, artifact_path=str(selected[0]),
                    resolve_local=lambda path: selected,
                    replace=write_mode == WriteMode.OVERWRITE,
                    prepare_file=prepare)
        return sql.qualified_name(config.catalog, schema, table)

    def publish_model(
        self,
        connector: SourceConnector,
        bundle_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        metadata: Mapping[str, str] = {},
        metrics: Mapping[str, float] = {},
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        """Register in Unity Catalog, or copy to a volume ``path``."""
        if location is None and path and self.holds_files(connector, path):
            return FilesystemSink(self).publish_model(
                connector, bundle_dir, path=path, metadata=metadata)
        return super().publish_model(connector, bundle_dir, path=path,
                                     location=location, metadata=metadata,
                                     metrics=metrics, project_id=project_id)

    def fetch_model(
        self,
        connector: SourceConnector,
        local_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        """Download from Unity Catalog, or from a volume ``path``."""
        if location is None and path and self.holds_files(connector, path):
            return FilesystemSink(self).fetch_model(connector, local_dir,
                                                    path=path)
        return super().fetch_model(connector, local_dir, path=path,
                                   location=location, project_id=project_id)

    def preflight_write_files(self, connector: SourceConnector,
                              table_name: str,
                              write_mode: Optional[WriteMode] = None) -> None:
        """Refuse an occupied volume directory; tables are not checked."""
        if self.holds_files(connector, table_name):
            FilesystemSink(self).preflight_write_files(connector, table_name)

    def holds_files(self, connector: SourceConnector, path: str) -> bool:
        """``/Volumes/<catalog>/<schema>/<volume>/...``; anything else is a
        table."""
        return is_volume_path(path)

    def filesystem(self, connector: SourceConnector,
                   path: str) -> tuple[FileSystem, str]:
        """A volume path as a filesystem, as the connector's identity."""
        if not is_volume_path(path):
            raise ValueError(
                f"{path!r} is not a Unity Catalog volume path; Databricks "
                f"artifacts go under /Volumes/<catalog>/<schema>/<volume>/.")
        return volume_filesystem(self._client(connector)), path.rstrip('/')

    def __repr__(self) -> str:
        return f'DatabricksSinkBackend(identity={self._identity!r})'


def _volume_runner(connection: Connection) -> RunVolume:
    r"""Runs volume commands on the one connection this process opened.

    The hosted worker fans these out over a pool it mints per batch, which is
    what its ``worker_count`` sizes; a local export stages a single file, so
    there is nothing to fan out and the arguments that describe the fan-out
    are accepted and ignored.
    """
    def run(commands: Sequence[str], *, worker_count: int = 1,
            staging_allowed_local_path: Optional[str] = None,
            ignore_missing: bool = False) -> None:
        del worker_count, staging_allowed_local_path
        for command in commands:
            try:
                _run(connection, command)
            except Exception:
                if not ignore_missing:
                    raise

    return run


def _run(connection: Connection, statement: str) -> None:
    r"""Runs one statement and reads nothing back.

    Deliberately not :func:`~kumo_databricks.session._rows`: a
    ``PUT`` is served by the driver rather than the warehouse and leaves no
    result set to fetch, and none of the statements the writer issues answers
    anything a caller wants.
    """
    cursor = connection.cursor()
    try:
        cursor.execute(statement)
    finally:
        cursor.close()
