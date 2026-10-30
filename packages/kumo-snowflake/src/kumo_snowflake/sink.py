# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Writing a produced artifact back to a Snowflake table.

The write direction of :mod:`~kumo_snowflake.source`, and the
``artifact_export`` half of this package. It is short because it has one job:
the destination is the connector's, the frame is whatever the run produced, and
the only real decision is what to do about a table already sitting there.
"""

from typing import Optional

import numpy as np
import pandas as pd
from kumo_core.object_store_sink import FilesystemSink
from kumo_snowflake.artifacts import _split_stage_location
from kumo_snowflake.errors import is_lapsed_credential
from kumo_snowflake.registry import SnowflakeModelRegistryMixin
from kumo_snowflake.session import (
    _owned_session,
    _require_snowpark,
    _SnowflakeSessions,
    _target_config,
    resolved,
)
from kumoapi.jobs import WriteMode
from kumoapi.plugin import SourceConnector

# Snowflake ARRAY and VARIANT are acquired as JSON strings, so nested
# predictions cannot round-trip as typed lists or mappings. NumPy arrays
# also cover nested values returned by Parquet reads.
_NESTED_TYPES = (list, dict, set, tuple, np.ndarray)


def _refuse_nested_columns(frame: pd.DataFrame) -> None:
    r"""Fails on a column whose values are lists or dicts, naming it.

    Checked against the first row rather than the whole column: an object
    column is homogeneous in practice, and scanning millions of rows to confirm
    it would cost more than the write.
    """
    if frame.empty:
        return
    nested = []
    for name in frame.columns:
        column = frame[name]
        if (pd.api.types.is_object_dtype(column)
                and isinstance(column.iloc[0], _NESTED_TYPES)):
            nested.append(str(name))
    if nested:
        raise ValueError(
            f"Cannot write {', '.join(repr(name) for name in nested)} to "
            f"Snowflake: nested values cannot round-trip as typed lists or "
            f"mappings; ARRAY and VARIANT are read as JSON strings. "
            f"Ranking predictions are the usual cause; "
            f"flatten them to one row per class, or drop the column, before "
            f"exporting.")


class SnowflakeSinkBackend(SnowflakeModelRegistryMixin, _SnowflakeSessions):
    r"""Writes a produced artifact to a Snowflake table.

    The write direction of
    :class:`~kumo_snowflake.backend.SnowflakeSourceBackend`, and
    registered beside it: a :class:`~kumoai_studio.backend.LocalBackend` gets one by
    default, so
    exporting a batch prediction back to the warehouse it was read from needs
    no more setup than reading did.

    It also publishes trained models to the Snowflake Model Registry, through
    :class:`~kumo_snowflake.registry.SnowflakeModelRegistryMixin`.
    That half needs ``snowflake-ml-python``, which the ``snowflake`` extra
    installs, and is imported only when a model actually moves -- so writing a
    table costs nothing for it.

    Identifiers are written **as given**, quoted. That is the one decision here
    a caller can see: a table exported as ``churn_predictions`` is
    ``"churn_predictions"`` in Snowflake and stays lower case, rather than
    being folded to upper as an unquoted identifier would be. It is the choice
    that round-trips -- the reader in this package looks tables and columns up
    in ``INFORMATION_SCHEMA`` by exact name -- so an exported table can be read
    straight back through the same connector.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. As with the reader, leaving it unset is the common
            case, since a bound
            :class:`~kumoai_studio.connector.SnowflakeConnector` records the identity
            it was created with. That identity needs ``CREATE TABLE`` on the
            connector's schema, which a read-only role used for ingest will
            not have.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumo_snowflake import SnowflakeSinkBackend  # doctest: +SKIP
        >>> kumo.set_backend(kumo.LocalBackend(  # doctest: +SKIP
        ...     storage=paths,
        ...     sinks={kumo.DataSourceType.SNOWFLAKE:
        ...            SnowflakeSinkBackend(identity='prod-sf')},
        ... ))
    """

    _default_hint = ('LocalBackend(sinks={DataSourceType.SNOWFLAKE: '
                     'SnowflakeSinkBackend(identity=identity)})')

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
        r"""Writes :obj:`frame` into the connector's database and schema.

        The destination is the connector's own -- writing elsewhere means
        binding another connector -- so only the table is named here.

        ``OVERWRITE`` drops and recreates the table, which is what makes a
        re-run idempotent, and also means grants on the destination do not
        survive one. Where they matter, ``APPEND`` with
        :attr:`~kumoapi.jobs.MetadataField.JOB_TIMESTAMP` distinguishes runs
        without replacing the table. ``APPEND`` to a table that does not exist
        creates it.

        Returns:
            The fully qualified, quoted name of the table written.

        Raises:
            ValueError: If a column holds lists or dicts, which do not
                round-trip as typed nested values.
            TypeError: If the registered factory returned something that is
                not a Snowpark session.
        """
        if self.holds_files(connector, table_name):
            return FilesystemSink(self).write_table(connector, table_name,
                                                    frame, write_mode,
                                                    export_id=export_id)
        # `export_id` identifies a partial result for a destination that can
        # hold one. A warehouse write is a single statement, so there is
        # nothing here for a rerun to recognise.
        del project_id, export_id
        with self._session(connector) as session:
            _require_snowpark(session)
            _refuse_nested_columns(frame)
            session.write_pandas(
                frame,
                table_name=table_name,
                database=resolved(_target_config(connector).database),
                schema=resolved(_target_config(connector).schema_name),
                # Creates the table on first export and, with `overwrite`, on
                # every later one. A prediction frame's schema is the model's
                # output schema, so declaring it here would only restate what
                # the frame already carries.
                auto_create_table=True,
                overwrite=write_mode == WriteMode.OVERWRITE,
                quote_identifiers=True,
                # Without it a tz-aware column lands as a number of nanoseconds
                # rather than a timestamp, which is how an anchor time silently
                # stops being a time.
                use_logical_type=True,
            )
        return self._qualified(connector, table_name)

    def write_files(self, connector, table_name, files,
                    write_mode=WriteMode.OVERWRITE, project_id=None, *,
                    export_id=None, extra_columns=None, **options):
        """Copy Parquet to a stage path, or COPY it into a table."""
        if self.holds_files(connector, table_name):
            return FilesystemSink(self).write_files(connector, table_name,
                                                    files, write_mode,
                                                    export_id=export_id,
                                                    extra_columns=extra_columns,
                                                    **options)
        del project_id, export_id
        from kumo_core.artifact_staging import selected_local_parquet_files
        from kumo_snowflake.write import export_table_files
        selected = [str(path) for path in selected_local_parquet_files(files)]

        def export(*, retry: bool = False) -> None:
            session_context = (_owned_session(self._principal(connector),
                                              connector)
                               if retry else self._session(connector))
            with session_context as session:
                _require_snowpark(session)
                target = _LocalArtifactConnection(session, connector,
                                                  extra_columns)
                export_table_files(target, table_name, selected[0],
                                   replace=write_mode == WriteMode.OVERWRITE,
                                   artifact_files=selected)

        try:
            export()
        except Exception as error:
            if (write_mode != WriteMode.OVERWRITE
                    or not is_lapsed_credential(error)):
                raise
            # The artifact producer has already finished.  Reconnect from the
            # refreshed mounted token and repeat only this idempotent
            # overwrite export; never repeat training or prediction.
            export(retry=True)
        return self._qualified(connector, table_name)

    def preflight_write_files(self, connector, table_name, write_mode=None):
        """Refuse an occupied stage directory; tables are not checked."""
        del write_mode
        from kumo_core.artifact_export import ArtifactDestinationConflict
        from pyarrow.fs import FileSelector

        if (not table_name.endswith('/')
                or not self.holds_files(connector, table_name)):
            return
        filesystem, path = self.filesystem(connector, table_name)
        if filesystem.get_file_info(FileSelector(path, allow_not_found=True)):
            raise ArtifactDestinationConflict(
                f"Output directory '{path}/' already contains data from "
                f"another export. Choose an empty destination path.")

    def holds_files(self, connector, path):
        """Whether ``path`` names a named internal stage location."""
        try:
            _split_stage_location(path)
        except (TypeError, ValueError):
            return False
        return True

    def filesystem(self, connector, path):
        """A pyarrow filesystem for an ``@DB.SCHEMA.STAGE/prefix`` path."""
        from kumo_snowflake.stage_fs import stage_filesystem
        with self._session(connector) as session:
            _require_snowpark(session)
            return stage_filesystem(session, path)

    def publish_model(self, connector, bundle_dir, *, path=None, location=None,
                      metadata={}, metrics={}, project_id=None):
        """Registers a model version, or uploads to a stage ``path``."""
        if location is None and path is not None and self.holds_files(
                connector, path):
            return FilesystemSink(self).publish_model(connector, bundle_dir,
                                                      path=path,
                                                      metadata=metadata)
        return super().publish_model(connector, bundle_dir, path=path,
                                     location=location, metadata=metadata,
                                     metrics=metrics, project_id=project_id)

    def fetch_model(self, connector, local_dir, *, path=None, location=None,
                    project_id=None):
        """Downloads a registry version, or the bundle at a stage ``path``."""
        if location is None and path is not None and self.holds_files(
                connector, path):
            return FilesystemSink(self).fetch_model(connector, local_dir,
                                                    path=path)
        return super().fetch_model(connector, local_dir, path=path,
                                   location=location, project_id=project_id)

    def __repr__(self) -> str:
        return f'SnowflakeSinkBackend(identity={self._identity!r})'


class _LocalArtifactConnection:
    """Bind shared Snowflake statements to the local connector's session."""
    def __init__(self, session, connector, extra_columns):
        self.context = session.connection
        self.connector = connector
        self.extra_columns = extra_columns

    def _qualified_table_name(self, table_name):
        return _SnowflakeSessions._qualified(self.connector, table_name)

    def execute_meta_query(self, context, statement, *, creds_clause=None):
        if creds_clause:
            raise ValueError(
                'Local Snowflake output uses its authenticated user stage')
        cursor = context.cursor()
        try:
            cursor.execute(statement)
        finally:
            cursor.close()

    def table_exists(self, table_name):
        from kumo_snowflake.catalog import table_exists
        config = _target_config(self.connector)
        cursor = self.context.cursor()
        try:
            return table_exists(cursor, database=resolved(config.database),
                                schema=resolved(config.schema_name),
                                table=table_name)
        finally:
            cursor.close()

    def table_columns(self, table_name):
        from kumo_snowflake.catalog import columns
        config = _target_config(self.connector)
        cursor = self.context.cursor()
        try:
            values = columns(cursor, database=resolved(config.database),
                             schema=resolved(config.schema_name),
                             table=table_name)
            return [column for (_table, column) in values]
        finally:
            cursor.close()

    def write_artifacts_to_table(self, table_name, artifact_path, replace=False,
                                 *, artifact_files=None):
        from functools import partial

        from kumo_core.artifact_export import prepared_artifact_file
        from kumo_snowflake.write import (
            _upload_local_parquet_to_user_stage,
            write_artifacts_to_table,
        )
        prepare = partial(prepared_artifact_file,
                          extra_columns=self.extra_columns)
        write_artifacts_to_table(
            self, table_name, artifact_path, replace,
            artifact_files=artifact_files,
            upload_local=partial(_upload_local_parquet_to_user_stage,
                                 prepare_file=prepare))

    def overwrite_table(self, source, target):
        from kumo_snowflake.write import overwrite_table
        overwrite_table(self, source, target, quote=self._qualified_table_name)

    def drop_table_if_exists(self, table_name):
        self.execute_meta_query(
            self.context,
            f'DROP TABLE IF EXISTS {self._qualified_table_name(table_name)}')
