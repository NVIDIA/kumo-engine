# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Writing prediction Parquet into a BigQuery table through
:mod:`kumo_bigquery.write`: the read path of
:class:`~kumo_bigquery.backend.BigQuerySourceBackend` in reverse.

Modes are chosen as reads choose them: ``STAGED`` leads when a probe finds
``staging_uri`` usable, ``STREAMED`` (the Storage Write API) otherwise, and an
explicit ``mode`` is refused rather than substituted.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator, Mapping, Optional

from kumo_bigquery import errors, staging, streaming, write
from kumo_bigquery.session import (
    _BigQuerySessions,
    _billing_project,
    _clients,
    _target_config,
)
from kumo_bigquery.staging import StagingLocation
from kumo_core.sources.diagnostics import translated
from kumo_core.sources.errors import SourceAccessDeniedError
from kumo_core.sources.principals import IdentityRef
from kumo_core.sources.protocol import require_mode, staged_first
from kumo_core.sources.types import Capability, ReadMode
from kumoapi.jobs import WriteMode
from kumoapi.plugin import SourceConnector

logger = logging.getLogger(__name__)


def _replace(write_mode: WriteMode) -> bool:
    if write_mode not in (WriteMode.OVERWRITE, WriteMode.APPEND):
        raise ValueError(f'Unsupported BigQuery write mode: {write_mode!r}')
    return write_mode == WriteMode.OVERWRITE


class _LocalError(Exception):
    """Carries a local file error past the BigQuery error translation."""


class BigQuerySinkBackend(_BigQuerySessions):
    r"""Writes prediction Parquet files to a table in the connector's
    dataset.

    Args:
        identity: A fallback identity for connectors that name none.
        staging_uri: Optional ``gs://`` prefix. Output is uploaded under
            ``<staging_uri>/_kumo.export/<id>/``, loaded by one job and
            deleted. Without one, output is streamed through the Storage
            Write API.
        mode: The mode every write uses, or ``None`` to take whichever
            :meth:`write_modes` offers first.
    """

    _default_hint = 'BigQuerySinkBackend(identity=identity)'

    def __init__(self, identity: Optional[IdentityRef] = None, *,
                 staging_uri: Optional[str] = None,
                 mode: Optional[ReadMode] = None) -> None:
        if mode is ReadMode.STAGED and not staging_uri:
            raise ValueError('mode=STAGED needs a staging_uri')
        super().__init__(identity)
        self.staging_uri = staging_uri
        self.mode = mode
        self._staging = (StagingLocation.parse(staging_uri)
                         if staging_uri else None)
        self._staging_capability: Optional[Capability] = None

    def preflight_write_files(
        self,
        connector: SourceConnector,
        table_name: str,
        write_mode: WriteMode = WriteMode.OVERWRITE,
    ) -> None:
        r"""Fails before any output exists if the destination cannot take
        it: the dataset is missing or elsewhere, the table is not writable,
        the chosen mode's load jobs or write streams cannot run, or a
        required staging prefix is unusable."""
        replace = _replace(write_mode)
        with self._translated(connector, table_name):
            client, dataset = self._dataset(connector, table_name)
            streamed = self._transport(
                connector, f'{dataset}.{table_name}') is ReadMode.STREAMED
            write.probe_write_access(
                client, dataset, table_name, replace=replace,
                create=replace and streamed, probe_stream=partial(
                    self._probe_stream, connector) if streamed else None)
            if replace or not streamed:
                write.require_job_access(client, _billing_project(connector))

    def write_files(
        self,
        connector: SourceConnector,
        table_name: str,
        files: list[str],
        write_mode: WriteMode = WriteMode.OVERWRITE,
        project_id: Optional[str] = None,
        *,
        export_id: Optional[str] = None,
        extra_columns: Optional[Mapping[str, Any]] = None,
    ) -> str:
        r"""Commits exactly ``files`` into ``table_name`` at once.
        ``OVERWRITE`` replaces the table's rows and schema; ``APPEND`` adds
        rows and must match the existing schema.

        Returns:
            The table written, as ``project.dataset.table``.
        """
        del project_id, export_id
        from kumo_core.artifact_staging import selected_local_parquet_files

        replace = _replace(write_mode)
        with self._translated(connector, table_name) as ref:
            client, dataset = self._dataset(connector, table_name)
            selected = selected_local_parquet_files(files)
            if not selected:
                raise ValueError('No Parquet files to export')
            existing = write.existing_table(client, ref)
            if self._transport(connector, ref) is ReadMode.STAGED:
                assert self._staging is not None
                write.staged_write(
                    client, self._storage(connector),
                    self._staging.child(
                        f'{write.EXPORT_PREFIX}/{uuid.uuid4().hex}'), selected,
                    ref, replace=replace, existing=existing,
                    extra_columns=extra_columns)
            else:
                write.streamed_write(
                    client, _clients.write_client(self._principal(connector)),
                    selected, ref, replace=replace, existing=existing,
                    extra_columns=extra_columns,
                    credentials=self._credentials(connector))
        return ref

    def write_table(
        self,
        connector: SourceConnector,
        table_name: str,
        frame: Any,
        write_mode: WriteMode = WriteMode.OVERWRITE,
        project_id: Optional[str] = None,
        *,
        export_id: Optional[str] = None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix='kumo-bq-frame-') as tmp:
            path = os.path.join(tmp, 'part-0.parquet')
            frame.to_parquet(path, index=False)
            return self.write_files(connector, table_name, [path], write_mode,
                                    project_id, export_id=export_id)

    def _dataset(self, connector: SourceConnector,
                 table_name: str) -> tuple[Any, str]:
        if not write.is_output_table_id(table_name):
            raise ValueError(
                f'BigQuery output table must be an unqualified table ID of '
                f'letters, numbers, underscores or hyphens; got '
                f'{table_name!r}.')
        config = _target_config(connector)
        dataset = f'{config.project_id}.{config.dataset_id}'
        client = self._client(connector)
        write.require_dataset(client, dataset, config.location)
        return client, dataset

    def _probe_stream(self, connector: SourceConnector, ref: str) -> None:
        from google.api_core.exceptions import Forbidden
        from google.cloud import bigquery

        write_client = _clients.write_client(self._principal(connector))
        table = bigquery.TableReference.from_string(ref)
        try:
            streaming.probe_write_stream(
                write_client,
                write_client.table_path(table.project, table.dataset_id,
                                        table.table_id))
        except Forbidden:
            raise SourceAccessDeniedError(
                f'This identity cannot write {ref} through the BigQuery '
                f'Storage Write API; enable the BigQuery Storage API and '
                f'grant bigquery.tables.updateData, or configure a GCS '
                f'staging_uri.', table=ref) from None

    def _storage(self, connector: SourceConnector) -> Any:
        return _clients.storage_client(self._principal(connector),
                                       project=_billing_project(connector))

    def probe_staging(self, connector: SourceConnector) -> Capability:
        """As :meth:`BigQuerySourceReader.probe_staging`: answered once and
        never persisted."""
        if self._staging is None:
            return Capability.DENIED
        if self._staging_capability is None:
            storage = self._storage(connector)
            answer = staging.probe_upload(storage, location=self._staging)
            if answer is Capability.UNKNOWN:
                return answer
            if answer is Capability.ALLOWED:
                answer = self._colocated(connector, storage)
            self._staging_capability = answer
        return self._staging_capability

    def _colocated(self, connector: SourceConnector,
                   storage: Any) -> Capability:
        assert self._staging is not None
        bucket = staging.bucket_location(storage, self._staging)
        dataset = _target_config(connector).location
        if bucket is None or staging.colocated(bucket, dataset):
            return Capability.ALLOWED
        logger.warning(
            'Staging bucket gs://%s is in %s, outside the dataset location '
            '%s; output will not be staged there', self._staging.bucket, bucket,
            dataset)
        return Capability.DENIED

    def write_modes(self, connector: SourceConnector) -> tuple[ReadMode, ...]:
        return staged_first(self.probe_staging(connector) is Capability.ALLOWED)

    def _transport(self, connector: SourceConnector, ref: str) -> ReadMode:
        if self.mode is None:
            return self.write_modes(connector)[0]
        if self.mode is ReadMode.STAGED:
            require_mode(ref, self.mode, self.write_modes(connector))
        return self.mode

    @staticmethod
    @contextmanager
    def _translated(connector: SourceConnector,
                    table_name: str) -> Iterator[str]:
        config = _target_config(connector)
        ref = f'{config.project_id}.{config.dataset_id}.{table_name}'
        try:
            with translated(f'bigquery export to {ref}', table=ref,
                            rules=errors.rules(), summarize=errors.summary):
                try:
                    yield ref
                except OSError as error:
                    if error.filename is None:
                        raise
                    raise _LocalError() from error
        except _LocalError as local:
            raise local.__cause__ from None  # type: ignore[misc]

    def __repr__(self) -> str:
        return (f'BigQuerySinkBackend(identity={self._identity!r}, '
                f'staging_uri={self.staging_uri!r}, mode={self.mode!r})')
