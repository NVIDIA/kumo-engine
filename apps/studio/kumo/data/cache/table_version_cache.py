# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from enum import Enum
from typing import List, Literal, Optional, Union

from cachetools import TTLCache
from kumo_core.types.schema import SourceTableID, VersionedSourceTableID
from wrapt import synchronized

from async_utils.sync import FutureDedupeOption, blocking_method
from io_utils import fsutil
from io_utils.json import fast_from_json, load_json, save_json
from kumo.data.cache import table_source_key
from kumo.data.cache.table_cache_dir import (
    CACHE_TTL,
    MAX_CACHE_SIZE,
    table_source_path_component,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from monitoring.tracing import TraceOptions

logger = logging.getLogger(__name__)


class TableIngestionStatus(Enum):
    """Each source table may be associated with many different data verions.
    For example, it may contain a pointer to the most recently successful
    version, and may actively be ingesting a newer version. This enum is used
    to described the various type of versions that may be associated with a
    source table.
    """
    # The most recent successfully ingested data version
    SUCCESSFUL = 'SUCCESSFUL'
    # The most recently failed ingested data version
    FAILED = 'FAILED'
    # The most recently completed ingested data version (regardless of
    # success or failed)
    COMPLETED = 'COMPLETED'

    # Note: it may make sense to add a "started" version, indicated the most
    # recently started data ingestion workflow (but not necessarily completed)

    @property
    def s3_file_name(self) -> str:
        # The name of the file containing the version string on s3
        if self == TableIngestionStatus.SUCCESSFUL:
            return "version"
        if self == TableIngestionStatus.FAILED:
            return "failure_version"
        if self == TableIngestionStatus.COMPLETED:
            return "completed_version"
        raise RuntimeError(f"Invalid TableIngestionStatus {self}")


class TableVersionCache():
    """The backend cache for everything related to a table version, include:
    table_versions and table_versions_for_job_id.
    """
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir

        # connector --> source name --> data version
        # This map doesn't need to be persisted, since a table's data version
        # can change once the underlying data gets refreshed. Instead, we
        # use a WeakKeyDictionary with connector object as the key.
        # This allows auto-purging entries in this map when the connector is
        # garbage collected.
        # TODO(dong) after more clean up, we can make it connector local.
        self._table_versions = TTLCache(MAX_CACHE_SIZE, CACHE_TTL)

    def reset(self, cache_dir: str | None = None):
        if cache_dir is not None:
            self._cache_dir = cache_dir
        self._table_versions = TTLCache(MAX_CACHE_SIZE, CACHE_TTL)

    # Persistent object-store I/O must not hold the in-memory cache lock.
    # Transfer workers can finalize connectors and call remove(), which needs
    # the same lock.
    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    def save_for_job(
        self,
        job_id: str,
        connector_name: str,
        table_source_name: str,
        data_version: str,
    ):
        r"""Cache and save the data version for the specified table associated
        with a given job_id. This is used to make sure the data version for a
        table is consistently used throughout the execution of training,
        prediction, or edge stats workflow.
        Args:
            job_id (str): The job ID the data version is associated with.
                Could be the training or prediction job ID, or the graph
                version in case of edge stats workflow.
            connector_name (str): Connector name of the table
            table_source_name (str): Source name of the table
            data_version (str): Data version of the table to be cached
        """
        self._save(job_id, connector_name, table_source_name,
                   TableIngestionStatus.SUCCESSFUL, data_version)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    def get_for_job(
        self,
        job_id: str,
        connector_name: str,
        table_source_name: str,
    ) -> Optional[str]:
        r"""Retrieve the cached data version for a table associated with a
        given job_id. It would first check if the requested data version has
        already been cached. If not, it will attempt to load the data version
        from the cache persist dir if it's persisted.
        Args:
            job_id (str): The job ID for which to retrieve the cached table
                data version
            connector_name (str): The connector name of the table
            table_source_name (str): Source name of the table
        Returns:
            Optional[str]: The cached data version. None if not present.
        """
        table_key = table_source_key(table_source_name, connector_name, '')
        file_path = self._version_file_path(job_id, connector_name,
                                            table_source_name,
                                            TableIngestionStatus.SUCCESSFUL)

        value = load_json(file_path, str, default=None, raise_on_missing=False)
        if value is None:
            logger.warning(f'No table versions found at {file_path}')
            return None
        else:
            logger.info(f'Loading table {table_key} version for {job_id} '
                        f'from {file_path}')
            return value

    def _save(self, job_id: str, connector_name: str, source_name: str,
              version_type: TableIngestionStatus,
              table_version: Optional[str]) -> None:
        table_key = table_source_key(source_name, connector_name, '')
        file_path = self._version_file_path(job_id, connector_name, source_name,
                                            version_type)
        logger.info(
            f'Saving table_version={table_version} for table_key={table_key} '
            f'with version_type={version_type} to {file_path}')
        save_json(table_version, file_path)

    @synchronized
    def save_in_memory(
        self,
        connector_name: str,
        table_source_name: str,
        data_version: str,
    ):
        r"""Add table data version to cache.

        Args:
        connector_name (str): ID of the connector for the table.
        table_source_name (str): Name of the source table whose version is
            being saved.
        data_version (str): The data version.
        """
        self._table_versions[(connector_name, table_source_name)] = data_version

    @synchronized
    def _get_in_memory(
        self,
        connector_name: str,
        table_source_name: str,
    ) -> Optional[str]:
        return self._table_versions.get((connector_name, table_source_name))

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def save(
        self,
        connector_name: str,
        source_name: str,
        table_version: Optional[str] = None,
        is_version_ingestion_successful: Optional[bool] = True,
    ):
        r"""
        Saves the table_version in the location
        data-persist-dir/data_backend_cache/versions/latest/<connector_name>/<table_name>/, # noqa
        after an ingestion attempt. If the ingestion is successful,
        the version is saved in a file name called 'version'. If it has failed,
        the version is saved in a file name 'failure_version'. Regardless of
        success or failure, it will save the version in a file called
        'latest_version', representing the most recent workflow execution.
        """
        version_type = TableIngestionStatus.SUCCESSFUL
        if not is_version_ingestion_successful:
            version_type = TableIngestionStatus.FAILED
        self._save('latest', connector_name, source_name, version_type,
                   table_version)
        self._save('latest', connector_name, source_name,
                   TableIngestionStatus.COMPLETED, table_version)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    def get(self, connector_name: str, table_source_name: str) -> Optional[str]:
        """
        Get the most recent ingested version for a table.

        Args:
            connector_name (str): The name of the connector for the table. If
                the connector is an unnamed s3 connector created from the SDK,
                the name should be generated using
                FileConnectorResourceConfig.synthesize. For all other
                connectors, the name should be the connector name set in the
                SDK. All UI based connectors including s3 connectors already
                have name.
            table_source_name (str): The name of the table in the source for
                which to get the latest ingested version.

        Returns:
            Optional[str]: The most recent ingested version for the table.
        """

        # Do not hold the in-memory cache lock during object-store I/O.
        # Connector finalizers remove entries from this cache and may run in
        # an S3 transfer worker. Holding the lock while waiting for that
        # transfer deadlocks the caller and finalizer.
        ret = self._get_in_memory(connector_name, table_source_name)
        if ret is None:
            file_path = self._version_file_path('latest', connector_name,
                                                table_source_name,
                                                TableIngestionStatus.SUCCESSFUL)

            value = load_json(file_path, str, default=None,
                              raise_on_missing=False)
            if value is None:
                logger.warning(f'No table versions found at {file_path}')
                return None

            logger.info(
                f'Loading latest ingested version for table {connector_name}'
                f'.{table_source_name} from {file_path}')
            self.save_in_memory(connector_name, table_source_name, value)
            return value
        return ret

    @synchronized
    def remove(self, connector_name: str):
        r"""Remove the cached table data versions for a connector from
        memory."""
        to_be_delete = [
            key for key in self._table_versions if key[0] == connector_name
        ]
        for key in to_be_delete:
            try:
                del self._table_versions[key]
            except Exception:
                # Ignore exception if the key is expired.
                pass

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def get_many(
        self,
        tables: List[SourceTableID],
        version_type: TableIngestionStatus = TableIngestionStatus.SUCCESSFUL,
    ) -> List[Optional[VersionedSourceTableID]]:
        """
        Retrieves success or failed versions of ingested tables. If
        `retrieved_failed_versions` is ommitted or set to `False`, all
        successful versions are retrieved, if it is set to `True` only failed
        versions are retrieved.
        `None` is returned for tables whose version is not found.
        """
        paths: List[str] = []
        for table in tables:
            path = self._version_file_path(
                'latest',
                table.connector_name,
                table.source_table_name,
                version_type,
            )
            paths.append(path)
        versions = fsutil.get_many_if_present(paths)

        # Parse the result as a json string
        results: List[Optional[VersionedSourceTableID]] = []
        for table, version in zip(tables, versions):
            if version is None:
                results.append(None)
                continue
            results.append(table.make_versioned_id(fast_from_json(version,
                                                                  str)))

        return results

    def _version_file_path(
        self,
        job_id: Union[str, Literal['latest']],
        connector_name: str,
        source_table_name: str,
        version_type: TableIngestionStatus,
    ):
        return osp.join(self._cache_dir, 'versions', job_id, connector_name,
                        table_source_path_component(source_table_name),
                        version_type.s3_file_name)
