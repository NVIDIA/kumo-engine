# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextvars
import os
import os.path as osp
import posixpath
import re
from abc import abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Union

import pandas as pd
from kumo_core.sources import layout
from kumo_core.sources.errors import SourceNotFoundError
from kumo_core.sources.types import StagedTable
from kumo_core.types.schema import IngestedTableFingerprint
from kumoapi.typing import Dtype

from io_utils import fsutil, logging
from io_utils.fsutil import s3_connection
from io_utils.fsutil.credentials import (
    ContextPropagatingThreadPoolExecutor,
    credential_scope,
    credentials_for_provider,
    with_filesystem_credentials,
)
from io_utils.fsutil.uri import GCSURI, S3URI, ParseURI
from kumo.config import BaseColumnConfig
from kumo.connector import DEFAULT_DATA_VERSION, Connector
from kumo.connector.blocking import blocking_method
from kumo.connector.connector import (
    ConnectorError,
    ExportMetrics,
    SourceTableNotFoundError,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from service_framework.deploy_env_config import deploy_env

logger = logging.getLogger(__name__)

MAX_NUM_FILES_TO_SAMPLE = 10
MAX_SUPPORTED_FILES = layout.MAX_SUPPORTED_FILES


def _diskgraph_refreshable_workload_method(source: Any) -> str | None:
    """Return a workload provider the DiskGraph child can reconstruct."""
    method = getattr(source, 'method', None)
    if method not in s3_connection.AWS_WORKLOAD_CREDENTIAL_METHODS:
        return None
    if method == 'assume-role-with-web-identity':
        required = ('AWS_ROLE_ARN', 'AWS_WEB_IDENTITY_TOKEN_FILE')
        if not all(os.environ.get(name) for name in required):
            return None
    if method == 'container-role':
        selectors = ('AWS_CONTAINER_CREDENTIALS_FULL_URI',
                     'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI')
        if not any(os.environ.get(name) for name in selectors):
            return None
    return method


def _scan_dataset_files(
    path: str,
    suffixes: tuple[str, ...],
    max_entries: int | None = None,
) -> list[tuple[str, str]]:
    """Return relative and physical files confined to one directory root."""
    root = path.rstrip('/')
    root_uri = ParseURI(root)
    root_prefix = root_uri.path.rstrip('/')
    if root_prefix or root_uri.path == '/':
        root_prefix += '/'
    root_identity = (type(root_uri),
                     root_uri.model_dump(exclude={'path', 'scheme'}))
    normalized_suffixes = tuple(f'.{suffix.lstrip(".").lower()}'
                                for suffix in suffixes)
    pending = deque([root])
    visited = {root}
    num_entries = 0
    files: list[tuple[str, str]] = []
    use_listing_types = isinstance(root_uri, (S3URI, GCSURI))
    while pending:
        directory = pending.popleft()
        if use_listing_types:
            entries = fsutil.listdir(directory, max_items=None, detail=True)
            children = [entry.path for entry in entries]
            directories = {
                entry.path
                for entry in entries if entry.is_directory
            }
        else:
            children = fsutil.listdir(directory, max_items=None)
            directories = None
        for child in sorted(children):
            child_uri = ParseURI(child)
            child_identity = (type(child_uri),
                              child_uri.model_dump(exclude={'path', 'scheme'}))
            relative = (child_uri.path.removeprefix(root_prefix)
                        if child_identity == root_identity
                        and child_uri.path.startswith(root_prefix) else '')
            components = relative.split('/')
            if (not relative or '\\' in relative
                    or any(part in {'', '.', '..'} for part in components)):
                raise ValueError(
                    'Dataset listing returned an unsafe relative path: '
                    f'{child!r}')
            num_entries += 1
            if max_entries is not None and num_entries > max_entries:
                raise OverflowError(
                    f'Dataset exceeds the supported entry limit: {max_entries}')
            if (child in directories
                    if directories is not None else fsutil.isdir(child)):
                canonical = child.rstrip('/')
                if canonical not in visited:
                    visited.add(canonical)
                    pending.append(canonical)
            elif relative.lower().endswith(normalized_suffixes):
                files.append((relative, child))

    return sorted(files)


def _list_dataset_files(
    path: str,
    suffixes: tuple[str, ...],
    max_entries: int | None = None,
) -> List[str]:
    return [
        file[1] for file in _scan_dataset_files(path, suffixes, max_entries)
    ]


def _flat_parquet_files(
    path: str,
    scanned_files: list[tuple[str, str]] | None = None,
) -> list[str] | None:
    """Return exact flat Parquet files, or None for non-flat inputs."""
    if fsutil.isdir(path):
        files = (scanned_files if scanned_files is not None else
                 _scan_dataset_files(path, ('parquet', )))
        files = [(relative, physical) for relative, physical in files
                 if relative.lower().endswith('.parquet')]
        if not files or layout.nested([relative for relative, _ in files]):
            return None
        return [physical for _, physical in files]
    if path.lower().endswith('.parquet') and fsutil.exists(path):
        return [path]
    return None


class FileConnector(Connector):
    r"""Establishes a connection to a set of files either on-disk or remote.

    Args:
        name (str): The name of the connector
        root_dir (str): Root directory where a set of files are stored.
        suffix (str): The suffix of the file.
    """
    def __init__(
        self,
        name: str,
        root_dir: str,
        suffix: str,
        **kwargs,
    ):
        super().__init__(name)

        if osp.exists(root_dir):
            root_dir = osp.expanduser(osp.normpath(root_dir))

        self.root_dir = root_dir
        self.suffix = suffix
        self._direct_parquet_paths_cache: dict[str, list[str] | None] = {}

    def _clear_direct_source_cache(self) -> None:
        self._direct_parquet_paths_cache.clear()

    def require_explicit_credentials(self) -> None:
        super().require_explicit_credentials()
        self._clear_direct_source_cache()

    def set_credentials(self, credentials: Optional[Dict]) -> None:
        super().set_credentials(credentials)
        self._clear_direct_source_cache()

    def _table_path(self, table_name: str) -> str:
        return osp.join(self.root_dir, table_name)

    def _supports_nested_dataset(self, path: str) -> bool:
        del path
        return False

    @contextmanager
    def filesystem_credential_scope(self,
                                    path: str | None = None) -> Iterator[None]:
        with credential_scope(
                getattr(self, '_credentials', None),
                require_explicit=getattr(self, '_require_explicit_credentials',
                                         False)):
            if path and getattr(self, '_provider_name', None) == 'S3':
                from io_utils.fsutil.s3_filesystem import select_s3_region
                select_s3_region(path)
            yield

    # Connecting and closing connections ######################################

    def _open(self, credentials: Optional[Dict] = None) -> Any:
        try:
            # TODO properly handle HTTPS://-based filepaths, for which we lose
            # significant connector functionality (e.g. isdir, listdir). For
            # now, ignore validation on these filepaths:
            if not re.match(
                    r'^https?://',
                    self.root_dir,
            ) and not fsutil.exists(self.root_dir):
                raise ConnectorError(f"Root dir {self.root_dir} does not exist")
        except PermissionError:
            raise ConnectorError(
                'The selected identity cannot access the configured file '
                'connector target.') from None

    def _close(self, context: Any):
        pass

    # Reading table metadata ##################################################

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _get_data_version(
        self,
        source_table_name: str,
    ) -> str:
        try:
            source_path = self._table_path(source_table_name)
            if isinstance(ParseURI(source_path), (S3URI, GCSURI)):
                from kumo.connector.object_store_access import source_reader

                if self.suffix:
                    reader = source_reader(self._get_path(source_table_name),
                                           file_format=self.suffix)
                    name = ''
                else:
                    reader = source_reader(self.root_dir or source_path)
                    name = source_table_name if self.root_dir else ''
                return reader.data_version(name)
            table_path = self._get_path(source_table_name)
            # Missing paths return the shared sentinel on local and object
            # stores, but valid nested datasets can use that sentinel too.
            # Confirm existence only on the ambiguous sentinel path.
            data_version = str(
                fsutil.get_last_modified_time(table_path)).split('.')[0]
            if (data_version == DEFAULT_DATA_VERSION
                    and not fsutil.exists(table_path)):
                raise SourceTableNotFoundError(source_table_name, self.name)
            # Returning only the pre-decimal part keeps version names sane.
            return data_version
        except (FileNotFoundError, SourceNotFoundError):
            raise SourceTableNotFoundError(source_table_name,
                                           self.name) from None

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    @with_filesystem_credentials
    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        # List all tables in the root directory. Dataset could be local or
        # in S3. A table is stored in one of the following formats:
        # a) Flat: One file per table. Filename corresponds to the table name.
        # All files are in the root directory. Only files with
        # b) Hierachical: One directory per table. The directory name
        # corresponds to the table name. There could be multiple files in the
        # directory. Internally, these files are sorted by their names and
        # concatenated to produce the table data in a deterministic order.
        table_names: Set[str] = set()
        root_dir_entries = fsutil.listdir(path=self.root_dir,
                                          max_items=MAX_SUPPORTED_FILES)
        for entry in root_dir_entries:
            entry_basename = osp.basename(entry)
            if entry_basename.endswith(self.suffix):
                table_names.add(osp.splitext(entry_basename)[0])
            else:
                table_names.add(entry_basename)

        logger.info(f"Connector: {self.name}, root_dir: {self.root_dir}, "
                    f"table_names: {table_names}")
        if not len(table_names):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' does not support the dataset"
                f" format. A table may be stored as a single file in the"
                f" the root_dir -or- as multiple files in a sub-directory"
                f" under root_dir.")

        return list(table_names)

    # TODO: add the suffix option to fsutil.listdir to combine functionality
    def table_path_to_files(self, path: str) -> List[str]:
        path_to_files = []
        if fsutil.isdir(path):
            for file in fsutil.listdir(path):
                if fsutil.isdir(file) or not file.endswith(self.suffix):
                    continue
                path_to_files.append(file)
        else:
            path_to_files.append(path)

        return path_to_files

    # Reading raw table data ##################################################

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _read_table(
        self,
        context: Any,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        return self._load(self._get_path(table_name), pkey, cols)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _read_table_chunk(
        self,
        context: Any,
        table_name: str,
        num_rows: int,
        offset: int = 0,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        return self._load(self._get_path(table_name), pkey, cols,
                          num_rows=num_rows, offset=offset)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _read_table_sample(
        self,
        context: Any,
        table_name: str,
        num_rows: Optional[int],
        frac: Optional[float],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        if not num_rows and not frac:
            raise ConnectorError(
                "Connector requires num_rows or frac for reading "
                "a table sample.")

        path = self._get_path(table_name)
        if (fsutil.isdir(path) and not self._supports_nested_dataset(path)
                and fsutil.contains_subdir(path)):
            raise ConnectorError(
                f"Table '{table_name}' contains nested sub-directories, which "
                "is not supported by this connector. If your data is "
                "partitioned, organize it under one directory and retry.")

        return self._load(path, pkey, cols, num_rows=num_rows, frac=frac,
                          sample=True)

    def _create_iterator(
        self,
        path: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> Iterator[pd.DataFrame]:
        raise NotImplementedError(
            f"Get iterator is not implemented for {self.__class__.__name__}.")

    def _get_iterator(
        self,
        context: Any,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> Iterator[pd.DataFrame]:
        return self._create_iterator(self._get_path(table_name), pkey, cols)

    @abstractmethod
    def _load(
        self,
        path: Union[str, StringIO],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
        num_rows: Optional[int] = None,
        offset: int = 0,
        frac: Optional[float] = None,
        sample: Optional[bool] = False,
        **kwargs,
    ) -> pd.DataFrame:
        pass

    def _get_path(self, table_name: str,
                  data_version: str | None = None) -> str:
        data_path = self._table_path(table_name)
        # Object-store source coordinates are connector-relative paths.  A
        # caller may therefore name either a dataset directory (``events``)
        # or a concrete object (``2026/07/events.parquet``).  Preserve an
        # explicit supported suffix instead of appending it a second time.
        if (fsutil.isdir(data_path)
                or table_name.lower().endswith(f'.{self.suffix.lower()}')):
            return data_path
        return f'{data_path}.{self.suffix}'

    def _direct_parquet_paths(self, source_name: str) -> list[str] | None:
        if self.suffix is None or self.suffix.lower() != 'parquet':
            return None
        return _flat_parquet_files(self._get_path(source_name))

    def _cached_direct_parquet_paths(self,
                                     source_name: str) -> list[str] | None:
        if source_name not in self._direct_parquet_paths_cache:
            self._direct_parquet_paths_cache[
                source_name] = self._direct_parquet_paths(source_name)
        return self._direct_parquet_paths_cache[source_name]

    def _has_ambient_diskgraph_identity(self, path: str) -> bool:
        credentials = getattr(self, '_credentials', None) or {}
        implicit_ambient = (
            not getattr(self, '_require_explicit_credentials', False)
            and not credentials)
        if fsutil.is_s3(path):
            # DiskGraph has one ambient S3 auth profile for both source reads
            # and its streaming dataplane mirror.  An identity-owned static
            # connection must therefore use the existing authorized staging
            # path so its credentials remain orthogonal to deployment S3.
            ambient = (implicit_ambient
                       or credentials.get('ambient_session') is not None)
            # DiskGraph has only one S3 environment for both reads and writes.
            # When deployment storage has its own connection, stage an ambient
            # AWS source through the connector so its credentials cannot
            # replace the MinIO dataplane credentials in the child process.
            if not ambient:
                return False
            if s3_connection.deployment_s3_connection() is None:
                return True
            deployment_root = deploy_env().data_plane_storage_root.rstrip('/')
            return (path == deployment_root
                    or path.startswith(deployment_root + '/'))
        if fsutil.is_gcs(path):
            return implicit_ambient or credentials.get('ambient') is True
        if fsutil.is_adls(path):
            # The implicit connector mode deliberately selects the worker's
            # native Azure credential chain. Explicit Azure identities still
            # fail closed here and remain staged until the credential-profile
            # follow-up can carry their resolved identity to DiskGraph.
            return implicit_ambient
        return False

    @with_filesystem_credentials
    def uses_direct_diskgraph_source_for(self, source_name: str) -> bool:
        """Whether one table is flat cloud Parquet with ambient identity."""
        paths = self._cached_direct_parquet_paths(source_name)
        return bool(paths and self._has_ambient_diskgraph_identity(paths[0]))

    @with_filesystem_credentials
    def diskgraph_source(self, *, source_name: str) -> Dict[str, Any]:
        """Build an exact non-secret object manifest under connector auth."""
        paths = self._cached_direct_parquet_paths(source_name)
        if not paths or not self._has_ambient_diskgraph_identity(paths[0]):
            raise ConnectorError(
                'Direct DiskGraph source requires flat cloud Parquet and an '
                'ambient connector identity.')

        def capture(path: str) -> Dict[str, Any]:
            identity = fsutil.object_identity(path)
            if identity.size <= 0:
                raise ConnectorError(
                    'Direct DiskGraph ingest requires non-empty Parquet '
                    f'objects: {path!r}.')
            if identity.etag is None and identity.version is None:
                raise ConnectorError(
                    'Cloud object metadata did not provide an immutable '
                    'identity for direct DiskGraph ingest.')
            # GCS JSON metadata ETags are not XML download If-Match values.
            # The generation already pins the exact object for DiskGraph.
            etag = (None if isinstance(ParseURI(path), GCSURI)
                    and identity.version is not None else identity.etag)
            return {
                'path': path,
                'size': identity.size,
                'e_tag': etag,
                'version': identity.version,
            }

        # Provider metadata calls are independent. Keep ordering stable while
        # avoiding a serial HEAD/reload bottleneck for large flat datasets.
        max_workers = min(32, len(paths))
        with ContextPropagatingThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix='DiskGraph-manifest') as executor:
            objects = list(executor.map(capture, paths))
        return {
            'type': 'object_store_parquet',
            'objects': objects,
            'auth_profile': 'ambient',
        }

    @with_filesystem_credentials
    def diskgraph_subprocess_environment(
        self,
        source_name: str,
    ) -> Dict[str, str]:
        """Prepare non-secret settings for one direct object source."""
        paths = self._cached_direct_parquet_paths(source_name)
        if not paths or not self._has_ambient_diskgraph_identity(paths[0]):
            raise ConnectorError(
                'DiskGraph subprocess credentials require an authorized '
                'direct object source.')
        path = paths[0]
        if not fsutil.is_s3(path):
            return {}

        scoped = credentials_for_provider('s3')
        credentials = dict(scoped.credentials or {}) if scoped else {}
        resolved_connection = credentials.get('s3_connection')
        if resolved_connection is not None:
            if not isinstance(resolved_connection, s3_connection.S3Connection):
                raise ConnectorError(
                    'Resolved S3 connector connection is invalid.')
            try:
                return s3_connection.s3_connection_environment(
                    resolved_connection)
            except RuntimeError:
                raise ConnectorError(
                    'Resolved S3 connector has no credentials.') from None
        connection = s3_connection.deployment_s3_connection()
        deployment_root = deploy_env().data_plane_storage_root.rstrip('/')
        is_deployment_source = (path == deployment_root
                                or path.startswith(deployment_root + '/'))
        if connection is not None and is_deployment_source:
            return s3_connection.s3_subprocess_environment()

        session = credentials.get('ambient_session')
        authorized_ambient_session = session is not None
        if not authorized_ambient_session:
            import boto3
            session = boto3.Session()
        source = session.get_credentials()
        if source is None:
            raise ConnectorError(
                'Authorized ambient AWS session has no credentials.')
        refreshable_method = (_diskgraph_refreshable_workload_method(source)
                              if authorized_ambient_session else None)
        frozen = None
        if refreshable_method is None:
            # Providers the child cannot reconstruct, along with the legacy
            # no-Identity connector path, retain the existing frozen handoff.
            frozen = source.get_frozen_credentials()
            if not (frozen.access_key and frozen.secret_key):
                raise ConnectorError(
                    'Authorized ambient AWS session has no credentials.')
            if not frozen.token:
                raise ConnectorError(
                    'Authorized ambient AWS session has no temporary '
                    'credentials.')

        region = credentials.get('region') or session.region_name
        if not region:
            uri = ParseURI(path)
            response = session.client('s3').head_bucket(Bucket=uri.bucket)
            region = (response.get('BucketRegion')
                      or response.get('ResponseMetadata', {}).get(
                          'HTTPHeaders', {}).get('x-amz-bucket-region')
                      or 'us-east-1')
        environment = {
            'AWS_REGION': str(region),
            'AWS_DEFAULT_REGION': str(region),
        }
        if refreshable_method is not None:
            # Preserve the provider boundary instead of freezing its current
            # credential. DiskGraph resolves the same pod identity in the child
            # and retains the SDK provider so temporary credentials can rotate.
            return environment

        assert frozen is not None
        environment.update({
            'AWS_ACCESS_KEY_ID': frozen.access_key,
            'AWS_SECRET_ACCESS_KEY': frozen.secret_key,
            'AWS_SESSION_TOKEN': frozen.token,
            'AWS_EC2_METADATA_DISABLED': 'true',
        })
        return environment

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _write_pandas(self, context: Any, table_name: str, df: pd.DataFrame,
                      replace: bool = False) -> None:
        path = self._get_path(table_name)
        if fsutil.exists(path) and not replace:
            # TODO(dong) implement the function to append table.
            raise ConnectorError(
                f"Cannot write table to {path} because the file "
                "already exists.")
        else:
            fsutil.rm(path, ignore_errors=True)

        self._write(path, df)

    def export_table(
        self,
        source_name: str,
        table_fp: IngestedTableFingerprint,
        data_version: str,
        *,
        graph_table_count: Optional[int] = None,
    ) -> ExportMetrics:
        from kumo.connector.source_bridge import publish_source
        from kumo.data import MaterializationStage
        from kumo.data.utils import get_materialize_path

        with self._local_source(source_name, columns=dict(table_fp.cols),
                                version=data_version,
                                graph_table_count=graph_table_count) as (staged,
                                                                         kind):
            destination = get_materialize_path(table_fp, data_version,
                                               MaterializationStage.INGEST)
            metrics = publish_source(staged, destination)
            if kind == layout.CSV:
                return metrics
            assert staged.scanned_bytes is not None
            return ExportMetrics(num_rows=staged.num_rows,
                                 source_size=staged.scanned_bytes)

    @contextmanager
    def _local_source(
        self,
        source_name: str,
        *,
        columns: Optional[Mapping[str, Dtype]] = None,
        version: Optional[str] = None,
        graph_table_count: Optional[int] = None,
        transform: bool = True,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> Iterator[tuple[StagedTable, str]]:
        from kumo_core.sources.types import ReadMode

        from kumo.connector.source_bridge import worker_ceiling

        with TemporaryDirectory(prefix='kumo-file-source-') as scratch:
            with self.filesystem_credential_scope(source_name):
                reader, name = self._source_reader(source_name)
                _, kind = reader.resolve(name)
                if columns is not None:
                    reader.validate_ingest_columns(name, columns)
                staged = reader.prepare(
                    name, osp.join(scratch, 'prepared'),
                    columns=columns if kind == layout.CSV else None,
                    mode=ReadMode.STREAMED, version=version,
                    worker_ceiling=worker_ceiling(graph_table_count),
                    transform=transform, max_entries=max_entries,
                    max_bytes=max_bytes)
                if kind == layout.PARQUET and staged.scanned_bytes is None:
                    # Preserve the existing best-effort size fallback only
                    # when the shared object's listing lacks sizes.
                    staged = replace(
                        staged,
                        scanned_bytes=self._get_table_source_size(source_name))
            yield staged, kind

    def _source_reader(self, source_name: str):
        from kumo.connector.object_store_access import source_reader

        path = self._get_path(source_name)
        reader = source_reader(path, file_format=self._source_format(path),
                               file_suffix=self.suffix)
        source = reader.root
        context = contextvars.copy_context()

        def fetch(source_path: str, target: str) -> None:
            relative = posixpath.relpath(source_path, source)
            remote = path if relative == '.' else path.rstrip(
                '/') + '/' + relative
            context.copy().run(fsutil.copy_file, remote, target, overwrite=True)

        return replace(reader, root=osp.dirname(source),
                       csv_options=self._csv_read_options(path),
                       fetch=reader.fetch or fetch), osp.basename(source)

    def _source_format(self, path: str):
        return layout.PARQUET

    def _csv_read_options(self, path: str):
        return None

    def download(
        self,
        source_name: str,
        table_fp,
        data_version: str,
        local_dest: str,
    ) -> None:
        """Copy from persistent storage to local_dest.

        Per-table NUM_ROWS / SOURCE_SIZE come from `export_table`'s
        `ExportMetrics`, so download itself returns nothing.
        """
        from kumo.data import MaterializationStage  # pants: no-infer-dep
        from kumo.data.utils import get_materialize_path  # pants: no-infer-dep
        source_path = get_materialize_path(table_fp, data_version,
                                           MaterializationStage.INGEST)
        if fsutil.isdir(source_path):
            fsutil.copy(source_path, local_dest, overwrite=True)
        else:
            fsutil.copy_file(source_path, local_dest, overwrite=True)
        logger.info('download: copied %s to %s', source_path, local_dest)

    # Python built-ins #######################################################

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(root_dir={self.root_dir})'
