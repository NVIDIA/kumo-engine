# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Dict, List, Optional

from io_utils import fsutil
from io_utils.fsutil.credentials import with_filesystem_credentials
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import ConnectorError
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.file_connector import FileConnector
from kumo.connector.parquet_connector import ParquetConnector


class _AnyURIFileMixin:
    """Exact-URI-only file behavior shared by cloud object stores."""

    _normalizer: Callable[[str], str]
    _provider_name: str

    def _table_path(self, table_name: str) -> str:
        path = self._normalizer(table_name)
        if self._provider_name == 'S3':
            from io_utils.fsutil.s3_filesystem import select_s3_region
            select_s3_region(path)
        return path

    @with_filesystem_credentials
    def _get_path(self, table_name: str,
                  data_version: str | None = None) -> str:
        del data_version
        return self._table_path(table_name)

    def _open(self, credentials: Optional[Dict] = None):
        # Identity was attested before construction. Rootless connectors have
        # no account-wide target to probe or discover at open time.
        return None

    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        raise ConnectorError(
            f'Any-URI {self._provider_name} connectors require an exact URI')


class _AnyURICSVConnector(_AnyURIFileMixin, CSVConnector):
    pass


class _AnyURIParquetConnector(_AnyURIFileMixin, ParquetConnector):
    pass


class AnyURIFileConnector(_AnyURIFileMixin, CompositeFileConnector):
    """Rootless connector resolving one canonical URI per operation."""
    def __init__(self, name: str, *, normalizer: Callable[[str], str],
                 provider_name: str):
        FileConnector.__init__(self, name=name, root_dir='', suffix=None)
        # This class intentionally bypasses CompositeFileConnector.__init__
        # because it has no discoverable root, but still uses the composite
        # dataset scan cache when resolving an exact directory URI.
        self._dataset_files_by_path: dict[str, list[tuple[str, str]]] = {}
        self._normalizer = normalizer
        self._provider_name = provider_name
        self.connectors = {
            'csv': _AnyURICSVConnector(name=name, root_dir=''),
            'parquet': _AnyURIParquetConnector(name=name, root_dir=''),
        }
        for connector in self.connectors.values():
            connector._normalizer = normalizer
            connector._provider_name = provider_name

    def validate_permissions(self) -> None:
        return None

    @with_filesystem_credentials
    def _get_path(self, table_name: str,
                  data_version: str | None = None) -> str:
        del data_version
        path, _ = self._resolve_table(table_name)
        return path

    @with_filesystem_credentials
    def _resolve_table(self, table_name: str) -> tuple[str, str]:
        data_path = self._table_path(table_name)
        if fsutil.isdir(data_path):
            return f'{data_path.rstrip("/")}/', self._get_path_type(data_path)
        if fsutil.exists(data_path):
            file_type = self._supported_file_type(data_path)
            if file_type is not None:
                return data_path, file_type
            raise ConnectorError(
                f'The exact {self._provider_name} URI must identify a '
                '.csv or .parquet source')
        raise FileNotFoundError(
            f'The exact {self._provider_name} source URI does not exist')

    def _validate_missing_path_region(self, table_name: str,
                                      data_path: str) -> None:
        # Exact-URI providers route credentials/client construction from each
        # URI rather than a configured connector root.
        return None
