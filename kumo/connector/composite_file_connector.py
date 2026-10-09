# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
from kumo_core.sources import layout
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.utils.registry import register

from io_utils import fsutil, logging
from io_utils.fsutil.credentials import with_filesystem_credentials
from kumo.config import BaseColumnConfig, ConnectorConfig, TableMetadataConfig
from kumo.connector import CSVConnector, ParquetConnector
from kumo.connector.blocking import blocking_method
from kumo.connector.connector import ConnectorError
from kumo.connector.file_connector import (
    MAX_SUPPORTED_FILES,
    FileConnector,
    _flat_parquet_files,
    _scan_dataset_files,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.utils import to_dataclass
from kumo.validation import ValidationException
from service_framework.aws_region import AWS_REGION

logger = logging.getLogger(__name__)

_PARQUET = layout.PARQUET
_CSV = layout.CSV


def _check_suffix_error(table_name: str) -> str:
    if table_name.endswith(_PARQUET):
        return f" Or remove the suffix {_PARQUET}"
    elif table_name.endswith(_CSV):
        return f" Or remove the suffix {_CSV}"
    return ""


class CompositeFileConnector(FileConnector):
    r"""Establishes a connection to a set of files either on-disk or remote.
    This connector support both parquet files and csv files. If there are
    tables with the same name, the following order is applied:
        - check if table is a parquet dataset.
        - check if table is a csv dataset.
        - check if table is a parquet file.
        - check if table is a csv file.

    Note that there's warning in rest/backends/data_backend/data_backend.py
    for duplicate table names.

    Args:
        name (str): The name of the connector
        root_dir (str): Root directory where a set of files are stored.
    """
    def __init__(
        self,
        name: str,
        root_dir: str,
        **kwargs,
    ):
        super().__init__(name, root_dir, suffix=None, **kwargs)
        self._dataset_files_by_path: dict[str, list[tuple[str, str]]] = {}

        self.connectors = {
            _CSV:
            CSVConnector(
                name=f"{name}",
                root_dir=root_dir,
                suffix='csv',
                **kwargs,
            ),
            _PARQUET:
            ParquetConnector(
                name=f"{name}",
                root_dir=root_dir,
                suffix='parquet',
                **kwargs,
            ),
        }

    def _clear_direct_source_cache(self) -> None:
        super()._clear_direct_source_cache()
        self._dataset_files_by_path.clear()

    def _get_connector_type(self) -> str:
        return "csv_or_parquet"

    def require_explicit_credentials(self) -> None:
        """Require the same explicit credential scope in every delegate."""
        super().require_explicit_credentials()
        for connector in self.connectors.values():
            connector.require_explicit_credentials()

    def set_credentials(self, credentials: Optional[Dict]) -> None:
        """Propagate the resolved identity to CSV and Parquet delegates."""
        super().set_credentials(credentials)
        for connector in self.connectors.values():
            connector.set_credentials(credentials)

    @with_filesystem_credentials
    def validate_permissions(self) -> None:
        """Verify read access at the exact configured root without writes."""
        # ADLS has real hierarchical directories (unlike S3/GCS virtual
        # prefixes), and its generic list helper intentionally represents a
        # missing path as an empty list.  Distinguish a missing/inaccessible
        # container or directory from a valid empty one before listing it.
        if fsutil.is_adls(self.root_dir) and not fsutil.exists(self.root_dir):
            raise ConnectorError(
                'The configured ADLS container or directory is unavailable')
        fsutil.listdir(path=self.root_dir, max_items=1)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    @with_filesystem_credentials
    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        """List candidate tables under the connector.

        Discovery identifies file and dataset boundaries, but does not check
        whether each table can be read and parsed.
        """
        from kumo.connector.object_store_access import source_reader

        try:
            table_names = source_reader(self.root_dir).tables()
        except (OverflowError, PermanentSourceError) as error:
            raise ConnectorError(message=str(error)) from None

        logger.info(f"Connector: {self.name}, root_dir: {self.root_dir}, "
                    f"table_names: {table_names}")
        if not len(table_names):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' does not support the dataset"
                f" format. A table may be stored as a single file in the"
                f" the root_dir -or- as multiple files in a sub-directory"
                f" under root_dir.")

        return table_names

    # Reading raw table data ##################################################

    def _get_path(self, table_name: str,
                  data_version: str | None = None) -> str:
        del data_version
        path, _ = self._resolve_table(table_name)
        return path.rstrip('/')

    def _direct_parquet_paths(self, source_name: str) -> list[str] | None:
        path, file_type = self._resolve_table(source_name)
        if file_type != _PARQUET:
            return None
        return _flat_parquet_files(
            path,
            self._dataset_files_by_path.get(path.rstrip('/')),
        )

    def _get_path_type(self, path: str, table_name: str = '') -> str:
        if fsutil.isdir(path):
            files = [(relative, physical)
                     for relative, physical in _scan_dataset_files(
                         path, (_PARQUET, _CSV))
                     if not layout.is_hidden(relative)]
            self._dataset_files_by_path[path.rstrip('/')] = files
            try:
                file_type = layout.classify(path,
                                            [relative for relative, _ in files],
                                            table_name)
            except PermanentSourceError as error:
                raise ValidationException(title="Bad Path",
                                          message=str(error)) from None
            if file_type is not None:
                return file_type
        else:
            file_type = self._supported_file_type(path)
            if file_type is not None:
                return file_type

        raise ValidationException(
            title="Bad Path",
            message=(f"No files directly under {path} have a .csv or "
                     ".parquet suffix."))

    def _supports_nested_dataset(self, path: str) -> bool:
        return self._get_path_type(path) == _PARQUET

    @staticmethod
    def _supported_file_type(path: str) -> Optional[str]:
        return layout.file_type(path)

    def _listed_unreadable_table_file(self, table_name: str) -> Optional[str]:
        try:
            root_entries = fsutil.listdir(path=self.root_dir,
                                          max_items=MAX_SUPPORTED_FILES)
        except Exception:
            return None

        expected_files = {f'{table_name}.{_PARQUET}', f'{table_name}.{_CSV}'}
        for entry in root_entries:
            if osp.basename(entry) in expected_files:
                return entry
        return None

    @with_filesystem_credentials
    def _resolve_table(self, table_name: str) -> tuple[str, str]:
        """Return the physical path and file format."""
        data_path = self._table_path(table_name)
        if fsutil.isdir(data_path):
            return (f'{data_path.rstrip("/")}/',
                    self._get_path_type(data_path, table_name))
        if fsutil.exists(data_path):
            file_type = self._supported_file_type(data_path)
            if file_type is not None:
                return data_path, file_type
            raise ConnectorError(
                message=f"Table {table_name} does not have a supported "
                f"file type (.csv or .parquet).")

        for file_type in (_PARQUET, _CSV):
            path = f'{data_path}.{file_type}'
            if fsutil.exists(path):
                return path, file_type

        unreadable_table_file = self._listed_unreadable_table_file(table_name)
        if unreadable_table_file:
            raise ConnectorError(
                message=f"Table {table_name} was found at "
                f"{unreadable_table_file}, but Kumo cannot read it. "
                "Verify that the object grants read access to this "
                "environment. For S3, the object must allow s3:GetObject "
                "through configured AWS credentials or public-read access.")

        self._validate_missing_path_region(table_name, data_path)
        raise FileNotFoundError(
            f"Table {table_name} does not exist in path {data_path}. Please "
            f"ensure that your file exists or that the connector directory"
            " path is set up correctly." + _check_suffix_error(table_name))

    @blocking_method()
    def get_path_and_type(self, table_name: str) -> Tuple[str, str]:
        """Get s3 path and file type ("csv" or "parquet") for the table_name.

        Raises:
            FileNotFoundError: if no file/dir path could be found for the given
            table_name

            ValidationException: if a directory is found for the table_name but
            does not contain valid csv or parquet file.
        """
        return self._resolve_table(table_name)

    def _validate_missing_path_region(self, table_name: str,
                                      data_path: str) -> None:
        if not fsutil.is_s3(data_path):
            return
        try:
            bucket, _ = fsutil.s3_filesystem.parse_s3_dir(data_path)
            bucket_region = fsutil.s3_filesystem.get_bucket_aws_region(bucket)
        except ValueError:
            return
        if bucket_region != AWS_REGION:
            raise ConnectorError(
                f"Kumo encountered an issue while reading the table '{table_name}'."
                f"The table '{table_name}' is in region {bucket_region}, "
                f"while your current deployment is in region '{AWS_REGION}'.")

    # CSV & Parquet ##########################################################

    @blocking_method(DATA_BACKEND_THREAD_POOL)
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
        return self.connectors[self._get_path_type(path)]._load(
            path, pkey, cols, num_rows, offset, frac, sample, **kwargs)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:
        _, typ = self._resolve_table(table_name)
        return self.connectors[typ]._get_table_num_rows(table_name)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        _, typ = self._resolve_table(table_name)
        return self.connectors[typ]._get_table_source_size(table_name)

    def _get_table_cols_metadata(
        self,
        table_name: str,
        skip_cache: bool = False,
        skip_inferral: bool = False,
    ) -> TableMetadataConfig:
        _, typ = self._resolve_table(table_name)
        return self.connectors[typ]._get_table_cols_metadata(
            table_name, skip_cache, skip_inferral)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _write(self, path: str, df: pd.DataFrame):
        self.connectors[self._get_path_type(path)]._write(path, df)

    def _create_iterator(
        self,
        path: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> Iterator[pd.DataFrame]:
        return self.connectors[self._get_path_type(path)]._create_iterator(
            path, pkey, cols)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    @with_filesystem_credentials
    def load_file(
        self,
        path: str,
        cols: Optional[List[str]] = None,
        frac: Optional[float] = None,
        num_rows: Optional[int] = None,
        sample: Optional[bool] = False,
        col_types: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        return self.connectors[self._get_path_type(path)]._load_file(
            path, cols, frac, num_rows, sample, col_types)

    def _source_format(self, path: str):
        return self._get_path_type(path)

    def _csv_read_options(self, path: str):
        return self.connectors[self._get_path_type(path)]._csv_read_options(
            path)


# Explicit dataclass definition ###############################################

CompositeFileConnectorConfig = to_dataclass(CompositeFileConnector,
                                            base_cls=ConnectorConfig)
register(CompositeFileConnector, CompositeFileConnectorConfig,
         group='connector')
