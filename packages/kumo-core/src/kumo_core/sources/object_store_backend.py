# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, Tuple

from kumoapi.plugin import PrepareIngestResult, SourceConnector
from kumoapi.source_table import (
    FileType,
    SourceTableConfigResponse,
    SourceTableDataResponse,
    SourceTableListResponse,
)
from kumoapi.typing import Dtype

from .backend import ingest_result, select_columns, source_columns
from .errors import PermanentSourceError
from .layout import staging_dir
from .object_store import ObjectStoreSourceReader, require_no_sql
from .object_store_sessions import ObjectStoreRoot
from .types import ReadMode


class ObjectStoreSourceBackend(ABC):
    scheme: str
    mode: Optional[ReadMode]
    relative_source: type[Any]
    uri_source: type[Any]
    source_table: type[Any]

    @abstractmethod
    def _reader(self, connector: SourceConnector,
                table_name: str) -> Tuple[ObjectStoreSourceReader, str]:
        raise NotImplementedError

    @abstractmethod
    def _discovery(self, connector: SourceConnector) -> ObjectStoreSourceReader:
        raise NotImplementedError

    @abstractmethod
    def _root_for(self,
                  connector: SourceConnector) -> Optional[ObjectStoreRoot]:
        raise NotImplementedError

    def get_table_config(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> SourceTableConfigResponse:
        del project_id
        reader, name = self._reader(connector, table_name)
        _, file_type = reader.resolve(name)
        source = (self.relative_source(path=name) if self._root_for(connector)
                  is not None else self.uri_source(uri=table_name))
        return SourceTableConfigResponse(source_table=self.source_table(
            connector_id=connector.id,
            file_type=FileType(file_type.upper()),
            source=source,
        ))

    def list_tables(
        self,
        connector: SourceConnector,
        project_id: Optional[str] = None,
    ) -> SourceTableListResponse:
        del project_id
        return SourceTableListResponse(
            table_names=self._discovery(connector).tables())

    def get_columns(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> SourceTableDataResponse:
        require_no_sql(sql)
        del project_id
        reader, name = self._reader(connector, table_name)
        columns = source_columns(reader.dtypes(name).items())
        return SourceTableDataResponse(table_name=table_name, cols=columns)

    def get_sample(
        self,
        connector: SourceConnector,
        table_name: str,
        sample_rows: int,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SourceTableDataResponse:
        """Return a bounded, nonrandom sample."""
        require_no_sql(sql)
        del project_id, timeout
        reader, name = self._reader(connector, table_name)
        head = reader.head(name, int(sample_rows))
        columns = reader.dtypes(name)
        return SourceTableDataResponse(
            table_name=table_name,
            cols=source_columns(columns.items()),
            sample_rows=head.to_pandas().to_json(orient='table'),
        )

    def get_download_urls(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        del project_id
        reader, name = self._reader(connector, table_name)
        return [f'{self.scheme}://{path}' for path in reader.object_paths(name)]

    def get_data_version(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> str:
        del project_id
        reader, name = self._reader(connector, table_name)
        return str(reader.data_version(name))

    def prepare_ingest(
        self,
        connector: SourceConnector,
        table_name: str,
        dest_dir: str,
        dtypes: Optional[Mapping[str, Dtype]] = None,
        project_id: Optional[str] = None,
        data_version: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> PrepareIngestResult:
        r"""Every local object-store read copies rather than pointing ingest at
        the bucket: the hosted no-copy manifest needs a per-object ``e_tag``,
        and ``pyarrow.fs.FileInfo`` does not expose one.
        """
        require_no_sql(sql)
        del project_id
        from kumo_core.sources.prepare import read_table

        reader, name = self._reader(connector, table_name)
        # One directory per table: a read empties what it is handed.
        destination = staging_dir(dest_dir, name)
        columns = select_columns(name, reader.dtypes(name), dtypes)
        reader.validate_ingest_columns(name, columns)
        staged = read_table(reader, name, str(destination), columns=columns,
                            mode=self.mode, version=data_version)
        if not staged.num_rows:
            raise PermanentSourceError(
                f'Reading {table_name} produced no rows. An empty source '
                f'table cannot be ingested; DiskGraph needs at least one row '
                f'to infer an encoding.')

        return ingest_result(staged)
