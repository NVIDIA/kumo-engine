# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ADLS range-read filesystem: parquet metadata must not read whole files."""
import io
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from azure.core.exceptions import ResourceNotFoundError
from kumo_core.sources.arrow_parquet import read_bounded_dataset
from pyarrow.fs import FSSpecHandler, PyFileSystem

from io_utils import fsutil
from io_utils.fsutil.adls_arrow import ADLSArrowFileSystem
from io_utils.fsutil.credentials import pyarrow_filesystem
from io_utils.fsutil.uri import ParseURI
from kumo.testing.adls import adls_test_base_uri, withADLSCredentials

from .common import random_name


class FakeFileSystemClient:
    """Container-scoped fake mirroring the DataLake SDK surface we use."""
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.fetched_ranges: list[tuple[int, int]] = []

    def create_file(self, path: str, data: bytes):
        self.files[path] = data
        parent = path.rsplit('/', 1)[0] if '/' in path else ''
        while parent:
            self.directories.add(parent)
            parent = parent.rsplit('/', 1)[0] if '/' in parent else ''

    def get_file_client(self, path: str):
        return SimpleNamespace(
            get_file_properties=lambda: self._properties(path),
            download_file=lambda offset, length: self._download(
                path, offset, length),
        )

    def get_paths(self, path=None, recursive=False):
        prefix = f'{path}/' if path else ''
        children = set()
        for name in list(self.files) + list(self.directories):
            if name.startswith(prefix) and name != path:
                children.add(prefix + name[len(prefix):].split('/')[0])
        if path and not children and path not in self.files:
            raise ResourceNotFoundError(f'not found: {path}')
        return [
            SimpleNamespace(
                name=child,
                is_directory=child in self.directories,
                content_length=len(self.files.get(child, b'')),
            ) for child in sorted(children)
        ]

    def _properties(self, path: str):
        if path in self.files:
            return SimpleNamespace(size=len(self.files[path]), metadata={})
        if path in self.directories:
            return SimpleNamespace(size=0, metadata={'hdi_isfolder': 'true'})
        raise ResourceNotFoundError(f'not found: {path}')

    def _download(self, path: str, offset: int, length: int):
        self.fetched_ranges.append((offset, offset + length))
        data = self.files[path][offset:offset + length]
        return SimpleNamespace(readall=lambda: data)


@pytest.fixture
def client() -> FakeFileSystemClient:
    return FakeFileSystemClient()


def _write_parquet(client: FakeFileSystemClient, path: str,
                   num_rows: int = 200_000,
                   wide_payload: bool = False) -> pa.Table:
    columns = {
        'id': pa.array(range(num_rows), type=pa.int64()),
        'value': pa.array((float(i) for i in range(num_rows)),
                          type=pa.float64()),
    }
    if wide_payload:
        columns['payload'] = pa.array([b'x' * (16 << 10)] * num_rows)
    table = pa.table(columns)
    buffer = io.BytesIO()
    # Uncompressed so the data section dwarfs the footer and byte-range
    # assertions stay meaningful.
    pq.write_table(table, buffer, compression='none',
                   use_dictionary=not wide_payload)
    client.create_file(path, buffer.getvalue())
    return table


def _pyarrow_fs(client: FakeFileSystemClient) -> PyFileSystem:
    return PyFileSystem(FSSpecHandler(ADLSArrowFileSystem(client)))


def test_schema_read_fetches_only_footer_ranges(client):
    _write_parquet(client, 'tables/events/part-0.parquet')
    file_size = len(client.files['tables/events/part-0.parquet'])

    # Mirror the kwargs ParquetConnector._parquet_dataset passes in
    # production so buffering options cannot silently regress to full reads.
    dataset = pq.ParquetDataset('tables/events/part-0.parquet',
                                partitioning=None, pre_buffer=False,
                                buffer_size=1024 * 1024 * 64,
                                filesystem=_pyarrow_fs(client))

    assert dataset.schema.names == ['id', 'value']
    assert client.fetched_ranges, 'expected range reads'
    fetched_bytes = sum(end - start for start, end in client.fetched_ranges)
    assert fetched_bytes < file_size * 0.1
    assert min(start for start, _ in client.fetched_ranges) > file_size * 0.5


def test_bounded_preview_reads_projected_ranges_only(client):
    _write_parquet(client, 'tables/events/part-0.parquet', num_rows=2048,
                   wide_payload=True)
    metadata = pq.ParquetFile(
        io.BytesIO(client.files['tables/events/part-0.parquet'])).metadata
    payload_size = metadata.row_group(0).column(2).total_compressed_size

    dataset = pq.ParquetDataset('tables/events/part-0.parquet',
                                pre_buffer=False,
                                filesystem=_pyarrow_fs(client))
    client.fetched_ranges.clear()
    head = read_bounded_dataset(dataset, ['payload'], 5)

    assert head.num_rows == 5
    assert head.column_names == ['payload']
    assert sum(end - start for start, end in
               client.fetched_ranges) < payload_size, client.fetched_ranges


def test_directory_dataset_discovers_files_via_listing(client):
    _write_parquet(client, 'tables/events/part-0.parquet', num_rows=10)
    _write_parquet(client, 'tables/events/part-1.parquet', num_rows=10)

    dataset = pq.ParquetDataset('tables/events', filesystem=_pyarrow_fs(client))

    assert dataset.schema.names == ['id', 'value']
    assert len(dataset.fragments) == 2


def test_missing_path_raises_file_not_found(client):
    with pytest.raises(FileNotFoundError):
        ADLSArrowFileSystem(client).info('tables/absent.parquet')


def test_write_mode_is_rejected(client):
    client.create_file('tables/file.parquet', b'PAR1')
    with pytest.raises(NotImplementedError):
        ADLSArrowFileSystem(client).open('tables/file.parquet', 'wb')


def test_from_uri_rejects_non_adls():
    with pytest.raises(ValueError, match='Not an ADLS URI'):
        ADLSArrowFileSystem.from_uri('s3://bucket/key.parquet')


@withADLSCredentials
def test_real_adls_parquet_dataset_range_reads():
    """Live ABFS: schema and bounded preview through the range-read filesystem."""
    table = pa.table({
        'id':
        pa.array(range(1000), type=pa.int64()),
        'name':
        pa.array((f'row-{i}' for i in range(1000)), type=pa.string()),
    })
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    test_prefix = f'{adls_test_base_uri()}/{random_name()}'
    uri = f'{test_prefix}/events.parquet'
    fsutil.write_binary(uri, buffer.getvalue())
    try:
        dataset = pq.ParquetDataset(
            ParseURI(uri).path, pre_buffer=False,
            filesystem=pyarrow_filesystem(uri))
        assert dataset.schema.names == ['id', 'name']
        head = read_bounded_dataset(dataset, None, 3)
        assert head.column('id').to_pylist() == [0, 1, 2]
        assert head.column('name').to_pylist() == ['row-0', 'row-1', 'row-2']
    finally:
        fsutil.rm(test_prefix, ignore_errors=True)
