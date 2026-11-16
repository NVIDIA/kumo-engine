# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live ADLS range-read filesystem coverage."""
import io

import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.sources.arrow_parquet import read_bounded_dataset

from io_utils import fsutil
from io_utils.fsutil.credentials import pyarrow_filesystem
from io_utils.fsutil.uri import ParseURI
from kumo.testing.adls import adls_test_base_uri, withADLSCredentials

from .common import random_name


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
    try:
        fsutil.write_binary(uri, buffer.getvalue())
        dataset = pq.ParquetDataset(
            ParseURI(uri).path, pre_buffer=False,
            filesystem=pyarrow_filesystem(uri))
        assert dataset.schema.names == ['id', 'name']
        head = read_bounded_dataset(dataset, None, 3)
        assert head.column('id').to_pylist() == [0, 1, 2]
        assert head.column('name').to_pylist() == ['row-0', 'row-1', 'row-2']
    finally:
        fsutil.rm(test_prefix, ignore_errors=True)
