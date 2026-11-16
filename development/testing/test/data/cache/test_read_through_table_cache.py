# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from io_utils import fsutil
from kumo.data.cache import ReadThroughTableSampleCache


def test_missing_table_sample_does_not_try_parquet_read(monkeypatch):
    cache = ReadThroughTableSampleCache('gs://bucket/cache', 'table_samples')
    monkeypatch.setattr(fsutil, 'exists', lambda _: False)

    def fail_read(_):
        raise AssertionError('missing table sample must not be read')

    monkeypatch.setattr(pd, 'read_parquet', fail_read)

    assert cache._load('connector', 'source', 'version') is None


def test_existing_table_sample_still_loads(monkeypatch):
    cache = ReadThroughTableSampleCache('gs://bucket/cache', 'table_samples')
    expected = pd.DataFrame({'value': [1]})
    monkeypatch.setattr(fsutil, 'exists', lambda _: True)
    monkeypatch.setattr(pd, 'read_parquet', lambda _: expected)

    assert cache._load('connector', 'source', 'version') is expected
