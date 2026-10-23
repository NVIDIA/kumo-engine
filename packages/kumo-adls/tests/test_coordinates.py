# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from kumo_adls.backend import ADLSSourceBackend
from kumo_adls.sink import ADLSSinkBackend
from kumoapi.source_table import ADLSRelativeSource, ADLSURISource

URI = 'abfss://curated@companylake.dfs.core.windows.net/orders/'


@pytest.mark.parametrize('prefix', ['', 'orders', 'orders/team'])
@pytest.mark.parametrize('filename', [
    'scores%23review.parquet',
    'scores%3Freview.parquet',
    'scores%25review.parquet',
    'scores%3Areview.parquet',
])
def test_full_uri_output_obeys_rooted_relative_policy_before_storage(
        connector, monkeypatch, prefix, filename):
    from unittest.mock import Mock

    target = connector(prefix=prefix)
    root = URI.removesuffix('orders/') + (prefix + '/' if prefix else '')
    uri = root + filename
    access = Mock(side_effect=AssertionError('must not access storage'))
    native = Mock(side_effect=AssertionError('must not bind native output'))
    monkeypatch.setattr('kumo_adls.backend.source_access', access)
    monkeypatch.setattr('kumo_adls.sink.filesystem_for', native)
    source = ADLSSourceBackend()
    sink = ADLSSinkBackend()

    with pytest.raises(ValueError):
        source.get_table_config(target, uri)
    with pytest.raises(ValueError):
        source.get_columns(target, uri)
    with pytest.raises(ValueError):
        sink.preflight_write_files(target, uri)
    with pytest.raises(ValueError):
        sink.write_table(target, uri, pd.DataFrame({'id': [1]}))

    access.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize('prefix', ['', 'orders/team'])
@pytest.mark.parametrize('filename',
                         ['scores review.parquet', 'résultats ü.parquet'])
def test_valid_full_uri_export_roundtrips_through_source_schema(
        service, connector, prefix, filename):
    from urllib.parse import quote

    target = connector(prefix=prefix)
    root = URI.removesuffix('orders/') + (prefix + '/' if prefix else '')
    uri = root + quote(filename)
    returned = ADLSSinkBackend().write_table(target, uri,
                                             pd.DataFrame({'id': [3, 4]}))
    assert returned == uri
    source = ADLSSourceBackend()
    config = source.get_table_config(target, returned).source_table
    assert isinstance(config.source, ADLSRelativeSource)
    assert config.source.path == filename
    assert config.source.path == source.get_table_config(
        target, filename).source_table.source.path
    assert '3' in source.get_sample(target, returned, 2).sample_rows
    assert '4' in source.get_sample(target, filename, 2).sample_rows


@pytest.mark.parametrize('coordinate', ['', 'orders', 'orders/'])
def test_empty_root_relative_output_fails_before_storage(
        connector, monkeypatch, coordinate):
    from unittest.mock import Mock

    target = connector()
    uri = URI.removesuffix('orders/') + coordinate if coordinate else ''
    access = Mock(side_effect=AssertionError('must not access storage'))
    monkeypatch.setattr('kumo_adls.backend.source_access', access)
    with pytest.raises(ValueError):
        ADLSSourceBackend().get_table_config(target, uri)
    with pytest.raises(ValueError):
        ADLSSinkBackend().write_table(target, uri, pd.DataFrame({'id': [1]}))
    access.assert_not_called()


def test_rootless_full_uri_keeps_exact_source_policy(service, connector):
    uri = URI + 'scores%23review.parquet'
    service.files['curated/orders/scores#review.parquet'] = service.files[
        'curated/orders/events.parquet']
    source = ADLSSourceBackend()
    target = connector(any_uri=True)
    config = source.get_table_config(target, uri).source_table
    assert isinstance(config.source, ADLSURISource)
    assert config.source.uri == uri
    assert '10.0' in source.get_sample(target, uri, 1).sample_rows
