# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumoapi.s3 import normalize_s3_source_uri

from io_utils.fsutil.s3_filesystem import parse_s3_file
from io_utils.fsutil.uri import ParseURI
from kumo.connector.any_uri_s3_connector import AnyURIS3Connector


@pytest.fixture
def connector() -> AnyURIS3Connector:
    value = AnyURIS3Connector(name='rootless_s3')
    value.set_credentials({
        'provider': 's3',
        'aws_access_key_id': 'attested-access',
        'aws_secret_access_key': 'attested-secret',
        'aws_session_token': 'attested-token',
    })
    value.require_explicit_credentials()
    return value


def test_any_uri_parquet_uses_actual_path_and_explicit_pyarrow_filesystem(
        connector, monkeypatch, tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.sources.object_store_access import ObjectStoreAccess
    from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

    from io_utils.fsutil.credentials import credentials_for_provider

    path = tmp_path / 'company-data/events/a b.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2]}), path)
    uri = normalize_s3_source_uri('s3://company-data/events/a%20b.parquet')
    filesystem = SubTreeFileSystem(str(tmp_path), LocalFileSystem())

    def access(actual):
        assert actual == uri
        assert credentials_for_provider('s3') is not None
        return ObjectStoreAccess(filesystem)

    monkeypatch.setattr('kumo.connector.object_store_access.source_access',
                        access)
    parquet = connector.connectors['parquet']
    with parquet.filesystem_credential_scope():
        result = parquet._parquet_dataset(uri)
    assert result.to_table().to_pydict() == {'id': [1, 2]}


@pytest.mark.parametrize(
    ('raw_uri', 'provider_key'),
    [
        ('s3://company-data/a%20b.parquet', 'a b.parquet'),
        ('s3://company-data/literal%25.parquet', 'literal%.parquet'),
        ('s3://company-data/caf%C3%A9.parquet', 'café.parquet'),
        ('s3://company-data/hash%23query%3F.parquet', 'hash#query?.parquet'),
    ],
)
def test_canonical_wire_uri_decodes_exactly_once_for_boto_and_pyarrow(
        raw_uri, provider_key) -> None:
    canonical = normalize_s3_source_uri(raw_uri)
    parsed = ParseURI(canonical)

    assert parsed.path == provider_key
    assert parse_s3_file(canonical) == ('company-data', provider_key)
    assert parsed.as_s3() == canonical


def test_encoded_slash_is_rejected_before_provider_parsing() -> None:
    with pytest.raises(ValueError, match='encoded slash'):
        normalize_s3_source_uri('s3://company-data/events%2Ftable.parquet')
