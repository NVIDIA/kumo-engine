# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import kumo_s3
import pytest
from kumo_core.sources.errors import (
    ReconnectRequiredError,
    TransientSourceError,
)
from kumo_s3.access import source_access
from pyarrow import fs
from pyarrow.fs import AwsDefaultS3RetryStrategy


class Rotating:
    """Session credentials whose frozen value can be rotated."""
    def __init__(self, token='first'):
        self.token = token

    def get_frozen_credentials(self):
        return SimpleNamespace(access_key='key', secret_key='secret',
                               token=self.token)


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """``bucket/`` objects served from a local directory, recording which
    credential each filesystem was built with."""
    (tmp_path / 'bucket').mkdir()
    built = []

    def filesystem(name, *, session, **kwargs):
        built.append(session.get_credentials().get_frozen_credentials().token)
        return fs.SubTreeFileSystem(str(tmp_path), fs.LocalFileSystem())

    monkeypatch.setattr(kumo_s3, 'filesystem', filesystem)

    def identity(source, path):
        import hashlib
        with source.open_input_file(path) as stream:
            contents = stream.read()
        return (hashlib.sha256(contents).hexdigest(), '', str(len(contents)))

    monkeypatch.setattr('kumo_s3.access._object_identity', identity)
    return SimpleNamespace(root=tmp_path / 'bucket', built=built,
                           out=tmp_path / 'out')


def session_with(credentials):
    return SimpleNamespace(get_credentials=lambda: credentials,
                           region_name='us-west-2')


def test_an_object_is_copied_whole_through_the_filesystem(bucket):
    (bucket.root / 'key').write_bytes(b'data' * 1000)
    bucket.out.mkdir()
    access = source_access('bucket', session=session_with(Rotating()))
    access.fetch('bucket/key', str(bucket.out / 'copy'))
    assert (bucket.out / 'copy').read_bytes() == b'data' * 1000
    assert [path.name for path in bucket.out.iterdir()] == ['copy']


def test_a_failed_copy_preserves_the_target(bucket):
    bucket.out.mkdir()
    target = bucket.out / 'copy'
    target.write_bytes(b'previous')
    access = source_access('bucket', session=session_with(Rotating()))
    with pytest.raises(OSError):
        access.fetch('bucket/missing', str(target))
    assert target.read_bytes() == b'previous'
    assert list(bucket.out.iterdir()) == [target]


def test_an_object_rewritten_during_the_copy_is_retried(bucket, monkeypatch):
    source = bucket.root / 'key'
    source.write_bytes(b'old')
    bucket.out.mkdir()
    copy = fs.copy_files

    def rewritten(*args, **kwargs):
        copy(*args, **kwargs)
        source.write_bytes(b'newer')

    monkeypatch.setattr(fs, 'copy_files', rewritten)
    access = source_access('bucket', session=session_with(Rotating()))
    with pytest.raises(TransientSourceError, match='replaced'):
        access.fetch('bucket/key', str(bucket.out / 'copy'))
    assert list(bucket.out.iterdir()) == []


def test_rotated_credentials_re_sign_the_next_copy(bucket):
    (bucket.root / 'key').write_bytes(b'data')
    bucket.out.mkdir()
    credentials = Rotating()
    access = source_access('bucket', session=session_with(credentials))
    access.fetch('bucket/key', str(bucket.out / 'a'))
    credentials.token = 'second'
    access.fetch('bucket/key', str(bucket.out / 'b'))
    access.fetch('bucket/key', str(bucket.out / 'c'))
    assert bucket.built == ['first', 'second']


def test_an_object_outside_the_bucket_is_refused(bucket):
    access = source_access('bucket', session=session_with(Rotating()))
    with pytest.raises(ValueError, match='outside the selected bucket'):
        access.fetch('other/key', str(bucket.out))


def test_no_credentials_without_anonymous_access_is_refused(monkeypatch):
    monkeypatch.setattr(kumo_s3, 'filesystem', lambda *args, **kwargs: None)
    with pytest.raises(ReconnectRequiredError):
        source_access('bucket', session=session_with(None))


@pytest.mark.parametrize('anonymous, options', [
    (False, {}),
    (True, {}),
    (False, {
        'tls_ca_file_path': '/trusted/ca.pem'
    }),
    (True, {
        'endpoint_override': 'https://override.private'
    }),
])
def test_an_explicit_endpoint_reaches_the_filesystem(monkeypatch, anonymous,
                                                     options):
    built = {}

    def filesystem(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(region=kwargs['region'])

    monkeypatch.setattr('kumo_s3.session.fs.S3FileSystem', filesystem)
    monkeypatch.setattr(kumo_s3, 'bucket_region',
                        Mock(side_effect=AssertionError('AWS region lookup')))
    session = session_with(None if anonymous else Rotating())
    source_access('bucket', session=session,
                  endpoint_url='https://default.private', options=options,
                  allow_anonymous=anonymous)
    assert built['endpoint_override'] == options.get('endpoint_override',
                                                     'https://default.private')
    assert built['force_virtual_addressing'] is True
    assert isinstance(built['retry_strategy'], AwsDefaultS3RetryStrategy)
    assert built.get('anonymous', False) is anonymous
    if 'tls_ca_file_path' in options:
        assert built['tls_ca_file_path'] == options['tls_ca_file_path']


def test_connection_options_keep_the_authorized_credentials(monkeypatch):
    built = {}

    def filesystem(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(region=kwargs['region'])

    monkeypatch.setattr('kumo_s3.session.fs.S3FileSystem', filesystem)
    source_access(
        'bucket', session=session_with(Rotating(token='token')),
        endpoint_url='https://s3.private', options={
            'access_key': 'unrelated-key',
            'secret_key': 'unrelated-secret',
            'session_token': 'unrelated-token',
            'region': 'unrelated-region',
            'anonymous': True,
            'endpoint_override': 's3.private',
            'scheme': 'https',
            'tls_ca_file_path': '/trusted/ca.pem',
            'force_virtual_addressing': False,
        })
    retry = built.pop('retry_strategy')
    assert isinstance(retry, AwsDefaultS3RetryStrategy)
    assert built == {
        'access_key': 'key',
        'secret_key': 'secret',
        'session_token': 'token',
        'region': 'us-west-2',
        'endpoint_override': 's3.private',
        'scheme': 'https',
        'tls_ca_file_path': '/trusted/ca.pem',
        'force_virtual_addressing': False,
    }


def test_rotated_credentials_re_sign_a_projected_read_midway(
        bucket, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.sources import object_store
    from kumo_core.sources.object_store import ObjectStoreSourceReader
    from kumo_core.sources.types import ReadMode
    from kumoapi.typing import Dtype

    (bucket.root / 'events').mkdir()
    for part in range(2):
        pq.write_table(pa.table({
            'id': [part],
            'unread': ['x']
        }), bucket.root / 'events' / f'part-{part}.parquet')
    credentials = Rotating()
    access = source_access('bucket', session=session_with(credentials))
    read = object_store._parquet_batches
    parts = []

    def rotating(fragment, schema):
        parts.append(fragment.path)
        if len(parts) == 2:
            credentials.token = 'second'
        yield from read(fragment, schema)

    monkeypatch.setattr(object_store, '_parquet_batches', rotating)
    staged = ObjectStoreSourceReader(
        access.filesystem, root='bucket', fetch=access.fetch,
        object_identity=access.object_identity).prepare(
            'events', str(bucket.out), columns={'id': Dtype.int64},
            mode=ReadMode.STREAMED, worker_ceiling=1)
    assert staged.num_rows == 2
    assert bucket.built == ['first', 'second']


def test_a_failed_copy_rechecks_identity_and_preserves_target(
        bucket, monkeypatch):
    source = bucket.root / 'key'
    source.write_bytes(b'old')
    bucket.out.mkdir()
    target = bucket.out / 'copy'
    target.write_bytes(b'previous')

    def failed(*args, **kwargs):
        source.write_bytes(b'new')
        raise OSError('failed range')

    monkeypatch.setattr(fs, 'copy_files', failed)
    access = source_access('bucket', session=session_with(Rotating()))
    with pytest.raises(TransientSourceError, match='replaced'):
        access.fetch('bucket/key', str(target))
    assert target.read_bytes() == b'previous'
    assert list(bucket.out.iterdir()) == [target]
