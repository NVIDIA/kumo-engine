# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from fakes import FakeSession
from kumo_s3.sink import S3SinkBackend


def test_deleting_an_object_deletes_only_that_key(monkeypatch):
    calls = []
    session = FakeSession(region_name='us-east-1')
    session.client = lambda service, endpoint_url=None: SimpleNamespace(
        delete_object=lambda **kwargs: calls.append((service, endpoint_url,
                                                     kwargs)))
    sink = S3SinkBackend()
    monkeypatch.setattr(sink, '_resolved_session',
                        lambda connector, bucket: (session, True, None))

    sink.delete_file(None, 'bucket/root/.probe')

    assert calls == [('s3', None, {'Bucket': 'bucket', 'Key': 'root/.probe'})]
