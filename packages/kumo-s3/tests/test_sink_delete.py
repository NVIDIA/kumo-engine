# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
from kumo_s3.sink import S3SinkBackend
from moto import mock_s3


@mock_s3
def test_deleting_the_last_object_under_a_prefix_leaves_no_marker(
        monkeypatch):
    client = boto3.client('s3', region_name='us-east-1')
    client.create_bucket(Bucket='bucket')
    client.put_object(Bucket='bucket', Key='root/.probe', Body=b'probe')
    sink = S3SinkBackend()
    monkeypatch.setattr(
        sink, '_resolved_session', lambda connector, bucket:
        (boto3.Session(region_name='us-east-1'), True, None))

    sink.delete_file(None, 'bucket/root/.probe')

    assert client.list_objects_v2(Bucket='bucket')['KeyCount'] == 0
