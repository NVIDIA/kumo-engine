# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from urllib.parse import urlsplit

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.stub import ANY, Stubber

from kumo.pytorch import parquet_sources as sources


@pytest.mark.parametrize(
    "endpoint,virtual,bucket_path",
    [
        ("https://fixture.s3.example.net/prefix", "true", "/prefix"),
        ("https://bucket-alias.example.net/prefix", "true", "/prefix"),
        ("https://s3.example.net/prefix", "false", "/prefix/fixture"),
    ],
)
def test_client_uses_same_bucket_endpoint_as_diskgraph(monkeypatch, endpoint, virtual, bucket_path):
    monkeypatch.setenv("S3_ENDPOINT", endpoint)
    monkeypatch.setenv("S3_VIRTUAL_HOSTED", virtual)
    monkeypatch.setenv("S3_REGION", "us-east-1")
    monkeypatch.setenv("S3_ACCESS_KEY", "test")
    monkeypatch.setenv("S3_SECRET_KEY", "test")
    client = sources._client()
    for operation, params, suffix in [
        ("list_objects_v2", {}, ""),
        ("get_object", {"Key": "a b.parquet"}, "/a%20b.parquet"),
    ]:
        url = urlsplit(
            client.generate_presigned_url(operation, Params={"Bucket": "fixture", **params})
        )
        assert url.netloc == urlsplit(endpoint).netloc
        assert url.path.rstrip("/") == bucket_path + suffix


@pytest.fixture
def s3(monkeypatch):
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    monkeypatch.setattr(sources, "_client", lambda: client)
    with Stubber(client) as stub:
        yield stub
        stub.assert_no_pending_responses()
    client.close()


def test_paginated_prefix_filters_and_orders_parquet_parts(s3):
    prefix = "table/"
    params = {"Bucket": "fixture", "Prefix": prefix}
    s3.add_client_error(
        "head_object",
        "NoSuchKey",
        http_status_code=404,
        expected_params={"Bucket": "fixture", "Key": "table"},
    )
    s3.add_response(
        "list_objects_v2",
        {
            "IsTruncated": True,
            "NextContinuationToken": "next",
            "Contents": [
                {"Key": prefix + "z.parquet", "Size": 3, "ETag": '"z"'},
                {"Key": prefix + "_SUCCESS", "Size": 0},
            ],
        },
        params,
    )
    s3.add_response(
        "list_objects_v2",
        {
            "IsTruncated": False,
            "Contents": [{"Key": prefix + "date=1/a.parquet", "Size": 2, "ETag": '"a"'}],
        },
        {**params, "ContinuationToken": "next"},
    )
    assert sources.parquet_objects("s3://fixture/table") == [
        {"path": "s3://fixture/table/date=1/a.parquet", "size": 2, "e_tag": '"a"'},
        {"path": "s3://fixture/table/z.parquet", "size": 3, "e_tag": '"z"'},
    ]


def test_replaced_exact_object_fails_conditional_footer_read(s3):
    params = {"Bucket": "fixture", "Key": "rows.parquet"}
    s3.add_response(
        "head_object", {"ContentLength": 200_000, "ETag": '"old"', "VersionId": "v1"}, params
    )
    (obj,) = sources.parquet_objects("s3://fixture/rows.parquet")
    s3.add_client_error(
        "get_object",
        "PreconditionFailed",
        http_status_code=412,
        expected_params={
            **params,
            "IfMatch": '"old"',
            "VersionId": "v1",
            "Range": ANY,
        },
    )
    with pytest.raises(OSError, match="PreconditionFailed"):
        sources.open_parquet_file(obj)


def test_listing_without_etag_fetches_immutable_version(s3):
    key = "table/part.parquet"
    s3.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": key, "Size": 20}]},
        {"Bucket": "fixture", "Prefix": "table/"},
    )
    s3.add_response(
        "head_object",
        {"ContentLength": 20, "VersionId": "v1"},
        {"Bucket": "fixture", "Key": key},
    )
    assert sources.parquet_objects("s3://fixture/table/") == [
        {"path": f"s3://fixture/{key}", "size": 20, "version": "v1"}
    ]


def test_footer_and_projected_reads_use_bounded_pinned_ranges(monkeypatch):
    ids = list(range(8192))
    buf = BytesIO()
    pq.write_table(
        pa.table({"id": ids, "unused": [b"x" * 128] * len(ids)}),
        buf,
        compression=None,
        use_dictionary=False,
        row_group_size=2048,
    )
    data = buf.getvalue()
    ranges = []

    class Client:
        def get_object(self, *, Bucket, Key, IfMatch, Range):
            assert (Bucket, Key, IfMatch) == ("fixture", "part.parquet", '"abc"')
            start, end = (int(offset) for offset in Range.removeprefix("bytes=").split("-"))
            body = BytesIO(data[start : end + 1])
            ranges.append((start, end, body))
            return {"ContentLength": end - start + 1, "Body": body}

    monkeypatch.setattr(sources, "_client", Client)
    part = {"path": "s3://fixture/part.parquet", "size": len(data), "e_tag": '"abc"'}
    with sources.open_parquet_file(part) as reader:
        assert reader.schema_arrow.names == ["id", "unused"]
        assert all(start > len(data) // 2 for start, _, _ in ranges)
        assert (
            pa.Table.from_batches(reader.iter_batches(columns=["id"])).column("id").to_pylist()
            == ids
        )
    assert sum(end - start + 1 for start, end, _ in ranges) < len(data) // 4
    assert all(body.closed for _, _, body in ranges)
