# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve local or S3 Parquet parts and read a pinned object through Arrow."""

from __future__ import annotations

import io
import os
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from kumo.pytorch.dataset import _local_parquet_files


def _client():
    # Clients and their connection pools can be reused across parts, but not
    # across a worker fork or a change in endpoint, credentials or profile.
    key_id = os.getenv("S3_ACCESS_KEY")
    secret = os.getenv("S3_SECRET_KEY")
    token = os.getenv("S3_SESSION_TOKEN")
    if not (key_id and secret):
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        token = os.getenv("AWS_SESSION_TOKEN")
    return _make_client(
        os.getpid(),
        os.getenv("S3_ENDPOINT"),
        os.getenv("S3_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        os.getenv("S3_PROFILE") or os.getenv("AWS_PROFILE"),
        os.getenv("S3_VIRTUAL_HOSTED"),
        os.getenv("S3_ALLOW_HTTP", os.getenv("AWS_ALLOW_HTTP", "false")),
        key_id,
        secret,
        token,
    )


@lru_cache(maxsize=1)
def _make_client(pid, endpoint, region, profile, virtual_hosted, allow_http, key_id, secret, token):
    import boto3
    from botocore.config import Config

    if endpoint and endpoint.startswith("http://") and allow_http.lower() not in {"true", "1"}:
        raise ValueError("an HTTP S3_ENDPOINT requires S3_ALLOW_HTTP=true")
    virtual = (
        virtual_hosted.lower() in {"true", "1"}
        if virtual_hosted is not None
        else endpoint is None or "amazonaws.com" in endpoint
    )
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client(
        "s3",
        region_name=region or session.region_name or "us-east-1",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        aws_session_token=token,
        # Native DiskGraph takes its endpoint exclusively from S3_ENDPOINT.
        config=Config(
            ignore_configured_endpoint_urls=True,
            s3={"addressing_style": "virtual" if virtual and not endpoint else "path"},
        ),
    )
    if endpoint and virtual:
        # Native DiskGraph treats a virtual-hosted endpoint as the bucket's
        # complete base URL. Remove the SDK's path-style bucket before signing.
        base_path = urlsplit(endpoint).path.rstrip("/")

        def use_bucket_endpoint(request, **kwargs):
            url = urlsplit(request.url)
            _, _, key = url.path.removeprefix(base_path + "/").partition("/")
            request.url = url._replace(path=f"{base_path}/{key}").geturl()

        client.meta.events.register("before-sign.s3", use_bucket_endpoint)
    return client


def _location(path):
    bucket, _, key = path.removeprefix("s3://").partition("/")
    if not bucket or any(c in bucket for c in "@:?#"):
        raise ValueError("an S3 Parquet URI requires a bucket name")
    return bucket, key


def _record(bucket, key, size, etag, version=None):
    record = {"path": f"s3://{bucket}/{key}", "size": size}
    if etag:
        record["e_tag"] = etag
    if version and version != "null":
        record["version"] = version
    if not ("e_tag" in record or "version" in record):
        raise ValueError(f"S3 Parquet object has no ETag or version: {record['path']!r}")
    return record


def _head(client, bucket, key):
    metadata = client.head_object(Bucket=bucket, Key=key)
    return _record(
        bucket, key, metadata["ContentLength"], metadata.get("ETag"), metadata.get("VersionId")
    )


def _s3_objects(path):
    from botocore.exceptions import BotoCoreError, ClientError

    bucket, key = _location(path)
    try:
        client = _client()
        if key and not key.endswith("/"):
            try:
                return [_head(client, bucket, key)]
            except ClientError as error:
                if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
                    raise
                key += "/"
        records = []
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key)
        for page in pages:
            for item in page.get("Contents", ()):
                if item["Key"].endswith(".parquet"):
                    record = (
                        _record(bucket, item["Key"], item["Size"], item["ETag"])
                        if item.get("ETag")
                        else _head(client, bucket, item["Key"])
                    )
                    records.append(record)
    except (BotoCoreError, ClientError) as error:
        raise OSError(f"cannot resolve S3 Parquet source {path!r}: {error}") from error
    if not records:
        raise FileNotFoundError(f"no S3 Parquet files match {path!r}")
    return sorted(records, key=lambda item: item["path"])


def parquet_objects(path: str) -> list[dict]:
    """Return a sorted snapshot, in DiskGraph's object manifest format for S3."""
    if path.startswith("s3://"):
        return _s3_objects(path)
    records = []
    for file in _local_parquet_files(path):
        stat = Path(file).stat()
        records.append({"path": file, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return records


class _S3RangeFile(io.RawIOBase):
    """Seekable input for Arrow. Every GET must match the listed object revision."""

    def __init__(self, obj):
        self._client = _client()
        bucket, key = _location(obj["path"])
        self._request = {"Bucket": bucket, "Key": key}
        if "e_tag" in obj:
            self._request["IfMatch"] = obj["e_tag"]
        if "version" in obj:
            self._request["VersionId"] = obj["version"]
        self._size = obj["size"]
        self._offset = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._offset

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_CUR:
            offset += self._offset
        elif whence == io.SEEK_END:
            offset += self._size
        elif whence != io.SEEK_SET:
            raise ValueError("invalid seek mode")
        if offset < 0:
            raise ValueError("negative seek position")
        self._offset = offset
        return offset

    def read(self, size=-1):
        from botocore.exceptions import BotoCoreError, ClientError

        remaining = self._size - self._offset
        size = remaining if size is None or size < 0 else min(size, remaining)
        if size <= 0:
            return b""
        byte_range = f"bytes={self._offset}-{self._offset + size - 1}"
        try:
            response = self._client.get_object(**self._request, Range=byte_range)
            with closing(response["Body"]) as body:
                if response["ContentLength"] != size:
                    raise OSError(f"S3 returned an unexpected length for {byte_range}")
                data = body.read()
        except (BotoCoreError, ClientError) as error:
            path = f"s3://{self._request['Bucket']}/{self._request['Key']}"
            raise OSError(f"cannot read pinned S3 Parquet object {path!r}: {error}") from error
        if len(data) != size:
            raise OSError(f"incomplete S3 range response for {byte_range}")
        self._offset += size
        return data


def open_parquet_file(obj: dict):
    """Open one resolved part; S3 footer and column reads use conditional byte ranges."""
    import pyarrow.parquet as pq

    if obj["path"].startswith("s3://"):
        return pq.ParquetFile(_S3RangeFile(obj), pre_buffer=True)
    return pq.ParquetFile(obj["path"])
