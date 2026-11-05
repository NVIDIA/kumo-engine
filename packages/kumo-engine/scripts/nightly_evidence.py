# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish immutable reference proposals before fitting, with verified readbacks."""

import hashlib
import io
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

SINGLE_PUT_BYTES = 5 * 1024**3
CHUNK_BYTES = 64 * 1024**2


def _put(client, bucket, key, stream, size):
    from botocore.exceptions import ClientError

    upload = None
    try:
        if size <= SINGLE_PUT_BYTES:
            client.put_object(Bucket=bucket, Key=key, Body=stream,
                              ContentLength=size, IfNoneMatch='*')
        else:
            upload = client.create_multipart_upload(Bucket=bucket,
                                                    Key=key)['UploadId']
            parts = []
            width = max(CHUNK_BYTES, math.ceil(size / 10000))
            while block := stream.read(width):
                number = len(parts) + 1
                item = client.upload_part(Bucket=bucket, Key=key,
                                          UploadId=upload, PartNumber=number,
                                          Body=block)
                parts.append(dict(PartNumber=number, ETag=item['ETag']))
            client.complete_multipart_upload(Bucket=bucket, Key=key,
                                             UploadId=upload,
                                             MultipartUpload=dict(Parts=parts),
                                             IfNoneMatch='*')
            upload = None
    except ClientError as error:
        if error.response['Error']['Code'] not in ('PreconditionFailed', '412'):
            raise
    finally:
        if upload is not None:
            client.abort_multipart_upload(Bucket=bucket, Key=key,
                                          UploadId=upload)


def _verify(client, bucket, key, checksum, size):
    response = client.get_object(Bucket=bucket, Key=key)
    body, digest, total = response['Body'], hashlib.sha256(), 0
    try:
        while block := body.read(CHUNK_BYTES):
            digest.update(block)
            total += len(block)
    finally:
        body.close()
    if total != size or digest.hexdigest() != checksum:
        raise ValueError('S3 reference proposal checksum differs')
    return response.get('VersionId')


def publish(directory, destination):
    """Return a proposal receipt only after every S3 object reads back intact."""
    parsed = urlsplit(destination)
    prefix = parsed.path.removeprefix('/').removesuffix('/')
    if (parsed.scheme != 's3' or
            not re.fullmatch('[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', parsed.netloc)
            or parsed.query or parsed.fragment or not prefix or '\\' in prefix
            or any(ord(c) <= 32 for c in prefix)
            or any(part in ('', '.', '..') for part in prefix.split('/'))):
        raise ValueError('Use a normalized S3 bucket and proposal prefix')
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Reference proposals need a regular data directory')
    paths = sorted(directory.rglob('*'))
    files = {}
    for path in paths:
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError('Reference proposals need regular data files')
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        if path.suffix not in ('.json', '.parquet') or any(
                ord(c) <= 32 for c in name):
            raise ValueError('Reference proposals contain data and JSON only')
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[name] = dict(sha256=checksum, bytes=path.stat().st_size,
                           key=f'{prefix}/objects/{checksum}/{name}')
    if not files:
        raise ValueError('Reference proposals need nonempty data files')
    manifest = json.dumps(
        dict(format='kumo-engine-reference-proposal-v1', status='PREPARED',
             files=files), sort_keys=True, allow_nan=False).encode()
    checksum = hashlib.sha256(manifest).hexdigest()
    key = f'{prefix}/proposals/{checksum}/manifest.json'
    import boto3

    client = boto3.client('s3')
    for name, item in files.items():
        with (directory / name).open('rb') as stream:
            _put(client, parsed.netloc, item['key'], stream, item['bytes'])
        _verify(client, parsed.netloc, item['key'], item['sha256'],
                item['bytes'])
    _put(client, parsed.netloc, key, io.BytesIO(manifest), len(manifest))
    version = _verify(client, parsed.netloc, key, checksum, len(manifest))
    return dict(status='PREPARED_IN_S3', uri=f's3://{parsed.netloc}/{key}',
                sha256=checksum, version_id=version)
