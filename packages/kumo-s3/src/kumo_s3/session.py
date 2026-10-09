# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Mapping
from urllib.parse import urlparse

import boto3
from pyarrow import fs

logger = logging.getLogger(__name__)

#: The storage namespace of ordinary AWS S3. A coordinate records this rather
#: than the principal that read it, so rotating a credential does not
#: re-identify the data.
AWS_NAMESPACE = "aws-s3"

S3_MAX_RETRIES = 10


def ambient_session() -> boto3.Session:
    """The AWS credential chain this host resolves to.

    ``boto3.Session`` raises ``ProfileNotFound`` for an ``AWS_PROFILE`` no
    config file defines, where pyarrow's own chain ignores it and signs as
    whatever ambient role it finds instead.
    """
    return boto3.Session()


# A bucket's region is immutable, so it is cached for the life of the process,
# decoupled from any credential lifetime. Exceptions propagate rather than
# being memoized, so one unreachable moment cannot pin a bucket to "no region".
@lru_cache(maxsize=64)
def bucket_region(bucket: str) -> str:
    """Resolve a bucket's region through pyarrow's unsigned resolver.

    Unlike a boto3 ``head_bucket`` it does not 403 for a cross-account bucket
    that grants ``s3:GetObject`` but not ``HeadBucket``, so the region is never
    pinned wrong.
    """
    from pyarrow.fs import resolve_s3_region

    return resolve_s3_region(bucket)


def filesystem(
    bucket: str,
    *,
    session: Any,
    endpoint_url: str | None = None,
    default_region: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> fs.S3FileSystem | None:
    """A credentialed ``pyarrow.fs.S3FileSystem``, or ``None`` when no
    credential resolves and the caller should fall back to pyarrow's own chain.

    pyarrow resolves credentials with the C++ AWS SDK, whose STS web-identity
    provider is not fork-safe: in a ``multiprocessing`` fork child it resolves
    anonymously and a cross-account read 403s. boto3's resolution is pure
    Python and fork-safe, so the same credentials are resolved here and handed
    to pyarrow explicitly.

    ``endpoint_url`` names a non-AWS S3 endpoint. The bucket's region is not
    resolved for one -- ``resolve_s3_region`` queries AWS S3 and would answer
    for a different bucket entirely. ``options`` are extra ``S3FileSystem``
    keywords and replace the endpoint handling derived here.
    """

    credentials = session.get_credentials()
    if credentials is None:
        return None
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        return None

    region = session.region_name or default_region
    if not endpoint_url:
        try:
            region = bucket_region(bucket) or region
        except Exception as error:  # noqa: BLE001 - best effort
            logger.debug("Could not resolve region for bucket %s: %s", bucket,
                         error)

    if options is not None:
        extra = dict(options)
    elif endpoint_url:
        # pyarrow-24 defaults to path-style for a custom endpoint_override,
        # where the boto3 client this signs alongside addresses virtually.
        extra = {
            "endpoint_override": endpoint_url,
            "force_virtual_addressing": True
        }
    else:
        extra = {}
    extra.setdefault("retry_strategy", retry_strategy())

    return fs.S3FileSystem(
        access_key=frozen.access_key,
        secret_key=frozen.secret_key,
        session_token=frozen.token,
        region=region,
        **extra,
    )


def retry_strategy() -> fs.S3RetryStrategy:
    # Not the default standard strategy: Arrow never refunds its retry quota.
    return fs.AwsDefaultS3RetryStrategy(max_attempts=S3_MAX_RETRIES)


def arrow_connection_options(endpoint_url: str | None, addressing_style: str,
                             ca_path: str | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        'force_virtual_addressing': addressing_style == 'virtual',
        'retry_strategy': retry_strategy(),
    }
    if endpoint_url:
        options['endpoint_override'] = endpoint_url
        options['scheme'] = urlparse(endpoint_url).scheme
    if ca_path:
        options['tls_ca_file_path'] = ca_path
    return options


def anonymous_filesystem(
    bucket: str,
    *,
    region: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> fs.S3FileSystem:
    options = dict(options or {})
    options.setdefault('retry_strategy', retry_strategy())
    if not region and not options.get('endpoint_override'):
        import kumo_s3
        region = kumo_s3.bucket_region(bucket)
    return fs.S3FileSystem(anonymous=True, region=region, **options)
