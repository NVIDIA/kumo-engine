# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from ipaddress import ip_address
from urllib.parse import quote, urlsplit, urlunsplit

from kumoapi.object_store import (
    decode_object_store_uri_path,
    normalize_object_store_destination_relative_path,
    normalize_object_store_path_components,
    normalize_object_store_root_prefix,
    normalize_object_store_source_relative_path,
)

_S3_PATH_SAFE = "/-._~!$&'()*+,;=:@"
_S3_BUCKET_RE = re.compile(r'^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$')
_S3_RESERVED_PREFIXES = ('xn--', 'sthree-', 'amzn-s3-demo-')
_S3_RESERVED_SUFFIXES = (
    '-s3alias',
    '--ol-s3',
    '.mrap',
    '--x-s3',
    '--table-s3',
)


def normalize_s3_bucket(value: str) -> str:
    """Validate one canonical general-purpose S3 bucket name."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Required provider field must not be blank')
    if value != value.strip():
        raise ValueError('S3 bucket must not contain surrounding whitespace')
    if (_S3_BUCKET_RE.fullmatch(value) is None or '..' in value
            or value.startswith(_S3_RESERVED_PREFIXES)
            or value.endswith(_S3_RESERVED_SUFFIXES)):
        raise ValueError('Invalid canonical S3 bucket name')
    try:
        ip_address(value)
    except ValueError:
        return value
    raise ValueError('S3 bucket name must not be an IP address')


def normalize_s3_root_prefix(value: str) -> str:
    """Validate an optional, already-decoded rooted connector prefix."""
    return normalize_object_store_root_prefix(value, provider_name='S3')


def normalize_s3_source_relative_path(value: str) -> str:
    """Canonicalize a decoded source path, preserving directory intent.

    Percent-encoded relative keys are intentionally unsupported to avoid
    ambiguous or repeated decoding across clients and services.
    """
    return normalize_object_store_source_relative_path(value,
                                                       provider_name='S3')


def normalize_s3_destination_relative_path(value: str) -> str:
    """Canonicalize a decoded destination path and preserve one final slash.

    Percent-encoded relative keys are intentionally unsupported to avoid
    ambiguous or repeated decoding across clients and services.
    """
    return normalize_object_store_destination_relative_path(
        value,
        provider_name='S3',
    )


def _normalize_s3_uri(value: str, *, trailing_slash: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            'S3 URI must be non-empty without surrounding whitespace')
    parsed = urlsplit(value)
    if (parsed.scheme.lower() != 's3' or not parsed.netloc or parsed.query
            or parsed.fragment or parsed.username is not None
            or parsed.password is not None or parsed.port is not None):
        raise ValueError(
            'S3 URI must use s3://bucket/path without query or fragment')
    bucket = normalize_s3_bucket(parsed.netloc)
    if not parsed.path.startswith('/') or parsed.path.startswith('//'):
        raise ValueError(
            'S3 URI must contain exactly one bucket/path separator')
    path = normalize_object_store_path_components(
        decode_object_store_uri_path(parsed.path[1:], provider_name='S3'),
        field_name='S3 URI path',
        trailing_slash=trailing_slash,
    )
    encoded_path = quote(path, safe=_S3_PATH_SAFE)
    return urlunsplit(('s3', bucket, f'/{encoded_path}', '', ''))


def normalize_s3_source_uri(value: str) -> str:
    """Canonicalize a source URI, preserving one directory slash."""
    return _normalize_s3_uri(value, trailing_slash='preserve')


def normalize_s3_destination_uri(value: str) -> str:
    """Canonicalize a destination URI and preserve one final slash."""
    return _normalize_s3_uri(value, trailing_slash='preserve')
