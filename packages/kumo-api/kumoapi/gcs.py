# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import re
from ipaddress import ip_address
from urllib.parse import quote, urlsplit, urlunsplit

from kumoapi.object_store import (
    decode_object_store_uri_path,
    normalize_object_store_path_components,
    normalize_object_store_source_relative_path,
)

_GCS_PATH_SAFE = "/-._~!$&'()*+,;=:@"
_GCS_BUCKET_RE = re.compile(r'^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$')


def normalize_gcs_bucket(value: str) -> str:
    """Validate a canonical Google Cloud Storage bucket name."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Required provider field must not be blank')
    if value != value.strip():
        raise ValueError('GCS bucket must not contain surrounding whitespace')
    labels = value.split('.')
    if (_GCS_BUCKET_RE.fullmatch(value) is None or '..' in value
            or any(len(label) > 63
                   for label in labels) or value.startswith('goog')
            or 'google' in value or 'g00gle' in value or 'g0ogle' in value
            or 'go0gle' in value or 'goog1e' in value):
        raise ValueError('Invalid canonical GCS bucket name')
    try:
        ip_address(value)
    except ValueError:
        return value
    raise ValueError('GCS bucket name must not be an IP address')


def normalize_gcs_source_relative_path(value: str) -> str:
    return normalize_object_store_source_relative_path(value,
                                                       provider_name='GCS')


def normalize_gcs_source_uri(value: str) -> str:
    """Canonicalize an exact GCS source URI."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            'GCS URI must be non-empty without surrounding whitespace')
    parsed = urlsplit(value)
    if (parsed.scheme.lower() != 'gs' or not parsed.netloc or parsed.query
            or parsed.fragment or parsed.username is not None
            or parsed.password is not None or parsed.port is not None):
        raise ValueError(
            'GCS URI must use gs://bucket/path without query or fragment')
    bucket = normalize_gcs_bucket(parsed.netloc)
    if not parsed.path.startswith('/') or parsed.path.startswith('//'):
        raise ValueError(
            'GCS URI must contain exactly one bucket/path separator')
    path = normalize_object_store_path_components(
        decode_object_store_uri_path(parsed.path[1:], provider_name='GCS'),
        field_name='GCS URI path',
        trailing_slash='preserve',
    )
    return urlunsplit(
        ('gs', bucket, f'/{quote(path, safe=_GCS_PATH_SAFE)}', '', ''))
