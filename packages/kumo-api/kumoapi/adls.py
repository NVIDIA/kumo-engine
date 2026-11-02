# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import quote, urlsplit, urlunsplit

from kumoapi.object_store import (
    decode_object_store_uri_path,
    normalize_object_store_path_components,
    normalize_object_store_source_relative_path,
)

ADLS_ENDPOINT_SUFFIXES = frozenset({
    'dfs.core.windows.net',
    'dfs.core.usgovcloudapi.net',
    'dfs.core.chinacloudapi.cn',
})
_ADLS_ACCOUNT_RE = re.compile(r'^[a-z0-9]{3,24}$')
_ADLS_FILESYSTEM_RE = re.compile(r'^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])$')
_ADLS_PATH_SAFE = "/-._~!$&'()*+,;=:@"


def normalize_adls_storage_account(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Required provider field must not be blank')
    if _ADLS_ACCOUNT_RE.fullmatch(value) is None:
        raise ValueError('Invalid canonical Azure Data Lake storage account')
    return value


def normalize_adls_filesystem(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Required provider field must not be blank')
    if _ADLS_FILESYSTEM_RE.fullmatch(value) is None or '--' in value:
        raise ValueError('Invalid canonical Azure Data Lake filesystem')
    return value


def normalize_adls_endpoint_suffix(value: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError('Unsupported Azure Data Lake endpoint suffix')
    normalized = value.lower()
    if normalized not in ADLS_ENDPOINT_SUFFIXES:
        raise ValueError('Unsupported Azure Data Lake endpoint suffix')
    return normalized


def normalize_adls_source_relative_path(value: str) -> str:
    return normalize_object_store_source_relative_path(
        value, provider_name='Azure Data Lake')


def normalize_adls_source_uri(value: str) -> str:
    """Canonicalize an exact hierarchical-namespace ADLS source URI."""
    return _normalize_adls_uri(value, allow_root=False)


def normalize_adls_root_uri(value: str) -> str:
    """Canonicalize a connector root, including a filesystem without a prefix."""
    return _normalize_adls_uri(value, allow_root=True)


def _normalize_adls_uri(value: str, *, allow_root: bool) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            'ADLS URI must be non-empty without surrounding whitespace')
    parsed = urlsplit(value)
    if (parsed.scheme.lower() != 'abfss' or parsed.query or parsed.fragment
            or not parsed.netloc or parsed.password is not None
            or parsed.port is not None or parsed.netloc.count('@') != 1):
        raise ValueError(
            'ADLS URI must use abfss://filesystem@account.endpoint/path')
    filesystem, host = parsed.netloc.split('@', 1)
    matching_suffix = next(
        (suffix for suffix in ADLS_ENDPOINT_SUFFIXES
         if host.lower().endswith(f'.{suffix}')),
        None,
    )
    if matching_suffix is None:
        raise ValueError('Unsupported Azure Data Lake endpoint suffix')
    account = host[:-(len(matching_suffix) + 1)]
    filesystem = normalize_adls_filesystem(filesystem)
    account = normalize_adls_storage_account(account)
    if allow_root and parsed.path in ('', '/'):
        return f'abfss://{filesystem}@{account}.{matching_suffix}/'
    if not parsed.path.startswith('/') or parsed.path.startswith('//'):
        raise ValueError(
            'ADLS URI must contain exactly one filesystem/path separator')
    path = normalize_object_store_path_components(
        decode_object_store_uri_path(parsed.path[1:], provider_name='ADLS'),
        field_name='ADLS URI path',
        trailing_slash='preserve',
    )
    authority = f'{filesystem}@{account}.{matching_suffix}'
    return urlunsplit(
        ('abfss', authority, f'/{quote(path, safe=_ADLS_PATH_SAFE)}', '', ''))
