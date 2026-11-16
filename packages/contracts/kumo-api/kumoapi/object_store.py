# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import unquote, urlsplit


def normalize_object_store_path_components(
    value: str,
    *,
    field_name: str,
    trailing_slash: str,
) -> str:
    """Validate and canonicalize decoded object-store path components."""
    has_trailing_slash = value.endswith('/')
    normalized = value[:-1] if has_trailing_slash else value
    if not normalized or normalized.startswith('/'):
        raise ValueError(f'{field_name} must identify a relative path')
    if ('\\' in normalized
            or any(ord(char) < 32 or ord(char) == 127 for char in normalized)):
        raise ValueError(f'{field_name} contains an unsafe character')
    components = normalized.split('/')
    if any(component in {'', '.', '..'} for component in components):
        raise ValueError(
            f'{field_name} contains an empty, current, or parent component')
    if has_trailing_slash:
        if trailing_slash == 'reject':
            raise ValueError(f'{field_name} must not end with a slash')
        if trailing_slash == 'preserve':
            return f'{"/".join(components)}/'
    return '/'.join(components)


def decode_object_store_uri_path(path: str, *, provider_name: str) -> str:
    """Decode one URI path exactly once, rejecting ambiguous separators."""
    if re.search(r'%(?![0-9a-fA-F]{2})', path):
        raise ValueError(
            f'{provider_name} path contains a malformed percent escape')
    if re.search(r'%(?:2f|5c)', path, flags=re.IGNORECASE):
        raise ValueError(
            f'{provider_name} path must not contain an encoded slash')
    try:
        decoded = unquote(path, errors='strict')
    except UnicodeDecodeError as error:
        raise ValueError(
            f'{provider_name} path contains invalid UTF-8') from error
    if re.search(r'%[0-9a-fA-F]{2}', decoded):
        raise ValueError(
            f'{provider_name} path must not contain nested percent encoding')
    return decoded


def validate_object_store_relative_input(
    value: str,
    *,
    field_name: str,
) -> str:
    """Require one decoded relative coordinate without URI delimiters."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f'{field_name} must be non-empty without surrounding whitespace')
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f'{field_name} must not be a URI, query, or fragment')
    # Relative coordinates cross several clients and services. Requiring the
    # decoded spelling prevents path checks from being bypassed by one or more
    # rounds of percent decoding (for example ``%2e%2e`` or ``%252f``).
    if '%' in value:
        raise ValueError(f'{field_name} must be decoded, not percent-encoded')
    return value


def normalize_object_store_root_prefix(
    value: str,
    *,
    provider_name: str,
) -> str:
    """Validate an optional canonical connector-root prefix."""
    field_name = f'{provider_name} root prefix'
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(
            f'{field_name} must not contain surrounding whitespace')
    if not value:
        return ''
    value = validate_object_store_relative_input(value, field_name=field_name)
    if value.startswith('/') or value.endswith('/'):
        raise ValueError(f'{field_name} must be a canonical relative path')
    return normalize_object_store_path_components(
        value,
        field_name=field_name,
        trailing_slash='reject',
    )


def normalize_object_store_source_relative_path(
    value: str,
    *,
    provider_name: str,
) -> str:
    """Canonicalize one decoded connector-relative source coordinate."""
    field_name = f'{provider_name} source path'
    value = validate_object_store_relative_input(value, field_name=field_name)
    return normalize_object_store_path_components(
        value,
        field_name=field_name,
        trailing_slash='preserve',
    )


def normalize_object_store_destination_relative_path(
    value: str,
    *,
    provider_name: str,
) -> str:
    """Canonicalize one decoded connector-relative output coordinate."""
    field_name = f'{provider_name} destination path'
    value = validate_object_store_relative_input(value, field_name=field_name)
    return normalize_object_store_path_components(
        value,
        field_name=field_name,
        trailing_slash='preserve',
    )
