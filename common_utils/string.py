# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import urlsplit

from kumoapi.adls import normalize_adls_source_uri
from kumoapi.gcs import normalize_gcs_source_uri
from kumoapi.s3 import normalize_s3_source_uri


def contains_time(name: str) -> bool:
    if name.lower() == 'ts':
        return True

    # Check whether we can detect timestamps via naming convention:
    return re.search('(^|_)(time|timedate|timestamp)(_|$)', name.lower(),
                     re.IGNORECASE) is not None


def contains_date(name: str) -> bool:
    # Check whether we can detect dates via naming convention:
    return re.search(r'(^|_)(date|dt|datetime|updated)($|_)',
                     name.lower()) is not None


def contains_number(name: str) -> bool:
    # Check whether we can detect numbers/ordinals via naming convention:
    name = name.lower()
    for key in [
            'num', 'number', 'count', 'total', 'sum', 'avg', 'price', 'score',
            'ratio', 'value'
    ]:
        if f'_{key}_' in name or name.startswith(key) or name.endswith(key):
            return True
    return False


# Column names suffixes that end in "id" but should not be given the ID stype.
# From https://scrabble.merriam.com/words/end-with/id
_IGNORED_ID_SUFFIXES = [
    'bid',
    'acid',
    'grid',
    'maid',
    'paid',
    'raid',
    'void',
    'avoid',
    'braid',
    'covid',
    'fluid',
    'rabid',
    'solid',
    'hybrid',
    'inlaid',
    'liquid',
]

_ID_PATTERN = [
    'hash', 'no', 'cate', 'category', 'class', 'label', 'key', 'code'
]

_MORE_RELAX_ID_PATTERN = [
    'key',
    'code',
]


def contains_identifier(name: str) -> bool:
    name = name.lower()
    if '_id_' in name:
        return True
    for pattern in _ID_PATTERN:
        if (name.endswith(f'_{pattern}') or name.startswith(f'{pattern}_')
                or f'_{pattern}_' in name):
            return True
    for pattern in _MORE_RELAX_ID_PATTERN:
        if (name.startswith(pattern) or name.endswith(pattern)):
            return True
    if not name.endswith('id'):
        return False
    for suffix in _IGNORED_ID_SUFFIXES:
        if name.endswith(suffix):
            return False
    return True


def validate_hex_fingerprint(hex_fp: str) -> None:
    validate_non_empty(hex_fp)
    if not re.fullmatch(r'[a-f0-9]{32}', hex_fp):
        raise ValueError(
            f"Fingerprint ID must be 32-char hex digits, got {hex_fp}")


def validate_non_empty(name: str) -> None:
    if len(name) == 0:
        raise ValueError("Please fill out this value")


def validate_id(name: str, entity: str = "") -> None:
    """Ensure that the string only contains "safe" characters.

    The resulting string should be safe to use for filesystem names, and can be
    used as an identifier in most programming languages (such as SQL), without
    requiring any escaping of the characters.

    Specifically, it may only contain alphanumeric characters and underscores.
    """
    validate_non_empty(name)
    if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', name):
        error_text = (
            "Identifier may only contain alphanumeric characters and "
            f'underscores, and must begin with a letter. Got: "{name}"')
        if entity:
            error_text = (
                f'{entity} identifier may only contain alphanumeric '
                f'characters and underscores, and must begin with a letter. '
                f'Got: "{name}"')
        raise ValueError(error_text)


def validate_sql_id(name: str, is_col_name: bool = False) -> None:
    """Ensure that the string only contains "safe" characters and is formatted
    according to SQL requirements. In particular, verifies that it contains
    only alphanumeric characters and underscores, and begins with a letter or
    underscore.

    Args:
        is_col_name(Optional[bool]): Set to `True` when `name` is a column
            name, used to print a more meaningful error message for columns
    """
    validate_non_empty(name)
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', name):
        error_text = (
            "Identifier may only contain alphanumeric characters and "
            f'underscores, and must begin with a letter or _. Got: "{name}"')
        if is_col_name:
            error_text = (
                "Column names may only contain alphanumeric characters "
                "and underscores, and must begin with a letter or _. "
                f'Please fix the column with name: "{name}"')
        raise ValueError(error_text)


def validate_source_name(name: str) -> None:
    validate_non_empty(name)
    if not re.fullmatch(r'[a-zA-Z0-9_-]*', name):
        raise ValueError(
            'Source table name may only contain alphanumeric characters, '
            f'dashes, and underscores. Got: "{name}"')


def validate_connector_relative_source_name(name: str) -> None:
    """Validate a table coordinate relative to a connector-owned root.

    One trailing slash is meaningful: it selects a directory-backed table and
    must survive request validation unchanged. Empty components anywhere else
    remain invalid.
    """
    validate_non_empty(name)
    parsed = urlsplit(name)
    if (name.startswith('/') or '\\' in name or '?' in name or '#' in name
            or re.search(r'%[0-9a-fA-F]{2}', name)
            or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ValueError(
            'Source table name must be relative to its connector root. '
            f'Got: "{name}"')
    path = name[:-1] if name.endswith('/') else name
    if any(part in ('', '.', '..') for part in path.split('/')):
        raise ValueError(
            'Source table name cannot contain empty, dot, or parent path '
            f'components. Got: "{name}"')


def validate_connector_source_name(name: str) -> None:
    """Validate a rooted-relative or supported rootless cloud coordinate."""
    if urlsplit(name).scheme:
        normalize_connector_source_name(name)
        return
    validate_connector_relative_source_name(name)


def normalize_connector_source_name(name: str) -> str:
    """Normalize an exact rooted-relative or rootless cloud coordinate."""
    scheme = urlsplit(name).scheme
    if scheme == 's3':
        return normalize_s3_source_uri(name)
    if scheme == 'gs':
        return normalize_gcs_source_uri(name)
    if scheme == 'abfss':
        return normalize_adls_source_uri(name)
    if scheme:
        raise ValueError('Source URI must use s3://, gs://, or abfss://')
    validate_connector_relative_source_name(name)
    return name


def validate_max_length(name: str, length: int) -> None:
    if len(name) > length:
        raise ValueError(f"Identifier must be fewer than {length} characters "
                         f"long. Got: {name}")
