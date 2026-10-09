# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import datetime
from enum import Enum
from typing import Optional

from io_utils import kumo_dataclass
from statedb.config.common import APIKeyID
from statedb.rest.api_key import APIKeyType

_API_KEY_USER_HASH_PATTERN = re.compile(r"^[0-9a-f]{8}$")


class APIKeyFormat(str, Enum):
    """Supported serialized API key wire formats."""

    OWNER_BOUND = "owner_bound"
    LEGACY_PREFIXED = "legacy_prefixed"
    INVALID = "invalid"


@kumo_dataclass(frozen=True)
class ParsedAPIKey:
    """Normalized parse result for DB-backed API keys."""
    format: APIKeyFormat
    secret: str
    name: str = "default"
    user_hash: Optional[str] = None

    @classmethod
    def classify_format(cls, api_key: str) -> APIKeyFormat:
        """Classify an API key string by its owner-bound wire shape."""
        parts = api_key.split(":")
        if len(parts) == 3 and all(
                parts) and _API_KEY_USER_HASH_PATTERN.fullmatch(parts[0]):
            return APIKeyFormat.OWNER_BOUND
        if len(parts) == 4 and all(
                parts) and _API_KEY_USER_HASH_PATTERN.fullmatch(parts[1]):
            return APIKeyFormat.LEGACY_PREFIXED
        return APIKeyFormat.INVALID

    @classmethod
    def parse(cls, api_key: str) -> 'ParsedAPIKey':
        """Parse a current or legacy-prefixed owner-bound key."""
        parts = api_key.split(':')
        if len(parts) == 3 and all(parts):
            user_hash, name, secret = parts
            if _API_KEY_USER_HASH_PATTERN.fullmatch(user_hash):
                return cls(
                    format=APIKeyFormat.OWNER_BOUND,
                    secret=secret,
                    name=name,
                    user_hash=user_hash,
                )
        elif len(parts) == 4 and all(parts):
            _legacy_prefix, user_hash, name, secret = parts
            if _API_KEY_USER_HASH_PATTERN.fullmatch(user_hash):
                return cls(
                    format=APIKeyFormat.LEGACY_PREFIXED,
                    secret=secret,
                    name=name,
                    user_hash=user_hash,
                )

        raise ValueError('API key must be an owner-bound key.')

    @classmethod
    def parse_supported(cls, api_key: str) -> Optional['ParsedAPIKey']:
        """Parse any supported owner-bound wire format."""
        if cls.classify_format(api_key) == APIKeyFormat.INVALID:
            return None
        return cls.parse(api_key)


@kumo_dataclass
class CreateAdminAPIKeyRequest:
    """Request body for creating an admin-managed API key."""

    name: str
    type: APIKeyType


@kumo_dataclass
class CreateUserAPIKeyRequest:
    """Request body for creating a user-scoped API key."""

    name: str


@kumo_dataclass
class APIKeySummary:
    """List/read API view of an API key without secret verification data.

    This shape is safe to return to clients. It intentionally excludes
    cryptographic fields that belong only in `APIKeySecretRecord`.
    """
    id: APIKeyID
    name: str
    type: APIKeyType
    owner_user_id: str
    owner_display_name: str
    created_time: datetime
    value_preview: str


def matches_api_key_search(record: APIKeySummary,
                           search: Optional[str]) -> bool:
    """Apply case-insensitive API key search matching on summary fields."""
    if search is None or not search.strip():
        return True
    search_term = search.lower().strip()
    return search_term in " ".join([
        record.id,
        record.name,
        record.owner_display_name,
        record.value_preview,
    ]).lower()


@kumo_dataclass
class CreateAPIKeyResponse:
    """Response returned when an API key is created."""

    id: APIKeyID
    name: str
    type: APIKeyType
    created_time: datetime
    value_preview: str
    value: str


@kumo_dataclass
class APIKeyAuthIdentity:
    """Auth-layer identity returned after verifying a DB-backed API key."""

    # Narrow auth-layer projection returned by API-key verification. Keeping
    # this separate avoids coupling auth resolution to DB storage record shapes.
    user_id: str
    is_admin: bool
