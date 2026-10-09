# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from io_utils import kumo_dataclass
from statedb.config.common import APIKeyID, DBEntry


class APIKeyType(str, Enum):
    """Logical API key category used for ownership and UI/admin filtering.

    The type is also a privilege ceiling during verification: a USER key
    never confers admin authority, regardless of its owner's role.
    """

    ADMIN = "admin"
    USER = "user"


_REDACTED = "<redacted>"


def _redacted_dataclass_dict(
    value: object,
    redacted_fields: set[str],
) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass instance, got {type(value)!r}")
    return {
        field.name:
        _REDACTED if field.name in redacted_fields else getattr(
            value, field.name)
        for field in fields(value)
    }


def _redacted_dataclass_repr(
    value: object,
    redacted_fields: set[str],
) -> str:
    fields_repr = ", ".join(f"{name}={field_value!r}"
                            for name, field_value in _redacted_dataclass_dict(
                                value, redacted_fields).items())
    return f"{type(value).__name__}({fields_repr})"


@kumo_dataclass
class APIKeyMetadataRecord(DBEntry):
    """Persisted metadata used for listing, search, and ownership checks.

    This record stores non-secret attributes. Secret verification material is
    kept separately in `APIKeySecretRecord`.
    """

    id: APIKeyID
    name: str
    type: APIKeyType
    owner_user_id: str
    owner_display_name: str
    created_time: datetime
    value_preview: str
    owner_user_hash: Optional[str] = None


_API_KEY_SECRET_REDACTED_FIELDS = {"salt_b64", "hash_b64"}


@kumo_dataclass(repr=False)
class APIKeySecretRecord(DBEntry):
    """Persisted secret verification and lifecycle state for an API key.

    This record intentionally excludes list/search ownership metadata, which is
    stored in `APIKeyMetadataRecord`.
    """

    id: APIKeyID
    hash_version: str
    hash_algorithm: str
    hash_params: Dict[str, int]
    salt_b64: str
    hash_b64: str
    last_used_time: Optional[datetime] = None
    expires_time: Optional[datetime] = None
    revoked_time: Optional[datetime] = None

    def __repr__(self) -> str:
        return _redacted_dataclass_repr(
            self,
            _API_KEY_SECRET_REDACTED_FIELDS,
        )

    __str__ = __repr__

    def to_log_dict(self) -> dict[str, Any]:
        """Return a log-safe view; persistence must keep using JSON helpers."""
        return _redacted_dataclass_dict(
            self,
            _API_KEY_SECRET_REDACTED_FIELDS,
        )
