# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import hmac
import os
import re
import secrets
import uuid
from http import HTTPStatus
from threading import Lock
from typing import Dict, List, NoReturn, Optional

import cachetools
from fastapi import HTTPException

from rest.config.api_key import (
    APIKeyAuthIdentity,
    APIKeySummary,
    CreateAPIKeyResponse,
    ParsedAPIKey,
)
from rest.config.firstboot_secrets import get_secret_prefer_firstboot
from rest.config.status import RestStatusCode
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from statedb.config.common import APIKeyID
from statedb.config.timestamp import ResourceTimestamps, now_utc
from statedb.db.runtime import StateDB
from statedb.db.services import (
    APIKeyNameAlreadyExistsError,
    APIKeyService,
    StateServices,
)
from statedb.rest.api_key import (
    APIKeyMetadataRecord,
    APIKeySecretRecord,
    APIKeyType,
)

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_API_KEY_HASH_VERSION = "v1"
_API_KEY_HASH_ALGORITHM = "scrypt"
_API_KEY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_DEVTEST_API_KEY_PEPPER = "kumo-devtest-api-key-pepper"

_FAILED_ATTEMPT_LIMIT = 10
_FAILED_ATTEMPT_TTL_SECONDS = 300
_FAILED_ATTEMPTS = cachetools.TTLCache(maxsize=1000,
                                       ttl=_FAILED_ATTEMPT_TTL_SECONDS)
_FAILED_ATTEMPTS_LOCK = Lock()
_API_KEY_PEPPER_CACHE: dict[tuple[str, str], str] = {}
_API_KEY_PEPPER_CACHE_LOCK = Lock()


def _build_api_key_value(
    user_hash: str,
    name: str,
    secret: str,
) -> str:
    """Build the full on-wire API key value."""
    return f"{user_hash}:{name}:{secret}"


def _to_preview(key_value: str) -> str:
    """Return a redacted preview of a key value."""
    if len(key_value) <= 12:
        return key_value
    return f"{key_value[:6]}...{key_value[-6:]}"


def _to_stub_record(record: APIKeyMetadataRecord) -> APIKeySummary:
    """Convert persistent metadata to API response summary."""
    return APIKeySummary(
        id=record.id,
        name=record.name,
        type=record.type,
        owner_user_id=record.owner_user_id,
        owner_display_name=record.owner_display_name,
        created_time=record.created_time,
        value_preview=record.value_preview,
    )


def _get_api_key_pepper_cache_key(
    secret_name: str,
    pepper_field: str,
) -> tuple[str, str]:
    return secret_name, pepper_field


def _clear_api_key_pepper_cache_for_test() -> None:
    """Clear the memoized API key pepper cache for tests."""
    with _API_KEY_PEPPER_CACHE_LOCK:
        _API_KEY_PEPPER_CACHE.clear()


def _get_api_key_pepper() -> str:
    """Return the pepper used for API key hashing."""
    secret_name = os.getenv("API_KEY_PEPPER_SECRET_NAME", "kumo-secret-key")
    pepper_field = os.getenv("API_KEY_PEPPER_SECRET_FIELD", "api_key_pepper")
    cache_key = _get_api_key_pepper_cache_key(secret_name, pepper_field)

    with _API_KEY_PEPPER_CACHE_LOCK:
        cached_pepper = _API_KEY_PEPPER_CACHE.get(cache_key)
        if cached_pepper is not None:
            return cached_pepper

        env_pepper = os.getenv(
            pepper_field.upper()) or os.getenv('API_KEY_PEPPER')
        if env_pepper:
            _API_KEY_PEPPER_CACHE[cache_key] = env_pepper
            return env_pepper

        if _DEPLOY_STAGE.parse_value() != DeployStage.PROD:
            return _DEFAULT_DEVTEST_API_KEY_PEPPER

        secret_values = get_secret_prefer_firstboot(secret_name)
        pepper = secret_values.get(pepper_field)
        if not pepper:
            raise RuntimeError(
                f"Missing API key pepper field '{pepper_field}' in secret "
                f"'{secret_name}'")

        pepper_str = str(pepper)
        _API_KEY_PEPPER_CACHE[cache_key] = pepper_str
        return pepper_str


def _hash_api_key_value(key_value: str) -> tuple[str, str]:
    """Hash an API key using scrypt and the deployment pepper."""
    salt = secrets.token_bytes(16)
    pepper = _get_api_key_pepper().encode("utf-8")
    password = key_value.encode("utf-8") + b":" + pepper
    digest = hashlib.scrypt(
        password,
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return (
        base64.b64encode(salt).decode("utf-8"),
        base64.b64encode(digest).decode("utf-8"),
    )


async def _run_api_key_crypto(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _hash_api_key_value_async(key_value: str, ) -> tuple[str, str]:
    return await _run_api_key_crypto(_hash_api_key_value, key_value)


def _verify_api_key_value(
    api_key_value: str,
    salt_b64: str,
    hash_b64: str,
    hash_params: Dict[str, int],
) -> bool:
    """Verify API key value against persisted hash material."""
    try:
        salt = base64.b64decode(salt_b64)
        expected_digest = base64.b64decode(hash_b64)
    except Exception:
        return False

    pepper = _get_api_key_pepper().encode("utf-8")
    password = api_key_value.encode("utf-8") + b":" + pepper
    digest = hashlib.scrypt(
        password,
        salt=salt,
        n=hash_params.get("n", _SCRYPT_N),
        r=hash_params.get("r", _SCRYPT_R),
        p=hash_params.get("p", _SCRYPT_P),
        dklen=hash_params.get("dklen", _SCRYPT_DKLEN),
    )
    return hmac.compare_digest(digest, expected_digest)


async def _verify_api_key_value_async(
    api_key_value: str,
    salt_b64: str,
    hash_b64: str,
    hash_params: Dict[str, int],
) -> bool:
    return await _run_api_key_crypto(
        _verify_api_key_value,
        api_key_value,
        salt_b64,
        hash_b64,
        hash_params,
    )


def _normalize_type(api_key_type: APIKeyType | str) -> APIKeyType:
    """Normalize incoming API key type values to enum."""
    if isinstance(api_key_type, APIKeyType):
        return api_key_type
    return APIKeyType(api_key_type)


def _compute_user_hash(user_id: str) -> str:
    """Compute the stable 8-hex owner hash used in serialized keys."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8]


def _validate_api_key_name(name: str) -> None:
    """Validate API key display name."""
    if not _API_KEY_NAME_PATTERN.fullmatch(name):
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            "API key name must match [A-Za-z0-9_-]+",
        )


def _build_failed_attempt_key(parsed: ParsedAPIKey) -> str:
    """Build throttling key for failed verification attempts."""
    assert parsed.user_hash is not None
    return f"{parsed.user_hash}:{parsed.name}"


def _get_failed_attempts(fail_key: str) -> int:
    """Get failed verification attempt count for a key identity."""
    with _FAILED_ATTEMPTS_LOCK:
        return _FAILED_ATTEMPTS.get(fail_key, 0)


def _increment_failed_attempts(fail_key: str) -> None:
    """Increment failed verification attempt count for a key identity."""
    with _FAILED_ATTEMPTS_LOCK:
        _FAILED_ATTEMPTS[fail_key] = _FAILED_ATTEMPTS.get(fail_key, 0) + 1


def _clear_failed_attempts(fail_key: str) -> None:
    """Clear failed verification attempt count after successful auth."""
    with _FAILED_ATTEMPTS_LOCK:
        _FAILED_ATTEMPTS.pop(fail_key, None)


def _raise_invalid_api_key_verification() -> NoReturn:
    """Raise a standard invalid API-key verification error."""
    raise APIKeyVerificationError(RestStatusCode.AUTH_INVALID_API_KEY.value)


def _fail_invalid_api_key_verification(fail_key: str) -> NoReturn:
    """Record a failed attempt and raise standard invalid API-key error."""
    _increment_failed_attempts(fail_key)
    _raise_invalid_api_key_verification()


class APIKeyVerificationError(Exception):
    """Raised when API key verification fails."""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class APIKeyDriver:
    """CRUD and verification logic for v2 DB-backed API keys."""
    def __init__(
        self,
        update_last_used_time: bool = True,
        direct_connect_auth_store: Optional[DirectConnectAuthStore] = None,
        state_db: Optional[StateDB] = None,
        state_services: Optional[StateServices] = None,
    ) -> None:
        if state_db is not None and state_services is not None:
            raise ValueError("Pass either state_db or state_services, not both")
        self._update_last_used_time = update_last_used_time
        self._direct_connect_auth_store = direct_connect_auth_store
        self._state_services = state_services or (
            StateServices(state_db) if state_db is not None else None)

    def configure_state_db(self, state_db: StateDB) -> None:
        self.configure_state_services(StateServices(state_db))

    def configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    @property
    def _api_key_service(self) -> APIKeyService:
        if self._state_services is None:
            raise RuntimeError("APIKeyDriver StateServices is not configured")
        return self._state_services.api_keys

    async def list_admin_api_keys(
        self,
        type: Optional[APIKeyType] = None,
        search: Optional[str] = None,
    ) -> List[APIKeySummary]:
        """List admin-visible API keys."""
        api_keys = self._api_key_service
        records = [
            _to_stub_record(record)
            for record in await api_keys.list_metadata(type=type, search=search)
        ]

        return records

    async def create_admin_api_key(
        self,
        name: str,
        type: APIKeyType | str,
        user_id: str,
        user_name: str = "",
    ) -> CreateAPIKeyResponse:
        """Create an admin/user API key via admin endpoint."""
        key_type = _normalize_type(type)
        return await self._create_api_key(
            name=name,
            type=key_type,
            owner_user_id=user_id,
            owner_user_name=user_name,
        )

    async def delete_admin_api_key(self, api_key_id: APIKeyID) -> None:
        """Delete an API key from admin endpoint."""
        api_keys = self._api_key_service
        record = await api_keys.get_metadata(api_key_id)
        if record is None:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                f"API key not found: {api_key_id}")
        await api_keys.delete_if(
            api_key_id,
            lambda record: True,
        )

    async def list_user_api_keys(self, user_id: str) -> List[APIKeySummary]:
        """List user-scoped API keys owned by the requesting user."""
        api_keys = self._api_key_service
        records = [
            _to_stub_record(record) for record in await api_keys.list_metadata(
                type=APIKeyType.USER,
                owner_user_id=user_id,
            )
        ]
        return records

    async def create_user_api_key(
        self,
        name: str,
        user_id: str,
        user_name: str = "",
    ) -> CreateAPIKeyResponse:
        """Create a user-scoped API key."""
        return await self._create_api_key(
            name=name,
            type=APIKeyType.USER,
            owner_user_id=user_id,
            owner_user_name=user_name,
        )

    async def delete_user_api_key(self, api_key_id: APIKeyID,
                                  user_id: str) -> None:
        """Delete a user-scoped API key when owned by requester."""
        api_keys = self._api_key_service
        record = await api_keys.delete_if(
            api_key_id,
            lambda record: (record.owner_user_id == user_id and record.type ==
                            APIKeyType.USER),
        )
        if record is None:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                f"API key not found: {api_key_id}")

    async def _create_api_key(
        self,
        name: str,
        type: APIKeyType,
        owner_user_id: str,
        owner_user_name: str = "",
    ) -> CreateAPIKeyResponse:
        """Persist a new owner-bound API key and return plaintext once."""
        _validate_api_key_name(name)
        key_id = APIKeyID(f"k-{uuid.uuid4().hex}")
        user_hash = _compute_user_hash(owner_user_id)
        key_secret = secrets.token_urlsafe(32)
        key_value = _build_api_key_value(user_hash, name, key_secret)
        preview = _to_preview(key_value)
        created_time = now_utc()

        display_name = owner_user_name.strip() or owner_user_id
        metadata_record = APIKeyMetadataRecord(
            timestamps=ResourceTimestamps(
                created_at=created_time,
                updated_at=created_time,
            ),
            id=key_id,
            name=name,
            type=type,
            owner_user_id=owner_user_id,
            owner_user_hash=user_hash,
            owner_display_name=display_name,
            created_time=created_time,
            value_preview=preview,
        )
        salt_b64, hash_b64 = await _hash_api_key_value_async(key_value)
        secret_record = APIKeySecretRecord(
            timestamps=ResourceTimestamps(
                created_at=created_time,
                updated_at=created_time,
            ),
            id=key_id,
            hash_version=_API_KEY_HASH_VERSION,
            hash_algorithm=_API_KEY_HASH_ALGORITHM,
            hash_params={
                "n": _SCRYPT_N,
                "r": _SCRYPT_R,
                "p": _SCRYPT_P,
                "dklen": _SCRYPT_DKLEN,
            },
            salt_b64=salt_b64,
            hash_b64=hash_b64,
        )

        try:
            api_keys = self._api_key_service
            try:
                await api_keys.create_unique(
                    metadata=metadata_record,
                    secret=secret_record,
                )
            except APIKeyNameAlreadyExistsError:
                raise HTTPException(
                    HTTPStatus.BAD_REQUEST,
                    f"API key name already exists: {name}",
                ) from None
        except HTTPException:
            raise
        except Exception as ex:
            raise HTTPException(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Failed to securely persist API key",
            ) from ex

        return CreateAPIKeyResponse(
            id=metadata_record.id,
            name=metadata_record.name,
            type=metadata_record.type,
            created_time=metadata_record.created_time,
            value_preview=metadata_record.value_preview,
            value=key_value,
        )

    async def verify_api_key(
        self,
        api_key: str,
    ) -> Optional[APIKeyAuthIdentity]:
        """Verify an owner-bound API key against persisted hash material."""
        parsed = ParsedAPIKey.parse_supported(api_key)
        if parsed is None:
            return None
        assert parsed.user_hash is not None
        fail_key = _build_failed_attempt_key(parsed)
        if _get_failed_attempts(fail_key) >= _FAILED_ATTEMPT_LIMIT:
            _raise_invalid_api_key_verification()

        api_keys = self._api_key_service
        candidate = await api_keys.find_verification_candidate(
            owner_user_hash=parsed.user_hash,
            name=parsed.name,
        )
        if candidate is None:
            _fail_invalid_api_key_verification(fail_key)
        metadata = candidate.metadata
        secret_record = candidate.secret

        current_time = now_utc()
        if (secret_record.revoked_time is not None
                and secret_record.revoked_time <= current_time):
            _fail_invalid_api_key_verification(fail_key)

        if (secret_record.expires_time is not None
                and secret_record.expires_time <= current_time):
            _fail_invalid_api_key_verification(fail_key)

        if not await _verify_api_key_value_async(
                api_key, salt_b64=secret_record.salt_b64,
                hash_b64=secret_record.hash_b64,
                hash_params=secret_record.hash_params):
            _fail_invalid_api_key_verification(fail_key)

        _clear_failed_attempts(fail_key)

        if self._update_last_used_time:
            await self._api_key_service.update_last_used_time(
                metadata.id, now_utc())

        # The key `type` is a privilege ceiling: a USER key can never be admin,
        # regardless of who owns it. An ADMIN key additionally requires that its
        # owner is still an admin (Direct Connect), so a demotion instantly
        # neuters the key.
        is_admin = metadata.type == APIKeyType.ADMIN
        if self._direct_connect_auth_store is not None:
            owner = await self._direct_connect_auth_store.get_user(
                metadata.owner_user_id)
            # Direct Connect principals must remain backed by an active local
            # user. Other authenticated principals (for example an external
            # OIDC user) legitimately have no Direct Connect user row, even
            # when the public app also has Direct Connect support configured.
            # Such a key may authenticate, but its persisted metadata cannot
            # confer administrator authority without a live owner-role source.
            if owner is None:
                if metadata.owner_user_id.startswith('direct_connect|'):
                    _fail_invalid_api_key_verification(fail_key)
                is_admin = False
            elif owner.disabled:
                _fail_invalid_api_key_verification(fail_key)
            else:
                is_admin = (metadata.type == APIKeyType.ADMIN
                            and owner.role == 'admin')

        return APIKeyAuthIdentity(
            user_id=metadata.owner_user_id,
            is_admin=is_admin,
        )
