# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime, timedelta
from http import HTTPStatus
from threading import Lock
from typing import Dict, List, NoReturn, Optional

from cryptography.fernet import InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.cmac import CMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from fastapi import HTTPException

from io_utils import logging
from rest.config.encryption import decrypt_secret_string, encrypt_secret_string
from rest.config.firstboot_secrets import get_secret_prefer_firstboot
from rest.config.status import RestStatusCode
from rest.utils.driver_common import get_current_timestamps
from service_framework.deploy_stage import _DEPLOY_STAGE, DeployStage
from statedb.config.direct_connect_auth import (
    DirectConnectAuthConfigRecord,
    DirectConnectPasswordCredentialRecord,
    DirectConnectPasswordLoginStatus,
    DirectConnectPasswordResetRecord,
    DirectConnectSessionRecord,
    DirectConnectUserRecord,
    DirectConnectUserResponse,
    DirectConnectUserRole,
    direct_connect_password_credential_tag,
)
from statedb.config.timestamp import now_utc
from statedb.db.direct_connect_auth import DirectConnectAuthService

log = logging.getLogger(__name__)

_PASSWORD_SCRYPT_N = 2**17
_PASSWORD_SCRYPT_R = 8
_PASSWORD_SCRYPT_P = 1
_PASSWORD_SCRYPT_DKLEN = 32
_PASSWORD_HASH_ALGORITHM = "scrypt"
# Current scrypt parameters. These are persisted on every credential
# (DirectConnectPasswordCredentialRecord.hash_params) and verification uses the
# params recorded on each credential, so the cost factor can be raised in future
# without invalidating already-stored hashes.
_PASSWORD_SCRYPT_PARAMS = {
    "n": _PASSWORD_SCRYPT_N,
    "r": _PASSWORD_SCRYPT_R,
    "p": _PASSWORD_SCRYPT_P,
    "dklen": _PASSWORD_SCRYPT_DKLEN,
}
_PASSWORD_MIN_LENGTH = 12
_PASSWORD_DENIED_TERMS = ("password", "temporary", "admin", "kumo")
_PASSWORD_TOO_WEAK_MESSAGE = "Password is too easy to guess"
_DUMMY_PASSWORD_SALT = b"direct-connect!!"
_DUMMY_PASSWORD_DIGEST = b"\x00" * _PASSWORD_SCRYPT_DKLEN
_PASSWORD_LOGIN_EMAIL_LIMIT = 5
_PASSWORD_LOGIN_IP_LIMIT = 30
_PASSWORD_LOGIN_WINDOW = timedelta(minutes=10)
_PASSWORD_LOGIN_LOCKOUT = timedelta(minutes=15)
_PASSWORD_LOGIN_UNKNOWN_IP = "unknown"
_NON_PROD_CUSTOMERS = {"dev", "test", "kumo", "local", "pytest"}
_DEFAULT_DEVTEST_PASSWORD_RESET_TOKEN_PEPPER = (
    "kumo-devtest-password-reset-token-pepper")
_PASSWORD_RESET_TOKEN_LOOKUP_PREFIX = "v2"
_PASSWORD_RESET_TOKEN_LOOKUP_KEY_LENGTH = 32
_PASSWORD_RESET_TOKEN_LOOKUP_KEY_SALT = b"kumo-dc-pw-reset-token-lookup"
_PASSWORD_RESET_TOKEN_LOOKUP_KEY_INFO = b"dc-pw-reset-token-lookup-v2"
_PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE: Optional[bytes] = None
_PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE_LOCK = Lock()


def _raise_wrong_credentials() -> NoReturn:
    raise HTTPException(HTTPStatus.UNAUTHORIZED,
                        RestStatusCode.AUTH_WRONG_CREDENTIALS.value)


def _raise_login_rate_limited() -> NoReturn:
    raise HTTPException(HTTPStatus.TOO_MANY_REQUESTS,
                        RestStatusCode.AUTH_LOGIN_RATE_LIMITED.value)


def _password_login_attempt_key(kind: str, value: str) -> str:
    normalized = (value or "").strip().lower() or "missing"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def _password_login_attempt_keys(
    email: str,
    source_ip: Optional[str],
) -> list[tuple[str, int]]:
    keys = [
        (_password_login_attempt_key("email",
                                     email), _PASSWORD_LOGIN_EMAIL_LIMIT),
    ]
    # None explicitly disables the IP bucket; an empty peer preserves the
    # historical shared "unknown" bucket for enabled/external deployments.
    if source_ip is not None:
        source_ip = source_ip or _PASSWORD_LOGIN_UNKNOWN_IP
        keys.append(
            (_password_login_attempt_key("ip",
                                         source_ip), _PASSWORD_LOGIN_IP_LIMIT))
    return keys


def _password_policy_error(
    password: str,
    user_email: str = "",
) -> Optional[str]:
    if not password:
        return "Password is required"
    if len(password) < _PASSWORD_MIN_LENGTH:
        return f"Password must be at least {_PASSWORD_MIN_LENGTH} characters"
    if len(set(password)) == 1:
        return _PASSWORD_TOO_WEAK_MESSAGE
    if (not any(char.isalpha() for char in password)
            or not any(char.isdigit() for char in password)):
        return "Password must include at least one letter and one number"

    normalized_password = password.lower()
    if any(term in normalized_password for term in _PASSWORD_DENIED_TERMS):
        return _PASSWORD_TOO_WEAK_MESSAGE

    email_local_part = normalize_direct_connect_email(user_email).split("@",
                                                                        1)[0]
    if len(email_local_part) >= 3 and email_local_part in normalized_password:
        return "Password must not contain your email address"
    return None


def _validate_password_policy(password: str, user_email: str = "") -> None:
    error = _password_policy_error(password, user_email=user_email)
    if error:
        raise HTTPException(HTTPStatus.BAD_REQUEST, error)


def _scrypt_maxmem(params: Dict[str, int]) -> int:
    """OpenSSL memory ceiling sized from the scrypt params.

    OpenSSL rejects scrypt when ``128 * r * (n + p + 2)`` exceeds ``maxmem``
    (default 32 MiB). That bound is exceeded once the cost factor is raised, so
    derive the ceiling from the params (with 1 MiB of slack) and raising the
    cost factor needs no change here.
    """
    n = params["n"]
    r = params["r"]
    p = params["p"]
    return 128 * r * (n + p + 2) + (1 << 20)


def _scrypt_digest(password: str, salt: bytes, params: Dict[str, int]) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=params["n"],
        r=params["r"],
        p=params["p"],
        dklen=params["dklen"],
        maxmem=_scrypt_maxmem(params),
    )


def _scrypt_hash(password: str) -> tuple[str, str]:
    """Derive a fresh salt + scrypt digest using the current parameters."""
    salt = secrets.token_bytes(16)
    digest = _scrypt_digest(password, salt, _PASSWORD_SCRYPT_PARAMS)
    return (
        base64.b64encode(salt).decode("utf-8"),
        base64.b64encode(digest).decode("utf-8"),
    )


def _hash_password(password: str, user_email: str = "") -> tuple[str, str]:
    _validate_password_policy(password, user_email=user_email)
    return _scrypt_hash(password)


async def _run_auth_crypto(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _hash_password_async(
    password: str,
    user_email: str = "",
) -> tuple[str, str]:
    return await _run_auth_crypto(_hash_password, password,
                                  user_email=user_email)


def _build_password_credential(
    password_hash_b64: str,
    salt_b64: str,
    *,
    must_reset_password: bool,
) -> DirectConnectPasswordCredentialRecord:
    """Build a credential record stamped with the current scrypt params."""
    return DirectConnectPasswordCredentialRecord(
        timestamps=get_current_timestamps(),
        password_hash_b64=password_hash_b64,
        salt_b64=salt_b64,
        hash_algorithm=_PASSWORD_HASH_ALGORITHM,
        hash_params=dict(_PASSWORD_SCRYPT_PARAMS),
        must_reset_password=must_reset_password,
    )


def _temporary_password_credential(
    password: str,
    user_email: str,
) -> DirectConnectPasswordCredentialRecord:
    salt_b64, password_hash_b64 = _hash_password(password,
                                                 user_email=user_email)
    return _build_password_credential(
        password_hash_b64,
        salt_b64,
        must_reset_password=True,
    )


async def _password_credential_async(
    password: str,
    user_email: str,
    *,
    must_reset_password: bool,
) -> DirectConnectPasswordCredentialRecord:
    salt_b64, password_hash_b64 = await _hash_password_async(
        password, user_email=user_email)
    return _build_password_credential(
        password_hash_b64,
        salt_b64,
        must_reset_password=must_reset_password,
    )


async def _temporary_password_credential_async(
    password: str,
    user_email: str,
) -> DirectConnectPasswordCredentialRecord:
    return await _password_credential_async(
        password,
        user_email,
        must_reset_password=True,
    )


def _password_credential_is_valid(
    credential: Optional[DirectConnectPasswordCredentialRecord], ) -> bool:
    if credential is None or credential.hash_algorithm != _PASSWORD_HASH_ALGORITHM:
        return False
    try:
        salt = base64.b64decode(credential.salt_b64, validate=True)
        digest = base64.b64decode(credential.password_hash_b64, validate=True)
    except Exception:
        return False
    return bool(salt) and len(digest) == credential.hash_params["dklen"]


def _verify_dummy_password(password: str) -> None:
    # Burn comparable scrypt work on the no-user path so response timing does
    # not reveal whether an email exists.
    digest = _scrypt_digest(password, _DUMMY_PASSWORD_SALT,
                            _PASSWORD_SCRYPT_PARAMS)
    hmac.compare_digest(digest, _DUMMY_PASSWORD_DIGEST)


async def _verify_dummy_password_async(password: str) -> None:
    await _run_auth_crypto(_verify_dummy_password, password)


def _verify_password(
    password: str,
    credential: DirectConnectPasswordCredentialRecord,
) -> bool:
    if credential.hash_algorithm != _PASSWORD_HASH_ALGORITHM:
        return False
    try:
        salt = base64.b64decode(credential.salt_b64)
        expected_digest = base64.b64decode(credential.password_hash_b64)
    except Exception:
        return False
    digest = _scrypt_digest(password, salt, credential.hash_params)
    return hmac.compare_digest(digest, expected_digest)


async def _verify_password_async(
    password: str,
    credential: DirectConnectPasswordCredentialRecord,
) -> bool:
    return await _run_auth_crypto(_verify_password, password, credential)


def _password_reset_token_key(token: str) -> str:
    mac = CMAC(algorithms.AES(_password_reset_token_lookup_key()))
    mac.update(token.encode("utf-8"))
    token_tag = base64.urlsafe_b64encode(
        mac.finalize()).decode("utf-8").rstrip("=")
    return f"{_PASSWORD_RESET_TOKEN_LOOKUP_PREFIX}:{token_tag}"


def _password_reset_token_lookup_key() -> bytes:
    global _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE
    with _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE_LOCK:
        cached_key = _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE
    if cached_key is not None:
        return cached_key

    pepper = _get_password_reset_token_pepper().encode("utf-8")
    lookup_key = HKDF(
        algorithm=hashes.SHA256(),
        length=_PASSWORD_RESET_TOKEN_LOOKUP_KEY_LENGTH,
        salt=_PASSWORD_RESET_TOKEN_LOOKUP_KEY_SALT,
        info=_PASSWORD_RESET_TOKEN_LOOKUP_KEY_INFO,
    ).derive(pepper)
    with _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE_LOCK:
        if _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE is None:
            _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE = lookup_key
        return _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE


async def _warm_password_reset_token_lookup_key() -> None:
    await _run_auth_crypto(_password_reset_token_lookup_key)


def _clear_password_reset_token_lookup_key_cache_for_test() -> None:
    global _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE
    with _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE_LOCK:
        _PASSWORD_RESET_TOKEN_LOOKUP_KEY_CACHE = None


def _get_password_reset_token_pepper() -> str:
    if _DEPLOY_STAGE.parse_value() != DeployStage.PROD:
        return _DEFAULT_DEVTEST_PASSWORD_RESET_TOKEN_PEPPER

    secret_name = os.getenv("PASSWORD_RESET_TOKEN_PEPPER_SECRET_NAME",
                            "kumo-secret-key")
    pepper_field = os.getenv("PASSWORD_RESET_TOKEN_PEPPER_SECRET_FIELD",
                             "credentials_encryption_key")
    env_pepper = (os.getenv(pepper_field.upper())
                  or os.getenv("PASSWORD_RESET_TOKEN_PEPPER"))
    if env_pepper:
        return env_pepper

    secret_values = get_secret_prefer_firstboot(secret_name)
    pepper = secret_values.get(pepper_field)
    if not pepper:
        raise RuntimeError(
            f"Missing password reset token pepper field '{pepper_field}' "
            f"in secret '{secret_name}'")
    return str(pepper)


def _local_user_id() -> str:
    return f"direct_connect|local|{uuid.uuid4().hex}"


def normalize_direct_connect_email(email: str) -> str:
    return email.strip().lower()


def _normalized_admin_emails(admin_emails: List[str]) -> List[str]:
    normalized_emails = []
    seen = set()
    for email in admin_emails:
        normalized_email = normalize_direct_connect_email(email)
        if normalized_email and normalized_email not in seen:
            normalized_emails.append(normalized_email)
            seen.add(normalized_email)
    return normalized_emails


def direct_connect_user_id_for_identity(issuer: str, subject: str) -> str:
    digest = hashlib.sha256(f"{issuer}|{subject}".encode("utf-8")).hexdigest()
    return f"direct_connect|{digest[:32]}"


def direct_connect_identity_key(issuer: str, subject: str) -> str:
    return hashlib.sha256(f"{issuer}|{subject}".encode("utf-8")).hexdigest()


def direct_connect_user_to_response(
    user_id: str,
    record: DirectConnectUserRecord,
) -> DirectConnectUserResponse:
    return DirectConnectUserResponse(
        user_id=user_id,
        email=record.email,
        display_name=record.display_name,
        role=record.role,
        disabled=record.disabled,
    )


class DirectConnectAuthStore:
    def __init__(
        self,
        auth_service: Optional[DirectConnectAuthService] = None,
    ) -> None:
        self._auth_service = auth_service

    def configure_auth_service(
        self,
        auth_service: DirectConnectAuthService,
    ) -> None:
        self._auth_service = auth_service

    @property
    def is_configured(self) -> bool:
        return self._auth_service is not None

    @property
    def _service(self) -> DirectConnectAuthService:
        if self._auth_service is None:
            raise RuntimeError(
                "DirectConnectAuthStore auth service is not configured")
        return self._auth_service

    async def init_once(self) -> None:
        await self._service.init_once()

    async def warm_password_reset_token_lookup_key(self) -> None:
        await _warm_password_reset_token_lookup_key()

    async def close(self) -> None:
        if self._auth_service is not None:
            await self._auth_service.close()

    async def get_config(self) -> Optional[DirectConnectAuthConfigRecord]:
        return await self._service.get_config()

    async def set_config(self, config: DirectConnectAuthConfigRecord) -> None:
        await self._service.set_config(config)

    async def check_password_login_allowed(self, *, email: str,
                                           source_ip: Optional[str]) -> None:
        now = now_utc()
        async with self._service.read() as db:
            for key, limit in _password_login_attempt_keys(email, source_ip):
                record = await db.get_password_login_attempt(key)
                if record is None:
                    continue
                if (record.locked_until is not None
                        and record.locked_until > now):
                    _raise_login_rate_limited()
                if (record.first_failed_time > now - _PASSWORD_LOGIN_WINDOW
                        and record.failure_count >= limit):
                    _raise_login_rate_limited()

    async def record_password_login_failure(self, *, email: str,
                                            source_ip: Optional[str]) -> None:
        now = now_utc()
        await self._service.record_password_login_failures(
            _password_login_attempt_keys(email, source_ip),
            now=now,
            window=_PASSWORD_LOGIN_WINDOW,
            lockout=_PASSWORD_LOGIN_LOCKOUT,
            stale_before=now -
            (_PASSWORD_LOGIN_WINDOW + _PASSWORD_LOGIN_LOCKOUT),
        )

    async def record_password_login_success(self, *, email: str) -> None:
        key = _password_login_attempt_key("email", email)
        async with self._service.write() as db:
            await db.pop_password_login_attempt(key)

    async def cleanup_expired_password_login_attempts(self,
                                                      limit: int = 100) -> int:
        now = now_utc()
        async with self._service.write() as db:
            return await db.delete_expired_password_login_attempts(
                stale_before=now -
                (_PASSWORD_LOGIN_WINDOW + _PASSWORD_LOGIN_LOCKOUT),
                now=now,
                limit=limit,
            )

    async def seed_config(
        self,
        *,
        issuer: str,
        client_id: str,
        client_secret: str,
        groups_claim: str,
        admin_emails: List[str],
        google_sa_key: str = "",
        google_admin_email: str = "",
    ) -> None:
        if not (issuer and client_id and client_secret):
            return
        existing = await self.get_config()
        if existing is not None:
            return
        timestamps = get_current_timestamps()
        await self.set_config(
            DirectConnectAuthConfigRecord(
                timestamps=timestamps,
                issuer=issuer.rstrip("/"),
                client_id=client_id,
                encrypted_client_secret=encrypt_secret_string(client_secret),
                groups_claim=groups_claim or "groups",
                admin_emails=[
                    normalize_direct_connect_email(email)
                    for email in admin_emails if email.strip()
                ],
                enabled=True,
                encrypted_google_sa_key=(encrypt_secret_string(google_sa_key)
                                         if google_sa_key else ""),
                google_admin_email=google_admin_email,
            ))

    async def seed_config_from_env(
        self,
        *,
        issuer: str,
        client_id: str,
        client_secret: str,
        groups_claim: str,
        admin_emails: List[str],
    ) -> None:
        await self.seed_config(
            issuer=issuer,
            client_id=client_id,
            client_secret=client_secret,
            groups_claim=groups_claim,
            admin_emails=admin_emails,
        )

    async def seed_or_reconcile_oidc_config(
        self,
        *,
        issuer: str,
        client_id: str,
        client_secret: str,
        groups_claim: str,
        admin_emails: List[str],
        google_sa_key: str = "",
        google_admin_email: str = "",
    ) -> None:
        normalized_issuer = issuer.rstrip("/") if issuer else ""
        normalized_groups_claim = groups_claim or "groups"
        normalized_admin_emails = _normalized_admin_emails(admin_emails)

        async with self._service.write() as db:
            existing = await db.get_config()
            if existing is None:
                if not (normalized_issuer and client_id and client_secret):
                    raise ValueError(
                        "Direct Connect OIDC config is incomplete; issuer, "
                        "client_id, and client_secret are required")
                if not normalized_admin_emails:
                    raise ValueError(
                        "Direct Connect OIDC config is incomplete; "
                        "admin_emails is required")
                await db.set_config(
                    DirectConnectAuthConfigRecord(
                        timestamps=get_current_timestamps(),
                        issuer=normalized_issuer,
                        client_id=client_id,
                        encrypted_client_secret=encrypt_secret_string(
                            client_secret),
                        groups_claim=normalized_groups_claim,
                        admin_emails=normalized_admin_emails,
                        enabled=True,
                        encrypted_google_sa_key=(
                            encrypt_secret_string(google_sa_key)
                            if google_sa_key else ""),
                        google_admin_email=google_admin_email,
                    ))
                return

            if not existing.enabled:
                raise ValueError(
                    "Direct Connect StateDB config is disabled; refusing to "
                    "enable or overwrite it during startup.")

            drift_fields = []
            if normalized_issuer and existing.issuer != normalized_issuer:
                existing.issuer = normalized_issuer
                drift_fields.append("issuer")
            if client_id and existing.client_id != client_id:
                existing.client_id = client_id
                drift_fields.append("client_id")
            try:
                existing_client_secret = decrypt_secret_string(
                    existing.encrypted_client_secret)
            except (InvalidToken, ValueError):
                log.warning("Stored Direct Connect client secret could not be "
                            "decrypted; treating it as drifted")
                existing_client_secret = None
            if client_secret and existing_client_secret != client_secret:
                existing.encrypted_client_secret = encrypt_secret_string(
                    client_secret)
                drift_fields.append("client_secret")
            if groups_claim and existing.groups_claim != normalized_groups_claim:
                existing.groups_claim = normalized_groups_claim
                drift_fields.append("groups_claim")
            try:
                existing_google_sa_key = (
                    decrypt_secret_string(existing.encrypted_google_sa_key)
                    if existing.encrypted_google_sa_key else "")
            except (InvalidToken, ValueError):
                log.warning("Stored Direct Connect Google SA key could not be "
                            "decrypted; treating it as drifted")
                existing_google_sa_key = None
            if google_sa_key and existing_google_sa_key != google_sa_key:
                existing.encrypted_google_sa_key = encrypt_secret_string(
                    google_sa_key)
                drift_fields.append("google_sa_key")
            if (google_admin_email
                    and existing.google_admin_email != google_admin_email):
                existing.google_admin_email = google_admin_email
                drift_fields.append("google_admin_email")
            if drift_fields:
                log.warning(
                    "Direct Connect OIDC config drifted from deploy config; "
                    "updating stored config from environment: %s",
                    ", ".join(drift_fields))
                await db.set_config(existing)

            users = await db.list_users()
            active_admin_exists = any(item.role == "admin" and not item.disabled
                                      for _, item in users)
            if active_admin_exists:
                return

            if not normalized_admin_emails:
                if existing.admin_emails:
                    return
                raise ValueError(
                    "Direct Connect OIDC config is incomplete; admin_emails "
                    "is required when no active admin exists")

            if existing.admin_emails != normalized_admin_emails:
                existing.admin_emails = normalized_admin_emails
                await db.set_config(existing)

    async def get_client_secret(self) -> str:
        config = await self.get_config()
        if config is None:
            return ""
        return decrypt_secret_string(config.encrypted_client_secret)

    async def get_google_groups_settings(self) -> tuple[str, str]:
        """Return (service-account key JSON, admin email); empty if unset."""
        config = await self.get_config()
        if config is None or not config.encrypted_google_sa_key:
            return "", ""
        return (
            decrypt_secret_string(config.encrypted_google_sa_key),
            config.google_admin_email,
        )

    async def list_users(self) -> List[DirectConnectUserResponse]:
        users = await self._service.list_users()
        return sorted(
            [
                direct_connect_user_to_response(user_id, record)
                for user_id, record in users
            ],
            key=lambda record: record.email,
        )

    async def get_user(self, user_id: str) -> Optional[DirectConnectUserRecord]:
        return await self._service.get_user(user_id)

    async def get_user_by_email(
            self, email: str) -> Optional[tuple[str, DirectConnectUserRecord]]:
        return await self._service.get_user_by_email(email)

    async def upsert_user_from_idp(
        self,
        *,
        issuer: str,
        subject: str,
        email: str,
        display_name: str,
        refresh_token: str,
        email_verified: Optional[bool] = None,
    ) -> tuple[str, DirectConnectUserRecord]:
        normalized_email = normalize_direct_connect_email(email)
        if not normalized_email:
            raise HTTPException(400, "IdP token is missing email")

        config = await self.get_config()
        admin_emails = set(config.admin_emails if config is not None else [])
        async with self._service.write() as db:
            user_id = await db.get_user_id_by_identity(issuer, subject)
            email_user_id = await db.get_user_id_by_email(normalized_email)
            if user_id and email_user_id and email_user_id != user_id:
                raise HTTPException(
                    HTTPStatus.CONFLICT,
                    f"Direct Connect identity email conflicts with existing user: {normalized_email}",
                )
            if not user_id:
                user_id = (email_user_id or direct_connect_user_id_for_identity(
                    issuer, subject))
            record = await db.get_user(user_id)
            users = await db.list_users()
            active_admin_exists = any(item.role == "admin" and not item.disabled
                                      for _, item in users)
            should_bootstrap_admin = (normalized_email in admin_emails
                                      and not active_admin_exists
                                      and email_verified is True)
            if record is None:
                record = DirectConnectUserRecord(
                    timestamps=get_current_timestamps(),
                    email=normalized_email,
                    display_name=display_name,
                    role="admin" if should_bootstrap_admin else "user",
                    disabled=False,
                )
            else:
                record.email = normalized_email
                record.display_name = display_name or record.display_name
                if should_bootstrap_admin:
                    record.role = "admin"

            await db.set_user(user_id, record)
            await db.set_identity_user_id(
                issuer,
                subject,
                user_id,
                encrypt_secret_string(refresh_token),
            )
            return user_id, record

    async def fetch_external_group_ids(
        self,
        user_id: str,
    ) -> Optional[List[str]]:
        """Fetch current IdP groups for a linked OIDC user.

        ``None`` means the user is native/password-only and has no external
        membership source. Provider failures are raised so request
        authorization can use an unexpired cache entry or fail closed.
        """
        config = await self.get_config()
        if config is None or not config.enabled:
            return None
        from rest.deps.direct_connect_oidc_client import (
            DirectConnectOIDCClient,
        )

        client = DirectConnectOIDCClient(
            issuer=config.issuer,
            client_id=config.client_id,
            client_secret=await self.get_client_secret(),
            groups_claim=config.groups_claim,
        )
        async with self._service.serialized_identity_refresh(user_id) as (
                db, linked):
            if linked is None:
                return None
            link_id, link = linked
            if not link.encrypted_refresh_token:
                raise RuntimeError(
                    "Direct Connect identity has no OIDC refresh token")
            old_refresh_token = decrypt_secret_string(
                link.encrypted_refresh_token)
            refreshed = await client.refresh_groups.coroutine(
                refresh_token=old_refresh_token)
            if refreshed.refresh_token != old_refresh_token:
                await db.set_identity_refresh_token(
                    link_id,
                    link,
                    encrypt_secret_string(refreshed.refresh_token),
                )
            return refreshed.external_group_ids

    async def seed_local_admins(
        self,
        *,
        admin_emails: List[str],
        temporary_password: str,
    ) -> None:
        normalized_emails = [
            normalize_direct_connect_email(email) for email in admin_emails
            if email.strip()
        ]
        if not normalized_emails or not temporary_password:
            return
        async with self._service.read() as db:
            users = await db.list_users()
            active_admin_exists = any(item.role == "admin" and not item.disabled
                                      for _, item in users)
            existing_admin_needing_password = None
            for email in normalized_emails:
                user_id = await db.get_user_id_by_email(email)
                record = await db.get_user(user_id) if user_id else None
                if (user_id and record is not None and record.role == "admin"
                        and not record.disabled):
                    if not _password_credential_is_valid(
                            await db.get_password_credential(user_id)):
                        existing_admin_needing_password = (user_id, record)
                        break
                    return

            if active_admin_exists and existing_admin_needing_password is None:
                return

            planned_admins = []
            if existing_admin_needing_password is None:
                for email in normalized_emails:
                    user_id = (await db.get_user_id_by_email(email)
                               or _local_user_id())
                    record = await db.get_user(user_id)
                    credential = (await db.get_password_credential(user_id)
                                  if record is not None else None)
                    planned_admins.append((email, user_id, record, credential))

        if existing_admin_needing_password is not None:
            user_id, record = existing_admin_needing_password
            password_credential = await _temporary_password_credential_async(
                temporary_password, record.email)
            async with self._service.write() as db:
                current = await db.get_user(user_id)
                if (current is not None and current.role == "admin"
                        and not current.disabled
                        and not _password_credential_is_valid(
                            await db.get_password_credential(user_id))):
                    await db.set_password_credential(user_id,
                                                     password_credential)
            return

        credentials_by_email = {}
        for email, _user_id, _record, credential in planned_admins:
            if not _password_credential_is_valid(credential):
                credentials_by_email[
                    email] = await _temporary_password_credential_async(
                        temporary_password, email)

        async with self._service.write() as db:
            for email, planned_user_id, _record, _credential in planned_admins:
                user_id = await db.get_user_id_by_email(email
                                                        ) or planned_user_id
                record = await db.get_user(user_id)
                if record is None:
                    record = DirectConnectUserRecord(
                        timestamps=get_current_timestamps(),
                        email=email,
                        display_name=email,
                        role="admin",
                        disabled=False,
                    )
                else:
                    record.role = "admin"
                    record.disabled = False
                await db.set_user(user_id, record)

                if not _password_credential_is_valid(
                        await db.get_password_credential(user_id)):
                    password_credential = credentials_by_email.get(email)
                    if password_credential is not None:
                        await db.set_password_credential(
                            user_id,
                            password_credential,
                        )

    async def seed_oidc_admins(
        self,
        *,
        admin_emails: List[str],
    ) -> None:
        """Dev-only: force ``admin_emails`` to role=admin for OIDC logins.

        IdPs such as Entra ID omit the ``email_verified`` claim, which the
        login-time first-admin bootstrap requires, so a configured admin
        otherwise lands as a plain ``user``.  This seeds/upserts the admin
        record by email (no password, never a downgrade); the OIDC callback
        then links the identity to it and preserves the admin role.  Gated by
        ``KUMO_DEV_TRUST_ADMIN_EMAILS`` and never enabled in production.
        """
        normalized_emails = [
            normalize_direct_connect_email(email) for email in admin_emails
            if email.strip()
        ]
        if not normalized_emails:
            return
        async with self._service.write() as db:
            for email in normalized_emails:
                user_id = await db.get_user_id_by_email(email
                                                        ) or _local_user_id()
                record = await db.get_user(user_id)
                if record is None:
                    record = DirectConnectUserRecord(
                        timestamps=get_current_timestamps(),
                        email=email,
                        display_name=email,
                        role="admin",
                        disabled=False,
                    )
                elif record.role == "admin" and not record.disabled:
                    continue
                else:
                    record.role = "admin"
                    record.disabled = False
                await db.set_user(user_id, record)

    async def create_local_user(
        self,
        *,
        email: str,
        display_name: str = "",
        role: DirectConnectUserRole = "user",
        disabled: bool = False,
        password: Optional[str] = None,
    ) -> DirectConnectUserResponse:
        normalized_email = normalize_direct_connect_email(email)
        if not normalized_email:
            raise HTTPException(400, "User email is required")
        async with self._service.read() as db:
            if await db.get_user_id_by_email(normalized_email):
                raise HTTPException(409,
                                    f"User already exists: {normalized_email}")
        password_credential = None
        if password is not None:
            password_credential = await _temporary_password_credential_async(
                password, normalized_email)
        async with self._service.write() as db:
            if await db.get_user_id_by_email(normalized_email):
                raise HTTPException(409,
                                    f"User already exists: {normalized_email}")
            user_id = _local_user_id()
            record = DirectConnectUserRecord(
                timestamps=get_current_timestamps(),
                email=normalized_email,
                display_name=display_name,
                role=role,
                disabled=disabled,
            )
            await db.set_user(user_id, record)
            if password_credential is not None:
                await db.set_password_credential(user_id, password_credential)
            return direct_connect_user_to_response(user_id, record)

    async def set_temporary_password(
        self,
        user_id: str,
        password: str,
    ) -> DirectConnectUserResponse:
        async with self._service.read() as db:
            record = await db.get_user(user_id)
            if record is None:
                raise HTTPException(404, f"User not found: {user_id}")
        password_credential = await _temporary_password_credential_async(
            password, record.email)
        async with self._service.write() as db:
            record = await db.get_user(user_id)
            if record is None:
                raise HTTPException(404, f"User not found: {user_id}")
            await db.set_password_credential(
                user_id,
                password_credential,
            )
            return direct_connect_user_to_response(user_id, record)

    async def password_login(
        self,
        *,
        email: str,
        password: str,
        reset_expires_time: datetime,
    ) -> tuple[DirectConnectPasswordLoginStatus, Optional[str], str,
               DirectConnectUserRecord]:
        normalized_email = normalize_direct_connect_email(email)
        async with self._service.read() as db:
            user_id = await db.get_user_id_by_email(normalized_email)
            user = await db.get_user(user_id) if user_id else None
            credential = (await db.get_password_credential(user_id)
                          if user_id else None)

        if not user_id or user is None or credential is None:
            await _verify_dummy_password_async(password)
            _raise_wrong_credentials()

        if not await _verify_password_async(password, credential):
            _raise_wrong_credentials()
        if user.disabled:
            _raise_wrong_credentials()
        if not credential.must_reset_password:
            return "ok", None, user_id, user

        reset_token = secrets.token_urlsafe(32)
        token_key = _password_reset_token_key(reset_token)
        async with self._service.write() as db:
            current_user = await db.get_user(user_id)
            current_credential = await db.get_password_credential(user_id)
            if (current_user is None or current_credential is None or
                    direct_connect_password_credential_tag(current_credential)
                    != direct_connect_password_credential_tag(credential)):
                _raise_wrong_credentials()
            if current_user.disabled:
                _raise_wrong_credentials()
            if not current_credential.must_reset_password:
                return "ok", None, user_id, current_user
            await db.set_password_reset_token(
                token_key,
                DirectConnectPasswordResetRecord(
                    timestamps=get_current_timestamps(),
                    user_id=user_id,
                    expires_time=reset_expires_time,
                    used=False,
                    credential_tag=direct_connect_password_credential_tag(
                        current_credential),
                ),
            )
            return "password_reset_required", reset_token, user_id, user

    async def complete_password_reset(
        self,
        *,
        reset_token: str,
        new_password: str,
    ) -> tuple[str, DirectConnectUserRecord]:
        token_key = _password_reset_token_key(reset_token)
        now = now_utc()
        async with self._service.read() as db:
            reset_record = await db.get_password_reset_token(token_key)
            if (reset_record is None or reset_record.used
                    or reset_record.expires_time <= now):
                raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                    RestStatusCode.AUTH_INVALID_TOKEN.value)
            user = await db.get_user(reset_record.user_id)
            if user is None:
                raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                    RestStatusCode.AUTH_INVALID_TOKEN.value)
            if user.disabled:
                raise HTTPException(HTTPStatus.FORBIDDEN,
                                    RestStatusCode.INVALID_PERMISSIONS.value)
            credential = await db.get_password_credential(reset_record.user_id)
        if (credential is not None
                and await _verify_password_async(new_password, credential)):
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                "New password must differ from the temporary password",
            )
        password_credential = await _password_credential_async(
            new_password,
            user.email,
            must_reset_password=False,
        )
        status, user_id, updated_user = await self._service.complete_password_reset(
            token_key=token_key,
            now=now,
            expected_credential=credential,
            credential=password_credential,
        )
        if status == "invalid_token":
            raise HTTPException(HTTPStatus.UNAUTHORIZED,
                                RestStatusCode.AUTH_INVALID_TOKEN.value)
        if status == "disabled":
            raise HTTPException(HTTPStatus.FORBIDDEN,
                                RestStatusCode.INVALID_PERMISSIONS.value)
        assert user_id is not None
        assert updated_user is not None
        return user_id, updated_user

    async def cleanup_expired_password_reset_tokens(self,
                                                    limit: int = 100) -> int:
        async with self._service.write() as db:
            return await db.delete_expired_password_reset_tokens(limit=limit)

    async def update_user(
        self,
        user_id: str,
        *,
        role: Optional[DirectConnectUserRole] = None,
        disabled: Optional[bool] = None,
    ) -> DirectConnectUserResponse:
        status, record = await self._service.update_user_if_not_last_admin(
            user_id,
            role=role,
            disabled=disabled,
        )
        if status == "not_found" or record is None:
            raise HTTPException(404, f"User not found: {user_id}")
        if status == "last_admin":
            raise HTTPException(400, "Cannot disable or demote last admin")
        return direct_connect_user_to_response(user_id, record)

    async def create_session(
        self,
        *,
        user_id: str,
        expires_time: datetime,
    ) -> str:
        session_id = uuid.uuid4().hex + uuid.uuid4().hex
        async with self._service.write() as db:
            await db.set_session(
                session_id,
                DirectConnectSessionRecord(
                    timestamps=get_current_timestamps(),
                    user_id=user_id,
                    expires_time=expires_time,
                ),
            )
        return session_id

    async def get_session(
            self, session_id: str) -> Optional[DirectConnectSessionRecord]:
        return await self._service.get_session(session_id)

    async def delete_session(self, session_id: str) -> None:
        await self._service.pop_session(session_id)

    async def cleanup_expired_sessions(self, limit: int = 100) -> int:
        async with self._service.write() as db:
            return await db.delete_expired_sessions(limit=limit)
