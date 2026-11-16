# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from base64 import b64decode, b64encode, urlsafe_b64encode

from cryptography.fernet import Fernet
from pydantic import SecretStr

from io_utils import logging
from io_utils.secrets import get_secrets
from service_framework.deploy_env_config import deploy_env
from service_framework.deploy_stage import DeployStage

logger = logging.getLogger(__name__)

# Constant, default encryption key that's only meant for dev/test
_DEFAULT_ENCRYPTION_KEY = urlsafe_b64encode(
    hashlib.md5("kumo-default-encryption-key".encode()).hexdigest().encode())


class GlobalEncryptor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self) -> None:
        cfg = deploy_env()
        deploy_mode = cfg.deploy_mode
        # Default, unsecure encryption key for dev/test purpose only.
        if deploy_mode != DeployStage.PROD:
            logger.warning('Using default encryption key outside production')
            encryption_key: bytes = _DEFAULT_ENCRYPTION_KEY
        else:
            secret_name = os.getenv('FERNET_SECRET_NAME', 'kumo-secret-key')
            secrets = get_secrets(secret_name)
            encryption_key: bytes = secrets[
                'credentials_encryption_key'].encode()
        try:
            self._cipher = Fernet(encryption_key)
        except ValueError as err:
            raise ValueError('Invalid credentials encryption key') from err

    def encrypt(self, secret: str) -> str:
        return b64encode(self._cipher.encrypt(
            secret.encode('utf-8'))).decode('utf-8')

    def decrypt(self, encrypted: str) -> str:
        try:
            return self._cipher.decrypt(b64decode(encrypted)).decode('utf-8')
        except Exception as ex:  # noqa: E722
            logger.error('Failed to decrypt[%s]: %s', type(ex), ex)
            raise ValueError(
                f'Failed to decrypt secret ({type(ex)}): {ex}') from ex


class EncryptedStr(SecretStr):
    PREFIX = '__encrypted__'

    def __init__(self, value: str | SecretStr):
        value_str = (value
                     if isinstance(value, str) else value.get_secret_value())
        if value_str.startswith(self.PREFIX):
            value_str = GlobalEncryptor().decrypt(value_str[len(self.PREFIX):])

        return super().__init__(value_str)

    # v1 used __get_validators__; v2 uses __get_pydantic_core_schema__.
    # ``_unwrap`` runs as a *before* validator so it sees the raw input (a
    # plain str, an EncryptedStr, or any SecretStr) and normalizes it to the
    # plain secret value BEFORE the inner ``str_schema()`` runs. (Wrapping
    # str_schema with an *after* validator instead would reject an already
    # wrapped SecretStr -- it is not a str to pydantic-core -- so the unwrap
    # branch was unreachable on model-from-model re-validation.) The result
    # is re-wrapped into an EncryptedStr by the after validator. A
    # field/plain serializer restores SecretStr's masking so model_dump does
    # not leak the plaintext (json mode emits the mask, like SecretStr).
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        str_schema = core_schema.no_info_before_validator_function(
            cls._unwrap, core_schema.str_schema())
        return core_schema.no_info_after_validator_function(
            cls._wrap,
            str_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=True, when_used='always'),
        )

    @classmethod
    def _unwrap(cls, value):
        """Normalize any accepted input to the plain secret ``str``."""
        if value is None:
            return None
        # An already-wrapped EncryptedStr / SecretStr (model-from-model
        # re-validation) is unwrapped to its secret value here so the inner
        # str schema accepts it.
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        if isinstance(value, str):
            return value
        raise TypeError(f"Unsupported value for EncryptedStr: {value!r}")

    @classmethod
    def _wrap(cls, value):
        if value is None:
            return None
        return cls(value)

    @staticmethod
    def _serialize(value, info):
        # Preserve SecretStr-style masking. Mirrors pydantic SecretStr: json
        # mode emits the masked display (never the plaintext); python mode
        # keeps the EncryptedStr object so callers that re-validate or call
        # ``as_encrypted()`` still get the real value.
        if info.mode == 'json':
            return str(value) if isinstance(value, SecretStr) else value
        return value

    def as_encrypted(self) -> str:
        secret_value = self.get_secret_value()
        return f'{self.PREFIX}{GlobalEncryptor().encrypt(secret_value)}'
