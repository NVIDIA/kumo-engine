# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from cryptography.hazmat.primitives import serialization


def load_private_key(pem: str, passphrase: Optional[str]) -> Any:
    return serialization.load_pem_private_key(
        pem.replace('\\n', '\n').encode('utf-8'),
        password=passphrase.encode('utf-8') if passphrase else None,
    )


def private_key_bytes(key: Any, encoding: serialization.Encoding) -> bytes:
    return key.private_bytes(
        encoding=encoding,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
