# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache a local Snowflake RSA key in the Linux user keyring for eight hours."""
import ctypes
import ctypes.util
import getpass
import hashlib
import os
import stat
import sys
import warnings
from pathlib import Path

from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)

USER_KEYRING = -4
CACHE_SECONDS = 8 * 60 * 60


def library():
    name = ctypes.util.find_library('keyutils')
    if not name:
        raise RuntimeError(
            'Linux keyring library unavailable; no disk cache will be used')
    lib = ctypes.CDLL(name, use_errno=True)
    for symbol, arguments, result in [
        ('keyctl_join_session_keyring', [ctypes.c_char_p], ctypes.c_int32),
        ('keyctl_link', [ctypes.c_int32, ctypes.c_int32], ctypes.c_int),
        ('request_key',
         [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
          ctypes.c_int32], ctypes.c_int32),
        ('add_key', [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int32
        ], ctypes.c_int32),
        ('keyctl_read', [ctypes.c_int32, ctypes.c_void_p,
                         ctypes.c_size_t], ctypes.c_ssize_t),
        ('keyctl_set_timeout', [ctypes.c_int32, ctypes.c_uint], ctypes.c_int),
        ('keyctl_revoke', [ctypes.c_int32], ctypes.c_int),
        ('keyctl_unlink', [ctypes.c_int32, ctypes.c_int32], ctypes.c_int),
    ]:
        function = getattr(lib, symbol)
        function.argtypes = arguments
        function.restype = result
    # Possession permits timeout management; isolate this process from inherited sessions.
    if lib.keyctl_join_session_keyring(None) < 0 or lib.keyctl_link(
            USER_KEYRING, -3) < 0:
        raise RuntimeError(
            'Could not access the Linux user keyring; no disk cache will be used'
        )
    return lib


def load_der(key_file, *, unlock=False):
    key_file = Path(key_file)
    with key_file.open('rb') as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid(
        ) or metadata.st_mode & 0o077:
            raise RuntimeError(
                'Private key must be an owner-only regular file owned by the current user'
            )
        encrypted = stream.read()
    description = ('kumo-snowflake-local:' +
                   hashlib.sha256(encrypted).hexdigest()).encode()
    lib = library()
    serial = lib.request_key(b'user', description, None, 0)
    if serial >= 0:
        size = lib.keyctl_read(serial, None, 0)
        if 0 < size <= 32768:
            buffer = ctypes.create_string_buffer(size)
            actual = lib.keyctl_read(serial, buffer, size)
            if actual == size:
                return buffer.raw
    if not unlock:
        raise RuntimeError(
            'Local RSA cache is locked or expired. Run key_cache.py --key-file KEY once in your terminal.'
        )
    if not sys.stdin.isatty():
        raise RuntimeError(
            'Unlock requires an interactive terminal; passphrase input will not be echoed'
        )
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        passphrase = getpass.getpass(
            'Private-key passphrase (cached key expires in 8 hours): ')
    try:
        key = load_pem_private_key(encrypted, password=passphrase.encode()
                                   or None)
    finally:
        del passphrase
    der = key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
    payload = ctypes.create_string_buffer(der)
    serial = lib.add_key(b'user', description, payload, len(der), USER_KEYRING)
    if serial < 0:
        raise RuntimeError(
            'Could not cache key in Linux keyring; no disk cache was written')
    if lib.keyctl_set_timeout(serial, CACHE_SECONDS) < 0:
        revoked = lib.keyctl_revoke(serial) == 0
        lib.keyctl_unlink(serial, USER_KEYRING)
        if not revoked:
            raise RuntimeError(
                'Could not set key expiry or revoke cache; end the local session and investigate keyring access'
            )
        raise RuntimeError('Could not set key expiry; cached key revoked')
    return der


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, required=True)
    args = parser.parse_args()
    load_der(args.key_file, unlock=True)
    print('Local RSA key cached in memory for up to 8 hours.')
