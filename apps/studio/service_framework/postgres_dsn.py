# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured Postgres connection parsing and DSN formatters.

This module is intentionally a dependency leaf: it uses only the stdlib
(``dataclasses``, ``urllib.parse``) so it can be imported by the
StateDB resolver without pulling in heavier dependencies.

The flow is: parse a secret mapping into a structured :class:`PostgresConn`,
then format it for the runtime driver. TLS is expressed as URL query params.
"""
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import quote, urlencode

_SUPPORTED_SSLMODES = {'verify-ca', 'verify-full'}
_STRUCTURED_SECRET_EXAMPLE = (
    '{"username":"kumo","password":"...","host":"postgres.example.com",'
    '"port":"5432","database":"kumo","tls":{"sslmode":"verify-full",'
    '"sslrootcert":"/path/to/ca.pem"}}')


@dataclass(frozen=True)
class PostgresConn:
    """Structured Postgres connection parameters.

    Plain stdlib dataclass (not ``kumo_dataclass``) to keep this module a
    dependency leaf.
    """
    username: str
    password: Optional[str]
    host: str
    port: Optional[str]
    database: str
    sslmode: Optional[str]
    sslrootcert: Optional[str]


def _value(secret: Mapping[str, object], key: str) -> Optional[str]:
    value = secret.get(key)
    return str(value) if value else None


def _parse_tls(secret: Mapping[str, object],
               secret_name: str) -> tuple[Optional[str], Optional[str]]:
    """Validate and extract (sslmode, sslrootcert) from a secret's tls block."""
    tls = secret.get('tls')
    if not tls:
        return None, None
    if not isinstance(tls, Mapping):
        raise ValueError(
            f'Postgres secret "{secret_name}" has invalid tls config: '
            f'expected an object like {{"sslmode":"verify-full",'
            f'"sslrootcert":"/path/to/ca.pem"}}.')

    sslmode = _value(tls, 'sslmode')
    if sslmode not in _SUPPORTED_SSLMODES:
        raise ValueError(
            f'Postgres secret "{secret_name}" has invalid tls.sslmode '
            f'{sslmode!r}: expected "verify-ca" or "verify-full". Use '
            f'"verify-full" when the Postgres hostname matches the server '
            f'certificate.')

    sslrootcert = _value(tls, 'sslrootcert')
    return sslmode, sslrootcert


def postgres_conn_from_secret(
    secret: Mapping[str, object],
    secret_name: str,
    *,
    default_database: Optional[str] = None,
) -> PostgresConn:
    """Parse structured secret fields into a :class:`PostgresConn`.

    Validates required fields (username, host, database) and the optional TLS
    block (only ``verify-ca`` / ``verify-full`` sslmode allowed; optional
    ``sslrootcert``).
    """
    username = _value(secret, 'username')
    password = _value(secret, 'password')
    host = _value(secret, 'host')
    database = _value(secret, 'database') or default_database
    port = _value(secret, 'port')

    missing = [
        name for name, value in {
            'username': username,
            'host': host,
            'database': database,
        }.items() if not value
    ]
    if missing:
        raise ValueError(
            f'Postgres secret "{secret_name}" is missing required structured '
            f'field(s): {", ".join(missing)}. Expected shape: '
            f'{_STRUCTURED_SECRET_EXAMPLE}.')

    assert username is not None
    assert host is not None
    assert database is not None

    sslmode, sslrootcert = _parse_tls(secret, secret_name)

    return PostgresConn(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )


def _userinfo(conn: PostgresConn) -> str:
    userinfo = quote(conn.username, safe='')
    if conn.password is not None:
        userinfo += f':{quote(conn.password, safe="")}'
    return userinfo


def _hostport(conn: PostgresConn) -> str:
    if conn.port:
        return f'{conn.host}:{conn.port}'
    return conn.host


def _tls_query_suffix(conn: PostgresConn) -> str:
    if not conn.sslmode:
        return ''
    params = {'sslmode': conn.sslmode}
    if conn.sslrootcert:
        params['sslrootcert'] = conn.sslrootcert
    return f'?{urlencode(params)}'


def _format_with_tls_query(conn: PostgresConn, scheme: str) -> str:
    return (f'{scheme}://{_userinfo(conn)}@{_hostport(conn)}/'
            f'{quote(conn.database, safe="")}{_tls_query_suffix(conn)}')


def postgresql_dsn(conn: PostgresConn) -> str:
    """Format a ``postgresql://`` DSN with TLS as URL query params.

    Reproduces the legacy ``postgres_dsn_from_secret(..., async_driver=False)``
    output byte-for-byte. Used for the raw-asyncpg State DB.
    """
    return _format_with_tls_query(conn, 'postgresql')


def psycopg2_dsn(conn: PostgresConn) -> str:
    """Format a ``postgresql+psycopg2://`` DSN with TLS as URL query params."""
    return _format_with_tls_query(conn, 'postgresql+psycopg2')
