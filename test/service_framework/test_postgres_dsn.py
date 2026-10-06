# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the structured Postgres parser and per-driver formatters."""

import pytest

from service_framework.postgres_dsn import (
    PostgresConn,
    postgres_conn_from_secret,
    postgresql_dsn,
    psycopg2_dsn,
)

# --- parser ---------------------------------------------------------------


def test_parser_extracts_fields():
    conn = postgres_conn_from_secret(
        {
            'username': 'u',
            'password': 'p',
            'host': 'h',
            'port': '6543',
            'database': 'db',
        }, 'SECRET')
    assert conn == PostgresConn(
        username='u',
        password='p',
        host='h',
        port='6543',
        database='db',
        sslmode=None,
        sslrootcert=None,
    )


def test_parser_default_database_applied_when_missing():
    conn = postgres_conn_from_secret({
        'username': 'u',
        'host': 'h',
    }, 'SECRET', default_database='fallback')
    assert conn.database == 'fallback'


def test_parser_missing_required_fields_raises():
    with pytest.raises(ValueError, match='missing required structured'):
        postgres_conn_from_secret({'host': 'h', 'password': 'p'}, 'SECRET')


def test_parser_alias_fields_are_not_supported():
    with pytest.raises(ValueError, match='missing required structured'):
        postgres_conn_from_secret(
            {
                'user': 'u',
                'POSTGRES_HOST': 'h',
                'dbname': 'db',
            }, 'SECRET')


def test_parser_bad_sslmode_raises():
    with pytest.raises(ValueError, match='tls.sslmode'):
        postgres_conn_from_secret(
            {
                'username': 'u',
                'host': 'h',
                'database': 'db',
                'tls': {
                    'sslmode': 'require'
                },
            }, 'SECRET')


def test_parser_invalid_tls_config_raises():
    with pytest.raises(ValueError, match='invalid tls config'):
        postgres_conn_from_secret(
            {
                'username': 'u',
                'host': 'h',
                'database': 'db',
                'tls': 'not-a-mapping',
            }, 'SECRET')


def test_parser_tls_fields():
    conn = postgres_conn_from_secret(
        {
            'username': 'u',
            'host': 'h',
            'database': 'db',
            'tls': {
                'sslmode': 'verify-full',
                'sslrootcert': '/opt/kumo/secrets/postgres-ca.pem',
            },
        }, 'SECRET')
    assert conn.sslmode == 'verify-full'
    assert conn.sslrootcert == '/opt/kumo/secrets/postgres-ca.pem'


# --- formatters: TLS placement -------------------------------------------

_TLS_CONN = PostgresConn(
    username='u',
    password='p',
    host='host',
    port='6543',
    database='kumo',
    sslmode='verify-full',
    sslrootcert='/opt/kumo/secrets/postgres-ca.pem',
)
_NO_PORT_QUOTING_CONN = PostgresConn(
    username='user@example.com',
    password='p@ss word',
    host='postgres.example.com',
    port='6543',
    database='kumo',
    sslmode=None,
    sslrootcert=None,
)


def test_postgresql_dsn_tls_in_query():
    assert postgresql_dsn(_TLS_CONN) == (
        'postgresql://u:p@host:6543/kumo'
        '?sslmode=verify-full'
        '&sslrootcert=%2Fopt%2Fkumo%2Fsecrets%2Fpostgres-ca.pem')


def test_postgresql_dsn_url_quotes_credentials():
    assert postgresql_dsn(_NO_PORT_QUOTING_CONN) == (
        'postgresql://user%40example.com:p%40ss%20word'
        '@postgres.example.com:6543/kumo')


def test_psycopg2_dsn_tls_in_query():
    assert psycopg2_dsn(_TLS_CONN) == (
        'postgresql+psycopg2://u:p@host:6543/kumo'
        '?sslmode=verify-full'
        '&sslrootcert=%2Fopt%2Fkumo%2Fsecrets%2Fpostgres-ca.pem')
