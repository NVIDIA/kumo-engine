# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Snowpark Session helper.

Builds a Snowpark ``Session`` directly from a :class:`SnowflakeConnector`
and provides a module-level session cache keyed by an auth-scope identity
(:class:`CredentialsKey`). Production and test callers both go through
:func:`get_or_build_session`.

Used by :meth:`SnowflakeConnector.aggregate` for one-round-trip SQL
pushdown of pre-ingest col-stats, and by ``kumo/testing/snowflake.py``
for test-fixture setup.
"""
from typing import Any, Dict, NamedTuple, Optional

from cryptography.hazmat.primitives import serialization
from snowflake.snowpark import Session as SnowparkSession

from io_utils.snowflake.session_logging import ensure_logged_snowpark_session
from kumo.connector import snowflake_connector
from kumo.connector.snowflake_connector import SnowflakeConnector
from service_framework.deploy_env import get_snowflake_connection_string


def build_snowpark_session_from_connector(
    snow_connector: SnowflakeConnector, ) -> SnowparkSession:
    r"""Build a Snowpark ``Session`` directly from a :class:`SnowflakeConnector`.

    Reuses the connector's credential helper to assemble the
    connection-parameters dict, then calls
    ``Session.builder.configs(...).create()``.
    """
    # Mirror SnowflakeConnector._open: pass the connector-stashed credentials
    # dict so keypair-auth-via-set_credentials (user + private_key) propagates,
    # falling back to env vars only when set_credentials was never called.
    auth_params: Dict[str, Any] = snow_connector.get_user_pass_or_key(
        snow_connector.__repr__(),
        snowflake_connector._ENV_SNOWFLAKE_USER,
        snowflake_connector._ENV_SNOWFLAKE_PASSWORD,
        credentials=getattr(snow_connector, '_credentials', None),
        key_encoding=serialization.Encoding.DER,
        env_var_private_key=snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY,
    )
    connection_parameters: Dict[str, Any] = {
        **auth_params,
        'account': snow_connector.account,
        # The driver receives declared identifier syntax so quotes retain
        # their semantic distinction. Cache identity below uses exact names.
        'warehouse': snow_connector.warehouse,
        'database': snow_connector.database,
        'schema': snow_connector.schema_name,
        'client_session_keep_alive': True,
        'application': get_snowflake_connection_string(),
    }
    if snow_connector.role is not None:
        connection_parameters['role'] = snow_connector.role
    return ensure_logged_snowpark_session(
        SnowparkSession.builder.configs(connection_parameters).create(), )


class CredentialsKey(NamedTuple):
    r"""Hashable identity of a Snowpark session's auth+access scope.

    Two :class:`SnowflakeConnector` instances map to the same cache
    entry iff they would build a session with equivalent (account,
    user, role, warehouse, database, schema, auth-method,
    auth-source) — i.e., the same effective Snowflake context.
    Includes ``auth_method`` + an ``auth_source_id`` (the connector
    name, which doubles as the env-var-fallback identity) but NEVER
    raw credential material; tokens rotate, secrets shouldn't be
    cached as keys.
    """
    account: str
    user: Optional[str]
    role: Optional[str]
    warehouse: str
    database: Optional[str]
    schema_name: Optional[str]
    auth_method: str
    auth_source_id: str


# Module-level Snowpark session cache. Keyed by
# :class:`CredentialsKey` so multiple :class:`SnowflakeConnector`
# instances with the same auth scope share one session.
_SESSION_CACHE: Dict[CredentialsKey, SnowparkSession] = {}


def _credentials_key(connector: SnowflakeConnector) -> CredentialsKey:
    r"""Derive the cache key from a connector's auth/access fields.

    Mirrors the fields read by
    :func:`build_snowpark_session_from_connector`: account / warehouse /
    database / schema_name / role come from the connector directly; user
    + auth-method come from
    :meth:`SnowflakeConnector.get_user_pass_or_key`.
    """
    # Resolve user + auth-method by replaying the connector credential
    # lookup. We don't store the secret material in the key — only an
    # ``auth_method`` discriminator (private_key vs password) so two
    # connectors that differ only in auth method don't share a session.
    try:
        auth_params: Dict[str, Any] = connector.get_user_pass_or_key(
            connector.__repr__(),
            snowflake_connector._ENV_SNOWFLAKE_USER,
            snowflake_connector._ENV_SNOWFLAKE_PASSWORD,
            credentials=getattr(connector, '_credentials', None),
            key_encoding=serialization.Encoding.DER,
            env_var_private_key=snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY,
        )
    except Exception:
        # If credentials can't be resolved we fall back to a key that
        # still identifies the access scope; the subsequent
        # ``build_snowpark_session_from_connector`` call inside
        # ``get_or_build_session`` will surface the real error.
        auth_params = {}
    user = auth_params.get('user')
    if 'private_key' in auth_params:
        auth_method = 'private_key'
    elif 'password' in auth_params:
        auth_method = 'password'
    else:
        auth_method = 'unknown'
    # Use the connector name as the auth-source identifier (it doubles
    # as the env-var fallback ID per the existing connector contract).
    auth_source_id = getattr(connector, 'name', '') or ''
    return CredentialsKey(
        account=connector.account,
        user=user,
        role=connector.resolved_role,
        warehouse=connector.resolved_warehouse,
        database=connector.resolved_database,
        schema_name=connector.resolved_schema,
        auth_method=auth_method,
        auth_source_id=auth_source_id,
    )


def get_or_build_session(connector: SnowflakeConnector, ) -> SnowparkSession:
    r"""Return a cached or freshly-built Snowpark session for ``connector``.

    Production and test callers both go through here.
    """
    key = _credentials_key(connector)
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = build_snowpark_session_from_connector(connector)
    return _SESSION_CACHE[key]


def close_all_sessions() -> None:
    r"""Close all cached Snowpark sessions. Test-teardown / atexit only.

    Must NEVER be called mid-activity — would break in-flight
    ``_get_snowpark_df`` reads. Production deployments rely on the
    cache living for the duration of the worker process.
    """
    for sess in list(_SESSION_CACHE.values()):
        try:
            sess.close()
        except Exception:
            pass
    _SESSION_CACHE.clear()
