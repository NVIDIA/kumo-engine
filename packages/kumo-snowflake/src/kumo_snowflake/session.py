# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Opening a Snowflake connection for a connector, and talking to it.

The bottom of the package: everything above it -- the reader, the sink, the
staging helpers -- reaches a warehouse through this module and nothing else.
That is what keeps one identity rule for both directions of the connection. A
reader that resolved an identity differently from the writer beside it is the
drift this arrangement exists to prevent.

Identity-owned sessions are built here so the target is set at connect time
and no other caller moves their context. They are cached per (identity,
target), because warehouse, role, database and schema are baked into the
connection.

Without an explicit or backend-default identity, a connector's carried
Snowpark session or the process's active session is shared. Shared sessions
must match the connector's warehouse and optional role; database and schema
may differ for table access. Kumo never changes their context, closes them,
or reconnects them. Kumo's DDL commits any open transaction on a shared session.
"""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from weakref import WeakSet

from kumo_core.sources.principals import IdentityRef, Principal, resolve
from kumo_core.sources.sql import (
    SNOWFLAKE_UNQUOTED_IDENTIFIER,
    resolve_snowflake_identifier,
)
from kumo_snowflake.keys import load_private_key, private_key_bytes
from kumoapi.data_source import (
    SnowflakeAuthenticationKind,
    SnowflakeConnectorResourceConfig,
    SnowflakeIdentityConfig,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
)
from kumoapi.plugin import SourceConnector
from pydantic import SecretStr
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSessionException

#: What this module builds and hands upward.
SnowflakeSession = Session

#: Where Snowpark Container Services mounts the OAuth token, when an identity
#: names no path of its own.
DEFAULT_TOKEN_PATH = '/snowflake/session/token'

#: What ``Session.builder.configs`` is handed. Values are mixed because a
#: key pair reaches the driver as DER bytes while everything else is a string.
_ConnectSettings = Dict[str, Union[str, bytes, bool]]

#: A target reduced to something hashable, so two connectors naming the same
#: warehouse, database, schema and role share one connection.
_TargetKey = Tuple[Tuple[str, str], ...]

#: Identity name and target -- what a connection *is*. Nothing about the
#: credential's freshness belongs here: a session already open authenticates as
#: the same principal after a rotation as before, so the identity area closes
#: these by name instead, when it has a reason to.
_CacheKey = Tuple[str, _TargetKey]


def _target_config(
        connector: SourceConnector) -> SnowflakeConnectorResourceConfig:
    r"""Narrows a connector's target to this provider's own.

    A host routes on ``connector.source_type``, so in practice this always
    holds. Checking it is what turns a host that got that wrong into a
    sentence naming what arrived, rather than an attribute error from
    somewhere further in -- the same reason the identity above narrows.
    """
    config = connector.config
    if not isinstance(config, SnowflakeConnectorResourceConfig):
        raise ValueError(
            f"Connector {connector.id!r} points at a "
            f"{config.type.value} target, so it cannot be read as Snowflake.")
    return config


def _quoted(name: str) -> str:
    r"""Quotes an identifier for use in Snowflake SQL."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _require_snowpark(session: object) -> None:
    r"""Refuses anything that is not a Snowpark session.

    Staging and transfers require Snowpark's connection and session context.
    """
    if not isinstance(session, Session):
        raise TypeError(
            f"A Snowflake session must be a Snowpark 'Session'; got "
            f"{type(session).__name__!r}.")


def _rows(session: SnowflakeSession, sql: str,
          params: Optional[List[Any]] = None,
          timeout: Optional[int] = None) -> List[Any]:
    r"""Runs ``sql`` and returns its rows as sequences.

    Rows are normalized to tuples so that no driver type escapes this function:
    everything above it works in terms of the schema this module advertised,
    not in terms of what the client produced.

    ``timeout`` bounds the statement in the warehouse rather than the wait
    here, so a query that outlives it is cancelled instead of left running
    while this process stops waiting for it.
    """
    _require_snowpark(session)
    result = session.sql(sql, params=params) if params else session.sql(sql)
    if timeout is None:
        return [tuple(row) for row in result.collect()]
    return [
        tuple(row) for row in result.collect(
            statement_params={
                'STATEMENT_TIMEOUT_IN_SECONDS': str(int(timeout))
            })
    ]


def resolved(value: str) -> str:
    return resolve_snowflake_identifier(value)


def _session_identifier(name: Optional[str]) -> Optional[str]:
    """Use unquoted spelling only when it preserves the session's identifier."""
    if name is None:
        return None
    value = resolved(name)
    if (value == value.upper()
            and SNOWFLAKE_UNQUOTED_IDENTIFIER.fullmatch(value)):
        return value
    return name


def _private_key_der(pem: str, passphrase: Optional[str]) -> bytes:
    r"""Snowflake's driver takes a key-pair credential as DER bytes.

    ``cryptography`` ships with ``snowflake-connector-python``, so parsing
    here costs no dependency the caller did not already have.
    """
    from cryptography.hazmat.primitives import serialization
    return private_key_bytes(load_private_key(pem, passphrase),
                             serialization.Encoding.DER)


def _secret(value: Optional[SecretStr]) -> Optional[str]:
    r"""Unwraps a ``SecretStr`` at the one point the material is used."""
    return None if value is None else value.get_secret_value()


def _snowflake_config(principal: Principal) -> SnowflakeIdentityConfig:
    r"""Narrows an identity to this warehouse's own configuration.

    A connector dispatches on its source type, so a mismatch here means the
    identity named on it belongs to a different provider. Saying so beats the
    attribute error that reading an account off, say, a Databricks config
    would otherwise raise from somewhere further in.
    """
    config = principal.config
    if not isinstance(config, SnowflakeIdentityConfig):
        raise ValueError(
            f"Identity {principal.name!r} is a {config.type.value} "
            f"identity, so it cannot open a Snowflake connection.")
    return config


def _authentication(principal: Principal) -> _ConnectSettings:
    r"""The connect parameters that prove who this identity is.

    Split from the target parameters below because they answer different
    questions -- *who* rather than *where* -- and because only this half
    touches secret material.
    """
    config = _snowflake_config(principal)
    credentials = principal.credentials
    kind = config.authentication
    settings: _ConnectSettings = {'account': config.account}

    if kind is SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE:
        # Ambient by construction: the token is a file the container runtime
        # mounts, and the identity holds only where to read it. Read on every
        # connect rather than cached, because the runtime refreshes the file
        # in place and a token read once goes stale.
        path = config.token_path or DEFAULT_TOKEN_PATH
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                token = handle.read().strip()
        except OSError as error:
            raise ValueError(
                f"Identity {principal.name!r} authenticates with the "
                f"Snowflake OAuth token at {path}, which cannot be read: "
                f"{error.strerror}. This identity works only inside Snowpark "
                f"Container Services, which mounts that file; outside it, "
                f"create an identity with a key pair or a password.") from None
        # A full graph materialization can run without issuing a Snowflake
        # statement for several hours.  Keep the server session alive so the
        # final artifact import does not discover that its mounted OAuth
        # token has expired after all model work has completed.
        settings.update(token=token, authenticator='oauth',
                        client_session_keep_alive=True)
        if config.host:
            settings['host'] = config.host
        return settings

    if config.username is None:
        raise ValueError(f"Identity {principal.name!r} authenticates with "
                         f"{kind.value} but names no username.")
    settings['user'] = config.username
    if isinstance(credentials, SnowflakePasswordCredentials):
        settings['password'] = _secret(credentials.password) or ''
    elif isinstance(credentials, SnowflakeKeyPairCredentials):
        settings['private_key'] = _private_key_der(
            _secret(credentials.private_key) or '',
            _secret(credentials.private_key_passphrase))
    else:
        raise ValueError(
            f"Identity {principal.name!r} declares {kind.value} but holds "
            f"{type(credentials).__name__} credentials.")
    return settings


def _target(connector: Optional[SourceConnector]) -> Dict[str, str]:
    r"""The connect parameters that say *where* statements run.

    Identity-owned sessions select their context at connect time. Shared
    sessions are checked against the target without changing their context.
    """
    if connector is None:
        return {}
    config = _target_config(connector)
    # Connection parameters are identifier *expressions*. Retain delimiters
    # so the driver can distinguish an unquoted name (which Snowflake folds)
    # from a quoted, case-sensitive one. Exact catalog names are derived only
    # for cache identity and SQL/catalog work.
    settings: Dict[str, str] = {
        'database': config.database,
        'schema': config.schema_name,
        # Required by the config, so there is no "no warehouse" case to carry:
        # a connector that named none could not have been constructed.
        'warehouse': config.warehouse,
    }
    if config.role:
        settings['role'] = config.role
    return settings


def _target_key(connector: Optional[SourceConnector]) -> _TargetKey:
    # Cache equivalent unquoted spellings together, but keep case-sensitive
    # quoted identifiers distinct. This is catalog identity, unlike _target's
    # declared connection syntax.
    settings = {
        name: resolved(value)
        for name, value in _target(connector).items()
    }
    return tuple(sorted(settings.items()))


def _is_live(session: Session) -> bool:
    r"""Whether a cached session is still usable, without asking Snowflake.

    ``is_closed`` is the driver's own view, so this costs no round trip --
    which matters because it runs on every cache hit, and the cache exists
    precisely to avoid paying for a connection.

    It catches a session closed underneath us, not one Snowflake expired
    server-side while the driver still believes it holds it. The driver
    refreshes its own token, so that window is narrow; a caller who lands in
    it sees the reconnect error the source area already translates.
    """
    try:
        return not session.connection.is_closed()
    except Exception:
        # A session too broken to report on is not one to hand out.
        return False


class _SessionCache:
    r"""Live sessions, by identity and target.

    Module-level rather than per backend: the source and the sink are separate
    objects, and a reader that opened its own connection while the writer
    beside it opened another would double the connections an account sees for
    no gain.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[_CacheKey, Session] = {}

    def get(self, principal: Principal, connector: Optional[SourceConnector], *,
            fresh: bool = False) -> Session:
        key: _CacheKey = (principal.name, _target_key(connector))
        with self._lock:
            session = self._sessions.get(key)
            if session is not None and not fresh and _is_live(session):
                return session
            if session is not None:
                del self._sessions[key]
                try:
                    session.close()
                except Exception:
                    pass
            session = _connect(principal, connector)
            self._sessions[key] = session
            return session

    def close(self, name: Optional[str] = None) -> None:
        r"""Closes and forgets sessions, for one identity or for all.

        This module opens the connections, so it owns closing them. A caller
        that never asks still leaks nothing indefinitely: Snowflake expires an
        idle session on its own.
        """
        with self._lock:
            keys = [
                key for key in self._sessions if name is None or key[0] == name
            ]
            for key in keys:
                session = self._sessions.pop(key)
                try:
                    session.close()
                except Exception:
                    # Already gone server-side, most likely. Nothing here can
                    # act on it, and raising would mask the caller's own work.
                    pass


_sessions = _SessionCache()
# Track uncached retry/probe sessions as well as cached sessions.
_kumo_sessions: WeakSet[Session] = WeakSet()
_ownership_lock = threading.RLock()


def _require_shared_session(session: object) -> None:
    """Require a Snowpark session whose lifetime belongs to its caller."""
    _require_snowpark(session)
    with _ownership_lock:
        if session in _kumo_sessions:
            raise ValueError(
                'This Snowpark session is owned by Kumo and may be closed or '
                'refreshed. Supply an externally owned session, or keep using '
                'the identity-backed connector.')


def _active_session(*, hint: str = '') -> Session:
    try:
        session = get_active_session()
    except SnowparkSessionException as error:
        if error.error_code == '1409':
            raise ValueError(
                f'More than one active Snowpark session exists. {hint}'
                .rstrip()) from None
        if error.error_code != '1403':
            raise
        session = None
    with _ownership_lock:
        if session is None or session in _kumo_sessions:
            raise ValueError(
                'No Snowflake identity or shared Snowpark session is '
                f'available. {hint}'.rstrip())
    return session


def _check_shared_context(session: Session, connector: SourceConnector, *,
                          sql: bool = False) -> None:
    config = _target_config(connector)
    for field in ('warehouse', 'role', 'database', 'schema'):
        if field in ('database', 'schema') and not sql:
            continue
        expected = getattr(config,
                           'schema_name' if field == 'schema' else field)
        if expected is None:
            continue
        current = getattr(session, f'get_current_{field}')()
        if current is not None and resolved(expected) == resolved(current):
            continue
        expected = _session_identifier(expected)
        current = _session_identifier(current)
        if field in ('database', 'schema'):
            if current is None:
                reason = (
                    f'The shared Snowpark session has no current {field}; '
                    f'SQL sources require the connector {field} {expected!r}.')
            else:
                reason = (
                    f'SQL sources require the shared session {field} '
                    f'{current!r} to match the connector {field} {expected!r}.')
            remedy = ('Call session.use_database(...) and '
                      'session.use_schema(...)')
            hint = getattr(connector, '_snowpark_sql_hint', '')
            if hint:
                remedy += f', {hint}'
            raise ValueError(f'{reason} {remedy}.')
        if current is None:
            reason = (f'The shared Snowpark session has no current {field}; '
                      f'the connector requires {expected!r}.')
        else:
            reason = (f'Connector {field} {expected!r} differs from the shared '
                      f'Snowpark session {field} {current!r}.')
        raise ValueError(
            f'{reason} Switch the session {field}, or supply an identity for '
            'a separate session. '
            f"{getattr(connector, '_snowpark_mismatch_hint', '')}".rstrip())


@contextmanager
def _owned_session(principal: Principal,
                   connector: Optional[SourceConnector]) -> Iterator[Session]:
    """Open one uncached session and close it when the caller finishes.

    Recovery uses this instead of replacing a cache entry because another
    request can legitimately still be using that shared session.
    """
    session = _connect(principal, connector)
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception:
            pass


def _connect(principal: Principal,
             connector: Optional[SourceConnector]) -> Session:
    settings = {
        **_authentication(principal),
        **_target(connector),
        'application': 'nvidia/kumo-engine',
        'ocsp_fail_open': True,
    }
    try:
        with _ownership_lock:
            session = Session.builder.configs(settings).create()
            _kumo_sessions.add(session)
        return session
    except Exception as error:
        where = f' for {connector.id!r}' if connector is not None else ''
        raise RuntimeError(
            f"Could not open a Snowflake session{where} with identity "
            f"{principal.name!r}: {error}") from error


def probe(principal: Principal) -> None:
    r"""Opens a connection for :obj:`principal` and closes it again.

    Half of the ``kumoai_studio.identities`` lifecycle: what proves a credential
    works, rather than reporting what was configured.

    Deliberately not cached: this is asked when a caller wants to know whether
    the credentials work *now*, so reusing a session opened earlier would
    answer a question nobody asked.
    """
    session = _connect(principal, None)
    try:
        session.sql('SELECT 1').collect()
    finally:
        try:
            session.close()
        except Exception:
            # The answer this call exists for is whether `SELECT 1` ran, and
            # that is already settled. A session that will not close cleanly
            # expires server-side on its own, and raising here would replace a
            # credential verdict with a teardown complaint.
            pass


def close(identity_name: Optional[str] = None) -> None:
    r"""Closes the sessions opened as :obj:`identity_name`, or all of them.

    The other half of the ``kumoai_studio.identities`` lifecycle, and the half only a
    plugin that caches defines. The identity area calls it when material is
    rotated or an identity is deleted -- neither of which this module can see
    for itself, and neither of which it is subscribed to.

    Rotation is not a correctness problem for a session already open: it
    authenticates as the same principal afterwards, since a rotation replaces
    material and never the config. What it is is a *recovery* problem. A caller
    who rotates has usually just been told a connection no longer
    authenticates, and answering the retry from that very connection would
    make the documented recovery impossible.
    """
    _sessions.close(identity_name)


class _SnowflakeSessions:
    r"""Resolving a connector to a live Snowpark session.

    Shared by the two areas that talk to a warehouse --
    :class:`~kumo_snowflake.backend.SnowflakeSourceBackend` and
    :class:`~kumo_snowflake.sink.SnowflakeSinkBackend` --
    because a session is resolved the same way whichever direction the data is
    moving. A reader that resolved identities differently from the writer
    beside it is the whole reason this class is not written twice.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts the identity object the SDK handed back, an
            identity ID, or the name it was created under.
    """

    #: How a subclass spells "supply a default identity" in its own terms. The
    #: message resolution raises is the only place a caller learns the
    #: constructor to reach for, and the reader and the sink are registered
    #: under different keyword arguments.
    _default_hint = ''

    def __init__(self, identity: Optional[IdentityRef] = None) -> None:
        self._identity = identity

    @contextmanager
    def _session(self, connector: SourceConnector, *,
                 fresh: bool = False,
                 sql: bool = False) -> Iterator[SnowflakeSession]:
        r"""Yields an identity-owned or context-checked shared session."""
        principal = self._principal(connector, optional=True)
        if principal is not None:
            yield _sessions.get(principal, connector, fresh=fresh)
            return
        session = getattr(connector, '_snowpark_session', None)
        if session is None:
            hint = getattr(connector, '_snowpark_session_hint', '')
            try:
                session = _active_session(hint=hint)
            except ValueError:
                if not hint:
                    self._principal(connector)
                raise
        _require_shared_session(session)
        _check_shared_context(session, connector, sql=sql)
        if fresh:
            raise ValueError(
                'A shared Snowpark session cannot be refreshed by Kumo; '
                'refresh it in the notebook runtime.')
        yield session

    def _principal(self, connector: SourceConnector, *,
                   optional: bool = False) -> Optional[Principal]:
        return resolve(
            (getattr(connector, 'identity', None),
             getattr(connector, 'identity_id', None), self._identity),
            connector=connector,
            optional=optional,
            config_example='SnowflakeIdentityConfig(...), credentials=...',
            connector_example='SnowflakeConnector(identity=identity, ...)',
            fallback=(f'A default for sources that name none is also '
                      f'accepted, as {self._default_hint}.'),
        )

    def table_location(self, connector: SourceConnector,
                       table_name: str) -> str:
        """``account/"DB"."SCHEMA"."TABLE"``; the account when it resolves."""
        try:
            principal = self._principal(connector)
            assert principal is not None
            account = _snowflake_config(principal).account
        except Exception:
            account = None
        qualified = self._qualified(connector, table_name)
        return f'{account}/{qualified}' if account else qualified

    @staticmethod
    def _qualified(connector: SourceConnector, table_name: str) -> str:
        return (f'{_quoted(resolved(_target_config(connector).database))}.'
                f'{_quoted(resolved(_target_config(connector).schema_name))}.'
                f'{_quoted(table_name)}')
