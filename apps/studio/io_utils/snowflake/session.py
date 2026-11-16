# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from collections.abc import Generator
from contextlib import contextmanager
from typing import Dict, Optional

from snowflake.connector.errors import Error as SnowflakeError
from snowflake.snowpark import Session

from io_utils import logging
from io_utils.retry import retry
from io_utils.snowflake.session_logging import ensure_logged_snowpark_session
from service_framework.deploy_env import (
    SNOWFLAKE_APP_SCHEMA,
    SNOWFLAKE_DATABASE,
    get_snowflake_connection_string,
)

logger = logging.getLogger(__name__)

#: Snowflake mounts a session OAuth token at this path for workloads running
#: inside a container. This module is its only consumer, so it lives here
#: rather than in the deployment-wide environment table.
SNOWFLAKE_TOKEN_FILE_PATH: str = '/snowflake/session/token'


def _get_login_token() -> Optional[str]:
    if not osp.exists(SNOWFLAKE_TOKEN_FILE_PATH):
        return None
    with open(SNOWFLAKE_TOKEN_FILE_PATH, 'r') as f:
        return f.read()


def _build_fsutil_session_parameters() -> Optional[Dict]:
    # Initializes a Snowpark session in-container which should be used for
    # file-system operations on stage. This session will be running with
    # the privileges of the Native app.
    # This requires environment variables SNOWFLAKE_ACCOUNT and SNOWFLAKE_HOST
    # set, and assumes SNOWFLAKE_TOKEN_FILE_PATH is the standard location.
    # Hence, this can only be used from inside the container and not
    # in devserver (where a session is patched in for testing purposes).

    SNOWFLAKE_ACCOUNT: Optional[str] = os.getenv('SNOWFLAKE_ACCOUNT')
    SNOWFLAKE_HOST: Optional[str] = os.getenv('SNOWFLAKE_HOST')
    SNOWFLAKE_DATABASE: Optional[str] = os.getenv('SNOWFLAKE_DATABASE')
    SNOWFLAKE_SCHEMA: Optional[str] = os.getenv('SNOWFLAKE_SCHEMA')

    token = _get_login_token()
    if not token:
        return None

    connection_params = {
        "host": SNOWFLAKE_HOST,
        "account": SNOWFLAKE_ACCOUNT,
        "token": token,
        "authenticator": 'oauth',
        "client_session_keep_alive": True,
        "application": get_snowflake_connection_string(),
        "database": SNOWFLAKE_DATABASE,
        "schema": SNOWFLAKE_SCHEMA,
    }

    return connection_params


@retry(exceptions=SnowflakeError, tries=10, delay=1, logger=logger)
def _build_fsutil_session(connection_parameters: Optional[Dict] = None):
    logger.debug("Initializing Snowflake session")
    params = connection_parameters or _build_fsutil_session_parameters()
    if params is None:
        return None
    session = ensure_logged_snowpark_session(
        Session.builder.configs(params).create(),  # type: ignore
    )
    session.sql(
        "ALTER SESSION SET TIMESTAMP_TYPE_MAPPING=TIMESTAMP_NTZ").collect()

    return session


_fsutil_session: Session | None = None


def setup_fsutil_session(connection_parameters=None):
    global _fsutil_session
    _fsutil_session = _build_fsutil_session(connection_parameters)


def has_fsutil_session() -> bool:
    global _fsutil_session
    if _fsutil_session is None:
        _fsutil_session = _build_fsutil_session()
    return _fsutil_session is not None


def get_fsutil_session() -> Session:
    global _fsutil_session
    if _fsutil_session is None:
        _fsutil_session = _build_fsutil_session()
    assert _fsutil_session is not None
    # Actual object will have a wrapped type that is compatible with Session
    # Use the type hint so type checking still works as previously
    return _fsutil_session  # type: ignore


@contextmanager
def snowpark_caller_rights_session(
        ingress_user_token: str) -> Generator[Session]:
    login_token = _get_login_token()
    if not login_token:
        raise ValueError("No login token found")

    params = _build_fsutil_session_parameters()
    if params is None:
        raise ValueError("No connection parameters found")

    token = f"{login_token}.{ingress_user_token}"
    params["token"] = token
    params.pop("database", None)
    params.pop("schema", None)
    session = _build_fsutil_session(params)
    if session is None:
        raise ValueError("Failed to create caller rights session")

    try:
        yield session
    finally:
        session.close()


@contextmanager
def snowpark_session_in_app_schema(session: Session) -> Generator[Session]:
    """Snowpark sessions may run in:
    - App Context: <Native App Database>.<Native App Schema (KUMO_APP_SCHEMA)>
    - Session Context: <Native App Database>.<Session Schema (USER_SCHEMA)>
    - Connector Context: <Connector Database>.<Connector Schema>

    Temporarily run the session with App Context. This is where we have the
    broadest permissions, which may be required for implicit operations like
    creating temporary File Formats for Parquet operations.
    """
    if SNOWFLAKE_DATABASE is None or SNOWFLAKE_APP_SCHEMA is None:
        logger.warning(f"unable to set db.schema to match native app, "
                       f"got {SNOWFLAKE_DATABASE}.{SNOWFLAKE_APP_SCHEMA}")
        yield session
        return

    recorded_database = session.get_current_database()
    recorded_schema = session.get_current_schema()
    if recorded_database is None or recorded_schema is None:
        logger.warning(f"unable to record db.schema to revert to, "
                       f"got {recorded_database}.{recorded_schema}")
        yield session
        return

    try:
        session.use_database(SNOWFLAKE_DATABASE)
        session.use_schema(SNOWFLAKE_APP_SCHEMA)
        yield session
    finally:
        session.use_database(recorded_database)
        session.use_schema(recorded_schema)
