# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import os.path as osp
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Optional

from snowflake.snowpark import Session as SnowparkSession

from io_utils import fsutil
from io_utils.secrets import get_secrets
from kumo.connector import snowflake_connector
from kumo.connector.connector import Connector
from kumo.connector.snowflake_connector import SnowflakeConnector
# Production AND tests share the module-level Snowpark session cache in
# :mod:`kumo.connector.snowflake_session`.
from kumo.connector.snowflake_session import (
    close_all_sessions,
    get_or_build_session,
)
from kumo.data import reset_active_cache_for_test
from kumo.home import get_home_dir
from service_framework.deploy_env import ENV_KUMO_HOME
from service_framework.deploy_env_config import _reset_deploy_env


@dataclass
class SnowflakeTableConnector:
    table_name: str
    connector: snowflake_connector.SnowflakeConnector


def withSnowflakeCredentials(func: Callable):
    from kumo.testing._skip import skip_if

    try:
        secrets = get_secrets("xva19026.snowflakecomputing.com")
        os.environ[snowflake_connector._ENV_SNOWFLAKE_USER] = "kumo"
        os.environ[snowflake_connector.
                   _ENV_SNOWFLAKE_PRIVATE_KEY] = secrets["kumo_user_secretkey"]
    except Exception:
        # Use built-in credentials if this env doesn't have access to read the
        # secrets from AWS
        pass

    has_no_credentials = (
        not os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER)
        or not os.getenv(snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY))
    return skip_if(has_no_credentials, "No Snowflake credentials found", func)


@contextlib.contextmanager
def disable_snowflake_env_credentials() -> Generator[None, None, None]:
    """ Disables the ability to fetch snowflake credentials from the env.

    This is necessary for writing unit tests that test the case where
    environment credentials are not available.
    """
    old_input = Connector._DISABLE_INTERACTIVE_INPUT
    old_user = snowflake_connector._ENV_SNOWFLAKE_USER
    old_password = snowflake_connector._ENV_SNOWFLAKE_PASSWORD
    old_private_key = snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY
    try:
        snowflake_connector._ENV_SNOWFLAKE_USER = "_UNSET_ENV_VAR"
        snowflake_connector._ENV_SNOWFLAKE_PASSWORD = "_UNSET_ENV_VAR"
        snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY = "_UNSET_ENV_VAR"
        Connector.disable_interactive_input()
        yield
    finally:
        Connector._DISABLE_INTERACTIVE_INPUT = old_input
        snowflake_connector._ENV_SNOWFLAKE_USER = old_user
        snowflake_connector._ENV_SNOWFLAKE_PASSWORD = old_password
        snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY = old_private_key


# Primes the warehouse-flavored test environment: configures local data state
# and returns the shared cached Snowpark session from
# :mod:`kumo.connector.snowflake_session`. Optionally takes a connector so
# tests can pin a specific auth scope (the session cache key is derived from
# the connector).
@withSnowflakeCredentials
def setup_full_snowflake_session(
    tempdir,
    snow_connector: Optional[SnowflakeConnector] = None,
    persist_dir: Optional[str] = None,
    timezone: str = 'UTC',
) -> SnowparkSession:
    r"""Initialize the warehouse-flavored test environment and return an
    active Snowpark session.

    Snowflake is still supported as a data warehouse / source; only the
    Snowflake *compute* backend was removed. Compute now
    runs on DiskGraph, so this initializes the local persist directory and
    returns the cached Snowpark session used by both production code and
    tests.
    """
    fsutil.rm(osp.join(get_home_dir(), 'train_table'))
    # Drop any prior cached sessions so a fresh fixture run rebuilds the
    # Snowpark connection.
    close_all_sessions()
    if not snow_connector:
        account = 'xva19026'
        warehouse = 'WH_XS'
        database = 'KUMO'
        schema = 'FINANCIAL_PYTEST'
        snow_connector = SnowflakeConnector('snowflake', account=account,
                                            warehouse=warehouse,
                                            database=database,
                                            schema_name=schema)
    # TODO: Accept monkeypatch fixture to avoid polluting os.environ for
    # other tests in the same process. Point the deploy environment's storage
    # root at the test dir so kumo.data.get_persist_dir() (which now derives
    # the data persist dir from deploy_env) resolves under it, then drop any
    # cached BackendCacheManager so it re-roots.
    _persist = tempdir.name if persist_dir is None else persist_dir
    os.environ[ENV_KUMO_HOME] = _persist
    _reset_deploy_env()
    reset_active_cache_for_test()
    # Route through the module-level cache so production-side
    # ``_get_snowpark_df`` calls observe the same session this fixture
    # primed (cache key derived from the connector's auth scope).
    session = get_or_build_session(snow_connector)
    # Set session timezone to UTC to respect timezones in the data.
    if timezone:
        session.sql(f"ALTER SESSION SET TIMEZONE = '{timezone}'").collect()
    return session
