# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable, Generator
from dataclasses import dataclass

from kumo.connector.connector import Connector
from kumo.connector.databricks_connector import DatabricksConnector
from service_framework.deploy_env import (
    DATABRICKS_CLIENT_ID,
    DATABRICKS_CLIENT_SECRET,
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
)


@dataclass
class DatabricksTableConnector:
    table_name: str
    connector: DatabricksConnector


def withDatabricksCredentials(func: Callable):
    """Skip unless DATABRICKS_HOST and DATABRICKS_TOKEN are set."""
    from kumo.testing._skip import skip_if
    has_no_credentials = not (DATABRICKS_HOST.parse_value()
                              and DATABRICKS_TOKEN.parse_value())
    return skip_if(has_no_credentials, "No Databricks credentials found", func)


def withDatabricksOauthCredentials(func: Callable):
    """Skip unless the host, client ID, and client secret variables are set."""
    from kumo.testing._skip import skip_if
    has_no_credentials = not (DATABRICKS_HOST.parse_value()
                              and DATABRICKS_CLIENT_ID.parse_value()
                              and DATABRICKS_CLIENT_SECRET.parse_value())
    return skip_if(has_no_credentials, "No Databricks credentials found", func)


@contextlib.contextmanager
def disable_databricks_env_credentials() -> Generator[None, None, None]:
    """ Disables the ability to fetch databricks credentials from the env.

    This is necessary for writing unit tests that test the case where
    environment credentials are not available.
    """
    old_input = Connector._DISABLE_INTERACTIVE_INPUT
    old_pat = DATABRICKS_TOKEN.name
    old_client_id = DATABRICKS_CLIENT_ID.name
    try:
        DATABRICKS_TOKEN.name = "_UNSET_ENV_VAR"
        DATABRICKS_CLIENT_ID.name = "_UNSET_ENV_VAR"
        Connector.disable_interactive_input()
        yield
    finally:
        Connector._DISABLE_INTERACTIVE_INPUT = old_input
        DATABRICKS_TOKEN.name = old_pat
        DATABRICKS_CLIENT_ID.name = old_client_id
