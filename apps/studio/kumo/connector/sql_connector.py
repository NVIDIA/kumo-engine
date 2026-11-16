# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from cryptography.hazmat.primitives import serialization
from pydantic import SecretStr

from io_utils import logging
from kumo.config import BaseColumnConfig
from kumo.connector import Connector
from kumo.connector.blocking import TraceOptions, blocking_method
from kumo.connector.connector import ConnectorError
from kumo.connector.utils import cast_df
from kumo.data.common import DATA_BACKEND_THREAD_POOL

logger = logging.getLogger(__name__)


# This needs to be in sync with the WriteMode enum in the REST API.
# The canonical alias is ``rest.config.jobs.WriteMode``.
class InternalWriteMode(Enum):
    OVERWRITE = 'overwrite'
    APPEND = 'append'


class SQLConnector(Connector):
    r"""Base class for connecting to various SQL data sources."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def _get_connector_type(self) -> str:
        return "Sql"

    def __getstate__(self):
        # Defer to the base Connector to strip process-local
        # synchronization primitives (e.g. `_open_lock`, a `threading.RLock`
        # that cannot be pickled). Then drop `_context` so the live DB
        # connection doesn't travel across the pickle boundary.
        state = super().__getstate__()
        state.pop("_context", None)
        return state

    def __setstate__(self, state):
        # Defer to the base Connector so it recreates the stripped
        # `_open_lock` on the revived instance.
        super().__setstate__(state)

    def _close(self, context: Any):
        context.close()

    # Abstract methods ########################################################

    @abstractmethod
    def _get_df_result(self, context: Any, query: str) -> pd.DataFrame:
        r"""
        Execute the query and return result in a dataframe
        """

    # Reading table data ######################################################

    def skip_list_table(self) -> bool:
        return True

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=['table_name']))
    def _read_table(
        self,
        context: Any,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        query = self.get_query(table_name, pkey, cols)
        return self.execute_query(context, query, pkey, cols)

    def _read_table_chunk(
        self,
        context: Any,
        table_name: str,
        num_rows: int,
        offset: int = 0,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        query = self.get_query(table_name, pkey, cols, limit=num_rows,
                               offset=offset)
        return self.execute_query(context, query, pkey, cols)

    def _read_table_sample(
        self,
        context: Any,
        table_name: str,
        num_rows: Optional[int],
        frac: Optional[float],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        if not num_rows and not frac:
            raise ValueError(
                "Connector requires num_rows or frac to read a table "
                "sample.")
        query = self.get_query(table_name, pkey, cols, limit=num_rows,
                               frac=frac, sample=True)
        return self.execute_query(context, query, pkey, cols)

    # Data statistics #########################################################

    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:
        query = f'SELECT COUNT(*) FROM {self.sql_table_name(table_name)}'
        df = self.execute_query(self.context, query)
        return df.iloc[0, 0].item()

    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        # TODO(Dulloor): Implement for SQL databases
        return 0

    # Utility functions #######################################################

    def get_user_pass_or_key(
            self, warehouse: str, env_var_user: str, env_var_pass: str,
            credentials: Optional[Dict[str, SecretStr]] = None,
            key_encoding: Optional[serialization.Encoding] = None,
            env_var_private_key: Optional[str] = None):
        r"""
        Used by _open in subclasses to get username and password for a
        given warehouse, or username and private key (decoded).
        """
        credentials = credentials or {}
        user: Optional[SecretStr] = credentials.get('user',
                                                    os.getenv(env_var_user))
        # Extract the secret values
        if user is not None and isinstance(user, SecretStr):
            user: str = user.get_secret_value()

        password: Optional[SecretStr] = credentials.get('password')
        if password is None:
            # no credential password -> try private key
            private_key: Optional[SecretStr] = credentials.get('private_key')
            if private_key is None and env_var_private_key:
                # Try to get private key from environment variable
                private_key = os.getenv(env_var_private_key)
                if not private_key:
                    logger.info("Attempted to load private key from env var, "
                                f"{env_var_private_key} not set.")
            if private_key:
                if isinstance(private_key, SecretStr):
                    private_key: str = private_key.get_secret_value()

                from kumo_snowflake.keys import (
                    load_private_key,
                    private_key_bytes,
                )

                pkey_pass = credentials.get('private_key_passphrase')
                if pkey_pass:
                    if isinstance(pkey_pass, SecretStr):
                        pkey_pass: str = pkey_pass.get_secret_value()
                try:
                    pkey_serialized = load_private_key(private_key, pkey_pass)
                except Exception:
                    # exception from cryptography package doesn't provide
                    # any meaningful information on why key is invalid
                    raise ConnectorError(
                        "The private key provided is invalid or in an "
                        "unsupported format. Please ensure the key is in PEM "
                        "format (starts with '-----BEGIN PRIVATE KEY-----' or "
                        "'-----BEGIN RSA PRIVATE KEY-----'). If the key is "
                        "encrypted, provide the passphrase via the "
                        "'private_key_passphrase' credential field.")

                if not user:
                    raise ConnectorError(
                        f"Private key provided for connecting to {warehouse} "
                        "but also need a username passed in via credentials. "
                        "Alternatively, provide username and password in the "
                        f"environment variables `{env_var_user}` and "
                        f"`{env_var_pass}`.")
                if key_encoding:
                    pkey_serialized = private_key_bytes(pkey_serialized,
                                                        key_encoding)

                return {'user': user, 'private_key': pkey_serialized}
            else:
                # no credential password or private key -> try env var
                password = os.getenv(env_var_pass)

        # either from credential arg or env:
        if password is not None:
            if isinstance(password, SecretStr):
                password: str = password.get_secret_value()

        # if user + private key auth, returned or raised error above
        if not user or not password:  # pragma: no cover
            if self._DISABLE_INTERACTIVE_INPUT:
                raise ConnectorError(
                    "No credentials were provided for connecting to "
                    f"{warehouse}. Please pass in credentials through "
                    "the constructor, or provide them in the environment "
                    f"variables `{env_var_user}` and `{env_var_pass}`.")
            import getpass
            user = input('Enter your username: ')
            password = getpass.getpass('Enter your password: ')

        return {'user': user, 'password': password}

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def execute_query(
        self,
        context: Any,
        query: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        r"""
        Executes a given SQL query string on the connector and returns a
        dataframe.
        """
        df = self._get_df_result(context, query)

        if pkey is not None:
            df.set_index(pkey.name, inplace=True)

        return cast_df(df, pkey, cols)

    def get_query(
        self,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        frac: Optional[float] = None,
        sample: Optional[bool] = False,
    ) -> str:
        r"""Generates and returns a SQL query."""
        if cols is None:
            select_query = '*'
        elif len(cols) == 0 and pkey is None:
            raise ValueError("Should specify at least some column or leave "
                             "cols as None to select all columns, but got "
                             "an empty list for cols and no pkey specified.")
        else:
            names = [col.name for col in cols]
            names += [pkey.name] if pkey is not None else []
            select_query = ', '.join(
                self.sql_column_name(name) for name in names)

        if sample:
            from kumo_core.custom_sql import sample_select
            if limit is None:
                raise ValueError('Source sampling requires a row limit.')
            query = sample_select(self.sql_table_name(table_name), limit,
                                  select_query)
            return f'{query} OFFSET {offset}' if offset > 0 else query
        query_body = f'SELECT {select_query} from ' \
                     f'{self.sql_table_name(table_name)}'
        if limit is not None and offset > 0:
            return f'{query_body} LIMIT {limit} OFFSET {offset}'
        elif limit is not None:
            return f'{query_body} LIMIT {limit}'
        else:
            return query_body

    def sql_table_name(self, table_name: str):
        return f'"{table_name}"'

    def sql_column_name(self, column_name: str) -> str:
        """Render a column in this connector's SQL dialect.

        Existing dialects retain their historical unquoted projection unless
        they override this hook. Providers whose exact catalog names require
        quoting, such as Snowflake, own that rule in their implementation.
        """
        return column_name

    def check_schema_exists(self, schema_name: str) -> None:
        raise NotImplementedError

    def check_table_write_permissions(
        self,
        table_name: str,
        table_schema: str,
        mode: InternalWriteMode,
    ) -> None:
        raise NotImplementedError

    def is_table_view(self, table_name: str) -> bool:
        r"""Check if the table is a view in the source system.

        Args:
            table_name (str): The name of the table.

        Returns:
            bool: True if the table is a view, False otherwise.
        """
        raise NotImplementedError

    def check_table_read_permissions(self, table_name: str) -> None:
        raise NotImplementedError
