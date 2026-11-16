# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
import urllib.parse as up
from typing import Any, Dict, Optional, Tuple, Union

import pyarrow
from pydantic import SecretStr
from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError

from io_utils import kumo_dataclass
from io_utils.errors import (
    WarehouseExtendedRetriableError,
    WarehouseMisconfigurationError,
    WarehouseRetriableError,
)
from io_utils.retry import retry
from io_utils.secrets import SecretAccessException, get_secrets
from service_framework.deploy_env import (
    DATABRICKS_CLIENT_ID,
    DATABRICKS_CLIENT_SECRET,
    DATABRICKS_CLUSTER_ID,
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    DATABRICKS_WAREHOUSE_ID,
    KUMO_TESTING,
)

logger = logging.getLogger(__name__)

# Databricks cluster http_path template
_DATABRICKS_HTTP_PATH_TEMPLATE = '/sql/1.0/warehouses/{warehouse_id}'

# Connection-retry bounds passed to ``databricks.sql.connect``.
#
# databricks-sql-connector 4.x defaults to a very aggressive request-retry
# policy (``_retry_stop_after_attempts_count=30`` over up to
# ``_retry_stop_after_attempts_duration=900`` seconds). For an *unreachable*
# host this turns a single connection attempt into a long DNS-retry storm,
# which (multiplied by the kumo-side ``@retry`` wrappers below) lets a bogus
# host hang a connection attempt for ~30 minutes instead of fast-failing.
#
# Bound the connector's own retry sequence so an unreachable host surfaces its
# underlying connection error ("Failed to establish a new connection") in a
# few seconds. Reachable-but-transiently-failing hosts still get a retry at
# both this layer and the ``@retry`` decorators on ``create_sql_connection``.
_DATABRICKS_CONNECT_RETRY_KWARGS: Dict[str, Any] = {
    '_socket_timeout': 60,  # seconds
    '_retry_stop_after_attempts_count': 2,
    '_retry_stop_after_attempts_duration': 20,  # seconds
    '_retry_delay_max': 5,  # seconds (connector policy minimum)
}


def validate_host(host: str) -> str:
    r"""
    Validate databricks workspace host. We are expecting host in the form
    of https://<databricks_workspace_hostname>
    Customers might enter:
    * https://<databricks_workspace_hostname>/
    * https://<databricks_workspace_hostname>/login.html
    * http://<databricks_workspace_hostname>  (converted to https)
    * <databricks_workspace_hostname>
    This function converts these into the expected format.
    """
    res = up.urlparse(host)
    if res.scheme:
        # Always use https regardless of input scheme
        return f'https://{res.netloc}'
    else:
        res = up.urlparse(f'https://{host}')
        return f'{res.scheme}://{res.netloc}'


@kumo_dataclass
class DatabricksConnectionInfo:
    host: str
    cluster_id: str
    warehouse_id: str
    catalog: str
    credentials: dict[str, SecretStr]

    def __post_init__(self):
        self.host = validate_host(self.host)

    def __repr__(self) -> str:
        return (f"Databricks connector: "
                f"(host={self.host}, cluster={self.cluster_id}, "
                f"warehouse={self.warehouse_id}, catalog={self.catalog})")

    @staticmethod
    def _make_credentials(token: Optional[str], client_id: Optional[str],
                          client_secret: Optional[str]) -> dict[str, SecretStr]:
        if token:
            return {"pat": SecretStr(token)}
        else:
            if client_id is None or client_secret is None:
                raise RuntimeError(
                    "either (token) or (client_id and client_secret) "
                    "must be provided")
            return {
                "client_id": SecretStr(client_id),
                "client_secret": SecretStr(client_secret),
            }

    @staticmethod
    def from_remote_secrets() -> 'DatabricksConnectionInfo':
        credentials = get_secrets('kumo-databricks-credentials')
        host = validate_host(credentials["host"])
        cluster_id = credentials["cluster_id"]
        token = credentials.get("pat", None)
        client_id = credentials.get('client_id', None)
        client_secret = credentials.get('client_secret', None)
        warehouse_id = credentials.get('warehouse_id', '')
        cred = DatabricksConnectionInfo._make_credentials(
            token=token, client_id=client_id, client_secret=client_secret)
        return DatabricksConnectionInfo(host=host, cluster_id=cluster_id,
                                        warehouse_id=warehouse_id, catalog='',
                                        credentials=cred)

    @staticmethod
    def from_environment() -> 'DatabricksConnectionInfo':
        host = validate_host(DATABRICKS_HOST.parse_value())
        cluster_id = DATABRICKS_CLUSTER_ID.parse_value()
        warehouse_id = DATABRICKS_WAREHOUSE_ID.parse_value()
        token = DATABRICKS_TOKEN.parse_value()
        client_id = DATABRICKS_CLIENT_ID.parse_value()
        client_secret = DATABRICKS_CLIENT_SECRET.parse_value()
        cred = DatabricksConnectionInfo._make_credentials(
            token=token, client_id=client_id, client_secret=client_secret)
        return DatabricksConnectionInfo(host=host, cluster_id=cluster_id,
                                        warehouse_id=warehouse_id, catalog='',
                                        credentials=cred)

    @staticmethod
    def get() -> 'DatabricksConnectionInfo':
        if KUMO_TESTING.parse_value():
            logger.info("Running in test mode; using environment variables "
                        "for Databricks connection info")
            return DatabricksConnectionInfo.from_environment()
        try:
            logger.info(
                "Getting Databricks connection info from secret manager")
            return DatabricksConnectionInfo.from_remote_secrets()
        except SecretAccessException as e:
            logger.error(
                "Failed to get Databricks connection info from secret manager "
                "(error_type=%s); using environment variables",
                type(e).__name__)
            return DatabricksConnectionInfo.from_environment()


class DatabricksClient:
    """A client class for common Databricks operations."""
    def __init__(self, connection_info: DatabricksConnectionInfo):
        self.connection_info = connection_info

    def _get_databricks_token(
            self, credentials: Optional[Dict] = None) -> Optional[str]:
        """Get Databricks PAT token from credentials or environment."""
        credentials = credentials or {}

        token = credentials.get('pat', DATABRICKS_TOKEN.parse_value())
        # Extract the secret values
        if token is not None and isinstance(token, SecretStr):
            token: str = token.get_secret_value()

        client_id = credentials.get('client_id',
                                    DATABRICKS_CLIENT_ID.parse_value())
        # Extract the secret values
        if client_id is not None and isinstance(client_id, SecretStr):
            client_id: str = client_id.get_secret_value()

        if not token and not client_id:  # pragma: no cover
            raise WarehouseMisconfigurationError(
                "No credentials were provided. Please provide either "
                f"Databricks PAT with env var `{DATABRICKS_TOKEN.name}` "
                "or OAuth client ID and secret with "
                f"env var `{DATABRICKS_CLIENT_ID.name}` and "
                f"`{DATABRICKS_CLIENT_SECRET.name}`.")
        return token

    def _get_databricks_oauth_secret(
            self, credentials: Optional[Dict] = None) -> Tuple[str, str]:
        """Get Databricks OAuth credentials from credentials or environment."""
        credentials = credentials or {}

        client_id = credentials.get('client_id',
                                    DATABRICKS_CLIENT_ID.parse_value())
        client_secret = credentials.get('client_secret',
                                        DATABRICKS_CLIENT_SECRET.parse_value())

        # Extract the secret values
        if client_id is not None and isinstance(client_id, SecretStr):
            client_id = client_id.get_secret_value()
        if client_secret is not None and isinstance(client_secret, SecretStr):
            client_secret = client_secret.get_secret_value()

        if not client_id or not client_secret:
            raise WarehouseMisconfigurationError(
                "Please provide both the service principal client ID "
                "and the OAuth secret when using OAuth for authentication.")
        return client_id, client_secret

    def _process_exception_for_oauth_retry(self, err: Exception):
        """Check if exception is oauth request related."""
        if 'Max retries exceeded with url: /oidc' in str(err):
            raise WarehouseExtendedRetriableError(
                'Databricks OAuth request failed') from None
        else:
            raise WarehouseRetriableError(
                'Databricks connection request failed') from None

    def resolved_auth(self) -> Dict[str, str]:
        """Resolve the single auth method used by SQL and Cloud Fetch."""
        credentials = self.connection_info.credentials
        token = self._get_databricks_token(credentials)
        # Explicit connector credentials do not fall back to environment
        # variables. OAuth is the production default when both methods exist.
        if credentials:
            has_oauth = bool(credentials.get('client_id'))
        else:
            has_oauth = bool(DATABRICKS_CLIENT_ID.parse_value())
        if has_oauth:
            client_id, client_secret = self._get_databricks_oauth_secret(
                credentials)
            return {
                'auth_type': 'oauth-m2m',
                'client_id': client_id,
                'client_secret': client_secret,
            }
        if token:
            return {'auth_type': 'pat', 'token': token}
        raise WarehouseMisconfigurationError(
            "No credentials were provided. Please provide either "
            "OAuth client ID and secret or a Databricks PAT.")

    def workspace_client_kwargs(self) -> Dict[str, str]:
        """Build SDK arguments from the identity used by SQL and Cloud Fetch."""
        return {
            'host': self.connection_info.host,
            **self.resolved_auth(),
        }

    @retry(exceptions=WarehouseRetriableError, tries=3, delay=2, backoff=2,
           logger=logger)
    @retry(exceptions=WarehouseExtendedRetriableError, tries=7, delay=5,
           backoff=2, logger=logger)
    def create_sql_connection(
        self,
        schema: Optional[str] = None,
        *,
        staging_allowed_local_path: Optional[str] = None,
    ) -> Any:
        """Create a Databricks SQL connection in an optional initial schema.

        ``staging_allowed_local_path`` enables the connector-native
        ``PUT INTO`` command. Databricks returns a presigned URL for the
        managed Unity Catalog volume and the SQL connector uploads the file
        directly to the volume's backing object store. Keep the option absent
        for ordinary query connections so they cannot read arbitrary local
        files.
        """
        from databricks import sql
        from databricks.sdk.core import Config, oauth_service_principal
        from databricks.sql.exc import RequestError

        try:
            staging_kwargs = ({
                'staging_allowed_local_path':
                staging_allowed_local_path
            } if staging_allowed_local_path is not None else {})
            auth = self.resolved_auth()
            if auth['auth_type'] == 'oauth-m2m':
                # Perform 2-legged OAuth token authentication
                config = Config(host=self.connection_info.host,
                                auth_type="oauth-m2m",
                                client_id=auth['client_id'],
                                client_secret=auth['client_secret'])
                creds = oauth_service_principal(config)
                connection = sql.connect(
                    server_hostname=self.connection_info.host.replace(
                        'https://', '', 1),
                    http_path=_DATABRICKS_HTTP_PATH_TEMPLATE.format(
                        warehouse_id=self.connection_info.warehouse_id),
                    credentials_provider=lambda: creds,
                    catalog=self.connection_info.catalog,
                    schema=schema,
                    **staging_kwargs,
                    **_DATABRICKS_CONNECT_RETRY_KWARGS,
                )
            else:
                # Perform PAT-based authentication
                connection = sql.connect(
                    server_hostname=self.connection_info.host.replace(
                        'https://', '', 1),
                    http_path=_DATABRICKS_HTTP_PATH_TEMPLATE.format(
                        warehouse_id=self.connection_info.warehouse_id),
                    access_token=auth['token'],
                    catalog=self.connection_info.catalog,
                    schema=schema,
                    **staging_kwargs,
                    **_DATABRICKS_CONNECT_RETRY_KWARGS,
                )
            return connection
        except (ValueError, MaxRetryError, ConnectionError,
                RequestError) as err:
            self._process_exception_for_oauth_retry(err)


def _execute_query(query: str,
                   fetch_one: bool = False) -> Union[tuple, pyarrow.Table]:
    """Execute a SQL query using Databricks warehouse compute.

    Args:
        query (str): The SQL query to execute.
        fetch_one (bool): Whether to fetch only a single row.
        Defaults to False.

    Returns:
        Union[tuple, pyarrow.Table]: The query results. If fetch_one is True,
            returns a single row as a tuple of values, or None if no results.
            Otherwise returns all rows as a pyarrow Table.

    Raises:
        Exception: Any exception that occurs during query execution.
    """
    connection = None
    try:
        client = DatabricksClient(DatabricksConnectionInfo.get())
        connection = client.create_sql_connection()

        with connection.cursor() as cursor:
            cursor.execute(query)
            if fetch_one:
                return cursor.fetchone()
            return cursor.fetchall_arrow()
    finally:
        if connection:
            connection.close()


def get_num_rows_databricks(table_path: str) -> int:
    """Get the number of rows in a parquet table using Databricks compute.

    Args:
        table_path (str): Path where the table to be checked is. This exists
        in Databricks.

    Returns:
        int: The number of rows in the parquet table.
    """
    from io_utils.fsutil import isdir
    try:
        if isdir(table_path):
            query = f"SELECT COUNT(*) FROM parquet.`{table_path}/*.parquet`"
        else:
            # Already checked before that it is a parquet file
            query = f"SELECT COUNT(*) FROM parquet.`{table_path}`"

        result = _execute_query(query, fetch_one=True)
        return result[0] if result else 0
    except Exception as e:
        logger.error(
            "Failed to get Databricks parquet row count for %s "
            "(error_type=%s)",
            table_path,
            type(e).__name__,
        )
        return 0


def get_parquet_head_databricks(table_path: str,
                                max_rows: int = 1000) -> pyarrow.Table:
    """Get the first N rows of a parquet table using Databricks compute.

    Args:
        table_path (str): Path where the table to be checked is. This exists
        in Databricks.
        max_rows (int): Maximum number of rows to return.

    Returns:
        pyarrow.Table: The first N rows of the parquet table.

    Raises:
        Exception: Any exception that occurs during query execution.
    """
    from io_utils.fsutil import isdir, listdir
    if isdir(table_path):
        query = (f"SELECT * FROM parquet.`{table_path}/*.parquet` "
                 f"LIMIT {max_rows}")
    # file or file prefix with wildcard
    else:
        # databricks sql doesn't expand globs - for now check that wildcard
        # match won't pick up non-parquet files; TODO to use alternative
        # load in this case
        if table_path.endswith('*'):
            dir_list = listdir(osp.dirname(table_path))
            table_path_match = table_path.replace('*', '')
            for file in dir_list:
                if file.startswith(
                        table_path_match) and not file.endswith('.parquet'):
                    raise ValueError(f"Found non-parquet file {file} matching"
                                     f" {table_path}; unable to load table via "
                                     f"Databricks SQL.")

        query = f"SELECT * FROM parquet.`{table_path}` LIMIT {max_rows}"

    try:
        return _execute_query(query)
    except Exception as e:
        logger.error(
            "Failed to get Databricks parquet head rows for %s "
            "(error_type=%s)",
            table_path,
            type(e).__name__,
        )
        if isinstance(e, ValueError):
            raise ValueError(
                'Databricks could not read the requested parquet data.'
            ) from None
        raise RuntimeError(
            'Databricks could not read the requested parquet data.') from None
