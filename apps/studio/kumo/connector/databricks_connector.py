# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import databricks.sql.exc as DatabricksErrors
import pandas as pd
import pyarrow as pa
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import PermissionDenied
from kumo_core.sources.identity import hour_version
from kumo_core.types import ConnectorType
from kumo_core.types.schema import (
    BaseColumnConfig,
    BaseTableFingerprint,
    IngestedTableFingerprint,
    SqlTableFingerprint,
)
from kumo_core.utils.registry import register
from kumo_databricks import identity as dbx_identity
from kumo_databricks import output as dbx_output
from kumo_databricks import sql as dbx_sql
from kumo_databricks import write as dbx_write
from kumo_databricks.errors import ArtifactValidationError, is_not_found
from kumo_databricks.types import dtype_for
from kumoapi.typing import Dtype
from pydantic import SecretStr
from urllib3.exceptions import HTTPError

from async_utils.sync import EnvVar
from io_utils import fsutil, logging
from io_utils.databricks import (
    DatabricksClient,
    DatabricksConnectionInfo,
    validate_host,
)
from io_utils.errors import (
    WarehouseExtendedRetriableError,
    WarehouseMisconfigurationError,
    WarehouseRetriableError,
    restricted_diagnostic_boundary,
)
from io_utils.pyarrow import (
    DecimalOutOfBoundsError,
    fix_types_for_pandas,
    safe_to_pandas,
)
from io_utils.retry import retry
from kumo.config import ConnectorConfig, TableMetadataConfig
from kumo.connector.artifact_staging import local_parquet_files
from kumo.connector.blocking import blocking_method
from kumo.connector.connector import (
    ConnectorError,
    ExportMetrics,
    SourceTableNotFoundError,
)
from kumo.connector.sql_connector import InternalWriteMode, SQLConnector
from kumo.connector.utils import cast_df
from kumo.utils import to_dataclass
from service_framework.deploy_env import (
    DATABRICKS_CLIENT_ID,
    DATABRICKS_CLIENT_SECRET,
    DATABRICKS_TOKEN,
)
from statedb.utils.errors import (
    ConnectorValidationDiagnosticCode,
    ConnectorValidationDiagnosticError,
)

logger = logging.getLogger(__name__)

# names of env vars used for connecting to databricks
_ENV_DATABRICKS_CLUSTER_ID = EnvVar('DATABRICKS_CLUSTER_ID', type=str,
                                    default_value='')
_ENV_DATABRICKS_WAREHOUSE_ID = EnvVar('DATABRICKS_WAREHOUSE_ID', type=str,
                                      default_value='')
_ENV_DATABRICKS_CATALOG = EnvVar('DATABRICKS_CATALOG', type=str,
                                 default_value='')

_DATABRICKS_SAMPLE_CATALOG = 'samples'
_SAMPLE_TABLE_PERMISSIONS = 'SAMPLE_TABLE_PERMISSIONS'
_SqlResult = TypeVar('_SqlResult')


def _dataframe_result(cursor: Any, query: str) -> pd.DataFrame:
    return safe_to_pandas(
        fix_types_for_pandas(cursor.execute(query).fetchall_arrow()))


def _sample_dataframe(cursor: Any, source: str, limit: int,
                      projection: str = '*') -> pd.DataFrame:
    dbx_sql.sample(cursor, source, limit, projection)
    return safe_to_pandas(fix_types_for_pandas(cursor.fetchall_arrow()))


# Session statement timeout (seconds) applied around a Custom SQL preview
# (`sample_query`). Databricks has no client-side per-query timeout, so the
# preview is bounded via the session `STATEMENT_TIMEOUT` and reset afterwards;
# a preview is LIMIT-bounded and should return quickly rather than pin the
# warehouse.
DATABRICKS_SAMPLE_QUERY_TIMEOUT_SECONDS = 60
# This is per producer worker. Four prediction workers therefore expose up to
# 32 concurrent presigned object uploads without opening 128 SQL sessions.
_DATABRICKS_VOLUME_UPLOAD_CONCURRENCY = EnvVar(
    'DATABRICKS_VOLUME_UPLOAD_CONCURRENCY',
    type=int,
    default_value=8,
)
_DATABRICKS_VOLUME_UPLOAD_MEMORY_BUDGET = EnvVar(
    'DATABRICKS_VOLUME_UPLOAD_MEMORY_BUDGET',
    type=int,
    default_value=4 << 30,
)
_DATABRICKS_VOLUME_UPLOAD_ATTEMPTS = 3

#: Stamp for a destination-addressed export. `stage_source` returns
#: `ExportMetrics` and drops `StagedTable.version`, so this is never compared
#: or persisted; it exists so the reader does not spend a round trip deriving
#: a value with no reader.
_CUSTOM_SOURCE_VERSION = 'custom-source'


def _validated_local_parquet_files(path: str) -> list[Path]:
    """Map local artifact validation failures to the connector boundary."""
    try:
        return local_parquet_files(path)
    except (FileNotFoundError, ValueError) as error:
        raise ConnectorError(str(error)) from None


def validate_databricks_auth(
    host: str,
    credentials: Optional[Dict],
) -> None:
    """Validate Databricks credentials without an execution connector."""
    host = validate_host(host)
    creds = credentials or {}
    token = creds.get('pat')
    client_id = creds.get('client_id')
    client_secret = creds.get('client_secret')
    token_value = token.get_secret_value().strip() if token else ""
    client_id_value = (client_id.get_secret_value().strip()
                       if client_id else "")
    client_secret_value = (client_secret.get_secret_value().strip()
                           if client_secret else "")

    has_pat = bool(token_value)
    has_oauth = bool(client_id_value and client_secret_value)
    has_partial_oauth = bool(client_id_value) != bool(client_secret_value)

    if has_partial_oauth:
        raise ConnectorError("Databricks OAuth credentials are incomplete. "
                             "Both client_id and client_secret are required.")
    if not has_pat and not has_oauth:
        raise ConnectorError("No Databricks credentials were provided.")

    resolved_credentials: Dict[str, SecretStr] = {}
    if has_pat:
        resolved_credentials['pat'] = SecretStr(token_value)
    if has_oauth:
        resolved_credentials['client_id'] = SecretStr(client_id_value)
        resolved_credentials['client_secret'] = SecretStr(client_secret_value)

    auth_mode = "oauth" if has_oauth else "pat"
    try:
        connection_info = DatabricksConnectionInfo(
            host=host,
            cluster_id='',
            warehouse_id='',
            catalog='',
            credentials=resolved_credentials,
        )
        client_args = DatabricksClient(
            connection_info).workspace_client_kwargs()
        client = WorkspaceClient(**client_args)
        client.current_user.me()
    except ConnectorError:
        raise
    except WarehouseMisconfigurationError as err:
        raise ConnectorError(str(err)) from None
    except PermissionDenied as err:
        if auth_mode == "pat":
            message = ("Databricks PAT authentication failed. "
                       "The PAT may be invalid, expired, revoked, or from a "
                       "different Databricks workspace.")
        else:
            message = ("Databricks OAuth authentication failed. "
                       "Please verify client_id, client_secret, and service "
                       "principal permissions.")
        raise ConnectorError(message, error=err) from None
    except Exception:
        raise ConnectorError(
            f"Failed to authenticate with Databricks. Authentication mode used is {auth_mode}. "
            "Please verify your host and credentials.", ) from None


class DatabricksConnector(SQLConnector):
    r"""Establishes a connection to a Databricks lakehouse.

    Args:
        name (str): The name of the connector.
        host (str): The workspace hostname.
        cluster_id (str): The Databricks cluster ID.
        warehouse_id (str): The Databricks SQL warehouse ID.
        catalog (str): The name of the UC catalogue.
    """
    #: `export_table_read_only` takes a `sql` argument; Snowflake's and
    #: BigQuery's exports do not.
    exports_custom_sql = True

    def __init__(
        self,
        name: str,
        host: str,
        cluster_id: str,
        warehouse_id: str,
        catalog: str,
        schema_name: str,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.host = validate_host(host)
        self.cluster_id = cluster_id
        self.warehouse_id = warehouse_id
        self.catalog = catalog
        if not schema_name or not schema_name.strip():
            raise ConnectorError('Databricks schema is required.')
        self.schema_name = schema_name

    def _get_connector_type(self) -> str:
        return "Databricks"

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(host={self.host}, catalog={self.catalog}, '
                f'schema={self.schema_name}, '
                f'cluster={self.cluster_id}, warehouse={self.warehouse_id})')

    # Connecting and closing connections ######################################

    # Databricks connector does not use username and pasword for authentication
    # Instead, it uses either Databricks personal access token or Databricks
    # OAuth token for authentication.

    @blocking_method()
    def _open(self, credentials: Optional[Dict] = None) -> Any:
        try:
            if credentials:
                self.set_credentials(credentials)
            client = DatabricksClient(self._to_databricks_connection_info())
            return client.create_sql_connection(schema=self.schema_name)
        except (DatabricksErrors.Error, WarehouseRetriableError,
                WarehouseExtendedRetriableError,
                WarehouseMisconfigurationError) as err:
            if isinstance(err, WarehouseMisconfigurationError):
                # These messages are authored locally and contain only
                # configuration guidance, never provider response text.
                raise ConnectorError(str(err)) from None
            if isinstance(err, DatabricksErrors.Error):
                raise ConnectorError(
                    'Databricks could not establish the configured '
                    'connection.',
                    error=err,
                ) from None
            raise ConnectorError(
                'Databricks could not establish the configured connection.'
            ) from None

    def _to_databricks_connection_info(self) -> DatabricksConnectionInfo:
        r"""Convert the current connector configuration into a
        DatabricksConnectionInfo object."""
        if self._credentials:
            creds = self._credentials
        else:
            if self._require_explicit_credentials:
                raise ConnectorError(
                    'Identity-backed Databricks connectors require explicit '
                    'OAuth credentials.')
            creds = {}
            if DATABRICKS_TOKEN.parse_value():
                creds['pat'] = SecretStr(DATABRICKS_TOKEN.parse_value())
            if DATABRICKS_CLIENT_ID.parse_value():
                creds['client_id'] = SecretStr(
                    DATABRICKS_CLIENT_ID.parse_value())
                creds['client_secret'] = SecretStr(
                    DATABRICKS_CLIENT_SECRET.parse_value())
        return DatabricksConnectionInfo(
            host=self.host,
            cluster_id=self.cluster_id,
            warehouse_id=self.warehouse_id,
            catalog=self.catalog,
            credentials=creds,
        )

    def _fully_qualify_table_name(self, table_name: str) -> str:
        """Fully qualify a provider-native table in this connector's schema."""
        schema_name, table_name = self._target_table_coordinate(table_name)
        return dbx_sql.qualified_name(schema_name, table_name)

    def _target_table_coordinate(self, table_name: str) -> tuple[str, str]:
        """Normalize a table coordinate after enforcing connector scope."""
        try:
            return dbx_sql.normalize_table_coordinate(
                table_name,
                catalog=self.catalog,
                schema=self.schema_name,
            )
        except ValueError as error:
            raise ConnectorError(str(error)) from None

    def _require_target_schema(self, schema_name: str) -> None:
        """Reject coordinates outside a fixed-schema connector target."""
        if schema_name.casefold() != self.schema_name.casefold():
            raise ConnectorError(
                'Databricks table schema does not match the connector schema.')

    # Overrides functions from sql connector ##################################
    def sql_table_name(self, table_name: str):
        return f'{self._fully_qualify_table_name(table_name)}'

    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        is_databricks_view = self.is_table_view(table_name)
        if not is_databricks_view:
            query = ('DESCRIBE DETAIL '
                     f'{self._fully_qualify_table_name(table_name)}')
            df = self._get_df_result(self.context, query)
            if len(df) > 0:
                return df.loc[0]['sizeInBytes']
        logger.warning(f"could not retrieve Databricks table information "
                       f"for table {table_name}")
        return super()._get_table_source_size(
            table_name=self._fully_qualify_table_name(table_name))

    # Reading table metadata ##################################################

    def _get_data_version(
        self,
        table_name: str,
    ) -> str:
        try:
            return self._run_sql(self.context, dbx_identity.table_version,
                                 self._fully_qualify_table_name(table_name))
        except Exception as error:
            if is_not_found(error):
                raise SourceTableNotFoundError(table_name, self.name) from None
            if 'EXPECT_TABLE_NOT_VIEW' in str(error):
                logger.info(
                    'Using default data version for Databricks view %r '
                    'because history is not available for views', table_name)
            else:
                logger.warning(
                    'Could not retrieve Databricks data version for '
                    'table %r (error_type=%s)', table_name,
                    type(error).__name__)
            return hour_version()

    def is_table_view(self, table_name: str) -> bool:
        r"""Check if a databricks table is a view or streaming table by
        checking if it has history available. Views and streaming tables do not
        support time travel operations."""
        try:
            extended_query = (f'DESCRIBE EXTENDED '
                              f'{self._fully_qualify_table_name(table_name)}')
            df = self._get_df_result(self.context, extended_query)

            if len(df) > 0:
                type_rows = df[df['col_name'] == 'Type']
                if len(type_rows) > 0:
                    table_type = type_rows.iloc[0]['data_type']
                    if (table_type == 'STREAMING_TABLE' or table_type == 'VIEW'
                            or table_type == 'MATERIALIZED_VIEW'):
                        return True
                else:
                    raise ConnectorError(
                        "Kumo was unable to determine the table type.")

            return False
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(f"Could not retrieve Databricks table history "
                                 f"for table {table_name}.") from None

    def has_links_support(self) -> bool:
        return True

    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        if schema_name:
            self._require_target_schema(schema_name)
        schema_name = schema_name or self.schema_name
        try:
            from kumo_databricks.catalog import table_names

            with self.context.cursor() as cursor:
                return table_names(cursor, catalog=self.catalog,
                                   schema=schema_name,
                                   target_schema=self.schema_name)
        except Exception:
            raise ConnectorError(
                'Could not list Databricks tables. Please verify your '
                'connection settings and permissions.') from None

    def table_exists(self, table_name: str) -> bool:
        r"""Return whether an exact fixed-schema table is visible.

        ``_get_table_metadata`` intentionally raises
        ``SourceTableNotFoundError`` when a requested name is absent because
        callers requesting metadata expect a table. Existence validation needs
        the corresponding boolean contract, while still allowing connection
        and permission failures from the metadata query to propagate.
        """
        self.open()
        schema_name, table_name = self._target_table_coordinate(table_name)
        metadata = self._fetch_table_metadata(
            [table_name],
            skip_inferral=True,
            schema_name=schema_name,
        )
        return table_name in metadata

    def validate_permissions(self) -> None:
        if not self.catalog or not self.catalog.strip():
            raise ConnectorError(
                'Databricks catalog is required for exact target validation.')
        self.check_schema_exists(self.schema_name)

    def _fetch_table_metadata(
        self,
        table_names: Optional[List[str]] = None,
        skip_inferral: bool = False,
        schema_name: Optional[str] = None,
    ) -> Dict[str, TableMetadataConfig]:
        # Fetch table metadata from databricks.
        column_metadata_dict = self.get_column_metadata(self.context,
                                                        schema_name,
                                                        table_names)

        table_metadata_dict: Dict[str, TableMetadataConfig] = {}

        for (table_name, column_name), dtype in column_metadata_dict.items():
            if table_name not in table_metadata_dict:
                table_metadata_dict[table_name] = TableMetadataConfig(
                    table_name=table_name, source_name=table_name,
                    data_source=self._name)
            table_metadata_dict[table_name].cols.append(
                BaseColumnConfig(name=column_name, dtype=dtype))

        return table_metadata_dict

    def _get_table_metadata(
        self,
        table_names: Optional[List[str]] = None,
        skip_cache: bool = False,
        skip_inferral: bool = False,
        schema_name: Optional[str] = None,
        ignore_errors: bool = False,
    ) -> List[TableMetadataConfig]:
        try:
            if schema_name:
                logger.info(f"fetching Databricks table metadata for "
                            f"{table_names} under schema {schema_name}")
            else:
                logger.info(f"fetching Databricks table metadata for "
                            f"{table_names}")

            requested_names = table_names
            lookup_names = None
            if requested_names is not None:
                lookup_names = [
                    self._target_table_coordinate(table_name)[1]
                    for table_name in requested_names
                ]
            table_metadata_dict = self._fetch_table_metadata(
                lookup_names, skip_inferral, schema_name)

            if requested_names:
                out: List[TableMetadataConfig] = []
                assert lookup_names is not None
                for requested_name, lookup_name in zip(requested_names,
                                                       lookup_names,
                                                       strict=True):
                    if lookup_name not in table_metadata_dict:
                        if ignore_errors:
                            continue
                        raise SourceTableNotFoundError(requested_name,
                                                       self.name)
                    metadata = deepcopy(table_metadata_dict[lookup_name])
                    metadata.table_name = requested_name
                    metadata.source_name = requested_name
                    out.append(metadata)
                return out

            return list(table_metadata_dict.values())
        except ConnectorError:
            raise
        except Exception as err:
            logger.error(
                "Error retrieving Databricks table metadata "
                "(error_type=%s)",
                type(err).__name__)
            schema_suffix = (f", schema {schema_name}" if schema_name else "")
            table_suffix = (f" for table(s) {table_names}"
                            if table_names else "")
            raise ConnectorError(
                message=(
                    'Kumo was unable to retrieve table information from '
                    f'Databricks catalog {self.catalog}'
                    f'{schema_suffix}'
                    f'{table_suffix}. '
                    'Please verify your connection settings and permissions.'
                ), ) from None

    def _read_table(
        self,
        context: Any,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        try:
            return super()._read_table(context, table_name, pkey, cols)
        except Exception:
            raise ConnectorError(
                message=(f'An error occurred while reading table {table_name} '
                         f'from Databricks. Please verify your connection '
                         f'settings and permissions.')) from None

    def _read_table_chunk(
        self,
        context: Any,
        table_name: str,
        num_rows: int,
        offset: int = 0,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        try:
            return super()._read_table_chunk(context, table_name, num_rows,
                                             offset, pkey, cols)
        except Exception:
            raise ConnectorError(
                message=(f'An error occurred while reading a chunk of table '
                         f'{table_name} from Databricks. Please verify your '
                         f'connection settings and permissions.')) from None

    def _read_table_sample(
        self,
        context: Any,
        table_name: str,
        num_rows: Optional[int],
        frac: Optional[float],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        if num_rows is None:
            raise ConnectorError(
                'Databricks source sampling requires num_rows; '
                'an unbounded fraction-only source read is not supported.')
        try:
            if cols is None:
                projection = '*'
            else:
                names = [col.name for col in cols]
                if pkey is not None:
                    names.append(pkey.name)
                if not names:
                    raise ValueError('At least one sample column is required.')
                projection = ', '.join(
                    dbx_sql.quote_identifier(name) for name in names)
            df = self._run_sql(context, _sample_dataframe,
                               self.sql_table_name(table_name), num_rows,
                               projection)
            if pkey is not None:
                df.set_index(pkey.name, inplace=True)
            return cast_df(df, pkey, cols)
        except Exception:
            raise ConnectorError(
                message=(f'An error occurred while reading a sample of table '
                         f'{table_name} from Databricks. Please verify your '
                         f'connection settings and permissions.')) from None

    # Writing to Databricks Delta table #######################################

    @staticmethod
    def parse_table_schema_name(table_schema_name: str) -> Tuple[str, str]:
        try:
            return dbx_output.split_output_table(table_schema_name)
        except ArtifactValidationError as error:
            raise ConnectorError(str(error)) from None

    def require_output_table_within_connector_target(
        self,
        table_schema_name: str,
    ) -> Tuple[str, str]:
        try:
            return dbx_output.within_schema(table_schema_name, self.schema_name)
        except ArtifactValidationError as error:
            raise ConnectorError(str(error)) from None

    def _fully_qualify_output_table_name(self, table_schema_name: str) -> str:
        """Quote a Kumo-internal output coordinate for SQL."""
        schema_name, table_name = (
            self.require_output_table_within_connector_target(table_schema_name)
        )
        return dbx_sql.qualified_name(schema_name, table_name)

    def managed_volume_stage(self, export_id: str | None = None) -> str:
        return dbx_output.staging_stage(self.catalog, self.schema_name,
                                        export_id)

    def ensure_managed_volume(self) -> None:
        """Ensure the recommended one managed staging volume per schema."""
        try:
            dbx_output.ensure_staging_volume(
                lambda statement: self._get_df_result(self.context, statement),
                self.catalog,
                self.schema_name,
            )
        except ArtifactValidationError as error:
            raise ConnectorError(str(error)) from None

    def _require_managed_stage(self, artifact_path: str) -> str:
        try:
            return dbx_write.within_stage(artifact_path, self.catalog,
                                          self.schema_name)
        except ArtifactValidationError as error:
            raise ConnectorError(str(error)) from None

    def _volume_worker_count(self, files: list[Path]) -> int:
        configured = _DATABRICKS_VOLUME_UPLOAD_CONCURRENCY.parse_value()
        memory_budget = (_DATABRICKS_VOLUME_UPLOAD_MEMORY_BUDGET.parse_value())
        if configured <= 0 or memory_budget <= 0:
            raise ConnectorError(
                'Databricks volume upload concurrency and memory budget must '
                'be positive.')
        largest_file = max((file.stat().st_size for file in files), default=1)
        # databricks-sql-connector 4.x currently reads one PUT source file
        # into memory before sending it to the presigned URL. Bound aggregate
        # upload buffers without reducing the number of producer-side shards.
        memory_limited = max(1, memory_budget // max(1, largest_file))
        return max(1, min(len(files), configured, memory_limited))

    def _run_volume_commands(
        self,
        commands: list[str],
        *,
        worker_count: int,
        staging_allowed_local_path: str | None = None,
        ignore_missing: bool = False,
    ) -> None:
        """Run independent volume commands over persistent worker sessions."""
        if not commands:
            return
        worker_count = max(1, min(worker_count, len(commands)))
        batches = [
            commands[index::worker_count] for index in range(worker_count)
        ]
        connection_info = self._to_databricks_connection_info()

        def run_batch(batch: list[str]) -> None:
            connection = None
            cursor = None
            try:
                for command in batch:
                    for attempt in range(_DATABRICKS_VOLUME_UPLOAD_ATTEMPTS):
                        try:
                            with restricted_diagnostic_boundary(
                                    exception_types=(
                                        DatabricksErrors.Error,
                                        HTTPError,
                                        OSError,
                                    ), safe_message=(
                                        'Databricks managed-volume operation '
                                        'failed')):
                                if connection is None:
                                    connection = DatabricksClient(
                                        connection_info,
                                    ).create_sql_connection(
                                        schema=self.schema_name,
                                        staging_allowed_local_path=(
                                            staging_allowed_local_path),
                                    )
                                    cursor = connection.cursor()
                                assert cursor is not None
                                cursor.execute(command)
                            break
                        except Exception as error:
                            missing = (ignore_missing
                                       and any(marker in str(error).upper()
                                               for marker in (
                                                   'PATH_NOT_FOUND',
                                                   'RESOURCE_DOES_NOT_EXIST',
                                                   'NO_SUCH_FILE',
                                                   'NO SUCH FILE OR DIRECTORY',
                                                   'DOES NOT EXIST',
                                               )))
                            if cursor is not None:
                                with suppress(BaseException):
                                    cursor.close()
                            if connection is not None:
                                with suppress(BaseException):
                                    connection.close()
                            cursor = None
                            connection = None
                            if missing:
                                break
                            if attempt + 1 == (
                                    _DATABRICKS_VOLUME_UPLOAD_ATTEMPTS):
                                raise
            finally:
                if cursor is not None:
                    with suppress(BaseException):
                        cursor.close()
                if connection is not None:
                    with suppress(BaseException):
                        connection.close()

        with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix='databricks-volume',
        ) as executor:
            futures = [
                executor.submit(run_batch, batch) for batch in batches if batch
            ]
            for future in as_completed(futures):
                future.result()

    def upload_local_artifacts_to_volume(
        self,
        artifact_path: str,
        *,
        stage: str | None = None,
        file_prefix: str = 'part',
        files: list[Path] | None = None,
    ) -> tuple[str, list[str]]:
        """Upload a local Parquet dataset to a unique managed-volume prefix.

        ``PUT INTO`` is the Databricks-recommended ingestion-partner protocol.
        The SQL warehouse returns a presigned URL and the connector sends file
        bytes directly to the volume's object storage.
        """
        files = (_validated_local_parquet_files(artifact_path)
                 if files is None else files)
        stage = (self.managed_volume_stage()
                 if stage is None else self._require_managed_stage(stage))
        try:
            return dbx_write.stage_files(self._run_volume_commands,
                                         self._volume_workers, files,
                                         stage=stage, file_prefix=file_prefix)
        except ArtifactValidationError as error:
            raise ConnectorError(str(error)) from None

    def _volume_workers(self, files: list[Path] | None, commands: int) -> int:
        """How wide a volume batch may go, which only this deployment knows.

        A ``PUT`` batch is bounded by the largest shard it buffers; a
        ``REMOVE`` batch reads nothing and is bounded only by the configured
        concurrency.
        """
        if files is None:
            return min(
                commands,
                _DATABRICKS_VOLUME_UPLOAD_CONCURRENCY.parse_value(),
            )
        return self._volume_worker_count(files)

    def _cleanup_volume_files(self, files: list[str]) -> None:
        dbx_write.cleanup(self._run_volume_commands, self._volume_workers,
                          files)

    def cleanup_managed_volume_files(self, files: list[str]) -> None:
        validated = [self._require_managed_stage(path) for path in files]
        self._cleanup_volume_files(validated)

    def clear_managed_volume_file_prefix(
        self,
        stage: str,
        file_prefix: str,
    ) -> None:
        """Remove one producer's files from a shared retry directory.

        Prediction workers use disjoint filename prefixes in one flat
        table-specific directory. Clearing only the current worker's prefix
        before retry prevents duplicate rows without racing sibling uploads,
        and lets the coordinator issue one non-recursive ``COPY INTO``.
        """
        stage = self._require_managed_stage(stage)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', file_prefix):
            raise ConnectorError(
                'Databricks managed-volume file prefix contains unsupported '
                'characters.')
        try:
            listing = self._get_df_result(
                self.context,
                f'LIST {_quote_string(stage)}',
            )
        except Exception as error:
            if any(marker in str(error).upper()
                   for marker in ('PATH_NOT_FOUND', 'RESOURCE_DOES_NOT_EXIST',
                                  'NO_SUCH_FILE', 'NO SUCH FILE OR DIRECTORY',
                                  'DOES NOT EXIST')):
                return
            raise
        if listing.empty:
            return
        path_column = next(
            (column for column in listing.columns
             if str(column).casefold() in ('path', 'name')),
            listing.columns[0],
        )
        stage_prefix = stage.rstrip('/') + '/'
        producer_pattern = re.compile(
            rf'{re.escape(file_prefix)}-\d{{8}}\.parquet',
            re.IGNORECASE,
        )
        files = []
        for value in listing[path_column].tolist():
            path = self._require_managed_stage(str(value))
            if not path.casefold().startswith(stage_prefix.casefold()):
                raise ConnectorError(
                    'Databricks returned a file outside the shared retry '
                    'directory.')
            relative_path = path[len(stage_prefix):]
            if '/' not in relative_path and producer_pattern.fullmatch(
                    relative_path):
                files.append(path)
        self._cleanup_volume_files(files)

    @blocking_method()
    def read_managed_volume_parquet_sample(
        self,
        stage: str,
        columns: list[BaseColumnConfig],
        *,
        num_rows: int,
    ) -> pd.DataFrame:
        """Read a bounded sample from this connector's managed-volume stage."""
        stage = self._require_managed_stage(stage)
        if num_rows <= 0:
            raise ValueError('Databricks volume sample size must be positive.')
        if not columns:
            raise ValueError(
                'Databricks volume sample requires at least one column.')
        select_list = ', '.join(
            dbx_sql.quote_identifier(column.name) for column in columns)
        query = (f'SELECT {select_list} '
                 f'FROM read_files({_quote_string(stage)}, format => '
                 f"{_quote_string('parquet')}) "
                 f'LIMIT {int(num_rows)}')
        try:
            sample = self._get_df_result(self.context, query)
        except Exception:
            raise ConnectorError(
                'Databricks could not read the staged prediction Parquet '
                'files for column statistics.') from None
        missing = [
            column.name for column in columns
            if column.name not in sample.columns
        ]
        if missing:
            raise ConnectorError(
                'Databricks staged prediction Parquet is missing expected '
                f'column(s): {", ".join(missing)}.')
        return sample

    @blocking_method()
    def write_artifacts_to_table(
        self,
        table_schema_name: str,
        artifact_path: str,
        replace: bool = False,
        *,
        staged_files: list[str] | None = None,
    ) -> None:
        """Write Parquet through Kumo's managed UC volume and ``COPY INTO``.

        Local artifacts are uploaded with concurrent connector-native
        ``PUT INTO`` commands. Distributed producers may pass the exact
        managed-volume directory and file manifest they already uploaded.
        S3, ADLS, arbitrary volumes, and inline cloud credentials are
        intentionally unsupported.
        """
        schema_name, table_name = (
            self.require_output_table_within_connector_target(table_schema_name)
        )
        destination_table = self._fully_qualify_output_table_name(
            table_schema_name)
        try:
            dbx_write.load(
                execute=lambda statement: self._get_df_result(
                    self.context, statement),
                run=self._run_volume_commands,
                workers=self._volume_workers,
                catalog=self.catalog,
                schema=schema_name,
                table=table_name,
                artifact_path=artifact_path,
                resolve_local=_validated_local_parquet_files,
                staged_files=staged_files,
                replace=replace,
            )
        except ConnectorError:
            raise
        except ArtifactValidationError as error:
            # Only the writer's own deliberate refusals -- an artifact
            # outside the managed stage, a manifest that does not match what
            # was uploaded, nothing to upload. `execute` and `run` above do
            # real provider and filesystem work and can raise a bare
            # `ValueError` of their own; that is not this type, and falls
            # through to the generic branch below rather than being
            # relabelled as one of these refusals.
            raise ConnectorError(str(error)) from None
        except Exception as e:
            if 'SCHEMA_NOT_FOUND' in str(e):
                raise ConnectorError(
                    f"Cannot write to table {destination_table} "
                    f"since schema {schema_name} does not exist.") from None
            logger.warning('Databricks output write failed (%s)',
                           type(e).__name__)
            raise ConnectorError(
                'Kumo encountered an error while writing data to the '
                'configured Databricks output table.') from None

    def check_schema_exists(self, schema_name: str) -> None:
        try:
            self._check_schema(schema_name)
        except WarehouseRetriableError:
            raise ConnectorError(
                f"Unable to verify schema '{schema_name}'") from None
        except DatabricksErrors.Error as e:
            raise ConnectorError(f"Unable to verify schema '{schema_name}'",
                                 error=e) from None

    @blocking_method()
    @retry(exceptions=WarehouseRetriableError, tries=3, delay=2, backoff=2,
           logger=None)
    def _check_schema(self, schema_name: str):
        r"""Check if the given schema exists under the catalog for the current
        connector"""
        if not schema_name:
            raise ValueError("Schema is unspecified.")
        context = self.context
        # Legacy databricks catalog requires special treatment
        is_legacy = (self.catalog == 'hive_metastore')
        with context.cursor() as cursor:
            try:
                query = (
                    f'show grants '
                    f'{dbx_sql.quote_identifier(self._retrieve_current_user())} '
                    f'on SCHEMA {dbx_sql.quote_identifier(self.catalog)}.'
                    f'{dbx_sql.quote_identifier(schema_name)}')
                df = cursor.execute(query).fetchall_arrow().to_pandas()
                if is_legacy:
                    # Legacy catalog won't raise permission denied exception
                    # if USAGE permission is not granted.
                    if len(df) == 0:
                        raise ConnectorError(
                            "Connector does not have USAGE permission on "
                            f"catalog {self.catalog} and/or USAGE permission "
                            f"on schema {schema_name}.",
                            diagnostic_code=ConnectorValidationDiagnosticCode.
                            TARGET_ACCESS_DENIED)
            except HTTPError:
                raise WarehouseRetriableError(
                    'Databricks request failed') from None
            except DatabricksErrors.ServerOperationError as e:
                error_code = _databricks_server_error_code(e)
                if error_code == _SAMPLE_TABLE_PERMISSIONS:
                    if ((self.catalog
                         or '').casefold() == _DATABRICKS_SAMPLE_CATALOG):
                        self._validate_sample_schema_read_access(schema_name)
                        return
                    raise ConnectorValidationDiagnosticError(
                        ConnectorValidationDiagnosticCode.
                        DATABRICKS_SAMPLE_TABLE_PERMISSIONS) from None
                error_class = _databricks_error_class(e)
                if _databricks_error_class_is(error_class,
                                              'CATALOG_DOES_NOT_EXIST'):
                    raise ConnectorError(
                        f"Catalog {self.catalog} does not exist or is not "
                        "accessible to the connector.",
                        diagnostic_code=ConnectorValidationDiagnosticCode.
                        TARGET_NOT_FOUND) from None
                # Unity Catalog reports a missing catalog *or* schema on this
                # ``SHOW GRANTS ON SCHEMA`` as SCHEMA_DOES_NOT_EXIST; legacy
                # hive_metastore uses SECURABLE_DOES_NOT_EXIST, or omits the
                # code entirely (its text is the only signal there).
                if (_databricks_error_class_is(error_class,
                                               'SCHEMA_DOES_NOT_EXIST')
                        or _databricks_error_class_is(
                            error_class, 'SECURABLE_DOES_NOT_EXIST')
                        or (error_class is None and is_legacy
                            and 'does not exist' in str(e))):
                    raise ConnectorError(
                        f"Schema {schema_name} does not exist "
                        f"under catalog {self.catalog}.",
                        diagnostic_code=ConnectorValidationDiagnosticCode.
                        TARGET_NOT_FOUND) from None
                if _databricks_error_class_is(error_class, 'PERMISSION_DENIED'):
                    raise ConnectorError(
                        "Connector does not have USE CATALOG "
                        f"permission on catalog {self.catalog} "
                        "and/or USE SCHEMA permission on schema "
                        f"{schema_name}.",
                        diagnostic_code=ConnectorValidationDiagnosticCode.
                        TARGET_ACCESS_DENIED) from None
                raise e

    def _validate_sample_schema_read_access(self, schema_name: str) -> None:
        """Validate a Databricks system sample target without grant lookup.

        Databricks rejects ``SHOW GRANTS`` for its read-only ``samples``
        catalog.  Visibility alone is insufficient, so this fallback first
        proves that the exact schema is listed, then performs a one-row,
        read-only query against one of its visible tables.
        """
        code = (ConnectorValidationDiagnosticCode.
                DATABRICKS_SAMPLE_TABLE_PERMISSIONS)
        try:
            with self.context.cursor() as cursor:
                schema_rows = cursor.execute(
                    f'SHOW SCHEMAS IN '
                    f'{dbx_sql.quote_identifier(self.catalog)}').fetchall()
            if not any(row and str(row[0]).casefold() == schema_name.casefold()
                       for row in schema_rows):
                raise ConnectorValidationDiagnosticError(code)

            with self.context.cursor() as cursor:
                table_rows = cursor.execute(
                    f'SHOW TABLES IN '
                    f'{dbx_sql.quote_identifier(self.catalog)}.'
                    f'{dbx_sql.quote_identifier(schema_name)}').fetchall()
            table_names = [str(row[1]) for row in table_rows if len(row) > 1]
            if not table_names:
                raise ConnectorValidationDiagnosticError(code)

            for table_name in table_names:
                try:
                    with self.context.cursor() as cursor:
                        cursor.execute(
                            f'SELECT 1 FROM '
                            f'{dbx_sql.quote_identifier(self.catalog)}.'
                            f'{dbx_sql.quote_identifier(schema_name)}.'
                            f'{dbx_sql.quote_identifier(table_name)} '
                            f'LIMIT 1').fetchone()
                    return
                except HTTPError:
                    raise WarehouseRetriableError(
                        'Databricks system sample read probe failed') from None
                except DatabricksErrors.Error:
                    continue
            raise ConnectorValidationDiagnosticError(code)
        except (ConnectorValidationDiagnosticError, WarehouseRetriableError):
            raise
        except HTTPError:
            raise WarehouseRetriableError(
                'Databricks system sample metadata probe failed') from None
        except DatabricksErrors.Error:
            raise ConnectorValidationDiagnosticError(code) from None

    def _retrieve_current_user(self) -> str:
        r"""Retrieve current user principal ID. It will be the email address
        for a normal account and the client ID for a service principal"""
        connection_info = self._to_databricks_connection_info()
        assert connection_info is not None
        host = connection_info.host
        token = connection_info.credentials.get("pat", None)
        client_id = connection_info.credentials.get('client_id', None)

        if not host:
            raise ConnectorError("No databricks hostname was provided.")
        elif client_id:
            # Service principal client ID is the current user ID
            return client_id.get_secret_value()
        elif token:
            # Get current user ID via databricks SDK if credential
            # only contains PAT
            client = WorkspaceClient(host=host, auth_type="pat",
                                     token=token.get_secret_value())
            return client.current_user.me().user_name
        else:
            raise ConnectorError("No databricks credentials were provided")

    def check_table_write_permissions(
        self,
        table_name: str,
        table_schema: str,
        mode: InternalWriteMode,
    ) -> None:
        if mode not in {
                InternalWriteMode.APPEND,
                InternalWriteMode.OVERWRITE,
        }:
            raise ConnectorError(
                'Invalid Databricks output write mode; expected APPEND or '
                'OVERWRITE.')
        if (not self.schema_name
                or table_schema.casefold() != self.schema_name.casefold()):
            raise ConnectorError(
                'Databricks output table schema does not match the '
                'connector schema.')
        try:
            self._check_table(table_name, table_schema)
        except WarehouseRetriableError:
            raise ConnectorError(
                f"Unable to verify write permissions for table "
                f"'{table_schema}.{table_name}'") from None
        except DatabricksErrors.Error as e:
            raise ConnectorError(
                f"Unable to verify write permissions for table "
                f"'{table_schema}.{table_name}'", error=e) from None

    @blocking_method()
    @retry(exceptions=WarehouseRetriableError, tries=3, delay=2, backoff=2,
           logger=None)
    def _check_table(self, table_name: str, schema_name: str):
        r"""Check if sufficient permission has been granted to create and
        write to the specified table under the given schema."""
        if not table_name:
            raise ValueError("Table name is unspecified.")
        context = self.context
        with context.cursor() as cursor:
            current_user = self._retrieve_current_user()
            is_legacy = (self.catalog == 'hive_metastore')
            schema_query = (f'show grants `{current_user}` on SCHEMA `'
                            f'{self.catalog}`.`{schema_name}`')
            table_query = (f'show grants `{current_user}` on TABLE `'
                           f'{self.catalog}`.`{schema_name}`.`{table_name}`')
            try:
                # Schema query should not throw exception after checking
                # schema existence
                schema_df: pd.DataFrame = cursor.execute(
                    schema_query).fetchall_arrow().to_pandas()
                table_df: pd.DataFrame = cursor.execute(
                    table_query).fetchall_arrow().to_pandas()
                if is_legacy:
                    # Legacy catalog won't raise permission denied exception
                    # if SELECT permission is not granted.
                    if len(table_df) == 0:
                        raise ConnectorError(
                            "Connector does not have SELECT "
                            f"permission on table {self.catalog}."
                            f"{schema_name}.{table_name}.")
                # Table exists and permission to read the table is granted.
                # Check permissions to append/overwrite the table.
                permissions = table_df['ActionType'].tolist()
                # Check if required permission to append to table is
                # granted
                if {'ALL PRIVILEGES', 'OWN',
                        'MODIFY'}.intersection(permissions):
                    return
                logger.warning(
                    'Databricks returned no direct MODIFY grant for '
                    '%s.%s.%s. The identity may inherit permission through '
                    'a group or an administrative role; deferring the '
                    'authoritative check to the output write.',
                    self.catalog,
                    schema_name,
                    table_name,
                )
                return
            except HTTPError:
                raise WarehouseRetriableError(
                    'Databricks request failed') from None
            except DatabricksErrors.ServerOperationError as e:
                # Either the table does not exist, or permission to read
                # table within the schema is not granted.
                if ('TABLE_DOES_NOT_EXIST' in str(e)
                        or is_legacy and 'does not exist' in str(e)):
                    # Table does not exist. Check required permission for
                    # creating the table in the schema.
                    permissions = schema_df['ActionType'].tolist()
                    if {'ALL PRIVILEGES', 'OWN', 'CREATE',
                            'CREATE TABLE'}.intersection(permissions):
                        return
                    logger.warning(
                        'Databricks returned no direct CREATE TABLE grant for '
                        '%s.%s. The identity may inherit permission through '
                        'a group or an administrative role; deferring the '
                        'authoritative check to the output write.',
                        self.catalog,
                        schema_name,
                    )
                    return
                elif 'PERMISSION_DENIED' in str(e):
                    raise ConnectorError("Connector does not have SELECT "
                                         f"permission on table {self.catalog}."
                                         f"{schema_name}.{table_name}.")
                else:
                    raise e

    # Utility functions #######################################################

    @blocking_method()
    def _run_sql(self, context: Any, operation: Callable[..., _SqlResult],
                 query: str, *args: Any) -> _SqlResult:
        return self._execute_sql(context, operation, query, *args)

    @retry(exceptions=WarehouseRetriableError, tries=3, delay=2, backoff=2,
           logger=None)
    def _execute_sql(self, context: Any, operation: Callable[..., _SqlResult],
                     query: str, *args: Any) -> _SqlResult:
        try:
            with restricted_diagnostic_boundary(
                    exception_types=(
                        DatabricksErrors.Error,
                        HTTPError,
                        OSError,
                        pa.ArrowException,
                    ), safe_message='Databricks SQL operation failed'):
                with context.cursor() as cursor:
                    return operation(cursor, query, *args)
        except HTTPError:
            raise WarehouseRetriableError('Databricks request failed') from None

    @blocking_method()
    def _get_df_result(self, context: Any, query: str) -> pd.DataFrame:
        return self._run_sql(context, _dataframe_result, query)

    def drop_table_if_exists(self, table_name: str) -> None:
        self._get_df_result(
            self.context,
            'DROP TABLE IF EXISTS '
            f'{self._fully_qualify_output_table_name(table_name)}',
        )

    def databricks_to_dtype(self, data_type: str) -> Dtype:
        dtype = dtype_for(data_type)
        if dtype == Dtype.unsupported:
            logger.warning(f"Unsupported databricks type: {data_type}")
        return dtype

    def get_column_metadata(
        self,
        context: Any,
        schema_name: Optional[str],
        table_names: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], Dtype]:
        from kumo_databricks.catalog import column_types

        if table_names and len(table_names) == 1:
            schema, table = self._target_table_coordinate(table_names[0])
            try:
                declared = self._run_sql(
                    context, dbx_identity.column_types,
                    dbx_sql.qualified_name(self.catalog, schema, table))
            except Exception as error:
                if is_not_found(error):
                    return {}
                raise
            columns = {
                (table, column): dtype
                for column, dtype in declared.items()
            }
        else:
            requested: Dict[str, List[str]] = {}
            for name in table_names or []:
                table = self._target_table_coordinate(name)[1]
                requested.setdefault(table.casefold(), []).append(table)
            with context.cursor() as cursor:
                columns = column_types(cursor, catalog=self.catalog,
                                       schema=schema_name,
                                       target_schema=self.schema_name)
            if requested:
                columns = {
                    (requested_name, column): dtype
                    for (table, column), dtype in columns.items()
                    for requested_name in requested.get(table.casefold(), [])
                }
        return {
            key: self.databricks_to_dtype(dtype)
            for key, dtype in columns.items()
        }

    # =========================================================================
    # Export
    # =========================================================================

    @blocking_method()
    def export_table(
        self,
        source_name: str,
        table_fp: IngestedTableFingerprint,
        data_version: str,
        *,
        graph_table_count: Optional[int] = None,
    ) -> ExportMetrics:
        from kumo.connector.source_bridge import read_source
        return read_source(self._source_reader(),
                           self._reader_table_name(source_name), table_fp,
                           data_version, graph_table_count=graph_table_count)

    @blocking_method()
    def export_query(
        self,
        sql: str,
        table_fp: SqlTableFingerprint,
        data_version: str,
    ) -> ExportMetrics:
        """Query counterpart of :meth:`export_table` for a Custom SQL Table.

        The same reader, given the SELECT in parenthesized subquery position
        instead of a table name, so the projection, the staged-then-streamed
        fallback, the clear-before-write and the row/byte counts are the ones
        `export_table` already gets. The fingerprint names the staging prefix
        because a query has no source table to name it after.
        """
        from kumo.connector.source_bridge import read_source
        return read_source(self._source_reader(), table_fp.ingest_path_fpid,
                           table_fp, data_version, sql=sql)

    def download(
        self,
        source_name: str,
        table_fp: BaseTableFingerprint,
        data_version: str,
        local_dest: str,
    ) -> None:
        from kumo.data import MaterializationStage
        from kumo.data.utils import get_materialize_path
        fsutil.copy(
            get_materialize_path(table_fp, data_version,
                                 MaterializationStage.INGEST), local_dest,
            overwrite=True)

    def _reader_table_name(self, source_name: str) -> str:
        """The reader qualifies with its own catalog and schema, so it is
        handed the bare table. A coordinate the rest of this connector accepts
        -- ``table``, ``schema.table``, ``catalog.schema.table`` -- would
        otherwise be re-qualified into ``catalog.schema.'catalog.schema.table'``
        and reported as a missing table.
        """
        return self._target_table_coordinate(source_name)[1]

    def _source_reader(self, *, staged: bool = True):
        # Lazy: keeps the reader package off the import path of every
        # process that imports this module.
        from kumo_databricks import DatabricksSourceReader

        # A reader with no volume offers only STREAMED, so declining staging
        # here is what keeps `ensure_managed_volume`'s CREATE from running at
        # all -- pinning the mode alone would still have issued it.
        volume, client = self._staging_volume() if staged else (None, None)
        return DatabricksSourceReader(self.context, catalog=self.catalog,
                                      schema=self.schema_name,
                                      workspace_client=client,
                                      staging_volume_path=volume)

    def _staging_volume(self) -> Tuple[Optional[str], Any]:
        """Offering staging we cannot actually write turns a read that would
        have streamed into a hard failure: a missing volume is reported as
        absent rather than as denied, and so never reaches the fallback.
        """
        try:
            self.ensure_managed_volume()
            connection = self._to_databricks_connection_info()
            client_args = DatabricksClient(connection).workspace_client_kwargs()
            client = WorkspaceClient(**client_args)
        except Exception as error:
            logger.info(
                'Databricks staging is unavailable, so %s will be streamed '
                '(error_type=%s)', self.schema_name,
                type(error).__name__)
            return None, None
        return (f'/Volumes/{self.catalog}/{self.schema_name}/'
                f'{dbx_output.STAGING_VOLUME}/source-reader'), client

    @blocking_method()
    def export_table_read_only(
        self,
        source_name: str,
        destination: str,
        cols: Optional[Dict[str, Dtype]] = None,
        sql: Optional[str] = None,
    ) -> ExportMetrics:
        """Stream a read-only SELECT into a Kumo-owned parquet path.

        :meth:`export_table` addressed by destination instead of by
        fingerprint, for a custom training or prediction table. ``sql`` reads a
        validated Custom SQL query in subquery position; the caller normalizes
        it, exactly as the graph-ingest activity does.

        Read-only is the contract the name states and three call sites rely
        on, so the mode is pinned rather than offered: a staged read unloads
        through ``INSERT OVERWRITE DIRECTORY`` into the customer's Unity
        Catalog, which is a write to the source account for a read this method
        promises does not make one.
        """
        from kumo_core.sources.types import ReadMode

        from kumo.connector.source_bridge import stage_source

        # Resolved before the schema round trip, so a coordinate outside this
        # connector's namespace is refused by name rather than after a query.
        # For a query, `source_name` is not a table -- it is only what staging
        # prefixes and errors key on -- so it is not a coordinate to normalize.
        table = (source_name
                 if sql is not None else self._reader_table_name(source_name))
        if cols is None:
            metadata = (self.get_query_metadata(sql) if sql is not None else
                        self.get_table_metadata([source_name])[0])
            cols = self._readable_columns(metadata, source_name)
        # A custom table is addressed by `destination`, not keyed by a data
        # version, so the reader's stamp is discarded by `stage_source`.
        # Supplying one skips deriving it, which is a `DESCRIBE HISTORY` per
        # export -- on top of the one `get_table_metadata` already issued
        # above when it resolved the schema.
        return stage_source(self._source_reader(staged=False), table,
                            destination, columns=cols, sql=sql,
                            mode=ReadMode.STREAMED,
                            version=_CUSTOM_SOURCE_VERSION)

    def _readable_columns(
        self,
        metadata: TableMetadataConfig,
        source_name: str,
    ) -> Dict[str, Dtype]:
        """The source's own schema, minus what Kumo cannot read.

        A custom table is exported without a caller-supplied projection, so
        the projection is the source's schema. A Databricks type Kumo has no
        ``Dtype`` for resolves to :attr:`Dtype.unsupported`, and selecting it
        would decode and persist a column no part of Kumo can consume, on a
        table the user may only have referenced three columns of. Dropping it
        here is the contract the retired direct path stated: unselected
        source columns do not affect ingestion.

        That set is wider than the composite types: ``databricks_to_dtype``
        matches array types by exact string, so only ``ARRAY<FLOAT|DOUBLE|
        INT|BIGINT|STRING>`` resolve and ``ARRAY<SMALLINT>``,
        ``ARRAY<BOOLEAN>``, ``ARRAY<DATE>``, ``ARRAY<TIMESTAMP>`` and nested
        ``ARRAY<ARRAY<...>>`` do not; nor does ``TIMESTAMP_LTZ``. Widening
        that allowlist is the fix and is not this change.

        A *required* column that is missing by name is caught before any
        export, by ``missing_source_column_errors``
        (``custom_table_validation.py``), which reads the same source schema.
        Its *type* is not judged there: a warehouse's type metadata is a
        second vocabulary, and the column is gated on the parquet that
        arrives, against the contract DiskGraph exports. So a column in this
        set surfaces at that gate rather than at registration.
        """
        readable = {
            column.name: column.dtype
            for column in metadata.cols if column.dtype != Dtype.unsupported
        }
        dropped = [
            column.name for column in metadata.cols
            if column.dtype == Dtype.unsupported
        ]
        if dropped:
            logger.info(
                'Databricks custom source %s has %d column(s) of a type Kumo '
                'cannot read; they are not selected: %s', source_name,
                len(dropped), ', '.join(sorted(dropped)))
        if not readable:
            raise ConnectorError(
                f"No column of Databricks custom source {source_name!r} has a "
                f"type Kumo can read: {', '.join(sorted(dropped))}. Cast or "
                "drop these columns in the source table or query.")
        return readable

    # Custom SQL query pushdown ###############################################

    @blocking_method()
    def get_query_metadata(self, sql: str) -> TableMetadataConfig:
        r"""Describe an ordered query schema without scanning rows."""
        from kumo_databricks.identity import query_columns

        try:
            columns = self._run_sql(self.context, query_columns, f'({sql})')
        except Exception as err:
            detail = _databricks_query_error_detail(err)
            if detail is not None:
                raise ConnectorError(detail) from None
            raise ConnectorError(
                "Databricks failed to describe the custom SQL query", error=err)

        cols = [
            BaseColumnConfig(name=name,
                             dtype=self.databricks_to_dtype(data_type))
            for name, data_type in columns
        ]
        return TableMetadataConfig(table_name='', source_name='',
                                   data_source=self._name, cols=cols)

    @blocking_method()
    def get_query_num_rows(self, sql: str) -> int:
        """Return the exact row count of a validated read-only query."""
        from kumo.data.custom_sql import (
            advisory_read_only_check,
            subquery_safe_sql,
            validate_connector_sql_scope,
        )
        connector_type = ConnectorType.DATABRICKS
        advisory_read_only_check(sql, connector_type)
        validate_connector_sql_scope(sql, connector_type, self)
        query = subquery_safe_sql(sql, connector_type)
        result = self._get_df_result(
            self.context,
            f'SELECT COUNT(*) AS row_count FROM ({query})',
        )
        return int(result.iloc[0]['row_count'])

    @blocking_method()
    def sample_query(self, sql: str, num_rows: int) -> pd.DataFrame:
        r"""Return a bounded preview of a read-only SELECT's result.

        Wraps ``sql`` in subquery position and bounds it — ``SELECT * FROM
        (<sql>) LIMIT <num_rows>`` — so the user's SQL is never rewritten.
        Databricks has no client-side per-query timeout, so the preview is
        bounded by the session ``STATEMENT_TIMEOUT`` (reset afterwards so the
        shared connection isn't left with a reduced ceiling).
        """
        num_rows = int(num_rows)
        if num_rows < 0:
            raise ValueError(
                f'sample_query requires a non-negative row limit, got {num_rows}.'
            )
        try:
            self._get_df_result(
                self.context, 'SET STATEMENT_TIMEOUT = '
                f'{DATABRICKS_SAMPLE_QUERY_TIMEOUT_SECONDS}')
            return self._run_sql(self.context, _sample_dataframe, f'({sql})',
                                 num_rows)
        except DecimalOutOfBoundsError as err:
            raise ConnectorError(
                f"Databricks DECIMAL column '{err.column_name}' contains a "
                "value outside Kumo's signed 64-bit integer range. Cast the "
                "column to DOUBLE or STRING in Custom SQL, or keep its values "
                "within the signed 64-bit range.") from None
        except Exception as err:
            detail = _databricks_query_error_detail(err)
            if detail is not None:
                raise ConnectorError(detail) from None
            raise ConnectorError("Databricks sample query failed", error=err)
        finally:
            self._reset_sample_statement_timeout()

    def _reset_sample_statement_timeout(self) -> None:
        r"""Clear the session ``STATEMENT_TIMEOUT`` set for a preview.

        Kumo never sets a session-level timeout on the connector's connection,
        so 0 (no session override) restores the as-connected state and lets any
        warehouse-level default apply — also what subsequent long-running
        operations (e.g. an export) on this shared connection want, so we clear
        rather than re-impose the short preview ceiling. If the reset itself
        fails, the connection is invalidated (closed, so the next use reopens a
        fresh session) — a *successful* preview must never leave the shared
        connection silently pinned to the 60s preview ceiling.
        """
        try:
            self._get_df_result(self.context, 'SET STATEMENT_TIMEOUT = 0')
        except Exception as reset_err:
            logger.warning(
                "Failed to reset Databricks STATEMENT_TIMEOUT after a sample "
                "query; invalidating the connection so it reopens with the "
                "default timeout: %s", reset_err)
            try:
                self.close()
            except Exception as close_err:
                # Swallow: this runs in sample_query's `finally`, so letting it
                # escape would mask a successful preview's return value. The
                # load-bearing step is dropping `_context` below.
                logger.warning(
                    "Error while closing the invalidated Databricks "
                    "connection after a failed STATEMENT_TIMEOUT reset: %s",
                    close_err)
            finally:
                # Force the poisoned session to be dropped even if close()
                # raised, so the next `context` access reopens cleanly.
                self._context = None


def _databricks_server_error_code(
        error: DatabricksErrors.ServerOperationError) -> Optional[str]:
    """Return one exact Databricks SQL error code without exposing details."""
    context = getattr(error, 'context', None)
    if isinstance(context, dict):
        error_code = context.get('error_code')
        if isinstance(error_code, str):
            return error_code

    # Current databricks-sql-python exposes the provider's structured code
    # only as the leading bracketed token in ``displayMessage``. Parse just
    # that token, then compare it against an allowlist at the callsite.
    message = getattr(error, 'message', None)
    if not isinstance(message, str):
        return None
    match = re.match(r'^\[([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)*)\](?:\s|$)',
                     message)
    return match.group(1) if match else None


def _databricks_error_class(
        error: DatabricksErrors.ServerOperationError) -> Optional[str]:
    """Return the structured error class for target-validation classification.

    Extends :func:`_databricks_server_error_code` (a context ``error_code`` or a
    clean leading ``[<code>]`` token) with the Unity Catalog message form
    ``[RequestId=... ErrorClass=<code>] <detail>``, where the structured code is
    embedded in the leading metadata bracket. Only that structured token — never
    the free-form detail that follows the bracket — is ever returned, so a
    schema or table name cannot be mistaken for the provider's verdict.
    """
    code = _databricks_server_error_code(error)
    if code is not None:
        return code
    message = getattr(error, 'message', None)
    if not isinstance(message, str):
        return None
    bracket = re.match(r'^\[([^\]]*)\]', message)
    if bracket is None:
        return None
    match = re.search(r'ErrorClass=([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)*)',
                      bracket.group(1))
    return match.group(1) if match else None


def _databricks_error_class_is(error_class: Optional[str], base: str) -> bool:
    """True when the structured error class is ``base`` or a ``base.<subclass>``
    specialization. Databricks appends dotted subclasses, e.g.
    ``SCHEMA_DOES_NOT_EXIST.RESOURCE_DOES_NOT_EXIST``.
    """
    return error_class is not None and (error_class == base
                                        or error_class.startswith(f'{base}.'))


_DATABRICKS_QUERY_ERRORS = {
    'PARSE_SYNTAX_ERROR':
    'Databricks could not parse the query.',
    'TABLE_OR_VIEW_NOT_FOUND':
    'Databricks could not find a referenced table or view.',
    'UNBOUND_SQL_PARAMETER':
    'Databricks does not accept unbound parameters in Custom SQL.',
    'UNRESOLVED_COLUMN':
    'Databricks could not resolve a referenced column.',
    'UNRESOLVED_ROUTINE':
    'Databricks could not resolve a referenced function.',
    'UNRESOLVED_TABLE_VALUED_FUNCTION':
    'Databricks could not resolve a referenced table function.',
    'TABLE_VALUED_FUNCTION_NOT_FOUND':
    'Databricks could not find a referenced table function.',
    'DELTA_TIMESTAMP_GREATER_THAN_COMMIT':
    'Databricks could not read the requested historical table version.',
    'DELTA_TIME_TRAVEL_INVALID_BEGIN_VALUE':
    'Databricks could not read the requested historical table version.',
    'INVALID_TEMPORAL_VERSION':
    'Databricks could not read the requested historical table version.',
}


def _databricks_query_error_detail(error: Exception) -> Optional[str]:
    if not isinstance(error, DatabricksErrors.ServerOperationError):
        return None
    code = _databricks_server_error_code(error)
    if code is None:
        return None
    return next((message for base, message in _DATABRICKS_QUERY_ERRORS.items()
                 if code == base or code.startswith(f'{base}.')), None)


def _quote_string(value: str) -> str:
    """Quote one Databricks SQL string literal."""
    return dbx_sql.quote_literal(value)


# Explicit dataclass definition ###############################################

DatabricksConnectorConfig = to_dataclass(DatabricksConnector,
                                         base_cls=ConnectorConfig)
register(DatabricksConnector, DatabricksConnectorConfig, group='connector')
