# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import snowflake.connector.errors as SnowflakeErrors
import snowflake.snowpark as snowpark
from cryptography.hazmat.primitives import serialization
from kumo_core.sources.identity import hour_version
from kumo_core.sources.sql import resolve_snowflake_identifier
from kumo_core.types.schema import (
    BaseColumnConfig,
    BaseTableFingerprint,
    IngestedTableFingerprint,
    SqlTableFingerprint,
)
from kumo_core.utils.registry import register
from kumoapi.typing import Dtype
from snowflake.connector import connect
from snowflake.connector.errors import DatabaseError, ProgrammingError

from io_utils import fsutil, kumo_dataclass, logging
from io_utils.errors import restricted_diagnostic_boundary
from io_utils.snowflake import session_logging as snowflake_session_logging
from kumo.config import ConnectorConfig, TableMetadataConfig
from kumo.config.graph import ForeignKeyConfig, PrimaryKeyConfig
from kumo.connector.artifact_staging import (
    local_parquet_files,
    selected_local_parquet_files,
)
from kumo.connector.blocking import blocking_method
from kumo.connector.connector import (
    ConnectorError,
    ExportMetrics,
    SourceTableNotFoundError,
    SourceTableUnavailableError,
)
from kumo.connector.sql_connector import InternalWriteMode, SQLConnector
from kumo.connector.utils import get_temporary_aws_credential
from kumo.utils import to_dataclass
from service_framework.deploy_env import get_snowflake_connection_string
from statedb.utils.errors import ConnectorValidationDiagnosticCode

logger = logging.getLogger(__name__)

# Snowflake collapses "missing object" and "ungranted object" into the same
# object-resolution compilation errors on the USE/SELECT validation path, so
# only these codes are a genuine target-configuration problem:
#   002003 - Object does not exist or not authorized (bad/ungranted
#            database/schema, the dominant case on USE DATABASE/USE SCHEMA).
#   002043 - Object does not exist, or operation cannot be performed (the
#            equivalent object-resolution failure, e.g. an ungranted or
#            missing warehouse on USE WAREHOUSE).
# Every other errno Snowflake can raise here (timeouts, cancellations,
# internal/transient failures) must stay unclassified so it is never mislabeled
# as a non-retriable target error.
_SNOWFLAKE_TARGET_INACCESSIBLE_ERRNOS = frozenset({2003, 2043})

# SQLSTATE 42501 is Snowflake's access-control violation ("Insufficient
# privileges to operate on ..."): the target exists but the role lacks the
# required grant — a genuine, actionable access denial distinct from the
# not-found/not-authorized compilation errors above.
_SNOWFLAKE_ACCESS_DENIED_SQLSTATE = '42501'

# Allowed path schemes for export/write operations. `dest_path` and
# `artifact_path` are interpolated into SQL strings (COPY INTO '<url>'),
# so any accepted value must be constrained to well-formed cloud storage
# URLs or Snowflake stage references to prevent SQL injection.
_ALLOWED_BLOB_PATH_RE = re.compile(
    r"^(s3://|gcs://|azure://|@)",
    flags=re.IGNORECASE,
)


def _microsecond_precision_kwargs(method: Any) -> Dict[str, bool]:
    from kumo_snowflake.source import microsecond_precision_kwargs
    return microsecond_precision_kwargs(method)


def _restore_in_range_timestamp_precision(table: pa.Table) -> pa.Table:
    from kumo_snowflake.results import restore_timestamp_precision
    return restore_timestamp_precision(table)


def _to_snowflake_stage_sql_ref(*args, **kwargs):
    from kumo_snowflake import write
    return write._to_snowflake_stage_sql_ref(*args, **kwargs)


def _validate_blob_path(*args, **kwargs):
    from kumo_snowflake import write
    try:
        return write._validate_blob_path(*args, **kwargs)
    except write.ArtifactWriteError as error:
        raise ConnectorError(str(error)) from None


# Snowflake caps an explicit COPY INTO file list at 1000 entries.
MAX_COPY_FILES = 1000


def _validate_selected_artifact_files(*args, **kwargs):
    from kumo_snowflake import write
    try:
        return write._validate_selected_artifact_files(*args, **kwargs)
    except write.ArtifactWriteError as error:
        raise ConnectorError(str(error)) from None


def _container(*args, **kwargs):
    from kumo_snowflake import write
    return write._container(*args, **kwargs)


def _common_prefix(*args, **kwargs):
    from kumo_snowflake import write
    try:
        return write._common_prefix(*args, **kwargs)
    except write.ArtifactWriteError as error:
        raise ConnectorError(str(error)) from None


def _relative_to(*args, **kwargs):
    from kumo_snowflake import write
    return write._relative_to(*args, **kwargs)


def _quoted_files_clause(*args, **kwargs):
    from kumo_snowflake import write
    return write._quoted_files_clause(*args, **kwargs)


def _upload_local_parquet_to_user_stage(
        context: Any, artifact_path: str, stage_ref: str, *,
        artifact_files: Optional[List[str]] = None) -> None:
    from kumo_snowflake.write import _upload_local_parquet_to_user_stage
    return _upload_local_parquet_to_user_stage(
        context, artifact_path, stage_ref, artifact_files=artifact_files,
        resolve_files=local_parquet_files)


# names of environment variables used for reading snowflake credentials
_ENV_SNOWFLAKE_USER = 'SNOWFLAKE_USER'
_ENV_SNOWFLAKE_PASSWORD = 'SNOWFLAKE_PASSWORD'
_ENV_SNOWFLAKE_PRIVATE_KEY = 'SNOWFLAKE_PRIVATE_KEY'

_SNOWFLAKE_CONNECTION_ERROR = 'Could not connect to Snowflake backend'
_SNOWFLAKE_OAUTH_ACCESS_ERROR = ('does not exist, or operation cannot ' +
                                 'be performed')

SNOWFLAKE_CONNECTOR_PERMISSIONS_CHECK_TIMEOUT = 30

# Client-side statement timeout (seconds) for a Custom SQL preview
# (`sample_query`). A preview is bounded by LIMIT and should return quickly;
# the timeout cancels a runaway query rather than pinning the warehouse.
SNOWFLAKE_SAMPLE_QUERY_TIMEOUT_SECONDS = 60


# TODO(Dulloor): Add these fields to TableMetadataConfig so they
# can be cached
@kumo_dataclass
class TableInfo:
    source_name: str
    kind: str
    # created_on: Dtype.time.numpy()
    created_on: str
    num_rows: int
    bytes: int


class SnowflakeConnector(SQLConnector):
    r"""Establishes a connection to a `Snowflake <https://www.snowflake.com/>`_
    database. The following credentials are supported for this database:
    * An account username, either passed as `user` or set as `SNOWFLAKE_USER`
    * An account password, either passed as `password` or set as
    `SNOWFLAKE_PASSWORD`

    Args:
        name (str): The name of the connector.
        account (str): The account name.
        warehouse (str): The name of the warehouse.
        database (str): The name of the database.
        schema_name (str): The name of the schema.
    """
    def __init__(
        self,
        name: str,
        account: str,
        warehouse: str,
        database: str,
        schema_name: str,
        role: str | None = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.account = account
        self.warehouse = warehouse
        # Keep identifier syntax in the serialized connector so quoted names
        # survive config round-trips. Preserve the historical canonical form
        # for ordinary database/schema names. Operational paths resolve these
        # fields through the properties below so the connector has one source
        # of truth for each configured target.
        # Empty targets are reserved for credential-only identity validation,
        # which opens a connection without selecting a database or schema.
        # Operational connectors still resolve every non-empty declaration.
        self.database = (database if not database or database.startswith('"')
                         else resolve_snowflake_identifier(database))
        self.schema_name = (schema_name
                            if not schema_name or schema_name.startswith('"')
                            else resolve_snowflake_identifier(schema_name))
        # TODO: Currently this role is propagated only to context created in
        # open() function. This needs to be included for all Session.builder
        # calls as well.
        self.role = role

    def _get_connector_type(self) -> str:
        return "Snowflake"

    @property
    def resolved_warehouse(self) -> str:
        """The exact catalog name, with connector syntax removed."""
        return resolve_snowflake_identifier(self.warehouse)

    @property
    def resolved_database(self) -> str:
        """The exact catalog name, with connector syntax removed."""
        return resolve_snowflake_identifier(self.database)

    @property
    def resolved_schema(self) -> str:
        """The exact catalog name, with connector syntax removed."""
        return resolve_snowflake_identifier(self.schema_name)

    @property
    def resolved_role(self) -> str | None:
        """The exact catalog name, with connector syntax removed."""
        return (resolve_snowflake_identifier(self.role)
                if self.role is not None else None)

    @staticmethod
    def _safe_table_info_int(value: Any, field_name: str,
                             source_name: str) -> int:
        """Best-effort cast for SHOW TABLES metadata fields."""
        numeric_value = pd.to_numeric(value, errors='coerce')
        if pd.isna(numeric_value):
            logger.warning(
                "Unable to parse Snowflake table metadata field '%s' for table "
                "'%s' from value %r. Defaulting to 0.",
                field_name,
                source_name,
                value,
            )
            return 0

        # Keep behavior predictable when metadata returns non-integer numerics.
        if isinstance(numeric_value, float) and not numeric_value.is_integer():
            logger.warning(
                "Snowflake table metadata field '%s' for table '%s' had "
                "non-integer value %r. Truncating to int.",
                field_name,
                source_name,
                value,
            )
        return int(numeric_value)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(account={self.account}, database={self.database}, '
                f'schema={self.schema_name}, role={self.role})')

    def validate_permissions(self):
        """Validate permissions with timeout using threads."""
        def _sync_validation():
            cursor = None
            try:
                cursor = self.context.cursor()
                cursor.execute(
                    f"USE DATABASE {_quote_id(self.resolved_database)}")
                cursor.execute(f"USE SCHEMA {_quote_id(self.resolved_schema)}")

                if self.warehouse:
                    cursor.execute(
                        f"USE WAREHOUSE {_quote_id(self.resolved_warehouse)}")

                cursor.execute(
                    """
                    SELECT TABLE_NAME
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = %s
                    AND TABLE_TYPE = 'BASE TABLE'
                    LIMIT 1
                    """, (self.resolved_schema, ))

                table_row = cursor.fetchone()
                if not table_row:
                    return

                table_name = table_row[0]
                cursor.execute(f'SELECT * FROM {_quote_id(table_name)} LIMIT 1')
                cursor.fetchone()

            except (ProgrammingError, DatabaseError) as error:
                # Direct Connector mode (SaaS/on-prem).
                #
                # Classify only Snowflake's genuine target-config failures. An
                # access-control violation (SQLSTATE 42501) is a clear access
                # denial on an existing object. A missing and an ungranted
                # database/schema/warehouse collapse into the same not-found
                # compilation code, so those are the ambiguous
                # ``target_inaccessible``. Operational/internal/transient
                # failures keep the same guidance message but stay unclassified
                # so they are not mislabeled as a non-retriable target error.
                if error.sqlstate == _SNOWFLAKE_ACCESS_DENIED_SQLSTATE:
                    diagnostic_code = (
                        ConnectorValidationDiagnosticCode.TARGET_ACCESS_DENIED)
                elif error.errno in _SNOWFLAKE_TARGET_INACCESSIBLE_ERRNOS:
                    diagnostic_code = (
                        ConnectorValidationDiagnosticCode.TARGET_INACCESSIBLE)
                else:
                    diagnostic_code = None
                raise ConnectorError(
                    f"Access checks failed. The user role does not have sufficient privileges "
                    f"on database '{self.database}', schema '{self.schema_name}', or warehouse '{self.warehouse}'.\n\n"
                    f"Please grant the required privileges by following the instructions at:\n"
                    "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/"
                    "kumo-studio-guide-coming-soon/preliminary-documentation/"
                    "connect-data/data-connectors"
                    "#minimum-privileges-required-for-the-connecting-user",
                    diagnostic_code=diagnostic_code,
                ) from None
            finally:
                if cursor:
                    cursor.close()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_sync_validation)
        timed_out = False
        try:
            future.result(timeout=SNOWFLAKE_CONNECTOR_PERMISSIONS_CHECK_TIMEOUT)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            raise ConnectorError(
                'Snowflake connection validation timed out') from None
        finally:
            # A ThreadPoolExecutor context manager always waits for a running
            # future on exit, which would negate the timeout.  Provider calls
            # cannot be forcefully stopped, so detach the timed-out worker and
            # fail the request immediately; normal completions are joined.
            executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

    @blocking_method()
    def _open(self, credentials: Optional[Dict] = None) -> Any:

        try:
            if self._require_explicit_credentials and not credentials:
                raise ConnectorError(
                    'Identity-backed Snowflake connectors require explicit '
                    'credentials.')
            # Authenticate via user/password or key auth directly.
            connection_args = self.get_user_pass_or_key(
                self.__repr__(), _ENV_SNOWFLAKE_USER, _ENV_SNOWFLAKE_PASSWORD,
                credentials, key_encoding=serialization.Encoding.DER,
                env_var_private_key=_ENV_SNOWFLAKE_PRIVATE_KEY)

            connection_args.update({
                'account':
                self.account,
                'warehouse':
                self.warehouse,
                'database':
                self.database,
                'schema':
                self.schema_name,
                'role':
                self.role,
                'client_session_keep_alive':
                True,
                'validate_default_parameters':
                True,
                'login_timeout':
                5,
                'application':
                get_snowflake_connection_string()
            })

            context = snowflake_session_logging.ensure_logged_connector_connection(
                connect(**connection_args), )
            return context
        except SnowflakeErrors.Error as err:
            if _SNOWFLAKE_CONNECTION_ERROR in str(err) or isinstance(
                    err, SnowflakeErrors.InterfaceError):
                raise ConnectorError(
                    "Error in connecting to the provided Snowflake account. " +
                    "This usually happens when the account identifier is " +
                    "incorrect or there is a network error in connecting to " +
                    "the provided Snowflake account. " +
                    "Check that the account identifier is the form " +
                    "ORGNAME-ACCOUNT_NAME. " +
                    "See https://docs.snowflake.com/en/user-guide/admin-account-identifier#finding-the-organization-and-account-name-for-an-account "  # noqa: E501
                    + "to retrieve ORGNAME and ACCOUNT_NAME.",
                    error=err) from None

            raise ConnectorError(
                "Error in connecting to the provided Snowflake account.",
                error=err) from None

    # Reading table metadata ##################################################

    # TODO(Dulloor): implement _get_table_version from table_info or
    # information_schema

    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:
        tables_info_dict = self.get_tables_info(self.context)
        table_info = tables_info_dict.get(table_name)
        if table_info is None or table_info.num_rows == 0:
            logger.warning(f"could not retrieve Snowflake table information "
                           f"for table {table_name}")
            # Depending on the Snowflake setup/permissions,
            # `show tables from schema ...` sometimes does not return the full
            # list of tables (https://github.com/kumo-ai/kumo/issues/3617).
            # Revert back to using `select count(*) ...` to get num_rows
            return super()._get_table_num_rows(table_name=table_name)

        return table_info.num_rows

    def _query_info_table(self, table_name: str, query_col: str) -> int:
        '''
        Queries the INFORMATION_SCHEMA.TABLES info table in the connector's
        data base and returns the value of query_col for the row corresponding
        to table with table_name

        Args:
            table_name (str): table to find in the info table
            query_col (str): column of the info table to get the value of
        Returns:
            value of query_col for table_name (default to 0 rather than raising
            an exception)
        '''
        try:
            if query_col not in {'BYTES', 'ROW_COUNT'}:
                raise ValueError(
                    f'Unsupported table metadata field: {query_col}')
            query = (f"select TABLE_NAME,{query_col} from "
                     f"{_quote_id(self.resolved_database)}"
                     ".INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA=%s "
                     "and TABLE_NAME=%s")
            df = self.execute_meta_query(
                self.context, query, params=(self.resolved_schema, table_name))
            return df.loc[0][query_col]
        except Exception as e:
            logger.warning(
                "Could not retrieve Snowflake table information column %s "
                "for table %r (error_type=%s)", query_col, table_name,
                type(e).__name__)
            return 0

    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        # method 1 for size: get_tables_info
        tables_info_dict = self.get_tables_info(self.context)
        table_info = tables_info_dict.get(table_name)
        table_bytes = table_info.bytes if table_info is not None else 0

        if table_bytes == 0 or table_bytes is None:
            # method 2: query the info table
            table_bytes = self._query_info_table(table_name, 'BYTES')

        if table_bytes == 0 or table_bytes is None:
            # method 3: default sql connector method
            table_bytes = super()._get_table_source_size(table_name=table_name)

        # never return None- missing source size isn't blocking
        return (table_bytes if table_bytes is not None else 0)

    # Reading table metadata ##################################################

    def _get_data_version(
        self,
        table_name: str,
    ) -> str:
        from kumo_core.sources.errors import SourceNotFoundError
        from kumo_snowflake.identity import data_version
        try:
            cursor = self.context.cursor()
            try:
                return data_version(cursor, database=self.resolved_database,
                                    schema=self.resolved_schema,
                                    table=table_name)
            finally:
                cursor.close()
        except SourceNotFoundError:
            raise SourceTableUnavailableError(table_name, self.name) from None
        except (SourceTableNotFoundError, SourceTableUnavailableError):
            raise
        except Exception as e:
            logger.warning(
                "Could not retrieve Snowflake data version for table %r "
                "(error_type=%s)", table_name,
                type(e).__name__)

        # Default to current time. This is for Snowflake views and
        # situations where the above query fails.
        logger.info(f"using default data version for Snowflake view "
                    f"{table_name} because last_altered is not updated"
                    f"for views")
        return hour_version()

    def has_links_support(self) -> bool:
        return True

    def is_table_view(self, table_name: str) -> bool:
        """Return whether an exact Snowflake source is a view.

        Raises:
            SourceTableUnavailableError: If the source is missing or hidden by
                the selected identity's access controls.
        """
        from kumo_snowflake.catalog import table_type
        from kumo_snowflake.identity import VIEW_TYPES
        with restricted_diagnostic_boundary(
                safe_message='Snowflake metadata query failed',
                exception_types=SnowflakeErrors.Error):
            cursor = self.context.cursor()
            try:
                kind = table_type(cursor, database=self.resolved_database,
                                  schema=self.resolved_schema, table=table_name)
            finally:
                cursor.close()
        if kind is None:
            raise SourceTableUnavailableError(table_name, self.name)
        return kind in VIEW_TYPES

    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        from kumo_snowflake.catalog import tables
        with restricted_diagnostic_boundary(
                safe_message='Snowflake metadata query failed',
                exception_types=SnowflakeErrors.Error):
            with self.context.cursor() as cursor:
                return tables(cursor, database=self.resolved_database,
                              schema=self.resolved_schema)

    def _fetch_table_metadata(
        self,
        table_names: Optional[List[str]] = None,
        skip_inferral: bool = False,
    ) -> Dict[str, TableMetadataConfig]:

        # Implemented as a separate function to enable mocking in tests
        primary_keys_dict = self.get_primary_keys(self.context)
        foreign_keys_dict = self.get_foreign_keys(self.context)
        column_metadata_dict = self.get_column_metadata(self.context)

        table_metadata_dict: Dict[str, TableMetadataConfig] = {}

        for (table_name, column_name), dtype in column_metadata_dict.items():
            if table_name not in table_metadata_dict:
                table_metadata_dict[table_name] = TableMetadataConfig(
                    table_name=table_name, source_name=table_name,
                    data_source=self._name)

            # NOTE: in case there are multiple primary keys, this will
            # add only one of them as pkey. The others will be added as
            # regular columns.
            if ((table_name, column_name) in primary_keys_dict
                    and table_metadata_dict[table_name].pkey is None):
                primary_key = primary_keys_dict[table_name, column_name]
                primary_key.dtype = dtype
                table_metadata_dict[table_name].pkey = primary_key
                del primary_keys_dict[table_name, column_name]

            elif (table_name, column_name) in foreign_keys_dict:
                foreign_key = foreign_keys_dict[table_name, column_name]
                foreign_key.dtype = dtype
                table_metadata_dict[table_name].fkeys.append(foreign_key)
                del foreign_keys_dict[table_name, column_name]

            else:
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
            logger.info(f"fetching Snowflake table metadata for {table_names}")

            # TODO: This function should not return metadata invalid for the
            # list of input tables. At the least we should filter out invalid
            # foreign keys which are pointing to tables outside of
            # table_names.

            # This function could be called on a per table basis.

            table_metadata_dict = self._fetch_table_metadata(
                table_names, skip_inferral)

            if table_names is not None and len(table_names) > 0:
                out: List[TableMetadataConfig] = []
                for table_name in table_names:
                    if table_name not in table_metadata_dict:
                        if ignore_errors:
                            continue
                        raise SourceTableUnavailableError(table_name, self.name)
                    out.append(table_metadata_dict[table_name])
                return out

            return list(table_metadata_dict.values())
        except ConnectorError:
            raise
        except Exception as err:
            logger.error(
                "Error retrieving Snowflake table metadata "
                "(error_type=%s)",
                type(err).__name__)
            raise ConnectorError(message=(
                f'Unable to retrieve table information from database '
                f'{self.database}, schema {self.schema_name}.')) from None

    def _read_table_sample(
        self,
        context: Any,
        table_name: str,
        num_rows: Optional[int],
        frac: Optional[float],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        try:
            return super()._read_table_sample(context, table_name, num_rows,
                                              frac, pkey, cols)
        except Exception:
            raise ConnectorError(
                message='Snowflake reading table sample failed') from None

    # Pre-ingest col-stats compute is the inherited default
    # ``Connector.compute_col_stats`` (sample via ``read_table_sample`` +
    # ``compute_col_stats_python`` on a pandas frame). A prior Snowpark-
    # pushdown override was over-engineered for the only use case here —
    # pre-ingest approx stats, where DG produces the exact stats post-
    # ingest. The pushdown also kept surfacing dtype gaps (binary cast
    # ``NotImplementedError``; ARRAY column multi-cat detection silently
    # falling through to ``Stype.text``). Pulling a 100k-row sample to
    # Python and reducing with pandas handles every Snowflake source
    # type (ARRAY → object, BINARY → bytes, etc.) without a cast layer.

    # Writing to Snowflake   ##################################################

    @blocking_method()
    def write_artifacts_to_table(
            self, table_name: str, artifact_path: str, replace: bool = False, *,
            artifact_files: Optional[List[str]] = None) -> None:
        """Delegate COPY policy to the shared Snowflake artifact owner."""
        from kumo_snowflake import write
        try:
            write.write_artifacts_to_table(
                self, table_name, artifact_path, replace,
                artifact_files=artifact_files,
                get_temporary_aws_credential=get_temporary_aws_credential,
                upload_local=_upload_local_parquet_to_user_stage,
                resolve_local=selected_local_parquet_files)
        except write.ArtifactWriteError as error:
            raise ConnectorError(str(error)) from None

    def overwrite_table(self, src_table_name: str, dst_table_name: str) -> None:
        from kumo_snowflake import write
        try:
            write.overwrite_table(self, src_table_name, dst_table_name,
                                  quote=_quote_id)
        except write.ArtifactWriteError as error:
            raise ConnectorError(str(error)) from None

    @blocking_method()
    def export_table_read_only(
        self,
        source_name: str,
        destination: str,
        cols: Optional[Dict[str, Dtype]] = None,
    ) -> ExportMetrics:
        from kumo.connector.source_bridge import stage_source

        sql = f'SELECT * FROM {self._qualified_table_name(source_name)}'
        if not cols:
            cols = {
                column.name: column.dtype
                for column in self.get_query_metadata(sql).cols
            }
        return stage_source(self._source_reader(), 'custom_table', destination,
                            columns=cols, sql=sql, version='custom-source')

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
        return read_source(self._source_reader(), source_name, table_fp,
                           data_version, graph_table_count=graph_table_count)

    def _source_reader(self):
        # Lazy: keeps `kumo_snowflake` off this module's import path. Not
        # the driver, which this module already imports at module scope --
        # what this defers is the reader package, and nothing more.
        from kumo_snowflake import SnowflakeSourceReader
        return SnowflakeSourceReader(self.context,
                                     database=self.resolved_database,
                                     schema=self.resolved_schema)

    def download(
        self,
        source_name: str,
        table_fp: BaseTableFingerprint,
        data_version: str,
        local_dest: str,
    ) -> None:
        """Copy Kumo-owned parquet shards to the pool-local ingest path."""
        from kumo.data import MaterializationStage
        from kumo.data.utils import get_materialize_path
        source = get_materialize_path(table_fp, data_version,
                                      MaterializationStage.INGEST)
        fsutil.copy(source, local_dest, overwrite=True)
        logger.info('Copied Snowflake parquet from Kumo persistence')

    # Custom SQL query pushdown ###############################################
    #
    # A Custom SQL Table pushes a read-only SELECT down to Snowflake and
    # ingests only its result. The customer's SQL is pass-through: it is sent
    # unchanged and only ever embedded in parenthesized subquery position
    # (`... FROM (<sql>)`), never rewritten, inlined, or concatenated into
    # DDL. Read-only enforcement lives upstream (sqlglot classification +
    # least-privilege credentials); these methods just describe, preview, and
    # unload the result.

    @blocking_method()
    def get_query_metadata(self, sql: str) -> TableMetadataConfig:
        r"""Describe a read-only SELECT's result schema without scanning data.

        Uses ``cursor.describe`` — Snowflake's describe-only compile — so the
        warehouse returns the output columns without executing the query or
        scanning any rows. Result columns use the provider's shared dtype
        mapping for downstream dtype/stype inference.

        ``sql`` is passed to describe unchanged. Only the result-set columns
        are populated here; ``table_name`` / ``source_name`` are naming
        concerns owned by the SQL-table definition (its synthetic alias) and
        are filled in by the caller.
        """
        from kumo_snowflake.sql import describe_columns

        try:
            cursor = self.context.cursor()
            try:
                columns = describe_columns(cursor, sql)
            finally:
                cursor.close()
        except Exception as err:
            raise ConnectorError(
                "Snowflake failed to describe the custom SQL query", error=err)

        cols = [
            BaseColumnConfig(name=name, dtype=dtype) for name, dtype in columns
        ]
        return TableMetadataConfig(table_name='', source_name='',
                                   data_source=self._name, cols=cols)

    @blocking_method()
    def sample_query(self, sql: str, num_rows: int) -> pd.DataFrame:
        r"""Return a bounded preview of a read-only SELECT's result.

        Wraps ``sql`` in subquery position and bounds it — ``SELECT * FROM
        (<sql>) LIMIT <num_rows>`` — so the user's SQL is never rewritten, and
        applies a warehouse statement timeout so a runaway preview is cancelled
        rather than pinning the warehouse.
        """
        from kumo_snowflake.results import sample
        try:
            return sample(self.context, f'({sql})', num_rows,
                          timeout=SNOWFLAKE_SAMPLE_QUERY_TIMEOUT_SECONDS)
        except Exception as err:
            raise ConnectorError("Snowflake sample query failed", error=err)

    @blocking_method()
    def export_query(
        self,
        sql: str,
        table_fp: SqlTableFingerprint,
        data_version: str,
    ) -> ExportMetrics:
        from kumo.connector.source_bridge import read_source
        return read_source(self._source_reader(), table_fp.ingest_path_fpid,
                           table_fp, data_version, sql=sql)

    # Utility functions #######################################################

    @blocking_method()
    def _get_df_result(self, context: Any, query: str,
                       timeout: Optional[int] = None):
        with restricted_diagnostic_boundary(
                safe_message='Snowflake query execution failed',
                exception_types=SnowflakeErrors.Error):
            cursor = context.cursor().execute(query, timeout=timeout)
        from kumo_snowflake.results import dataframe
        with restricted_diagnostic_boundary(
                safe_message='Snowflake query fetch failed',
                exception_types=SnowflakeErrors.Error):
            return dataframe(cursor)

    @blocking_method()
    def execute_meta_query(
        self,
        context: Any,
        query: str,
        creds_clause: Optional[str] = None,
        *,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> pd.DataFrame:
        r"""Execute a SQL query against Snowflake.

        When `creds_clause` is provided, it is appended to `query` only at the
        execution boundary. Callers should use this parameter for queries that
        embed STS tokens (COPY INTO or CREATE STAGE with CREDENTIALS). Kumo
        never copies either SQL argument into centralized logs.

        Catalog names and other values belong in ``params`` rather than SQL
        text. This matters even after identifier syntax has been resolved:
        exact quoted names may legally contain an apostrophe.
        """
        # Query text can contain customer literals or temporary credentials.
        # Execution metadata is logged by the wrapped Snowflake cursor without
        # copying SQL into centralized logs.
        logger.info("Executing Snowflake metadata query")
        full_sql = f"{query}\n{creds_clause}" if creds_clause else query
        with restricted_diagnostic_boundary(
                safe_message='Snowflake metadata query failed',
                exception_types=SnowflakeErrors.Error):
            cursor = context.cursor()
            if params is None:
                cursor.execute(full_sql)
            else:
                cursor.execute(full_sql, params)
            hdrs = pd.DataFrame(cursor.description)
            metadata = cursor.fetchall()
        df = pd.DataFrame(metadata, columns=hdrs['name'])
        return df

    def drop_table_if_exists(self, table_name: str) -> None:
        self.execute_meta_query(
            self.context,
            f'DROP TABLE IF EXISTS {self._qualified_table_name(table_name)}')

    def table_exists(self, table_name: str) -> bool:
        """Check one exact Snowflake table or view without listing schema."""
        cursor = None
        try:
            cursor = self.context.cursor()
            from kumo_snowflake.catalog import table_exists
            return table_exists(cursor, database=self.resolved_database,
                                schema=self.resolved_schema, table=table_name)
        except (ProgrammingError, DatabaseError) as error:
            raise ConnectorError(
                'Unable to validate exact Snowflake table access.',
                error=error,
            ) from None
        finally:
            if cursor is not None:
                cursor.close()

    def table_columns(self, table_name: str) -> List[str]:
        """Adapt hosted metadata to the shared artifact-writer interface."""
        values = self.get_column_metadata(self.context, table_name=table_name)
        return [column for (_table, column) in values]

    def get_tables_info(self, context: Any) -> Dict[str, Any]:
        tables_info_dict: Dict[str, Any] = {}

        query = f'show tables in schema {self._qualified_schema_name()};'
        df = self.execute_meta_query(context, query)

        for i in range(len(df)):
            source_name = df.loc[i]['name']
            kind = df.loc[i]['kind']
            created_on = str(df.loc[i]['created_on'])
            num_rows = self._safe_table_info_int(df.loc[i]['rows'], 'rows',
                                                 source_name)
            bytes = self._safe_table_info_int(df.loc[i]['bytes'], 'bytes',
                                              source_name)

            tables_info_dict[source_name] = TableInfo(source_name=source_name,
                                                      kind=kind,
                                                      created_on=created_on,
                                                      num_rows=num_rows,
                                                      bytes=bytes)
        return tables_info_dict

    # Note: PK and FK constraints are not enforced by Snowflake. It is possible
    # that these constraints are not even set, in which case users can revert
    # back to using infer_links or manually setting primary keys
    # and pk-fk links.
    def get_primary_keys(
            self, context: Any) -> Dict[Tuple[str, str], PrimaryKeyConfig]:
        primary_keys_dict: Dict[Tuple[str, str], PrimaryKeyConfig] = {}

        query = f'show primary keys in schema {self._qualified_schema_name()};'
        df = self.execute_meta_query(context, query)

        for i in range(len(df)):
            table_name = df.loc[i]['table_name']
            column_name = df.loc[i]['column_name']
            assert primary_keys_dict.get((table_name, column_name)) is None
            # pkey dtype is not available yet
            primary_keys_dict[table_name,
                              column_name] = PrimaryKeyConfig(name=column_name)

        return primary_keys_dict

    def get_foreign_keys(
            self, context: Any) -> Dict[Tuple[str, str], ForeignKeyConfig]:
        # For each table, return the list of foreign keys that are pointing to
        # a primary key in one of the other tables.
        foreign_keys_dict: Dict[Tuple[str, str], ForeignKeyConfig] = {}

        query = f'show imported keys in schema {self._qualified_schema_name()};'
        df = self.execute_meta_query(context, query)

        for i in range(len(df)):
            pk_table_name = df.loc[i]['pk_table_name']
            pk_column_name = df.loc[i]['pk_column_name']
            fk_table_name = df.loc[i]['fk_table_name']
            fk_column_name = df.loc[i]['fk_column_name']
            foreign_keys_dict[fk_table_name, fk_column_name] = ForeignKeyConfig(
                name=fk_column_name, dst_table=pk_table_name,
                dst_key=pk_column_name)
        return foreign_keys_dict

    @blocking_method()
    def get_column_metadata(
        self,
        context: Any,
        table_name: Optional[str] = None,
    ) -> Dict[Tuple[str, str], Dtype]:
        from kumo_snowflake.catalog import columns

        with restricted_diagnostic_boundary(
                safe_message='Snowflake metadata query failed',
                exception_types=SnowflakeErrors.Error):
            with context.cursor() as cursor:
                return columns(cursor, database=self.resolved_database,
                               schema=self.resolved_schema, table=table_name)

    def _table_id(self, table_name: str) -> list[str]:
        return [
            _quote_id(self.resolved_database),
            _quote_id(self.resolved_schema),
            _quote_id(table_name),
        ]

    def _qualified_schema_name(self) -> str:
        return '.'.join((_quote_id(self.resolved_database),
                         _quote_id(self.resolved_schema)))

    def _qualified_table_name(self, table_name: str) -> str:
        return '.'.join(self._table_id(table_name))

    def sql_table_name(self, table_name: str) -> str:
        """Render one exact source name independent of session defaults."""
        return self._qualified_table_name(table_name)

    def sql_column_name(self, column_name: str) -> str:
        """Quote an exact column name, including embedded quote characters."""
        return _quote_id(column_name)

    def _get_snowpark_df(self, source_name: str,
                         version: Optional[str] = None) -> snowpark.DataFrame:
        """Versioned (or current) Snowpark DataFrame for ``source_name``.

        Uses the shared Snowpark session from
        :func:`kumo.connector.snowflake_session.get_or_build_session`.
        Issues a versioned ``AT (TIMESTAMP => …)`` query with fallback to
        current-version ``session.table(...)``.
        """
        from kumo.connector.snowflake_session import get_or_build_session

        session = get_or_build_session(self)
        if version is not None:
            try:
                # We need to increment the data version by 1 to get the
                # data at the version.
                query_version = str(int(version) + 1)
                df = session.sql(
                    f"SELECT * FROM "
                    f"{self._qualified_table_name(source_name)} "
                    f"AT (TIMESTAMP => TO_TIMESTAMP('{query_version}'))")
                # Trigger evaluation so we discover unavailable
                # versions immediately rather than at a later point.
                df.count()
                return df
            except Exception as e:
                logger.info(
                    "Could not retrieve data from Snowflake at version %s "
                    "(error_type=%s); using the latest version",
                    version,
                    type(e).__name__,
                )

        try:
            return session.table(self._qualified_table_name(source_name))
        except SnowflakeErrors.Error as e:
            raise ConnectorError(
                f"Failed to access table '{source_name}'",
                error=e,
            ) from None

    def check_table_write_permissions(
        self,
        table_name: str,
        table_schema: str,
        mode: InternalWriteMode,
    ):
        """
        Check if the connector has write permissions for the specified table
        in the given mode.

        Args:
            table_name (str): The name of the table to check permissions for.
            table_schema (str): The schema of the table.
            mode (InternalWriteMode): The mode of operation, enum.

        Raises:
            ValueError: If the mode is not 'append' or 'overwrite' or if the
                connector does not have the required permissions.
        """
        if mode not in {
                InternalWriteMode.APPEND,
                InternalWriteMode.OVERWRITE,
        }:
            raise ConnectorError(
                'Invalid Snowflake output write mode; expected APPEND or '
                'OVERWRITE.')
        # Ensure the schema matches the connector's schema:
        if resolve_snowflake_identifier(table_schema) != self.resolved_schema:
            raise ConnectorError(f"Table schema {table_schema} does not match "
                                 f"connector schema {self.schema_name}")
        # For both APPEND/OVERWRITE modes, we might be using a staging table,
        # which requires CREATE TABLE permission on the schema.
        # Check this by creating a dummy table and then dropping it
        dummy_table_name = f"DUMMY_TABLE_{uuid.uuid4().hex[:8]}"
        qualified_probe_name = self._qualified_table_name(dummy_table_name)
        probe_created = False
        cleanup_error = False
        try:
            self.execute_meta_query(
                self.context,
                f'CREATE TABLE {qualified_probe_name} (dummy_col INT)')
            probe_created = True
        except SnowflakeErrors.Error:
            logger.warning('Snowflake output CREATE TABLE probe was denied')
            raise ConnectorError(
                f"Connector does not have CREATE TABLE "
                f"privileges on schema: {self.schema_name}. "
                f"Grant this privilege and retry the operation.") from None
        finally:
            if probe_created:
                try:
                    self.execute_meta_query(
                        self.context, f'DROP TABLE {qualified_probe_name}')
                except SnowflakeErrors.Error:
                    cleanup_error = True
                    logger.warning(
                        'Snowflake output write probe cleanup failed')
        if cleanup_error:
            raise ConnectorError(
                'Snowflake output authorization probe cleanup failed.')
        try:
            table_exists = self.table_exists(table_name)
        except SnowflakeErrors.Error:
            logger.warning(
                'Snowflake output table metadata authorization failed')
            raise ConnectorError(
                'Unable to verify the Snowflake output table permissions.') \
                from None
        # If the table doesn't exist, we don't need to check for any other
        # permissions since the table will be created by the connector.
        if not table_exists:
            logger.info(f"Table {table_name} does not exist. "
                        "No further permissions check required.")
            return

        # Fully qualify every target probe so session defaults cannot redirect
        # authorization to another database or schema.
        table_name = self._qualified_table_name(table_name)
        # APPEND mode requires INSERT permission on the table. Check this by
        # using INSERT INTO with a LIMIT 0 SELECT statement.
        # NOTE: Snowflake exposes some tables like `grants_to_roles` etc.
        # which might have some information, but all such metadata only works
        # when the user/role is explicitly granted the required permisisons.
        # Instead, calling INSERT OVERWRITE with a LIMIT 0 SELECT statement
        # is the most reliable way to check if we have the correct permissions.
        if mode == InternalWriteMode.APPEND:
            try:
                self.execute_meta_query(
                    self.context,
                    f'INSERT INTO {table_name} SELECT * FROM {table_name} '
                    'LIMIT 0')
            except SnowflakeErrors.Error:
                logger.warning('Snowflake output INSERT probe was denied')
                raise ConnectorError(
                    (f"Cannot INSERT INTO table {table_name}. "
                     f"Grant this privilege and retry the operation.")) \
                    from None
        elif mode == InternalWriteMode.OVERWRITE:
            # OVERWRITE mode requires INSERT, DELETE permissions on the table.
            try:
                self.execute_meta_query(self.context,
                                        (f'INSERT OVERWRITE INTO {table_name} '
                                         f'SELECT * FROM {table_name} LIMIT 0'))
            except SnowflakeErrors.Error:
                logger.warning('Snowflake output OVERWRITE probe was denied')
                raise ConnectorError(
                    (f"Connector does not have permission to overwrite "
                     f"existing table: {table_name}. "
                     f"Grant INSERT and DELETE privileges and retry the "
                     f"operation.")) from None

        logger.info("Write permissions check passed for table " +
                    f"{table_name} in {mode} mode.")

    def check_schema_exists(self, schema_name: str) -> None:
        # The only validation we need to do is to check if the schema name
        # matches the connector's schema name. If the schema didn't exist
        # the connector would not have been able to connect to the database.
        if resolve_snowflake_identifier(schema_name) != self.resolved_schema:
            raise ConnectorError(
                f"Schema {schema_name} does not match connector "
                f"schema {self.schema_name}")


def _quote_id(identifier: str) -> str:
    """Quotes snowflake identifiers in a way that is compliant with the
    following guideline.

    https://docs.snowflake.com/en/sql-reference/identifiers-syntax.html

    Note, this should only be used with identifiers such as db, table, and
    column names. It should not be used to quote values, as these have a
    different syntax.
    """
    if len(identifier) > 255:
        raise ValueError(f"Snowflake identifier can only have a maximum length "
                         f"of 255, but got: '{identifier}'")
    return '"' + identifier.replace('"', '""') + '"'


# Explicit dataclass definition ###############################################

SnowflakeConnectorConfig = to_dataclass(SnowflakeConnector,
                                        base_cls=ConnectorConfig)
register(SnowflakeConnector, SnowflakeConnectorConfig, group='connector')
