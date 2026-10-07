# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, List, NoReturn, Optional, Union

import db_dtypes
import google.cloud.bigquery as bigquery
import pandas as pd
from google.api_core.exceptions import (
    BadRequest,
    Forbidden,
    GoogleAPIError,
    NotFound,
    PermissionDenied,
)
from google.auth import impersonated_credentials, load_credentials_from_file
from google.cloud.bigquery.client import Client
from google.cloud.bigquery.table import Table as BQTable
from google.cloud.bigquery_storage_v1 import BigQueryReadClient, types
from google.oauth2 import service_account
from kumo_bigquery.sql import (
    dry_run,
    query_result,
    read_only_job,
    sample_request,
    table_names,
)
from kumo_bigquery.staging import StagingLocation
from kumo_bigquery.types import dtype_for
from kumo_bigquery.write import (
    MissingWritePermissions,
    OutputSchemaMismatchError,
    ProbeCleanupError,
    existing_table,
    is_output_table_id,
    load_uris,
    probe_write_access,
    staged_write,
)
from kumo_core.custom_sql import SQL_SAMPLE_MAX_SCAN_BYTES
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.sources.identity import hour_version
from kumo_core.types.schema import (
    BaseTableFingerprint,
    IngestedTableFingerprint,
    SqlTableFingerprint,
)
from kumo_core.utils.registry import register
from kumoapi.typing import Dtype
from pydantic import SecretStr

from io_utils import fsutil, logging
from io_utils.errors import restricted_diagnostic_boundary
from io_utils.secrets import get_secrets
from kumo.config import BaseColumnConfig, ConnectorConfig
from kumo.config.graph import TableMetadataConfig
from kumo.connector.artifact_staging import (
    is_local_artifact_path,
    local_parquet_files,
)
from kumo.connector.blocking import blocking_method
from kumo.connector.connector import (
    ConnectorError,
    ExportMetrics,
    SourceTableNotFoundError,
)
from kumo.connector.sql_connector import InternalWriteMode, SQLConnector
from kumo.utils import to_dataclass
from service_framework.deploy_env import EnvVar
from statedb.utils.errors import ConnectorValidationDiagnosticCode

ENV_GOOGLE_CREDENTIALS_JSON = "GOOGLE_CREDENTIALS_JSON"
ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH = os.path.join(os.path.expanduser("~"),
                                                   ".google_credentials.json")
ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT = "GOOGLE_IMPERSONATE_SERVICE_ACCOUNT"
GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# This is the GCP project ID used by Kumo for internal operations.
ENV_GCP_KUMO_PROJECT_ID = EnvVar("GOOGLE_CLOUD_PROJECT", type=str,
                                 default_value="kumo-engineering")
ENV_GCS_KUMO_BUCKET = EnvVar("GCP_STORAGE_TRANSFER_BUCKET", type=str,
                             default_value="")

BIGQUERY_CONNECTOR_PERMISSIONS_CHECK_TIMEOUT = 30

logger = logging.getLogger(__name__)


def _bq_query_error_reason(err: Exception) -> Optional[str]:
    """Return one structured provider reason, never free-form detail."""
    try:
        errors = getattr(err, 'errors', None)
        if errors:
            reason = errors[0].get('reason')
            return reason if isinstance(reason, str) else None
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    return None


_BQ_SAFE_QUERY_REASON_MESSAGES = {
    'resourcesExceeded':
    ('BigQuery could not plan or execute this query within its resource '
     'limits. Simplify the query, especially recursive or deeply nested '
     'subqueries, and try again.'),
    'responseTooLarge':
    ('BigQuery produced a result that is too large for this operation. '
     'Reduce or aggregate the selected output.'),
}


def _bq_safe_query_diagnostic(err: Exception) -> Optional[str]:
    """Map only structured query failures whose user guidance is allowlisted."""
    reason = _bq_query_error_reason(err)
    if reason == 'invalidQuery':
        return f"BigQuery could not run the query: {_bq_query_error_detail(err)}"
    if isinstance(err, NotFound) or reason == 'notFound':
        return ('BigQuery could not resolve a referenced table in the '
                "connector's configured location. Check the table name and "
                'dataset location.')
    return _BQ_SAFE_QUERY_REASON_MESSAGES.get(reason)


def _raise_bq_query_failure(err: Exception, fallback: str) -> NoReturn:
    diagnostic = _bq_safe_query_diagnostic(err)
    if diagnostic is not None:
        raise ConnectorError(diagnostic) from None
    raise ConnectorError(fallback, error=err) from None


def _bq_query_error_detail(err: Exception) -> str:
    r"""Clean, user-actionable reason from a BigQuery query-compile error.

    A Custom SQL dry-run/compile failure is a ``BadRequest`` (HTTP 400) whose
    payload is a diagnostic about the caller's *own* SQL (e.g. "Unrecognized
    name: a at [1:8]") — safe and actionable to surface, unlike auth/permission
    (``Forbidden``) failures. Prefer the structured compile diagnostic
    (``err.errors[0]["message"]``), then the exception's ``message``, then
    ``str(err)``. Defensive: BigQuery may omit ``errors`` or leave it empty.
    """
    try:
        errors = getattr(err, 'errors', None)
        if errors:
            msg = errors[0].get('message')
            if msg and msg.strip():
                return msg.strip()
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    msg = getattr(err, 'message', None)
    if msg and str(msg).strip():
        return str(msg).strip()
    return str(err).strip()


class WrappedClient:
    r"""
    Wrapper class for the big query connector that tries to handle
    failures/aborts gracefully.
    Args:
        client (Client): Client object to be wrapped.
    """

    from rest.backends import (
        DATA_BACKEND_INGEST_TIMEOUT_HOURS,
        DATA_BACKEND_IO_TIMEOUT_HOURS,
        DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS,
    )

    MAX_SECONDS_BEFORE_TIMEOUT = 10
    r"""
    The constants below take their values from temporal constants. The
    intent is that these constants have slightlly smaller values than
    the corresponding temporal constants. This ensures that the job times
    out with a useful/actionable error message before being killed due
    to the temporal timeout.
    """
    BQ_INGEST_TIMEOUT_SECONDS = (DATA_BACKEND_INGEST_TIMEOUT_HOURS * 3600 -
                                 MAX_SECONDS_BEFORE_TIMEOUT)
    BQ_MATERIALIZE_TIMEOUT_SECONDS = (
        DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS * 3600 -
        MAX_SECONDS_BEFORE_TIMEOUT)
    BQ_PROGRESS_POLL_INTERVAL_SECONDS = 10

    BQ_BACKEND_IO_TIMEOUT_SECONDS = int(DATA_BACKEND_IO_TIMEOUT_HOURS * 3600 /
                                        2)

    def __init__(self, client: Client):
        self._client = client

    def list_tables(self, dataset: str):
        logger.info("Listing tables in the configured BigQuery dataset")
        return self._client.list_tables(dataset=dataset)

    def get_table(self, table: str):
        return self._client.get_table(table=table)

    def _get_query_source_size_internal(
            self, query: str, job_config: bigquery.QueryJobConfig | None,
            surface_query_diagnostics: bool = False) -> Optional[int]:

        if job_config is None:
            job_config = bigquery.QueryJobConfig()

        try:
            job_config.dry_run = True
            query_job = self._client.query(query, job_config=job_config)
            return query_job.total_bytes_processed
        except Exception as e:
            logger.error(
                "Failed to perform BigQuery dry run (error_type=%s)",
                type(e).__name__,
            )
            if surface_query_diagnostics:
                _raise_bq_query_failure(
                    e, "Failed to estimate the BigQuery query size")
            raise ConnectorError(
                "Failed to estimate query size for BigQuery dry run") from None
        finally:
            job_config.dry_run = False

    def poll_job_completion(
        self,
        executing_job: Union[bigquery.QueryJob, bigquery.LoadJob],
        timeout_in_seconds: int,
        log_message: str,
    ):
        assert timeout_in_seconds >= 0

        start_time = time.perf_counter()

        while not executing_job.done():
            curr_time = time.perf_counter()

            elapsed_time = curr_time - start_time

            if elapsed_time > timeout_in_seconds:
                logger.warning(
                    f"Timeout {timeout_in_seconds} seconds exceeded. "
                    "Attempting to cancel the BigQuery connector query.")
                executing_job.cancel()
                logger.warning(f"Timeout {timeout_in_seconds} seconds exceeded."
                               " Canceled the BigQuery connector query.")
                raise TimeoutError(
                    "We hit a timeout while running an operation on your "
                    "BigQuery warehouse and it was automatically canceled "
                    f"after {timeout_in_seconds} seconds. Please try "
                    "rerunning the operation and check if your project has "
                    "enough BigQuery resources to complete it.")

            logger.info("Still executing the BigQuery connector query")
            time.sleep(WrappedClient.BQ_PROGRESS_POLL_INTERVAL_SECONDS)

        end_time = time.perf_counter()
        logger.info("BigQuery connector query finished after %s seconds",
                    end_time - start_time)
        return executing_job

    def query(
        self,
        query: str,
        job_config: bigquery.QueryJobConfig | None = None,
        timeout_in_seconds: int = BQ_INGEST_TIMEOUT_SECONDS,
        surface_query_diagnostics: bool = False,
    ) -> bigquery.QueryJob:

        self._get_query_source_size_internal(query, job_config,
                                             surface_query_diagnostics)

        with restricted_diagnostic_boundary(
                safe_message='BigQuery query execution failed',
                exception_types=GoogleAPIError):
            query_job: bigquery.QueryJob = (self._client.query(
                query=query,
                job_config=job_config,
            ) if job_config is not None else self._client.query(query=query))
            self.poll_job_completion(query_job, timeout_in_seconds, query)

        return query_job

    def load_table_from_uri(
        self,
        source_uri: str | list[str],
        destination: str,
        job_config: bigquery.LoadJobConfig,
    ) -> bigquery.LoadJob:

        with restricted_diagnostic_boundary(
                safe_message='BigQuery load operation failed',
                exception_types=GoogleAPIError):
            load_job = self._client.load_table_from_uri(
                source_uris=source_uri,
                destination=destination,
                job_config=job_config,
            )

            self.poll_job_completion(
                load_job, WrappedClient.BQ_INGEST_TIMEOUT_SECONDS,
                f"Load from uri job with source {source_uri} "
                f"and destination {destination}.")

        return load_job


class BigQueryConnector(SQLConnector):
    r"""
    Args:
        name (str): The name of the connector.
        project_id (str): The project ID.
        dataset_id (str): The dataset ID.
    """
    #: ``LIMIT`` does not bound a BigQuery read, so a Custom SQL sample is
    #: costed before it runs.
    gates_query_scan = True

    def __init__(
        self,
        name: str,
        project_id: str,
        dataset_id: str,
        location: str = 'US',
        billing_project_id: str | None = None,
    ):
        super().__init__(name)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.billing_project_id = billing_project_id or project_id

        self._credentials: Dict[str, Any] = {}

    @property
    def context(self) -> Client:
        r"""Returns the connector context."""
        return super().context

    # Required abstract methods from base-class ###############################

    def validate_permissions(self) -> None:
        """
        Validate that the service account has necessary permissions on the BigQuery dataset.

        Args:
            client: bigquery.Client instance
        """
        def _sync_validation():

            try:
                dataset_ref = f"{self.project_id}.{self.dataset_id}"
                # Bypassing WrappedClient here because we don't need
                # the logging overhead.
                real_client = self.client._client

                query = f"""
                    SELECT table_name
                    FROM `{dataset_ref}.INFORMATION_SCHEMA.TABLES`
                    WHERE table_type = 'BASE TABLE'
                    LIMIT 1
                """
                query_job = real_client.query(query)
                results = list(query_job.result())

                if not results:
                    # While extremely unusual we might have a situation
                    # where the user only has access to views and/or
                    # the project/dataset itself only has views.
                    # So we don't treat this situation as an error.
                    return

                table_name = results[0].table_name

                # Try to read one record from the table
                test_query = f"""
                    SELECT * FROM `{dataset_ref}.{table_name}`
                    LIMIT 1
                """

                test_job = real_client.query(test_query)
                list(test_job.result())

                return

            except PermissionDenied:
                # A gRPC/IAM PERMISSION_DENIED is unambiguously an access
                # failure. Provider exception strings may carry request
                # metadata, so only the sanitized category is recorded.
                logger.warning(
                    'BigQuery target validation was denied by the provider')
                raise ConnectorError(
                    'The selected identity does not have the required '
                    'BigQuery dataset read permissions.',
                    diagnostic_code=ConnectorValidationDiagnosticCode.
                    TARGET_ACCESS_DENIED) from None
            except Forbidden as err:
                # A REST 403 is an access denial only when the structured
                # reason says so. The same status also covers quota,
                # rate-limit, and billing failures, which are operational:
                # keep those a plain ConnectorError with no diagnostic code so
                # they are never surfaced as a non-retriable target-config
                # problem.
                logger.warning(
                    'BigQuery target validation was denied by the provider')
                if _bq_query_error_reason(err) == 'accessDenied':
                    raise ConnectorError(
                        'The selected identity does not have the required '
                        'BigQuery dataset read permissions.',
                        diagnostic_code=ConnectorValidationDiagnosticCode.
                        TARGET_ACCESS_DENIED) from None
                raise ConnectorError(
                    'BigQuery connector access validation failed.') from None
            except NotFound:
                raise ConnectorError(
                    'The configured BigQuery dataset was not found or is not '
                    'accessible to the selected identity.',
                    diagnostic_code=ConnectorValidationDiagnosticCode.
                    TARGET_NOT_FOUND) from None
            except Exception:
                raise ConnectorError(
                    'BigQuery connector access validation failed.') from None

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_sync_validation)
        timed_out = False
        try:
            future.result(timeout=BIGQUERY_CONNECTOR_PERMISSIONS_CHECK_TIMEOUT)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            raise ConnectorError(
                'BigQuery connection validation timed out') from None
        finally:
            # A context-managed executor blocks in shutdown(wait=True), which
            # would turn this timeout into an unbounded wait.  A running
            # provider call cannot be killed, so detach it on timeout and fail
            # the validation request immediately.
            executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

    def _open(self, credentials: Optional[Dict[str, Any]] = None):
        # Override the existing credentials if provided. This is okay since
        # _open is only called when the connector is not already opened.
        if credentials is not None:
            self._credentials = credentials
        if len(self._credentials) > 0:
            context = self._open_with_credentials()
        else:
            if self._require_explicit_credentials:
                raise ConnectorError(
                    'Identity-backed BigQuery connectors require explicit '
                    'credentials.')
            try:
                context = self._use_credentials(
                    'bigquery', Client.from_service_account_json,
                    Client.from_service_account_info, True)
            except Exception:
                raise ConnectorError(
                    message="An error occured while trying to connect " +
                    "to BigQuery." +
                    " Please try by providing service account credentials instead.",
                ) from None

        return context

    def _close(self, context: Any):
        context.close()

    @property
    def client(self) -> WrappedClient:
        context: Client = self.context
        return WrappedClient(context)

    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        r"""
        Returns a list of table names within the project and dataset.
        """
        # Client.list_tables is a lightweight API to get limited information
        # about tables in a dataset. It does not return heavyweight metadata
        # like table schema, cols etc.
        try:
            return table_names(self.client, self.dataset_id)
        except Exception:
            raise ConnectorError(
                message=(
                    'Kumo could not list tables in the configured BigQuery '
                    'dataset.'), ) from None

    @classmethod
    def to_kumo_dtype(cls, bq_dtype: str, mode: str,
                      precision: Optional[int] = None,
                      scale: Optional[int] = None) -> Dtype:
        return dtype_for(bq_dtype, mode, precision, scale)

    def is_table_view(self, table_name: str) -> bool:
        bq_table = self._get_bigquery_table(table_name)
        return str(bq_table.table_type) == "VIEW"

    def table_exists(self, table_name: str) -> bool:
        try:
            self.client.get_table(self.full_table_name(table_name))
            logger.info("The requested BigQuery table exists")
            return True
        except NotFound:
            logger.info("The requested BigQuery table does not exist")
            return False

    def check_table_write_permissions(
        self,
        table_name: str,
        table_schema: str,
        mode: InternalWriteMode,
    ) -> None:
        """Verify write access to the connector's exact BigQuery dataset
        through the shared :func:`kumo_bigquery.write.probe_write_access`.
        """
        self.require_output_table_within_connector_target(
            table_name,
            table_schema,
        )
        if mode not in {
                InternalWriteMode.APPEND,
                InternalWriteMode.OVERWRITE,
        }:
            raise ConnectorError(
                'Invalid BigQuery output write mode; expected APPEND or '
                'OVERWRITE.')
        if not self.is_connected:
            self.open()

        try:
            probe_write_access(self.context,
                               f'{self.project_id}.{self.dataset_id}',
                               table_name,
                               replace=mode == InternalWriteMode.OVERWRITE,
                               create=True)
        except MissingWritePermissions:
            raise ConnectorError(
                'The selected identity cannot perform the requested write '
                'mode on the existing BigQuery output table.') from None
        except ProbeCleanupError:
            raise ConnectorError(
                'BigQuery output authorization probe cleanup failed. The '
                'probe is configured to expire automatically.') from None
        except Exception as error:
            logger.warning(
                'BigQuery output write authorization failed (%s)',
                type(error).__name__,
            )
            raise ConnectorError(
                'The selected identity cannot create or modify tables in '
                'the configured BigQuery dataset.') from None

    def require_output_table_within_connector_target(
        self,
        table_name: str,
        table_schema: str | None = None,
    ) -> None:
        if table_schema is not None and table_schema != self.dataset_id:
            raise ConnectorError(
                'Output table dataset does not match the connector dataset.')
        if not is_output_table_id(table_name):
            raise ConnectorError(
                'BigQuery output table must be an unqualified table ID '
                'containing only letters, numbers, underscores, or hyphens.')

    def overwrite_table(self, src_table_name: str, dst_table_name: str) -> None:
        r"""Overwrite `dst_table_name` with the content from `src_table_name`.
        Args:
            src_table_name (str): The name of the source table.
            dst_table_name (str): The name of the destination table.
        """
        if not self.table_exists(src_table_name):
            raise ConnectorError(
                message='The requested BigQuery source table does not exist.')

        logger.info("Overwriting one BigQuery table from another")
        query = f"""
            CREATE OR REPLACE TABLE {self.sql_table_name(dst_table_name)}
            CLONE {self.sql_table_name(src_table_name)}
        """
        # NOTE(jana): delete before merging PR. Method used internally.
        with restricted_diagnostic_boundary(
                safe_message='BigQuery table overwrite failed',
                exception_types=GoogleAPIError):
            self.context.query(query).result()

    def _cols_from_schema(
        self,
        schema: List[bigquery.SchemaField],
    ) -> List[BaseColumnConfig]:
        r"""Map a BigQuery schema — a physical table's or a query result's — to
        Kumo column configs, warning on any unsupported source type."""
        from kumo_bigquery.types import schema_columns

        cols = []
        for name, dtype in schema_columns(schema):
            col = BaseColumnConfig(name=name, dtype=dtype)
            if dtype == Dtype.unsupported:
                logger.warning("Unsupported BigQuery dtype for %s", name)
            cols.append(col)
        return cols

    def _get_column_metadata(
        self,
        table_obj: BQTable,
    ) -> List[BaseColumnConfig]:
        return self._cols_from_schema(table_obj.schema)

    def _get_one_table_metadata(self, table_name: str) -> TableMetadataConfig:
        r"""
        Returns metadata of a single table within the dataset.
        """
        try:
            bq_table = self._get_bigquery_table(table_name)
            return TableMetadataConfig(
                table_name=table_name,
                source_name=bq_table.table_id,
                data_source=self.name,
                cols=self._get_column_metadata(bq_table),
            )
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(
                message='Kumo could not read BigQuery table metadata.',
            ) from None

    # TODO: BigQuery only recently introduced the concept of primary key and
    # foreign key and it doesn't seem to be available from the Python API.
    # Hence, we cannot get this information directly from the database.
    def _get_table_metadata(
        self,
        table_names: Optional[List[str]] = None,
        skip_cache: bool = False,
        skip_inferral: bool = False,
        schema_name: Optional[str] = None,
        ignore_errors: bool = False,
    ) -> List[TableMetadataConfig]:
        r"""
        Returns a list of :class:`TableMetadataConfig` objects which describe
        the metadata of each table.

        Args:
            table_names (List[str]): List of table names to get metadata for.
                If :obj:`None`, returns metadata for all tables in the dataset
            ignore_errors (bool): Omit requested tables that do not exist.
                Other metadata failures still propagate.

        Raises:
            SourceTableNotFoundError: If a requested table does not exist and
                ``ignore_errors`` is :obj:`False`.
        """
        if not table_names:
            table_names = self.list_tables()
        result = []
        for table in table_names:
            try:
                result.append(self._get_one_table_metadata(table))
            except SourceTableNotFoundError:
                if not ignore_errors:
                    raise
        return result

    @blocking_method()
    def _get_df_result(
        self,
        context: Any,
        query: str,
        job_config: Optional[bigquery.QueryJobConfig] = None,
        *,
        surface_query_diagnostics: bool = False,
    ) -> pd.DataFrame:
        r"""
        Executes the SQL query and returns the result as a pandas dataframe.

        ``job_config`` is optional; Custom SQL previews pass one carrying a
        ``default_dataset`` so unqualified table names resolve to the
        connector's home scope.

        ``surface_query_diagnostics`` lets Custom SQL preview expose only the
        structured reasons accepted by :func:`_bq_safe_query_diagnostic`.
        Physical-table reads keep the generic redacted contract.
        """
        try:
            execute = partial(
                self.client.query,
                timeout_in_seconds=WrappedClient.BQ_BACKEND_IO_TIMEOUT_SECONDS,
                surface_query_diagnostics=surface_query_diagnostics)
            df = query_result(
                execute, query,
                job_config).to_dataframe(create_bqstorage_client=False)
        except Forbidden as e:
            logger.error(
                "BigQuery permission error while executing query "
                "(error_type=%s)",
                type(e).__name__)
            raise ConnectorError(
                'The selected identity does not have the required BigQuery '
                'table read permission.') from None
        except TimeoutError:
            raise
        except Exception as e:
            # Custom SQL exposes only allowlisted structured reasons. Physical
            # reads and unknown provider failures remain generic.
            if surface_query_diagnostics:
                if isinstance(e, ConnectorError):
                    raise
                diagnostic = _bq_safe_query_diagnostic(e)
                if diagnostic is not None:
                    raise ConnectorError(diagnostic) from None
                if isinstance(e, BadRequest):
                    _raise_bq_query_failure(
                        e, 'BigQuery failed while reading table data.')
            logger.error("BigQuery query failed (error_type=%s)",
                         type(e).__name__)
            raise ConnectorError(
                message='BigQuery failed while reading table data.', ) from None

        # For certain db datatypes, we convert them to native pandas dtype.
        for col in df.columns:
            if df[col].dtype == db_dtypes.DateDtype():
                df[col] = pd.to_datetime(df[col])
        return df

    # Overrides of functions from SQLConnector ################################
    def sql_table_name(self, table_name):
        r"""Table name to be used in SQL queries in BigQuery."""
        return f"`{self.project_id}.{self.dataset_id}.{table_name}`"

    def _get_bigquery_table(self, table_name) -> BQTable:
        logger.info("Started BigQuery get_table API")
        try:
            table_info = self.client.get_table(self.full_table_name(table_name))
            logger.info("Finished BigQuery get_table API")
            return table_info
        except NotFound:
            raise SourceTableNotFoundError(table_name, self.name) from None
        except Exception:
            raise ConnectorError(
                message='Kumo could not read BigQuery table metadata.',
            ) from None

    def _get_table_num_rows(
        self,
        table_name: str,
        data_version: str | None = None,
    ) -> int:
        logger.info("Started BigQuery row-count lookup")
        bq_table = self._get_bigquery_table(table_name)
        logger.info("Finished BigQuery row-count lookup")
        if bq_table.num_rows is None or bq_table.num_rows == 0:
            logger.warning("Could not retrieve BigQuery table row count")
            # Revert back to SELECT COUNT(*) for getting num_rows.
            logger.info("Started BigQuery fallback row-count query")
            try:
                num_rows = super()._get_table_num_rows(table_name)
                logger.info("Finished BigQuery fallback row-count query")
                return num_rows
            except Exception as e:
                logger.info(
                    "Failed BigQuery fallback API for retrieving row count "
                    "(error_type=%s)",
                    type(e).__name__)
        assert bq_table.num_rows
        return bq_table.num_rows

    def _get_table_source_size(
        self,
        table_name: str,
        data_version: str | None = None,
    ) -> int:
        logger.info("Started BigQuery table-size lookup")
        bq_table = self._get_bigquery_table(table_name)
        logger.info("Finished BigQuery table-size lookup")
        num_bytes = bq_table.num_bytes
        if num_bytes is None:
            logger.warning("Could not retrieve BigQuery table size")
            try:
                logger.info("Started BigQuery fallback table-size query")
                num_bytes = super()._get_table_source_size(table_name)
                logger.info("Finished BigQuery fallback table-size query")
                return num_bytes
            except Exception as e:
                logger.info(
                    "Failed BigQuery fallback table-size query "
                    "(error_type=%s)",
                    type(e).__name__)
                raise ConnectorError(
                    message='Failed to retrieve the BigQuery table size.',
                ) from None
        if str(bq_table.table_type) != "VIEW":
            assert num_bytes
        return num_bytes

    def _get_selected_table_source_size(
        self,
        table_name: str,
        cols: List[BaseColumnConfig],
    ) -> int:
        if not self.is_connected:
            self.open()
        if not cols:
            logger.warning("No columns specified for BigQuery table size")
            return 0
        query = self.get_query(table_name=table_name, pkey=None, cols=cols)
        query_job_config = bigquery.QueryJobConfig()

        try:
            estimated_bytes = self.client._get_query_source_size_internal(
                query, query_job_config)
            if estimated_bytes is None:
                logger.warning(
                    "BigQuery dry run did not return total_bytes_processed; "
                    "defaulting to 0")
                return 0
            return estimated_bytes
        except Exception as e:
            logger.error("Failed to perform BigQuery dry run (error_type=%s)",
                         type(e).__name__)
            raise

    def _get_data_version(
        self,
        table_name: str,
    ) -> str:
        logger.info("Started BigQuery data-version lookup")
        bq_table = self._get_bigquery_table(table_name)

        logger.info("Finished BigQuery data-version lookup")

        if str(bq_table.table_type) == "VIEW":
            # The shared window, so this platform and an in-process
            # kumo_core stamp a versionless source alike and either can
            # reuse what the other ingested.
            logger.info("Computed BigQuery view data version")
            return hour_version()
        if bq_table.modified is None:
            try:
                logger.warning("Could not retrieve BigQuery table data version")
                return super()._get_data_version(table_name)
            except Exception:
                logger.warning("Could not retrieve BigQuery data version")
        return str(bq_table.modified.timestamp()).split(".")[0]

    # Helper methods ########################################

    def _get_plain_credentials(self) -> Dict[str, str]:
        r"""
        Parses the stored credentials into a Dict[str, str], which can be
        directly used for BigQuery connection.
        """
        parsed_credentials: Dict[str, str] = {}
        for key, value in self._credentials.items():
            # EMR scripts do not use SecretStr i.e. the pass the credentials
            # as string. Hence we need this check to avoid calling
            # get_secret_value always.
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            parsed_credentials[key] = value
        return parsed_credentials

    def _open_with_credentials(self):
        r"""Opens a BigQuery connection with the provided credentials."""
        credential = self._credentials.get('credential')
        if credential is not None:
            if self._credentials.get('provider') != 'bigquery':
                raise ConnectorError(
                    'BigQuery connector received credentials for another '
                    'provider')
            return Client(
                project=self.project_id,
                credentials=credential,
                location=self.location,
            )

        # Validate whether all the required fields are present in the
        # credentials.
        #
        # NOTE: These fields should match the ones in BigQueryCredentials
        # dataclass in rest/config/credentials.py.
        required_fields = {
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "project_id",
        }
        credentials_without_type = {
            k: v
            for k, v in self._credentials.items() if k != "_type"
        }
        if not required_fields.issubset(credentials_without_type):
            raise ConnectorError(
                'BigQuery credentials are incomplete or invalid')

        # Add the standard service-account discriminator if omitted. The
        # service-account project must be supplied by the identity;
        # it must never be inferred from the connector's data project.
        if self._credentials.get("type") is None:
            self._credentials["type"] = SecretStr("service_account")

        credential_info = self._get_plain_credentials()
        google_credentials = service_account.Credentials.from_service_account_info(
            credential_info)
        google_credentials = google_credentials.with_quota_project(
            self.billing_project_id)
        return Client(
            project=self.project_id,
            credentials=google_credentials,
            location=self.location,
        )

    def full_table_name(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    def drop_table_if_exists(self, table_name: str):
        try:
            table_ref = self._context.dataset(self.dataset_id).table(table_name)
            self._context.delete_table(table_ref)
        except Exception as e:
            logger.warning('BigQuery table cleanup failed (%s)',
                           type(e).__name__)

    def _initialize_gcs_client(self) -> Any:
        r"""Initialize GCS with the activity-local BigQuery credential.

        BigQuery diskgraph ingestion stages exported parquet files in GCS.
        That staging access is part of the authorized connector operation, so
        it must use the same credential resolved by the activity credential
        broker.  Looking up host ADC or an internal service-account secret
        here would create a second, unrelated authorization boundary.
        """
        from google.cloud import storage

        credential = self._credentials.get('credential')
        if (self._credentials.get('provider') != 'bigquery'
                or credential is None):
            raise ConnectorError(
                'Authorized BigQuery staging credentials are unavailable.')

        return storage.Client(
            project=ENV_GCP_KUMO_PROJECT_ID.parse_value(),
            credentials=credential,
        )

    def _write_artifacts_from_gcs(
            self, table_name: str, gcs_path: str, replace: bool = False, *,
            artifact_files: list[str] | None = None) -> None:
        r"""Fast path for artifacts already resident in a Kumo-owned GCS
        bucket: skip STS and load directly from ``gs://`` into BigQuery.
        """
        # Selected BP artifacts can span multiple immutable attempt roots,
        # including directories named predictions.parquet. Load only the
        # selected objects; retain URI inference for legacy callers.
        source_uri: str | list[str]
        if artifact_files is not None:
            source_uri = list(artifact_files)
            if (not gcs_path.startswith('gs://') or not source_uri
                    or len(set(source_uri)) != len(source_uri)):
                raise ConnectorError(
                    'Selected BigQuery artifacts require non-empty, unique '
                    'GCS object URIs.')
            for uri in source_uri:
                bucket, _, object_name = uri.removeprefix('gs://').partition(
                    '/')
                if (not uri.startswith('gs://') or not bucket or not object_name
                        or uri.endswith('/') or '*' in uri):
                    raise ConnectorError(
                        'Selected BigQuery artifacts require concrete GCS '
                        'object URIs without wildcards.')
        else:
            source_uri = (gcs_path if gcs_path.endswith('.parquet') else
                          gcs_path.rstrip('/') + '/*.parquet')

        # Preserve the existing table schema if present (mirrors the
        # STS path's behavior — important for embedding array columns).
        schema = None
        if self.table_exists(table_name):
            table = self.client.get_table(self.full_table_name(table_name))
            schema = table.schema

        logger.info("Loading artifacts into BigQuery (replace=%s)", replace)
        try:
            load_uris(self.client, source_uri, self.full_table_name(table_name),
                      replace=replace, schema=schema)
        except Exception as e:
            logger.warning('BigQuery load job failed (%s)', type(e).__name__)
            raise ConnectorError(
                'Failed to load artifacts into the configured BigQuery '
                'output table.') from None

    @blocking_method()
    def write_artifacts_to_table(
            self, table_name: str, artifact_path: str, replace: bool = False, *,
            artifact_files: list[str] | None = None) -> None:
        r"""Upload parquet artifacts into a customer's BigQuery table.

        Artifacts already staged in ``gs://`` load directly. Worker-local
        Parquet is first uploaded to a unique prefix in the configured Kumo
        transfer bucket using this connector's brokered Google credential,
        then deleted after the load. The connector intentionally does not
        bridge artifacts from another cloud.

        ``artifact_files`` is the exact activity-resolved GCS selection for
        BP output. It avoids root inference and must not contain wildcards.
        Omit it to retain legacy single-file/prefix and local-staging behavior.
        """
        self.require_output_table_within_connector_target(table_name)
        self.require_supported_artifact_path(artifact_path)
        if artifact_files is not None:
            return self._write_artifacts_from_gcs(table_name, artifact_path,
                                                  replace=replace,
                                                  artifact_files=artifact_files)
        if not is_local_artifact_path(artifact_path):
            return self._write_artifacts_from_gcs(table_name, artifact_path,
                                                  replace=replace)

        bucket_name = ENV_GCS_KUMO_BUCKET.parse_value()
        if not bucket_name:
            raise ConnectorError(
                'GCP_STORAGE_TRANSFER_BUCKET is not configured; local '
                'BigQuery artifact output cannot be staged.')

        try:
            staged_write(
                self.client, self._initialize_gcs_client(),
                StagingLocation(bucket_name,
                                f'artifact-export/{uuid.uuid4().hex}'),
                local_parquet_files(artifact_path),
                self.full_table_name(table_name), replace=replace,
                existing=existing_table(self.client,
                                        self.full_table_name(table_name)))
        except ConnectorError:
            raise
        except OutputSchemaMismatchError as error:
            message = (
                'Prediction output does not fit the existing BigQuery output '
                f'table: {"; ".join(error.problems)}. APPEND needs every '
                'output column in the table with the same type; use '
                'OVERWRITE or another table.')
            logger.warning(message)
            raise BigQueryOutputSchemaError(message) from None
        except Exception as error:
            logger.warning(
                'BigQuery local artifact export failed (%s)',
                type(error).__name__,
            )
            raise ConnectorError(
                'Failed to load local artifacts into the configured BigQuery '
                'output table.') from None

    @staticmethod
    def require_supported_artifact_path(artifact_path: str) -> None:
        r"""Allow direct GCS loads or controlled worker-local staging."""
        if artifact_path.startswith('gs://'):
            return
        if is_local_artifact_path(artifact_path):
            return
        if artifact_path.startswith('s3://'):
            raise ConnectorError(
                'BigQuery output does not support s3:// artifact paths; '
                'use gs://.')
        raise ConnectorError(
            'Unsupported artifact scheme for BigQuery output; expected '
            'gs://.')

    @blocking_method()
    def export_table_to_blob_storage(
        self,
        table_name: str,
        dest_path: str,
        columns: Optional[List[str]] = None,
    ) -> None:
        from kumo_bigquery.sql import (
            home_dataset,
            qualified_name,
            quote_identifier,
        )
        from kumo_bigquery.staging import StagingLocation, export

        if not dest_path.startswith('gs://'):
            raise ConnectorError(
                'Unsupported destination scheme for BigQuery export; expected '
                'gs://.')
        try:
            # Custom training runs on DataBackend, without whole-table scratch.
            export(
                self.client,
                location=StagingLocation.parse(dest_path),
                projection=(', '.join(map(quote_identifier, columns))
                            if columns else '*'),
                source=qualified_name(self.project_id, self.dataset_id,
                                      table_name),
                job_config=home_dataset(self.project_id, self.dataset_id),
            )
        except Exception as error:
            logger.error("BigQuery table export failed (error_type=%s)",
                         type(error).__name__)
            raise ConnectorError(
                'Failed to export data from the configured BigQuery table.'
            ) from None

    @contextmanager
    def _source_reader(self, source_key: str, data_version: str):
        # Lazy: keeps the reader package off the import path of every
        # process that imports this module.
        from kumo_bigquery import BigQuerySourceReader

        staging_uri, storage_client = None, None
        bucket = ENV_GCS_KUMO_BUCKET.parse_value()
        if bucket:
            # Scoped per attempt as well as per table and version: the
            # reader clears its prefix before and after the unload, so two
            # attempts sharing one would delete each other's export mid-fetch.
            staging_uri = (f'gs://{bucket}/source-reader/'
                           f'{source_key}/{data_version}/'
                           f'{uuid.uuid4().hex}')
            storage_client = self._initialize_gcs_client()
        # Passed explicitly rather than read back off the read client, where
        # an SDK rename would silently cost every worker its own channel.
        credentials = self.context._credentials
        reader = BigQuerySourceReader(
            self.context,
            project=self.project_id,
            dataset=self.dataset_id,
            read_client=BigQueryReadClient(credentials=credentials),
            credentials=credentials,
            storage_client=storage_client,
            staging_uri=staging_uri,
            billing_project=self.billing_project_id,
        )
        try:
            yield reader
        finally:
            try:
                reader.read_client.transport.close()
            except Exception as error:
                logger.warning(
                    "BigQuery read client cleanup failed (error_type=%s)",
                    type(error).__name__)

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
        with self._source_reader(table_fp.ingest_path_fpid,
                                 data_version) as reader:
            return read_source(reader, source_name, table_fp, data_version,
                               graph_table_count=graph_table_count)

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

    # Custom SQL query pushdown ###############################################

    def _validate_read_only_query(self, sql: str) -> bigquery.QueryJob:
        r"""Compile with BigQuery and require one SELECT result statement."""
        try:
            query_job = read_only_job(self.context, sql,
                                      project=self.project_id,
                                      dataset=self.dataset_id)
        except PermanentSourceError:
            raise ConnectorError(
                "Custom SQL must be a single read-only SELECT statement; "
                "BigQuery scripts, procedures, DDL, and DML are unsupported."
            ) from None
        except Exception as err:
            _raise_bq_query_failure(
                err, "BigQuery failed to compile the custom SQL query")
        return query_job

    @blocking_method()
    def get_query_metadata(self, sql: str) -> TableMetadataConfig:
        r"""Describe a query's ordered result schema with a dry run."""
        query_job = self._validate_read_only_query(sql)
        return TableMetadataConfig(
            table_name='', source_name='', data_source=self.name,
            cols=self._cols_from_schema(query_job.schema))

    @blocking_method()
    def get_query_scan_estimate_bytes(self, sql: str) -> Optional[int]:
        r"""Estimate scanned bytes with a free dry run.

        BigQuery ``LIMIT`` does not reduce bytes scanned, so callers use this
        estimate to gate preview and stype sampling.
        """
        try:
            query_job = dry_run(self.context, sql, project=self.project_id,
                                dataset=self.dataset_id)
        except Exception as err:
            _raise_bq_query_failure(
                err, "BigQuery failed to compile the custom SQL query")
        return query_job.total_bytes_processed

    @blocking_method()
    def sample_query(self, sql: str, num_rows: int) -> pd.DataFrame:
        r"""Return a bounded query preview in subquery position.

        Under a ceiling BigQuery enforces itself, because the dry-run estimate
        the caller gated on is advisory.
        """
        query, config = sample_request(
            sql, num_rows, project=self.project_id, dataset=self.dataset_id,
            max_bytes_billed=SQL_SAMPLE_MAX_SCAN_BYTES)
        try:
            return self._get_df_result(self.context, query, job_config=config,
                                       surface_query_diagnostics=True)
        except ConnectorError:
            raise
        except Exception as err:
            _raise_bq_query_failure(
                err, "BigQuery failed to compile the custom SQL query")

    @blocking_method()
    def export_query(
        self,
        sql: str,
        table_fp: SqlTableFingerprint,
        data_version: str,
    ) -> ExportMetrics:
        from kumo.connector.source_bridge import read_source

        try:
            self._validate_read_only_query(sql)
            with self._source_reader(table_fp.ingest_path_fpid,
                                     data_version) as reader:
                return read_source(reader, table_fp.ingest_path_fpid, table_fp,
                                   data_version, sql=sql)
        except ConnectorError:
            raise
        except Exception as e:
            logger.error(
                'BigQuery custom SQL query export failed (error_type=%s)',
                type(e).__name__)
            raise ConnectorError(
                'Failed to read the configured BigQuery custom SQL query.',
                error=e) from None

    def check_table_read_permissions(self, table_name):
        """
        Check if the connector has read permissions for the specified table
        in the given mode.

        Args:
            table_name (str): The name of the table to check permissions for.

        Raises:
            PermissionError: If the connector does not have the required
                permissions.
        """
        if not self.is_connected:
            self.open()
        try:
            bq_table = self._get_bigquery_table(table_name)
            if str(bq_table.table_type) == "TABLE":
                # A read-session probe validates BigQuery Storage Read API
                # permissions for physical tables.
                project_path = f"projects/{self.project_id}"
                table_path = f"{project_path}/datasets/{self.dataset_id}/tables/{table_name}"
                storage_client = BigQueryReadClient(
                    credentials=self.context._credentials)
                storage_client.create_read_session(
                    parent=project_path, read_session=types.ReadSession(
                        table=table_path, data_format=types.DataFormat.ARROW))
            else:
                # Use dry run for views
                table_ref = self.full_table_name(table_name)
                query = f"SELECT 1 FROM `{table_ref}` LIMIT 1"
                job_config = bigquery.QueryJobConfig(dry_run=True,
                                                     use_query_cache=False)
                self.context.query(query, job_config=job_config)
        except ConnectorError:
            raise
        except PermissionDenied as e:
            logger.error(
                "BigQuery table permission denied (error_type=%s)",
                type(e).__name__,
            )
            raise ConnectorError(
                'The selected identity does not have the required BigQuery '
                'table read permission.') from None
        except Forbidden:
            raise ConnectorError(
                'The selected identity does not have the required BigQuery '
                'table metadata permission.') from None
        except Exception as e:
            logger.error(
                "Error while checking BigQuery read permissions "
                "(error_type=%s)",
                type(e).__name__)
            raise ConnectorError('Kumo was unable to read the configured '
                                 'BigQuery table.') from None

    def _use_credentials(self, service: str, from_json: Callable[[str], Any],
                         from_account_info: Callable[[Dict[str, str]], Any],
                         store_credentials: bool = False) -> Any:
        r"""Uses credentials associated with a Google Cloud service.
        NOTE: The `from_json` path should only be used in tests. Production
        workflow only uses credentials in AWS service account keys."""
        def _store_credentials(parsed_credentials: Dict[str, str]):
            for key, value in parsed_credentials.items():
                self._credentials[key] = SecretStr(value)

        service_account_json = os.getenv(ENV_GOOGLE_CREDENTIALS_JSON,
                                         ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH)
        # If the service account json exists, use it to initialize the client.
        # This is used only in tests.
        if fsutil.exists(service_account_json):
            with open(service_account_json, "r") as f:
                parsed_credentials: Dict[str, str] = json.load(f)

            # GitHub Workload Identity Federation produces an
            # ``external_account`` ADC file, not a service-account key. Load
            # it through google-auth and optionally impersonate one of the
            # deliberately restricted BigQuery test identities.
            if parsed_credentials.get("type") != "service_account":
                credential, _ = load_credentials_from_file(
                    service_account_json,
                    scopes=[GOOGLE_CLOUD_PLATFORM_SCOPE],
                )
                target_principal = os.getenv(
                    ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT)
                if target_principal:
                    credential = impersonated_credentials.Credentials(
                        source_credentials=credential,
                        target_principal=target_principal,
                        target_scopes=[GOOGLE_CLOUD_PLATFORM_SCOPE],
                    )
                if store_credentials:
                    self._credentials["provider"] = "bigquery"
                    self._credentials["credential"] = credential
                return Client(
                    project=self.project_id,
                    credentials=credential,
                    location=self.location,
                )

            if store_credentials:
                _store_credentials(parsed_credentials)
            return from_json(service_account_json)
        elif service_account_json == ENV_GOOGLE_CREDENTIALS_JSON:
            raise ConnectorError(
                message='The configured service-account key file does not '
                'exist.')

        # Otherwise use service account keys.
        service_account_key = f'kumo-{service}-service-account-key'
        try:
            credentials = get_secrets(service_account_key)
            if "json" in credentials:
                # Support having the json stored in json key instead.
                credentials = json.loads(credentials["json"])
            if store_credentials:
                _store_credentials(credentials)
            service_account_info = credentials
            return from_account_info(service_account_info)
        except Exception as e:
            logger.error(
                "Error accessing stored BigQuery credentials "
                "(error_type=%s)",
                type(e).__name__)
            raise ConnectorError(
                message=f"Kumo failed to retrieve stored {service} credentials.",
            ) from None


class BigQueryOutputSchemaError(ConnectorError):
    """Prediction output does not fit an existing output table; names the
    columns and types, never the data."""


# Explicit dataclass definition ###############################################

BigQueryConnectorConfig = to_dataclass(BigQueryConnector,
                                       base_cls=ConnectorConfig)
register(BigQueryConnector, BigQueryConnectorConfig, group="connector")
