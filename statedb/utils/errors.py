# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import re
from enum import Enum
from typing import Callable, List, NoReturn, Optional, Tuple, Type

from kumoapi.data_snapshot import GraphSnapshotID, TableSnapshotID

from common_utils.error import ErrorInfo, PublicException
from io_utils import logging
from io_utils.errors import mark_exception_restricted
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    ConnectorID,
    GraphID,
    IdentityID,
    PredictiveQueryID,
    TableID,
    TrainingJobID,
)


def extract_public_exception(error):
    """Load workflow-specific error traversal only when it is requested."""
    from temporalio_commons.error_utils import extract_public_exception as impl
    return impl(error)


def is_error_non_retryable(error):
    """Load Temporal retry classification only on an error handling path."""
    from temporalio_commons.error_utils import is_error_non_retryable as impl
    return impl(error)


def _import_optional(module_path: str):
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None


# Import error types needed to error message parsing
SnowflakeErrors = _import_optional("snowflake.connector.errors")
DatabricksErrors = _import_optional("databricks.sql.exc")
# Databricks SDK platform errors (`databricks.sdk.errors.*`) are a different
# package from the SQL connector errors (`databricks.sql.exc`) imported above.
# Unity Catalog file connectors still use the Files API, so retain separate
# classifiers for those source/file-connector operations.
DatabricksSdkErrors = _import_optional("databricks.sdk.errors")
DatabricksSdkError = getattr(DatabricksSdkErrors, "DatabricksError",
                             None) if DatabricksSdkErrors else None
DatabricksSdkPlatform = _import_optional("databricks.sdk.errors.platform")
DatabricksSdkPermissionDenied = getattr(DatabricksSdkPlatform,
                                        "PermissionDenied",
                                        None) if DatabricksSdkPlatform else None
DatabricksSdkUnauthenticated = getattr(DatabricksSdkPlatform, "Unauthenticated",
                                       None) if DatabricksSdkPlatform else None
DatabricksSdkNotFound = getattr(DatabricksSdkPlatform, "NotFound",
                                None) if DatabricksSdkPlatform else None
RequestsErrors = _import_optional("requests.exceptions")
RequestsHTTPError = (getattr(RequestsErrors, "HTTPError", None)
                     if RequestsErrors else None)
RequestsRetryError = (getattr(RequestsErrors, "RetryError", None)
                      if RequestsErrors else None)
GoogleAPIErrors = _import_optional("google.api_core.exceptions")
GoogleAuthErrors = _import_optional("google.auth.exceptions")
BotocoreErrors = _import_optional("botocore.exceptions")
BotocoreError = (getattr(BotocoreErrors, "BotoCoreError", None)
                 if BotocoreErrors else None)
BotocoreClientError = (getattr(BotocoreErrors, "ClientError", None)
                       if BotocoreErrors else None)
AzureErrors = _import_optional("azure.core.exceptions")
AzureError = (getattr(AzureErrors, "AzureError", None) if AzureErrors else None)
Py4JProtocol = _import_optional("py4j.protocol")
Py4JJavaError = getattr(Py4JProtocol, "Py4JJavaError",
                        None) if Py4JProtocol else None

DiskgraphCErrors = _import_optional("diskgraph._c")
DiskgraphDataError = getattr(DiskgraphCErrors, "DiskgraphDataError",
                             None) if DiskgraphCErrors else None
DiskgraphTargetCategoryLimitError = getattr(
    DiskgraphCErrors, "DiskgraphTargetCategoryLimitError",
    None) if DiskgraphCErrors else None
DiskgraphConfigError = getattr(DiskgraphCErrors, "DiskgraphConfigError",
                               None) if DiskgraphCErrors else None
DiskgraphSystemError = getattr(DiskgraphCErrors, "DiskgraphSystemError",
                               None) if DiskgraphCErrors else None

logger = logging.getLogger(__name__)


class AuthenticationError(ValueError):
    pass


class UnauthorizedError(AuthenticationError):
    pass


class NotSupportedError(AuthenticationError):
    pass


class NotFoundError(AuthenticationError):
    pass


class ForbiddenError(AuthenticationError):
    pass


class ExternalMembershipUnavailableError(AuthenticationError):
    """Authorization needs an external membership result that is unavailable."""


class EntityDoesNotExistError(ValueError):
    def __init__(self, entity_name: str, entity_id: str) -> None:
        msg = f"{entity_name} with id='{entity_id}' does not exist"
        return super().__init__(msg)


class ConnectorDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: ConnectorID) -> None:
        return super().__init__("Connector", id)


class IdentityDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: IdentityID) -> None:
        return super().__init__("Identity", id)


class IdentityForbiddenError(ForbiddenError):
    pass


class IdentityInUseError(ValueError):
    pass


class IdentityConflictError(ValueError):
    pass


class ResourceSlugConflictError(PublicException):
    """An immutable project, group, or identity slug is already reserved."""

    http_code = 409


class IdentityValidationError(PublicException):
    """Stored identity credentials could not be resolved or authenticated.

    This error can originate in the credential-broker activity, so it must be
    a ``PublicException`` to retain its sanitized message and HTTP status when
    Temporal serializes it.  Callers must never attach the provider exception
    or credential payload to this error.
    """

    http_code = 422


class ConnectorValidationError(PublicException):
    """An explicitly selected identity cannot access a connector target."""

    http_code = 422


# Upper bound on the warehouse's own error text surfaced to the authoring UI:
# enough to carry the actionable reason (unknown function, missing table,
# syntax error) without pasting a multi-KB provider dump into the client.
_MAX_SQL_QUERY_ERROR_CHARS = 400


def sql_query_error_message(error: Exception) -> str:
    """User-facing message for a warehouse-rejected Custom SQL query.

    Surfaces the warehouse's *own* reason (unknown function, missing table,
    syntax error) behind one actionable prefix, so the authoring UI can tell a
    bad query apart from a credential/connection problem. Only ever called with
    a post-``open`` query-execution error (describe/sample) — never a
    pre-connect/connection error, whose text can carry secrets. The raw reason
    is flattened to a single whitespace-collapsed, length-bounded line; the
    traceback stays in the server log.
    """
    reason = ' '.join(str(error).split()).strip()
    if len(reason) > _MAX_SQL_QUERY_ERROR_CHARS:
        reason = reason[:_MAX_SQL_QUERY_ERROR_CHARS].rstrip() + '…'
    prefix = 'The warehouse could not run this query.'
    return f'{prefix} {reason}' if reason else prefix


class CustomSqlQueryError(ConnectorValidationError):
    """A Custom SQL query the warehouse rejected while describing or sampling it.

    A subclass of :class:`ConnectorValidationError` so the connector activity
    boundary's ``except (…, ConnectorValidationError): raise`` allowlist
    propagates it unflattened — but it carries the warehouse's own actionable
    reason (built with :func:`sql_query_error_message`) instead of the generic
    "source access failed" that a genuine connection/auth failure keeps. Raised
    only for a failure in the describe/sample step that runs *after*
    ``connector.open()``, so its message never carries pre-connect secrets.
    """

    http_code = 422


class ConnectorSourceTableNotFoundError(PublicException, KeyError):
    """Connector discovery could not find or access the requested source.

    The columns and sample activities translate both typed source-table errors
    to this established HTTP 404 contract.
    """

    http_code = 404


class ConnectorValidationDiagnosticCode(str, Enum):
    """Allowlisted, non-secret connector validation failure categories.

    These codes are internal signals from native connectors to the identity
    resolver.  They must contain no target, principal, SQL, or provider
    response data.
    """

    DATABRICKS_SAMPLE_TABLE_PERMISSIONS = (
        'databricks_sample_table_permissions')

    #: The configured target (schema / dataset / catalog) does not exist or is
    #: not visible to the selected identity.
    TARGET_NOT_FOUND = 'target_not_found'

    #: The target exists but the selected identity lacks the grants required to
    #: read it.
    TARGET_ACCESS_DENIED = 'target_access_denied'

    #: The target could not be accessed and the provider error does not let us
    #: distinguish "not found" from "access denied" (e.g. Snowflake reports a
    #: single privilege-or-existence failure for the database/schema/warehouse).
    TARGET_INACCESSIBLE = 'target_inaccessible'


class ConnectorValidationDiagnosticError(Exception):
    """A connector validation failure with an allowlisted safe category."""
    def __init__(self, code: ConnectorValidationDiagnosticCode) -> None:
        self.code = code
        super().__init__('Connector target validation failed')


def connector_validation_diagnostic_value(
        error: BaseException) -> Optional[str]:
    """Return the safe diagnostic-code string for a connector target-validation
    failure, or ``None`` when the failure is unclassified.

    Normalizes the two connector-side signalling shapes: the standalone
    :class:`ConnectorValidationDiagnosticError`, and any exception (in practice
    a :class:`~kumo.connector.connector.ConnectorError`) carrying a
    ``diagnostic_code`` attribute. The attribute form lets a connector keep
    raising its detailed public message for non-masked callers (e.g. ingest)
    while still signalling a safe category to the masking boundaries. Only the
    enum *value* is returned; callers must never surface the exception message
    itself, which may embed provider response data.
    """
    code = (error.code if isinstance(error, ConnectorValidationDiagnosticError)
            else getattr(error, 'diagnostic_code', None))
    # Enforce the allowlist for both shapes: only a genuine enum member — never
    # an arbitrary attribute value — is ever surfaced through a masking boundary.
    if isinstance(code, ConnectorValidationDiagnosticCode):
        return code.value
    return None


class NoIdentityValidatesTargetError(PublicException):
    """No group-bound identity can access an automatic connector target."""

    http_code = 422


class MultipleIdentitiesValidateTargetError(PublicException):
    """More than one group-bound identity can access a connector target."""

    http_code = 409


class IdentityResolutionUnavailableError(PublicException):
    """The trusted workload-identity broker is temporarily unavailable."""

    http_code = 503


class ConnectorNameConflictError(PublicException, ValueError):
    http_code = 409


class ConnectorInUseError(ValueError):
    pass


class PredictiveQueryDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: PredictiveQueryID) -> None:
        return super().__init__("Predictive Query", id)


class TableDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: TableID) -> None:
        return super().__init__("Table", id)


class GraphDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: GraphID) -> None:
        return super().__init__("Graph", id)


class BatchPredictionDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: BatchPredictionJobID) -> None:
        return super().__init__("Batch Prediction", id)


class BatchPredictionWorkflowDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: BatchPredictionWorkflowID) -> None:
        return super().__init__("Batch Prediction", id)


class TrainingJobDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: TrainingJobID) -> None:
        return super().__init__("Training Job", id)


class GraphSnapshotDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: GraphSnapshotID) -> None:
        return super().__init__("Graph Snapshot", id)


class TableSnapshotDoesNotExistError(EntityDoesNotExistError):
    def __init__(self, id: TableSnapshotID) -> None:
        return super().__init__("Table Snapshot", id)


def rethrow_or_wrap(
    e: Exception,
    wrapper_exc_type: type[PublicException],
    fallback: str,
) -> NoReturn:
    """Re-raise if e contains a PublicException; otherwise wrap with fallback message."""
    if extract_public_exception(e):
        raise

    raise wrapper_exc_type(
        message=fallback,
        error=e,
        suffix_unclassified_error=True,
        non_retryable=is_error_non_retryable(e),
    ) from None


def maybe_get_public_message(error: BaseException) -> str:
    """Return the user-facing message from a PublicException, or empty string.

    Accepts ``BaseException`` because workflow crash handlers catch
    ``BaseException``; ``extract_public_exception`` already takes it.
    """
    public_ex = extract_public_exception(error)
    return public_ex.get_user_message() if public_ex else ""


Classifier = Tuple[str, Type[Exception], Optional[Callable[[Exception],
                                                           ErrorInfo]]]


def _classifiers(*classifiers: Optional[Classifier], ) -> List[Classifier]:
    return [classifier for classifier in classifiers if classifier is not None]


# Friendly messages for Databricks Files API errors used by UC file connectors.
def _databricks_sdk_permission_denied_info(err: Exception) -> ErrorInfo:
    return ErrorInfo(
        source="Databricks",
        exception_msg=(
            "Databricks denied access to the Unity Catalog volume over the "
            "Files API. Verify the connector's Databricks token has the "
            "Files API ('files') scope and the required USE CATALOG, USE "
            "SCHEMA, and READ VOLUME or WRITE VOLUME privileges."),
        friendly=None,
        stop_chain=True,
    )


def _databricks_sdk_not_found_info(err: Exception) -> ErrorInfo:
    return ErrorInfo(
        source="Databricks",
        exception_msg=(
            "Databricks could not find the requested Unity Catalog volume or "
            "path. Verify it exists and the connector principal has USE "
            "CATALOG, USE SCHEMA, and the required volume privilege."),
        friendly=None,
        stop_chain=True,
    )


def _fixed_provider_error_info(
    source: str,
    exception_msg: str,
    friendly: str,
) -> Callable[[Exception], ErrorInfo]:
    """Build a classifier that never reads an untrusted provider message."""
    def classify(err: Exception) -> ErrorInfo:
        logger.error(
            "%s provider operation failed (error_type=%s)",
            source,
            type(err).__name__,
        )
        return ErrorInfo(
            source=source,
            exception_msg=exception_msg,
            friendly=friendly,
            stop_chain=True,
        )

    return classify


_databricks_provider_error_info = _fixed_provider_error_info(
    "Databricks",
    "Databricks could not complete the connector operation.",
    "Verify the selected identity's Databricks permissions and connector "
    "target, then retry.",
)


def _snowflake_provider_error_info(err: Exception) -> ErrorInfo:
    """Classify a Snowflake failure using only allowlisted scalar fields."""
    safe_fields = []
    errno = getattr(err, "errno", None)
    if isinstance(errno, int):
        safe_fields.append(f"code={errno}")
    sqlstate = getattr(err, "sqlstate", None)
    if isinstance(sqlstate, str) and re.fullmatch(r"[0-9A-Z]{5}", sqlstate):
        safe_fields.append(f"sqlstate={sqlstate}")
    query_id = getattr(err, "sfqid", None) or getattr(err, "query_id", None)
    if isinstance(query_id, str) and re.fullmatch(
            r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
            r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}", query_id):
        safe_fields.append(f"query_id={query_id}")

    logger.error(
        "Snowflake provider operation failed (error_type=%s%s)",
        type(err).__name__,
        f" {' '.join(safe_fields)}" if safe_fields else "",
    )
    details = f" ({', '.join(safe_fields)})" if safe_fields else ""
    return ErrorInfo(
        source="Snowflake",
        exception_msg=f"Snowflake could not complete the operation{details}.",
        friendly=("Verify the selected identity's Snowflake permissions and "
                  "connector target, then retry."),
        stop_chain=True,
    )


_google_api_error_info = _fixed_provider_error_info(
    "Google Cloud",
    "The Google Cloud provider rejected or could not complete the connector "
    "operation.",
    "Verify the selected identity's Google Cloud permissions and the "
    "connector target, then retry.",
)
_aws_provider_error_info = _fixed_provider_error_info(
    "AWS",
    "AWS could not complete the connector operation.",
    "Verify the selected identity's AWS permissions and connector target, "
    "then retry.",
)
_azure_provider_error_info = _fixed_provider_error_info(
    "Azure",
    "Azure could not complete the connector operation.",
    "Verify the selected identity's Azure permissions and connector target, "
    "then retry.",
)


class ProviderBackendError(PublicException):
    """Provider-safe base for errors that can wrap connector or fsutil I/O."""
    CLASSIFIERS = _classifiers(
        SnowflakeErrors and
        ("Snowflake", SnowflakeErrors.Error, _snowflake_provider_error_info),
        DatabricksErrors and
        ("Databricks", DatabricksErrors.Error, _databricks_provider_error_info),
        # Databricks SDK platform errors (UC Files API, most specific first).
        DatabricksSdkPermissionDenied and
        ("Databricks", DatabricksSdkPermissionDenied,
         _databricks_sdk_permission_denied_info),
        DatabricksSdkUnauthenticated
        and ("Databricks", DatabricksSdkUnauthenticated,
             _databricks_sdk_permission_denied_info),
        DatabricksSdkNotFound and
        ("Databricks", DatabricksSdkNotFound, _databricks_sdk_not_found_info),
        # Broad SDK base last: retain a fixed provider summary rather than
        # exposing a response body or request payload from the SDK exception.
        DatabricksSdkError and
        ("Databricks", DatabricksSdkError, _databricks_provider_error_info),
        GoogleAPIErrors and ("Google Cloud", GoogleAPIErrors.GoogleAPIError,
                             _google_api_error_info),
        GoogleAuthErrors and ("Google Cloud", GoogleAuthErrors.GoogleAuthError,
                              _google_api_error_info),
        BotocoreClientError
        and ("AWS", BotocoreClientError, _aws_provider_error_info),
        BotocoreError and ("AWS", BotocoreError, _aws_provider_error_info),
        AzureError and ("Azure", AzureError, _azure_provider_error_info),
    )

    def __init__(self, *args, **kwargs):
        error = kwargs.get('error')
        if isinstance(error, BaseException):
            marked_error = mark_exception_restricted(
                error,
                safe_message='Provider operation failed',
            )
            if marked_error is not error:
                kwargs['error'] = marked_error
            logger.debug(
                'Restricted raw provider failure (wrapper=%s, '
                'error_type=%s)',
                type(self).__name__,
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )
        super().__init__(*args, **kwargs)


class TableIngestionError(ProviderBackendError):
    """Error class for table ingestion failures across all backends."""

    # yapf: disable
    CLASSIFIERS: List[Classifier] = list()
    # yapf: enable

    def __init__(self, *args, connector_type: Optional[str] = None, **kwargs):
        self.connector_type = connector_type
        error = kwargs.get('error')
        if error and connector_type:
            try:
                error._kumo_connector_type = connector_type
            except (AttributeError, TypeError):
                pass
        super().__init__(*args, **kwargs)


class TableIngestionOutOfMemoryError(PublicException):
    """The worker ran out of memory while exporting a source table.

    Raised for an in-process allocation failure (``MemoryError``, including
    pyarrow's ``ArrowMemoryError``) caught before the kernel's OOM killer can
    intervene. Always constructed with a fixed local message and never carries
    the provider exception or provider payload, so the table-ingestion
    sanitizer can propagate it verbatim. Non-retryable because retrying the
    same data deterministically runs out of memory again (kumo#30084).
    """

    non_retryable = True
    http_code = 422


class TableInvalidColumnNameError(PublicException):
    """Raised when table contains an invalid column name."""


class TableInvalidName(PublicException):
    """Raised when table ID or name is invalid."""


class TableDefinitionAlreadyExists(PublicException):
    """Raised when table definition is already saved in the DB."""


# =============================================================================
# Project and identity authorization errors
# =============================================================================


class TableExistsInProjectError(PublicException):
    """Raised when table already exists in a project."""


class TableDoesntExistsInProjectError(PublicException):
    """Raised when table doesn't exist in a project."""


class GroupAlreadyExists(PublicException):
    """Raised when an IdP group has already been mapped to a Kumo group."""


class GroupInputError(ValueError):
    """Raised when a native-group relation input is empty or malformed."""


class GroupRelationConflictError(ValueError):
    """Raised when a group relation is already owned elsewhere."""


class GroupNotFoundError(PublicException):
    """Raised when a Kumo group cannot be found by the given ID."""

    http_code = 404


class ProjectNotFoundError(PublicException):
    """Raised when a project cannot be found by the given ID."""

    http_code = 404


class ProjectForbiddenError(PublicException):
    """Raised when the current user does not have permission for a project operation."""

    http_code = 403


class ProjectNotAvailableError(PublicException):
    """Raised when an archived project cannot accept mutations."""

    http_code = 422


class ProjectNameConflictError(PublicException):
    """Raised when a globally normalized project name is unavailable."""

    http_code = 409


class ArtifactExportJobNotFoundError(PublicException):
    """Raised when an artifact export is absent from the requested project."""

    http_code = 404


# =============================================================================
# Infrastructure Errors
# =============================================================================


class DatabricksClusterError(PublicException):
    """Raised when Databricks cluster operations fail."""


# =============================================================================
# Data Ingestion and Refresh Errors
# =============================================================================


class ColumnStatsError(ProviderBackendError):
    """Raised when column statistics computation fails."""


class EdgeStatsComputationError(ProviderBackendError):
    """Raised when edge statistics computation fails."""


class DiskgraphError(PublicException):
    """Raised when DiskGraph backend operations fail."""

    CLASSIFIERS: List[Classifier] = list(
        filter(
            None,
            (DiskgraphDataError and
             ("DiskGraph", DiskgraphDataError, lambda e: ErrorInfo(
                 source="DiskGraph Data",
                 exception_msg=str(e),
                 friendly="The requested data is invalid or not found in the "
                 "DiskGraph index. Please check your schema.",
             )), DiskgraphConfigError and
             ("DiskGraph", DiskgraphConfigError, lambda e: ErrorInfo(
                 source="DiskGraph Config",
                 exception_msg=str(e),
                 friendly="The DiskGraph job configuration is invalid.",
             )), DiskgraphSystemError and
             ("DiskGraph", DiskgraphSystemError,
              lambda e: ErrorInfo(source="DiskGraph System", exception_msg=str(
                  e), friendly=None)))))


class UnsupportedInDiskgraphError(PublicException):
    """Raised when a connector or feature has no DiskGraph-native path.

    DiskGraph mode requires source parquet in blob storage. Warehouse
    connectors must implement ``export_table_to_blob_storage``; file /
    blob connectors must expose a parquet path.
    """


class LLMEmbeddingError(PublicException):
    """Raised when LLM embedding generation fails."""


class GraphRefreshError(PublicException):
    """Raised when graph-level ingestion orchestration fails."""


class GraphExistsInProjectError(PublicException):
    """Raised when graph already exists in a project."""


class GraphDoesntExistsInProjectError(PublicException):
    """Raised when graph doesn't exist in a project."""


class GraphDefinitionAlreadyExists(PublicException):
    """Raised when graph definition is already saved in the DB."""


class GraphNotInProjectError(PublicException):
    """Raised when a training job references a graph not in the specified project."""

    http_code: int = 403


class TableNotInProjectError(PublicException):
    """Raised when a training job references a table not in the specified project."""

    http_code: int = 403


class TrainingJobNotInProjectError(PublicException):
    """Raised when a batch prediction job references a training job from a different project."""

    http_code: int = 403


class BatchPredictionNotInProjectError(PublicException):
    """Raised when a batch prediction job is outside the requested project."""

    http_code: int = 403


class DataRefreshError(PublicException):
    """Raised when critical data refresh failures occur."""


# =============================================================================
# Training Errors
# =============================================================================


class TrainingTableGenerationError(ProviderBackendError):
    """Raised when training table generation fails."""


class TrainingDataMaterializationError(ProviderBackendError):
    """Raised when training data materialization fails."""


class TrainingJobOutputError(PublicException):
    """Raised when training job outputs are missing or invalid."""


class InvalidGraphError(PublicException):
    """Raised when graph is in an invalid state."""


class GraphMaterializationError(ProviderBackendError):
    """Raised when graph materialization fails."""


# =============================================================================
# Batch Prediction Errors
# =============================================================================


class BatchPredictionJobError(PublicException):
    """Raised when batch prediction job fails."""


class PredictionDataMaterializationError(ProviderBackendError):
    """Raised when prediction data materialization fails."""


class IncompatiblePredictionTable(PublicException):
    """Raised when prediction table is incompatible with the trained model."""


class PredictionOutputProcessingError(ProviderBackendError):
    """Raised when prediction output postprocessing fails."""


class MissingTrainingJobError(PublicException):
    """Raised when a required training job cannot be found."""


class EmptyPredictionTableError(PublicException):
    """Raised when custom prediction table is empty."""

    template = ("The custom prediction table at '{path}' is empty. "
                "Please provide a table containing at least one row of data "
                "to generate predictions.")


class InvalidPredictionTableError(ProviderBackendError):
    """Raised when custom prediction table cannot be read or validated."""

    template = ("Unable to read the custom prediction table at '{path}'. "
                "Please verify the path is correct, the table exists, "
                "and you have read permissions.")


# =============================================================================
# Data Export Errors
# =============================================================================


class ArtifactExportError(ProviderBackendError):
    """Raised when artifact export to customer warehouse fails."""


class ArtifactNotFoundError(PublicException):
    """Raised when a required artifact file or directory is not found."""

    template = ("Unable to find the expected artifact at '{path}'. "
                "Please verify the job completed successfully.")


class TrainingJobStartError(PublicException):
    """Raised when training job fails to start within timeout."""

    template = (
        "Training job failed to start within {timeout_seconds} seconds. "
        "Check compute capacity, job queue status, or try re-running the job.")


class BestTrainerCfgFileMissingError(PublicException):
    """Raised when best trainer config file is missing after training."""

    template = (
        "Training job completed but did not produce a valid model "
        "configuration. Please re-run the training job or contact support "
        "if the issue persists.")
