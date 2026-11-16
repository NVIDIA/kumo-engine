# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

from io_utils import kumo_dataclass
from statedb.config.common import (
    ConnectorAuthorizationID,
    ConnectorAuthorizationUseID,
    DBEntry,
)


class ConnectorAuthorizationResourceKind(str, Enum):
    CONNECTOR = 'connector'
    PROJECT_CONNECTOR_CREATE = 'project_connector_create'
    IDENTITY_VALIDATE = 'identity_validate'


class ConnectorCapabilityAction(str, Enum):
    """Closed activity operations allowed to exercise credential authority."""

    DELEGATE_CHILD = 'delegate_connector_authorizations'
    LIST_CONNECTOR_SOURCE_TABLES = 'list_connector_source_tables'
    GET_CONNECTOR_SOURCE_TABLE_FILE_TYPE = (
        'get_connector_source_table_file_type')
    STAGE_CONNECTOR_SOURCE_DOWNLOAD = 'stage_connector_source_download'
    GET_CONNECTOR_SOURCE_COLUMNS = 'get_connector_source_columns'
    VALIDATE_CONNECTOR_SOURCE_TABLE_EXISTS = (
        'validate_connector_source_table_exists')
    GET_CONNECTOR_SAMPLE_ROWS = 'get_connector_sample_rows'
    GET_CONNECTOR_TABLE_DATA_VERSION = 'get_connector_table_data_version'
    IS_CONNECTOR_SOURCE_TABLE_VIEW = 'is_connector_source_table_view'
    VALIDATE_CONNECTOR_REFERENCE = 'validate_connector_reference'
    GET_TABLE_COL_STATS_AND_SAMPLE_ROWS = (
        'get_table_col_stats_and_sample_rows_activity')
    VALIDATE_CUSTOM_SQL = 'validate_custom_sql'
    COMPUTE_STANDALONE_TABLE_COL_STATS = ('compute_standalone_table_col_stats')
    START_BATCH_PREDICTION_STATS = 'start_bp_stats_activity'
    INFER_TABLE_METADATA = 'infer_table_metadata_activity'
    VALIDATE_TABLE_DEFINITION = 'validate_table_definition_activity'
    INFER_GRAPH_LINKS = 'infer_graph_links_activity'
    VALIDATE_GRAPH_DEFINITION = 'validate_graph_definition_activity'
    VALIDATE_PREDICTIVE_QUERY = 'validate_predictive_query_activity'
    STAGE_CUSTOM_PREDICTION_TABLE = 'stage_custom_prediction_table'
    INGEST_CUSTOM_TRAIN_TABLE = 'ingest_custom_train_table'
    VALIDATE_CUSTOM_TRAIN_TABLE = 'validate_custom_train_table_activity'
    UPDATE_TABLE_TIMESTAMP_UNITS = 'update_table_timestamp_units'
    VALIDATE_TABLE_INGEST = 'validate_table_ingest'
    BUILD_GRAPH_CONFIG = 'build_graph_config_activity'
    REFRESH_GRAPH_DATA_VERSION = 'refresh_graph_data_version'
    GENERATE_PREDICTION_CONFIG = 'generate_prediction_config'
    BUILD_TTNG_CUSTOM_PRED_TABLE_BUNDLE = (
        'build_ttng_custom_pred_table_bundle')
    GENERATE_TRAINING_CONFIG = 'generate_training_config'
    BUILD_TTNG_CUSTOM_TRAIN_TABLE_BUNDLE = (
        'build_ttng_custom_train_table_bundle')
    MATERIALIZE_TRAINING_TABLE = 'materialize_training_table'
    EXECUTE_SPLIT = 'execute_split'
    POOL_GENERATE_TARGET_TABLE = 'gen_tt'
    POOL_MATERIALIZE_TRAINING_TASK = 'materialize_training_task'
    POOL_MATERIALIZE_PREDICTION_TASK = 'materialize_prediction_task'
    POOL_TRAIN = 'train'
    POOL_PREDICT = 'predict'
    POSTPROCESS_TRAINING_JOB_OUTPUT = 'postprocess_training_job_output'
    ENSURE_SNAPSHOT = 'ensure_snapshot'
    VALIDATE_OBJECT_PREDICTION_OUTPUT = ('validate_object_prediction_output')
    VALIDATE_WAREHOUSE_PREDICTION_OUTPUT = (
        'validate_warehouse_prediction_output')
    INGEST_TABLE_DATA = 'ingest_table_data_activity_v2'
    WRITE_ARTIFACTS_TO_CUSTOMER = 'write_artifacts_to_customer'
    CLEANUP_DATABRICKS_VOLUME_OUTPUTS = 'cleanup_databricks_volume_outputs'
    CREATE_PROJECT_CONNECTOR = 'create_project_connector_activity'
    VALIDATE_IDENTITY = 'validate_identity_credentials_activity'


CONNECTOR_DISCOVERY_ACTIONS = frozenset({
    ConnectorCapabilityAction.LIST_CONNECTOR_SOURCE_TABLES,
    ConnectorCapabilityAction.GET_CONNECTOR_SOURCE_TABLE_FILE_TYPE,
    ConnectorCapabilityAction.GET_CONNECTOR_SOURCE_COLUMNS,
    ConnectorCapabilityAction.VALIDATE_CONNECTOR_SOURCE_TABLE_EXISTS,
    ConnectorCapabilityAction.GET_CONNECTOR_SAMPLE_ROWS,
    ConnectorCapabilityAction.IS_CONNECTOR_SOURCE_TABLE_VIEW,
    ConnectorCapabilityAction.VALIDATE_CONNECTOR_REFERENCE,
})

CONNECTOR_DOWNLOAD_ACTIONS = frozenset({
    ConnectorCapabilityAction.STAGE_CONNECTOR_SOURCE_DOWNLOAD,
})

CONNECTOR_TABLE_SOURCE_ACTIONS = frozenset({
    *CONNECTOR_DISCOVERY_ACTIONS,
    ConnectorCapabilityAction.GET_CONNECTOR_TABLE_DATA_VERSION,
    ConnectorCapabilityAction.GET_TABLE_COL_STATS_AND_SAMPLE_ROWS,
    ConnectorCapabilityAction.COMPUTE_STANDALONE_TABLE_COL_STATS,
    ConnectorCapabilityAction.START_BATCH_PREDICTION_STATS,
    ConnectorCapabilityAction.INFER_TABLE_METADATA,
    ConnectorCapabilityAction.VALIDATE_TABLE_DEFINITION,
    ConnectorCapabilityAction.VALIDATE_CUSTOM_SQL,
    ConnectorCapabilityAction.UPDATE_TABLE_TIMESTAMP_UNITS,
    ConnectorCapabilityAction.VALIDATE_TABLE_INGEST,
    ConnectorCapabilityAction.INGEST_TABLE_DATA,
    ConnectorCapabilityAction.DELEGATE_CHILD,
})

CONNECTOR_GRAPH_ACTIONS = frozenset({
    *CONNECTOR_TABLE_SOURCE_ACTIONS,
    ConnectorCapabilityAction.INFER_GRAPH_LINKS,
    ConnectorCapabilityAction.VALIDATE_GRAPH_DEFINITION,
    ConnectorCapabilityAction.BUILD_GRAPH_CONFIG,
    ConnectorCapabilityAction.REFRESH_GRAPH_DATA_VERSION,
    ConnectorCapabilityAction.GENERATE_PREDICTION_CONFIG,
    ConnectorCapabilityAction.BUILD_TTNG_CUSTOM_PRED_TABLE_BUNDLE,
    ConnectorCapabilityAction.GENERATE_TRAINING_CONFIG,
    ConnectorCapabilityAction.BUILD_TTNG_CUSTOM_TRAIN_TABLE_BUNDLE,
    ConnectorCapabilityAction.MATERIALIZE_TRAINING_TABLE,
    ConnectorCapabilityAction.EXECUTE_SPLIT,
    ConnectorCapabilityAction.POOL_GENERATE_TARGET_TABLE,
    ConnectorCapabilityAction.POOL_MATERIALIZE_TRAINING_TASK,
    ConnectorCapabilityAction.POOL_MATERIALIZE_PREDICTION_TASK,
    ConnectorCapabilityAction.POOL_TRAIN,
    ConnectorCapabilityAction.POOL_PREDICT,
    ConnectorCapabilityAction.POSTPROCESS_TRAINING_JOB_OUTPUT,
    ConnectorCapabilityAction.ENSURE_SNAPSHOT,
    ConnectorCapabilityAction.INGEST_CUSTOM_TRAIN_TABLE,
    ConnectorCapabilityAction.VALIDATE_CUSTOM_TRAIN_TABLE,
    ConnectorCapabilityAction.STAGE_CUSTOM_PREDICTION_TABLE,
    ConnectorCapabilityAction.DELEGATE_CHILD,
})

CONNECTOR_PQUERY_VALIDATION_ACTIONS = frozenset({
    ConnectorCapabilityAction.VALIDATE_PREDICTIVE_QUERY,
})

CONNECTOR_OUTPUT_ACTIONS = frozenset({
    ConnectorCapabilityAction.VALIDATE_OBJECT_PREDICTION_OUTPUT,
    ConnectorCapabilityAction.VALIDATE_WAREHOUSE_PREDICTION_OUTPUT,
    ConnectorCapabilityAction.WRITE_ARTIFACTS_TO_CUSTOMER,
    ConnectorCapabilityAction.DELEGATE_CHILD,
})

CONNECTOR_BATCH_PREDICTION_OUTPUT_ACTIONS = frozenset({
    *CONNECTOR_OUTPUT_ACTIONS,
    ConnectorCapabilityAction.START_BATCH_PREDICTION_STATS,
    ConnectorCapabilityAction.CLEANUP_DATABRICKS_VOLUME_OUTPUTS,
})
CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS = frozenset({
    *CONNECTOR_BATCH_PREDICTION_OUTPUT_ACTIONS,
    ConnectorCapabilityAction.POOL_PREDICT,
})

# Every closed activity capability that may be issued in a
# ``ConnectorExecutionRef``. Keep activity-name lookup and issuance coverage
# derived from the enum/action bundles so an activity cannot silently gain
# broker access through an unregistered string.
CONNECTOR_REFERENCE_ACTIONS = frozenset({
    *CONNECTOR_DISCOVERY_ACTIONS,
    *CONNECTOR_DOWNLOAD_ACTIONS,
    *CONNECTOR_TABLE_SOURCE_ACTIONS,
    *CONNECTOR_GRAPH_ACTIONS,
    *CONNECTOR_PQUERY_VALIDATION_ACTIONS,
    *CONNECTOR_OUTPUT_ACTIONS,
    *CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
})
CONNECTOR_ACTIVITY_AUTHORIZATION_ACTIONS = frozenset({
    ConnectorCapabilityAction.CREATE_PROJECT_CONNECTOR,
    ConnectorCapabilityAction.VALIDATE_IDENTITY,
})
CONNECTOR_ISSUABLE_ACTIONS = frozenset({
    *CONNECTOR_REFERENCE_ACTIONS,
    *CONNECTOR_ACTIVITY_AUTHORIZATION_ACTIONS,
})
CONNECTOR_CAPABILITY_ACTION_BY_ACTIVITY_TYPE = {
    action.value: action
    for action in ConnectorCapabilityAction
}


def connector_capability_action_for_activity_type(
    activity_type: str, ) -> ConnectorCapabilityAction:
    """Return the exact closed action for a credential-bearing activity."""
    try:
        return CONNECTOR_CAPABILITY_ACTION_BY_ACTIVITY_TYPE[activity_type]
    except KeyError:
        raise RuntimeError(
            'Activity is not allowed to resolve connector credentials') \
            from None


@kumo_dataclass
class ConnectorAuthorizationDBEntry(DBEntry):
    """Hashed, workflow-bound authority for one credential-bearing scope."""

    id: ConnectorAuthorizationID
    token_hash: str
    principal_id: str
    principal_name: str
    auth_backend: str
    resource_kind: ConnectorAuthorizationResourceKind
    resource_id: str
    project_id: str
    request_digest: str
    allowed_actions: Tuple[ConnectorCapabilityAction, ...]
    deployment_id: str
    temporal_namespace: str
    workflow_id: str
    workflow_run_id: str
    expires_at: datetime
    revoked_at: Optional[datetime] = None
    parent_authorization_id: Optional[ConnectorAuthorizationID] = None


@kumo_dataclass
class ConnectorAuthorizationUseDBEntry(DBEntry):
    """Atomic first-use/run binding and retry identity for an activity."""

    id: ConnectorAuthorizationUseID
    authorization_id: ConnectorAuthorizationID
    workflow_id: str
    workflow_run_id: str
    activity_id: str
    action: ConnectorCapabilityAction
