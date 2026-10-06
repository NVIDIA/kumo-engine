# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from temporalio import activity

from common_utils.error import PublicException
from rest.backends.data_backend.connector_activity_registry import (
    connector_authorized_activity,
)
from rest.backends.data_backend.connector_broker import (
    connector_credential_broker,
)
from rest.backends.data_backend.principal_authority import (
    PrincipalAuthorityDeniedError,
)
from rest.config.data_backend import ConnectorExecutionRef
from statedb.db.connector_authorizations import ConnectorAuthorizationError
from statedb.rest.connector_authorization import (
    CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
    CONNECTOR_GRAPH_ACTIONS,
    CONNECTOR_OUTPUT_ACTIONS,
    CONNECTOR_TABLE_SOURCE_ACTIONS,
    ConnectorCapabilityAction,
)
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    GroupNotFoundError,
    IdentityDoesNotExistError,
    IdentityForbiddenError,
    ProjectForbiddenError,
    ProjectNotAvailableError,
    ProjectNotFoundError,
)


class ConnectorDelegationDeniedError(PublicException):
    """A durable connector grant became permanently unavailable."""

    http_code = 403


_PERMANENT_DELEGATION_DENIALS = (
    ConnectorAuthorizationError,
    ConnectorDoesNotExistError,
    GroupNotFoundError,
    IdentityDoesNotExistError,
    IdentityForbiddenError,
    PrincipalAuthorityDeniedError,
    ProjectForbiddenError,
    ProjectNotAvailableError,
    ProjectNotFoundError,
)

_TABLE_WORKFLOWS = frozenset({
    'TableRefreshWorkflow',
    'TableIngestionWorkflow',
})
_SNAPSHOT_WORKFLOWS = frozenset({'DiskgraphIngestWorkflow'})
_TRAINING_JOB_WORKFLOWS = frozenset({'TrainingJobWorkflow'})
_OUTPUT_DELEGATING_WORKFLOWS = frozenset({'BatchPredictionJobWorkflow'})
_OUTPUT_WORKFLOWS = frozenset({
    'ArtifactExportWorkflow',
})
_CHILD_DELEGATING_WORKFLOWS = frozenset({
    *_OUTPUT_DELEGATING_WORKFLOWS,
    'BatchPredictionParentWorkflow',
    'GeneratePredictionTableWorkflow',
    'GenerateTrainingTableWorkflow',
    'GraphRefreshWorkflow',
    'TableRefreshWorkflow',
    'TrainingWorkflow',
})
_KNOWN_CONNECTOR_WORKFLOWS = frozenset({
    *_TABLE_WORKFLOWS,
    *_SNAPSHOT_WORKFLOWS,
    *_TRAINING_JOB_WORKFLOWS,
    *_OUTPUT_DELEGATING_WORKFLOWS,
    *_OUTPUT_WORKFLOWS,
    *_CHILD_DELEGATING_WORKFLOWS,
    'BatchPredictionStatsWorkflow',
    'CustomTrainTableWorkflow',
    'MaterializePredictionTaskWorkflow',
    'MaterializeTrainingTaskWorkflow',
    'SplitTrainingTableWorkflow',
})


def _actions_for_child_workflow(
    workflow_type: str, ) -> set[ConnectorCapabilityAction]:
    if workflow_type not in _KNOWN_CONNECTOR_WORKFLOWS:
        raise ValueError(
            f'Workflow {workflow_type!r} cannot receive connector authority')
    if workflow_type in _SNAPSHOT_WORKFLOWS:
        # The pool-side snapshot activity downloads every warehouse source
        # before DiskGraph ingest.  This child carries no other connector-
        # bearing work, so delegate only that exact closed action.
        actions = {ConnectorCapabilityAction.ENSURE_SNAPSHOT}
    elif workflow_type in _TRAINING_JOB_WORKFLOWS:
        actions = {
            ConnectorCapabilityAction.POSTPROCESS_TRAINING_JOB_OUTPUT,
        }
    elif workflow_type in _OUTPUT_DELEGATING_WORKFLOWS:
        # Batch-prediction workers may upload their final Parquet shards
        # directly into a Databricks managed volume before the coordinator
        # starts the output workflow.
        actions = set(CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS)
    elif workflow_type in _TABLE_WORKFLOWS:
        actions = set(CONNECTOR_TABLE_SOURCE_ACTIONS)
    elif workflow_type in _OUTPUT_WORKFLOWS:
        actions = set(CONNECTOR_OUTPUT_ACTIONS)
    else:
        actions = set(CONNECTOR_GRAPH_ACTIONS | CONNECTOR_OUTPUT_ACTIONS)
    if workflow_type not in _CHILD_DELEGATING_WORKFLOWS:
        actions.discard(ConnectorCapabilityAction.DELEGATE_CHILD)
    return actions


@connector_authorized_activity
def delegate_connector_authorizations(
    references: list[ConnectorExecutionRef],
    child_workflow_id: str,
    child_workflow_type: str,
) -> dict[str, str]:
    """Replace parent tokens with exact child-workflow capabilities."""
    actions = _actions_for_child_workflow(child_workflow_type)
    try:
        return connector_credential_broker.delegate_connector_references(
            references,
            child_workflow_id=child_workflow_id,
            child_actions=actions,
        )
    except _PERMANENT_DELEGATION_DENIALS:
        # Do not expose connector IDs, group IDs, identity IDs, principal
        # authorization context, or capability validation reasons through
        # Temporal history.
        raise ConnectorDelegationDeniedError(
            message=('Connector access changed before the child workflow could '
                     'start. Submit a new job with a connector that is still '
                     'available to this project.'),
            non_retryable=True,
        ) from None


@activity.defn
def revoke_connector_authorizations(tokens: list[str]) -> None:
    """Best-effort child-start rollback without exposing token values."""
    async def revoke(services) -> None:
        for token in tokens:
            await services.connector_authorizations.revoke(token)

    connector_credential_broker.run(revoke)
