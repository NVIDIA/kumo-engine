# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Execution-backend contract for the GNN training and prediction path.

The SDK's relational objects (:class:`~kumoai_studio.graph.Table`,
:class:`~kumoai_studio.graph.Graph`, :class:`~kumoai_studio.pquery.PredictiveQuery`,
:class:`~kumoai_studio.trainer.Trainer`) describe *what* to compute. A
:class:`Backend` decides *where* it is computed: the Kumo cloud platform.

The protocols below name exactly the operations the GNN path depends on, so
the backend has a finite, checkable contract. They are deliberately
narrower than :class:`~kumoai_studio.client.client.KumoClient`: connector and
identity administration and project management are **not** part of this
contract, and those call sites continue to reach the client directly.

Types crossing this boundary are :mod:`kumoapi` dataclasses, which are shared
data contracts rather than transport details. What a backend hides is
transport: endpoints, HTTP status handling, and how a job identifier resolves.

The two areas that a *plugin* implements -- :class:`~kumoapi.plugin.SourceBackend`
and :class:`~kumoapi.plugin.ArtifactSink`, with the data that crosses them --
live in :mod:`kumoapi.plugin` rather than here, because a plugin ships in its
own distribution and must not import the SDK to satisfy them. They are not
re-exported: a plugin author imports them from ``kumoapi.plugin``, which is
the one name for them, and a second spelling here would suggest the SDK owns
a contract it does not.

The protocols are structural, so the typed API classes in :mod:`kumoai_studio.client`
satisfy them without modification; see
:class:`~kumoai_studio.backend.cloud.CloudBackend`.
"""

from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import kumoapi.data_snapshot as snapshot_api
import kumoapi.graph as graph_api
import kumoapi.table as table_api
from kumoapi.common import JobStatus, ValidationResponse
from kumoapi.data_source import (
    ConnectorResponse,
    CreateConnectorArgs,
    CreateIdentityArgs,
    IdentityGroupsResponse,
    ProjectUploadConnectorResponse,
    ReplaceIdentityGroupsArgs,
    RotateIdentityCredentialsArgs,
    UpdateIdentityArgs,
)
from kumoapi.jobs import (
    ArtifactExportRequest,
    BatchPredictionJobResource,
    BatchPredictionRequest,
    CancelBatchPredictionJobResponse,
    ErrorDetails,
    GeneratePredictionTableJobResource,
    GeneratePredictionTableRequest,
    GenerateTrainTableJobResource,
    GenerateTrainTableRequest,
    JobResourceBase,
    TrainingJobRequest,
    TrainingJobResource,
    TrainingTableSpec,
)
from kumoapi.model_plan import (
    ModelPlan,
    SuggestModelPlanRequest,
    TrainingTableGenerationPlan,
)
from kumoapi.plugin import ModelRegistryLocation, PublishedModel, SourceBackend
from kumoapi.pquery import PQueryResource
from kumoapi.source_table import LocalFileSourceTable, SourceTableType
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage

from kumoai_studio.identity import Identity

__all__ = [
    'ArtifactExportBackend',
    'Backend',
    'ConnectorBackend',
    'GraphBackend',
    'IdentityBackend',
    'JobBackend',
    'PQueryBackend',
    'PredictionBackend',
    'PredictionTableBackend',
    'TableBackend',
    'TrainTableBackend',
    'TrainingBackend',
]


@runtime_checkable
class ConnectorBackend(Protocol):
    r"""Binding a connector target to a usable handle.

    *Binding* is not *administering*. Turning a target into something that can
    be read from is execution-adjacent -- no table can be listed without it --
    so it belongs in this contract. Listing, renaming and credential rotation
    remain resource management and stay outside it.

    What varies is where the handle's identity comes from: the Kumo service
    assigns an ID and resolves the bound principal server-side, while an
    in-process runtime derives one from the target itself. File-upload
    *transport* is deliberately absent; it is a cloud-only concept.
    Whether a managed-upload connector exists is not: a caller rebuilding a
    definition has to ask, and a runtime hosting no uploads answers by
    saying so.
    """
    def create_if_not_exist(
        self,
        args: CreateConnectorArgs,
        project_id: Optional[str] = None,
    ) -> ConnectorResponse:
        r"""Returns a connector for the target, creating it if needed."""
        ...

    def get(self, connector_id: str,
            project_id: Optional[str] = None) -> Optional[ConnectorResponse]:
        r"""Returns one connector by immutable ID, or ``None``."""
        ...

    def get_by_name(
        self,
        name: str,
        project_id: Optional[str] = None,
    ) -> Optional[ConnectorResponse]:
        r"""Returns one connector by display name, or ``None``."""
        ...

    def delete_if_exists(self, connector_id: str,
                         project_id: Optional[str] = None) -> bool:
        r"""Removes a connector, returning whether it existed."""
        ...

    def get_upload_connector(
        self,
        project_id: Optional[str] = None,
    ) -> Optional[ProjectUploadConnectorResponse]:
        r"""Returns the project's managed-upload connector, or ``None``.

        ``None`` where the runtime hosts no managed uploads, which is every
        runtime but the Kumo service: an upload is a file the service holds,
        so a backend that holds none has no connector to name.
        """
        ...


@runtime_checkable
class IdentityBackend(Protocol):
    r"""Resolution and administration of the principals connectors reference.

    An identity has two halves, and only one of them crosses this boundary. The
    **principal** -- an ID, a name, a provider type -- is safe, serializable
    data, and it is what a connector refers to. The **credential** goes in on
    :meth:`create` and never comes back out: the cloud stores it encrypted and
    an in-process backend keeps it in memory, and neither returns it, so an
    :class:`~kumoai_studio.identity.Identity` is credential-free either way.

    Both backends therefore implement the whole area, including
    :meth:`create`, and with the same arguments. What differs is only where
    the material comes to rest. The two operations an in-process backend
    refuses are the two that are platform concepts rather than credential
    ones: group binding, and renaming an identity whose name a saved graph
    records.

    Everything downstream of resolution is identical on both: a connector names
    an identity ID, and the backend turns it into access.
    """
    def list(self, project_id: Optional[str] = None) -> List[Identity]:
        r"""Returns the identities available to one project.

        ``project_id`` is ``None`` on a backend that has no projects.
        """
        ...

    def list_admin(self) -> List[Identity]:
        r"""Returns every identity in the tenant (administrator only)."""
        ...

    def get(self, identity_id: str) -> Identity:
        r"""Returns one identity by immutable ID."""
        ...

    def get_by_slug(self, project_id: Optional[str], slug: str) -> Identity:
        r"""Returns one identity by its project-scoped canonical slug.

        ``project_id`` is ``None`` on a backend that has no projects, as it is
        for :meth:`list`: both are reached from the same project scope, which
        a projectless runtime answers with ``None``.
        """
        ...

    def create(self, args: CreateIdentityArgs) -> Identity:
        r"""Provisions an identity from credentials (administrator only)."""
        ...

    def update(self, identity_id: str, args: UpdateIdentityArgs) -> Identity:
        r"""Renames or re-describes an identity."""
        ...

    def delete(self, identity_id: str) -> Identity:
        r"""Removes an identity."""
        ...

    def rotate_credentials(self, identity_id: str,
                           args: RotateIdentityCredentialsArgs) -> Identity:
        r"""Replaces the authentication behind an identity.

        ``args`` is :class:`~kumoapi.data_source.RotateIdentityCredentialsArgs`
        on both backends: rotation replaces the secret behind a name, and
        leaves the principal and authentication kind it was created with
        alone.
        """
        ...

    def test_credentials(self, identity_id: str) -> Identity:
        r"""Checks that the identity can currently authenticate."""
        ...

    def list_groups(self, identity_id: str) -> IdentityGroupsResponse:
        r"""Returns the groups an identity is bound to."""
        ...

    def replace_groups(
            self, identity_id: str,
            args: ReplaceIdentityGroupsArgs) -> IdentityGroupsResponse:
        r"""Replaces an identity's group bindings."""
        ...


@runtime_checkable
class TableBackend(Protocol):
    r"""Table metadata inference, validation, and snapshotting."""
    def create_table(
        self,
        table_def: table_api.TableDefinition,
        *,
        name_alias: str,
        force_rename: bool = False,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get_table_if_exists(
        self,
        table_id_or_name: str,
        project_id: Optional[str] = None,
    ) -> Optional[table_api.TableResource]:
        ...

    def create_snapshot(
        self,
        table_definition: table_api.TableDefinition,
        *,
        refresh_source: bool = False,
        project_id: Optional[str] = None,
    ) -> snapshot_api.TableSnapshotID:
        ...

    def get_snapshot(
        self,
        snapshot_id: snapshot_api.TableSnapshotID,
        project_id: Optional[str] = None,
    ) -> snapshot_api.TableSnapshotResource:
        ...

    def infer_metadata(
        self,
        request: table_api.TableMetadataRequest,
        project_id: Optional[str] = None,
    ) -> table_api.TableMetadataResponse:
        ...

    def validate_table(
        self,
        request: table_api.TableValidationRequest,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        ...

    def resolve_sql(
        self,
        request: table_api.SqlTableResolveRequest,
        project_id: Optional[str] = None,
        timeout: Optional[Tuple[float, float]] = None,
    ) -> table_api.SqlTableResolveResponse:
        ...


@runtime_checkable
class GraphBackend(Protocol):
    r"""Graph definition, link inference, validation, and snapshotting."""
    def create_graph(
        self,
        graph_def: graph_api.GraphDefinition,
        *,
        name_alias: Optional[str] = None,
        force_rename: bool = False,
        project_id: Optional[str] = None,
        skip_validation: bool = False,
    ) -> str:
        ...

    def get_graph_if_exists(
        self,
        graph_id_or_name: str,
        project_id: Optional[str] = None,
    ) -> Optional[graph_api.GraphResource]:
        ...

    def create_snapshot(
        self,
        graph_id: str,
        *,
        refresh_source: bool = False,
        project_id: Optional[str] = None,
    ) -> snapshot_api.GraphSnapshotID:
        ...

    def get_snapshot(
        self,
        snapshot_id: snapshot_api.GraphSnapshotID,
        project_id: Optional[str] = None,
    ) -> snapshot_api.GraphSnapshotResource:
        ...

    def get_edge_stats(
        self,
        graph_snapshot_id: snapshot_api.GraphSnapshotID,
        project_id: Optional[str] = None,
    ) -> graph_api.EdgeHealthResponse:
        ...

    def validate_graph(
        self,
        request: graph_api.GraphValidationRequest,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        ...

    def infer_links(
        self,
        graph: graph_api.GraphDefinition,
        project_id: Optional[str] = None,
    ) -> graph_api.GraphDefinition:
        ...


@runtime_checkable
class PQueryBackend(Protocol):
    r"""Predictive-query persistence, validation, and plan suggestion."""
    def create(
        self,
        pquery: PQueryResource,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get_if_exists(
        self,
        name_or_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[PQueryResource]:
        ...

    def validate(
        self,
        pquery: PQueryResource,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        ...

    def infer_task_type(
        self,
        pquery_string: str,
        graph_id: str,
        project_id: Optional[str] = None,
    ) -> Tuple[TaskType, str]:
        ...

    def suggest_model_plan(
        self,
        req: SuggestModelPlanRequest,
        project_id: Optional[str] = None,
    ) -> ModelPlan:
        ...

    def suggest_training_table_plan(
        self,
        req: SuggestModelPlanRequest,
        project_id: Optional[str] = None,
    ) -> TrainingTableGenerationPlan:
        ...


@runtime_checkable
class JobBackend(Protocol):
    r"""Operations shared by every job area.

    :class:`~kumoai_studio.jobs.JobInterface` drives job tagging and lookup through
    whichever area a job object belongs to, so the four job areas below must
    all provide these.
    """
    def get(self, id: str,
            project_id: Optional[str] = None) -> JobResourceBase:
        ...

    def list(
        self,
        *,
        pquery_name: Optional[str] = None,
        pquery_id: Optional[str] = None,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        additional_tags: Mapping[str, str] = {},
        project_id: Optional[str] = None,
    ) -> Sequence[JobResourceBase]:
        ...

    def delete_tags(self, job_id: str, tags: List[str],
                    project_id: Optional[str] = None) -> bool:
        ...

    def update_tags(self, job_id: str, custom_job_tags: Mapping[str,
                                                                Optional[str]],
                    project_id: Optional[str] = None) -> bool:
        ...


@runtime_checkable
class TrainTableBackend(JobBackend, Protocol):
    r"""Training-table generation jobs and their outputs."""
    def create(
        self,
        request: GenerateTrainTableRequest,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> GenerateTrainTableJobResource:
        ...

    def get_table_data(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        ...

    def get_split_masks(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> Dict[TrainingStage, str]:
        ...

    def validate_custom_train_table(
        self,
        id: str,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        extensive_validation: bool,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        ...

    def stage_custom_table(
        self,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        pquery_id: str,
        *,
        id: Optional[str] = None,
    ) -> str:
        ...

    def register_custom_table(
        self,
        request: GenerateTrainTableRequest,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        project_id: Optional[str] = None,
    ) -> str:
        r"""Records supplied rows as a generation job, and returns its ID.

        Nothing is generated: the rows are the caller's. The record is what
        carries the plan -- and so the split -- to the run that names it,
        which is why a supplied table has a job ID at all.

        The job is not necessarily finished when this returns. A backend
        that reads the rows in the calling process has nothing left to do,
        while one that reads them server-side answers with a job that is
        still running; the caller waits on it either way.
        """
        ...

    def get_progress(self, id: str,
                     project_id: Optional[str] = None) -> Dict[str, int]:
        ...

    def get_job_error(self, id: str,
                      project_id: Optional[str] = None) -> ErrorDetails:
        ...

    def get_config(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> GenerateTrainTableRequest:
        ...

    def cancel(self, id: str, project_id: Optional[str] = None) -> None:
        ...


@runtime_checkable
class TrainingBackend(JobBackend, Protocol):
    r"""GNN training jobs."""
    def create(
        self,
        request: TrainingJobRequest,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> TrainingJobResource:
        ...

    def get_progress(self, id: str, project_id: Optional[str] = None) -> Any:
        ...

    def get_encoders(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        ...

    def holdout_data_url(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get_config(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> TrainingJobRequest:
        ...

    def cancel(self, id: str, project_id: Optional[str] = None) -> Any:
        ...

    def graph_definition(self, job_id: str) -> graph_api.GraphDefinition:
        r"""Returns the graph one job's model was trained over.

        Data, not a live :class:`~kumoai_studio.graph.Graph`: it names the tables,
        columns, keys and links, and says nothing about where that data is
        *now*. Read back by :func:`~kumoai_studio.import_model`, which rebinds it to
        connectors the caller names.

        Only a backend that can import a model has anything to rebind, so a
        hosted backend raises rather than omitting the method, for the same
        reason as :meth:`ArtifactExportBackend.import_model`: a caller gets
        that reason instead of an attribute error.
        """
        ...

    def query_for_job(self, job_id: str) -> str:
        r"""Returns the predictive query one job's model answers.

        The companion of :meth:`graph_definition` -- a rebuilt model needs
        both -- and local-only for the same reason.
        """
        ...

    def get_artifacts(self, job_id: str) -> Dict[str, str]:
        r"""Local-only for the same reason as :meth:`graph_definition`: the
        roles are paths on the machine that trained, and a hosted job's are
        inside the service.
        """
        ...


@runtime_checkable
class PredictionTableBackend(JobBackend, Protocol):
    r"""Prediction-table generation jobs and their outputs."""
    def create(
        self,
        request: GeneratePredictionTableRequest,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> GeneratePredictionTableJobResource:
        ...

    def get_table_data(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        ...

    def get_anchor_time(
            self, id: str,
            project_id: Optional[str] = None) -> Optional[datetime]:
        ...

    def get_job_error(self, id: str,
                      project_id: Optional[str] = None) -> ErrorDetails:
        ...

    def get_config(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> GeneratePredictionTableRequest:
        ...

    def cancel(self, id: str, project_id: Optional[str] = None) -> None:
        ...


@runtime_checkable
class PredictionBackend(JobBackend, Protocol):
    r"""Batch prediction jobs and their outputs."""
    def maybe_create(
        self,
        request: BatchPredictionRequest,
        project_id: Optional[str] = None,
    ) -> Tuple[Optional[str], ValidationResponse]:
        ...

    def get(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> BatchPredictionJobResource:
        ...

    def get_progress(self, id: str, project_id: Optional[str] = None) -> Any:
        ...

    def get_batch_predictions_url(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        ...

    def get_batch_embeddings_url(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        ...

    def get_config(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> BatchPredictionRequest:
        ...

    def cancel(
            self, id: str, project_id: Optional[str] = None
    ) -> CancelBatchPredictionJobResponse:
        ...


@runtime_checkable
class ArtifactExportBackend(Protocol):
    r"""Export of a produced artifact to a caller-owned destination.

    One area serves every artifact kind -- training tables, batch-prediction
    outputs, and offline model bundles -- because they differ only in the
    request payload, not in how the export is launched or awaited.
    """

    #: Whether this backend can publish a model to a *registry* -- a
    #: destination named by model and version rather than by path. Declared
    #: rather than inferred from the backend's name, and false by default, so
    #: that a backend which cannot serve one refuses by simply not claiming it.
    #: Path-addressed model exports are served by every backend and are not
    #: gated on this.
    supports_model_registry: bool = False

    def create(
        self,
        request: ArtifactExportRequest,
        project_id: Optional[str] = None,
    ) -> str:
        ...

    def get(self, id: str, project_id: Optional[str] = None) -> JobStatus:
        ...

    def cancel(self, id: str, project_id: Optional[str] = None) -> JobStatus:
        ...

    def import_model(
        self,
        connector_id: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id: Optional[str] = None,
    ) -> str:
        r"""Fetches a published model bundle and adopts it as a training job.

        The mirror of exporting one, and the only operation on this protocol
        that a hosted backend cannot serve: the Kumo service holds every
        training job itself and addresses one by ID, so there is nothing for it
        to import *into*. It raises rather than omitting the method, so a
        caller gets that reason instead of an attribute error.

        Args:
            connector_id: The connector to read the bundle through.
            path: Source for a path-addressed target.
            location: A
                :class:`~kumoapi.plugin.ModelRegistryLocation` for a
                registry-addressed one.
            project_id: Authorizing project, where one applies.

        Returns:
            The ID of the training job the bundle was adopted as, which is what
            the rest of the SDK addresses a model by.
        """
        ...

    def list_models(
        self,
        connector_id: str,
        *,
        project_id: Optional[str] = None,
    ) -> List[PublishedModel]:
        r"""Lists durable model versions available for post-restart import."""
        ...

    def published_model(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> Optional[PublishedModel]:
        r"""Returns the model this export published, or ``None``.

        ``None`` is the ordinary answer, not a gap: a prediction or
        training-table export produces a table, and a model written to a path
        was addressed in full by the caller. Only a registry assigns something
        the caller could not already know.
        """
        ...


@runtime_checkable
class Backend(Protocol):
    r"""One execution target for the GNN training and prediction path.

    A backend groups the eleven areas that path depends on. An implementation
    that supports only part of the path should still expose every property and
    raise :class:`NotImplementedError` from the operations it cannot serve, so
    callers fail with a clear message rather than an attribute error.
    """
    @property
    def name(self) -> str:
        r"""Short, stable identifier used in diagnostics."""
        ...

    @property
    def identity(self) -> IdentityBackend:
        ...

    @property
    def connector(self) -> ConnectorBackend:
        ...

    @property
    def source(self) -> SourceBackend:
        ...

    @property
    def table(self) -> TableBackend:
        ...

    @property
    def graph(self) -> GraphBackend:
        ...

    @property
    def pquery(self) -> PQueryBackend:
        ...

    @property
    def train_table(self) -> TrainTableBackend:
        ...

    @property
    def training(self) -> TrainingBackend:
        ...

    @property
    def prediction_table(self) -> PredictionTableBackend:
        ...

    @property
    def prediction(self) -> PredictionBackend:
        ...

    @property
    def artifact_export(self) -> ArtifactExportBackend:
        ...
