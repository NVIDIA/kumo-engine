# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from kumoapi.common import ValidationResponse
from kumoapi.jobs import (
    ArtifactExportRequest,
    ArtifactExportResponse,
    AutoTrainerProgress,
    BatchPredictionJobResource,
    BatchPredictionRequest,
    CancelBatchPredictionJobResponse,
    CancelTrainingJobResponse,
    CustomTrainingTable,
    ErrorDetails,
    GeneratePredictionTableJobResource,
    GeneratePredictionTableRequest,
    GenerateTrainTableJobResource,
    GenerateTrainTableRequest,
    GetEmbeddingsDfUrlResponse,
    GetPredictionsDfUrlResponse,
    JobRequestBase,
    JobResourceBase,
    JobStatus,
    PredictionProgress,
    TrainingJobRequest,
    TrainingJobResource,
    TrainingTableSpec,
)
from kumoapi.json_serde import from_json, to_json_dict
from kumoapi.source_table import LocalFileSourceTable, SourceTableType
from kumoapi.train import TrainingStage
from typing_extensions import override

from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import (
    Returns,
    parse_patch_response,
    parse_response,
    raise_on_error,
    response_json,
)

TrainingJobID = str
BatchPredictionJobID = str
GenerateTrainTableJobID = str
GeneratePredictionTableJobID = str

JobRequestType = TypeVar('JobRequestType', bound=JobRequestBase)
JobResourceType = TypeVar('JobResourceType', bound=JobResourceBase)


class CommonJobAPI(Generic[JobRequestType, JobResourceType]):
    def __init__(self, client: Transport, base_endpoint: str,
                 res_type: Type[JobResourceType]) -> None:
        self._client = client
        self._base_endpoint = base_endpoint
        self._res_type = res_type

    @staticmethod
    def _params(project_id: Optional[str],
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = dict(params or {})
        if project_id is not None:
            out['project_id'] = project_id
        return out

    def create(
        self,
        request: JobRequestType,
        project_id: Optional[str] = None,
    ) -> str:
        body = to_json_dict(request)

        response = self._client._post(self._base_endpoint,
                                      params=self._params(project_id),
                                      json=body)
        raise_on_error(response)
        return parse_response(Dict[str, str], response)['id']

    def get(self, id: str,
            project_id: Optional[str] = None) -> JobResourceType:
        response = self._client._get(f'{self._base_endpoint}/{id}',
                                     params=self._params(project_id))
        raise_on_error(response)
        return parse_response(self._res_type, response)

    def list(
        self,
        *,
        pquery_name: Optional[str] = None,
        pquery_id: Optional[str] = None,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        additional_tags: Mapping[str, str] = {},
        project_id: Optional[str] = None,
    ) -> List[JobResourceType]:
        params: Dict[str, Any] = {
            'pquery_name': pquery_name,
            'pquery_id': pquery_id,
            'job_status': job_status,
            'limit': limit
        }
        params.update(additional_tags)
        params = self._params(project_id, params)
        response = self._client._get(self._base_endpoint, params=params)
        raise_on_error(response)
        resource_elements = response_json(response)
        assert isinstance(resource_elements, list)
        return [from_json(e, self._res_type) for e in resource_elements]

    def delete_tags(self, job_id: str, tags: List[str],
                    project_id: Optional[str] = None) -> bool:
        r"""Removes the tags from the job.

        Args:
            job_id (str): The ID of the job.
            tags (List[str]): The tags to remove.
            project_id (Optional[str]): Exact owning project for the job.
        """
        return self.update_tags(job_id, {t: 'none'
                                         for t in tags}, project_id=project_id)

    def update_tags(self, job_id: str, custom_job_tags: Mapping[str,
                                                                Optional[str]],
                    project_id: Optional[str] = None) -> bool:
        r"""Updates the tags of the job.

        Args:
            job_id (str): The ID of the job.
            custom_job_tags (Mapping[str, Optional[str]]): The tags to update.
                Note that the value 'none' will remove the tag. If the tag is
                not present, it will be added.
            project_id (Optional[str]): Exact owning project for the job.
        """
        response = self._client._patch(
            f'{self._base_endpoint}/{job_id}/tags',
            data=None,
            params=self._params(project_id, {
                k: str(v)
                for k, v in custom_job_tags.items()
            }),
        )
        raise_on_error(response)
        return parse_patch_response(response)

    def get_encoders(
            self, job_id: str,
            project_id: Optional[str] = None) -> Dict[str, str] | None:
        response = self._client._get(
            f'{self._base_endpoint}/{job_id}/encoders',
            params=self._params(project_id))
        if (response is None or response.text is None
                or response.text == "null"):
            return None
        if response.status_code == 404:
            return None
        raise_on_error(response)
        return parse_response(Dict[str, str], response)


class TrainingJobAPI(CommonJobAPI[TrainingJobRequest, TrainingJobResource]):
    r"""Typed API definition for the training job resource."""
    def __init__(self, client: Transport) -> None:
        super().__init__(client, '/training_jobs', TrainingJobResource)

    def get_progress(self, id: TrainingJobID,
                     project_id: Optional[str] = None) -> AutoTrainerProgress:
        response = self._client._get(f'{self._base_endpoint}/{id}/progress',
                                     params=self._params(project_id))
        raise_on_error(response)
        return parse_response(AutoTrainerProgress, response)

    def holdout_data_url(
        self,
        id: TrainingJobID,
        project_id: Optional[str] = None,
    ) -> str:
        """Return a safe reference to a completed model's holdout data.

        The server chooses a reference suitable for the active deployment,
        such as a short-lived signed download URL or a customer-facing
        warehouse path. Internal data-plane paths are never requested.

        Args:
            id: ID of the completed training job.
            project_id: Exact project owning the job. ``None`` uses the active
                project.

        Returns:
            A server-authorized reference to the holdout parquet data.
        """
        response = self._client._get(f'{self._base_endpoint}/{id}/holdout',
                                     params=self._params(project_id))
        raise_on_error(response)
        return response.text

    def cancel(self, id: str,
               project_id: Optional[str] = None) -> CancelTrainingJobResponse:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel',
                                      params=self._params(project_id))
        raise_on_error(response)
        return parse_response(CancelTrainingJobResponse, response)

    def get_config(self, job_id: str,
                   project_id: Optional[str] = None) -> TrainingJobRequest:
        """Load the configuration for a training job by ID."""
        resource = self.get(job_id, project_id=project_id)
        return resource.config

    def graph_definition(self, job_id: str) -> Any:
        r"""Refuses: a hosted job is loaded by ID, not rebuilt.

        Reading a job's graph back exists so an *imported* model can be rebound
        to data in the process that imported it, and the service has nothing to
        import into.
        """
        del job_id
        raise NotImplementedError(_HOSTED_REBUILD_ERROR)

    def query_for_job(self, job_id: str) -> str:
        r"""Refuses for the same reason as :meth:`graph_definition`."""
        del job_id
        raise NotImplementedError(_HOSTED_REBUILD_ERROR)

    def get_artifacts(self, job_id: str) -> Dict[str, str]:
        del job_id
        raise NotImplementedError(
            "The Kumo service holds a training job's files, so they are not "
            "addressable as paths. Export what you need -- "
            "kumoai_studio.export_model(...) or holdout_data_url() -- or use the "
            "local backend, which trains onto your own disk.")


#: Why the hosted backend serves neither half of a model rebuild. Shared by
#: both methods so the two cannot drift.
_HOSTED_REBUILD_ERROR = (
    "The Kumo service holds every training job, so a model trained there is "
    "loaded by its ID -- Trainer.load('trainingjob-...') -- with its graph "
    "and query already attached. Reading them back separately exists for the "
    "local backend, where import_model() adopts a published bundle.")


class BatchPredictionJobAPI(CommonJobAPI[BatchPredictionRequest,
                                         BatchPredictionJobResource]):
    r"""Typed API definition for the prediction job resource."""
    def __init__(self, client: Transport) -> None:
        super().__init__(client, '/prediction_jobs',
                         BatchPredictionJobResource)

    @override
    def create(
        self,
        request: BatchPredictionRequest,
        project_id: Optional[str] = None,
    ) -> str:
        # TODO(manan): eventually, all `create` methods should
        # return a validation response:
        raise NotImplementedError

    def maybe_create(
        self,
        request: BatchPredictionRequest,
        project_id: Optional[str] = None,
    ) -> Tuple[Optional[str], ValidationResponse]:
        body = to_json_dict(request)

        response = self._client._post(self._base_endpoint,
                                      params=self._params(project_id),
                                      json=body)
        raise_on_error(response)
        return parse_response(
            Returns[Tuple[Optional[str], ValidationResponse]], response)

    def list(
        self,
        *,
        model_id: Optional[TrainingJobID] = None,
        pquery_name: Optional[str] = None,
        pquery_id: Optional[str] = None,
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        additional_tags: Mapping[str, str] = {},
        project_id: Optional[str] = None,
    ) -> List[BatchPredictionJobResource]:
        if model_id:
            additional_tags = {**additional_tags, 'model_id': model_id}
        return super().list(pquery_name=pquery_name, pquery_id=pquery_id,
                            job_status=job_status, limit=limit,
                            additional_tags=additional_tags,
                            project_id=project_id)

    def get_progress(self, id: str,
                     project_id: Optional[str] = None) -> PredictionProgress:
        response = self._client._get(f'{self._base_endpoint}/{id}/progress',
                                     params=self._params(project_id))
        raise_on_error(response)
        return parse_response(PredictionProgress, response)

    def cancel(
        self,
        id: str,
        project_id: Optional[str] = None,
    ) -> CancelBatchPredictionJobResponse:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel',
                                      params=self._params(project_id))
        raise_on_error(response)
        return parse_response(CancelBatchPredictionJobResponse, response)

    def get_batch_predictions_url(
            self, id: str, project_id: Optional[str] = None) -> List[str]:
        """Return safe references to stored batch predictions.

        The server chooses references suitable for the active deployment,
        such as short-lived signed download URLs or customer-facing Snowflake
        stage and Databricks Unity Catalog volume paths. Internal data-plane
        paths are never returned.

        Args:
            id: ID of the batch prediction job.
            project_id: Exact owning project for the job. ``None`` uses the
                active project.

        Returns:
            Server-authorized references to every prediction partition.
        """
        response = self._client._get(
            f'{self._base_endpoint}/{id}/get_prediction_df_urls',
            params=self._params(project_id))
        raise_on_error(response)
        return parse_response(
            GetPredictionsDfUrlResponse,
            response,
        ).prediction_partitions

    def get_batch_embeddings_url(
            self, id: str, project_id: Optional[str] = None) -> List[str]:
        """Return safe references to stored batch-prediction embeddings.

        The server chooses references suitable for the active deployment,
        such as short-lived signed download URLs or customer-facing Snowflake
        stage and Databricks Unity Catalog volume paths. Internal data-plane
        paths are never returned.

        Args:
            id: ID of the batch prediction job.
            project_id: Exact owning project for the job. ``None`` uses the
                active project.

        Returns:
            Server-authorized references to every embedding partition.
        """
        response = self._client._get(
            f'{self._base_endpoint}/{id}/get_embedding_df_urls',
            params=self._params(project_id))
        raise_on_error(response)
        return parse_response(
            GetEmbeddingsDfUrlResponse,
            response,
        ).embedding_partitions

    def get_config(self, job_id: str,
                   project_id: Optional[str] = None) -> BatchPredictionRequest:
        """Load the configuration for a batch prediction job by ID."""
        resource = self.get(job_id, project_id=project_id)
        return resource.config


class GenerateTrainTableJobAPI(CommonJobAPI[GenerateTrainTableRequest,
                                            GenerateTrainTableJobResource]):
    r"""Typed API definition for training table generation job resource."""
    def __init__(self, client: Transport) -> None:
        super().__init__(client, '/gentraintable_jobs',
                         GenerateTrainTableJobResource)

    def get_table_data(
        self,
        id: GenerateTrainTableJobID,
        project_id: Optional[str] = None,
    ) -> List[str]:
        """Return safe references to generated training-table parquet data.

        The server returns one or more references appropriate for the active
        deployment. The result may be a customer-facing dataset directory or
        short-lived signed URLs for individual parquet partitions. Internal
        data-plane paths are never exposed.

        Args:
            id: ID of the training-table generation job.
            project_id: Exact project owning the job. ``None`` uses the active
                project.

        Returns:
            One or more server-authorized references to the generated data.
        """
        resp = self._client._get(f'{self._base_endpoint}/{id}/table_data',
                                 params=self._params(project_id))
        raise_on_error(resp)
        return parse_response(List[str], resp)

    def get_split_masks(
        self,
        id: GenerateTrainTableJobID,
        project_id: Optional[str] = None,
    ) -> Dict[TrainingStage, str]:
        """Return safe split-mask references keyed by training stage.

        Each reference identifies a torch-serialized (default pickle protocol)
        mask tensor. The server returns short-lived signed download URLs or
        customer-facing warehouse paths, never internal data-plane paths.

        Example:
            >>> # code to load a mask tensor:
            >>> import io
            >>> import torch
            >>> import requests
            >>> masks = get_split_masks('some-gen-traintable-job-id')
            >>> data_bytes = requests.get(masks[TrainingStage.TEST]).content
            >>> test_mask_tensor = torch.load(io.BytesIO(data))

        Args:
            id: ID of the training-table generation job.
            project_id: Exact project owning the job. ``None`` uses the active
                project.
        """
        resp = self._client._get(f'{self._base_endpoint}/{id}/split_masks',
                                 params=self._params(project_id))
        raise_on_error(resp)
        return parse_response(Dict[TrainingStage, str], resp)

    def get_progress(self, id: str,
                     project_id: Optional[str] = None) -> Dict[str, int]:
        response = self._client._get(f'{self._base_endpoint}/{id}/progress',
                                     params=self._params(project_id))
        raise_on_error(response)
        return parse_response(Dict[str, int], response)

    def cancel(self, id: str, project_id: Optional[str] = None) -> None:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel',
                                      params=self._params(project_id))
        raise_on_error(response)

    def validate_custom_train_table(
        self,
        id: str,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        extensive_validation: bool,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        response = self._client._post(
            f'{self._base_endpoint}/{id}/validate_custom_train_table',
            json=to_json_dict({
                'custom_table': source_table_type,
                'train_table_mod': train_table_mod,
                'extensive_validation': extensive_validation,
            }),
            params=self._params(project_id),
        )
        return parse_response(ValidationResponse, response)

    def stage_custom_table(
        self,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        pquery_id: str,
        *,
        id: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "Staging a custom training table server-side is not implemented. "
            "The platform reads the rows from the connector they were written "
            "to when the training job starts, rather than staging them here.")

    def register_custom_table(
        self,
        request: GenerateTrainTableRequest,
        source_table_type: Union[SourceTableType, LocalFileSourceTable],
        train_table_mod: TrainingTableSpec,
        project_id: Optional[str] = None,
    ) -> str:
        r"""Records supplied rows as a generation job, and returns its ID.

        The same endpoint a generating job is created through, carrying the
        rows instead of asking for them: the server resolves the query, the
        graph snapshot and the plan exactly as it does for a generated table,
        reads the rows into storage it owns, and completes the job. The ID is
        the server's, as it is for every other job.

        The job is running when this returns. A training table names a
        completed job, so the caller waits on it -- which is what
        :meth:`TrainingTable.from_source_table` does.
        """
        return self.create(
            dataclasses.replace(
                request,
                custom_train_table=CustomTrainingTable(
                    source_table=source_table_type,
                    table_mod_spec=train_table_mod,
                ),
            ),
            project_id=project_id,
        )

    def get_job_error(self, id: str,
                      project_id: Optional[str] = None) -> ErrorDetails:
        """Thin API wrapper for fetching errors from the jobs.

        Args:
            id (str): ID of the job whose related errors are expected to be
                queried.
            project_id (Optional[str]): Exact owning project for the job.
        """
        response = self._client._get(f'{self._base_endpoint}/{id}/get_errors',
                                     params=self._params(project_id))
        raise_on_error(response)
        return parse_response(ErrorDetails, response)

    def get_config(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> GenerateTrainTableRequest:
        """Load the configuration for a training table generation job by ID."""
        resource = self.get(job_id, project_id=project_id)
        return resource.config


class GeneratePredictionTableJobAPI(
        CommonJobAPI[GeneratePredictionTableRequest,
                     GeneratePredictionTableJobResource]):
    r"""Typed API definition for prediction table generation job resource."""
    def __init__(self, client: Transport) -> None:
        super().__init__(client, '/genpredtable_jobs',
                         GeneratePredictionTableJobResource)

    def get_anchor_time(
            self, id: BatchPredictionJobID,
            project_id: Optional[str] = None) -> Optional[datetime]:
        response = self._client._get(
            f'{self._base_endpoint}/{id}/get_anchor_time',
            params=self._params(project_id))
        raise_on_error(response)
        return parse_response(Returns[Optional[datetime]], response)

    def get_table_data(
        self,
        id: GeneratePredictionTableJobID,
        project_id: Optional[str] = None,
    ) -> List[str]:
        """Return safe references to generated prediction-table data.

        The server returns one or more references appropriate for the active
        deployment. The result may be a customer-facing dataset directory or
        short-lived signed URLs for individual parquet partitions. Internal
        data-plane paths are never exposed.

        Args:
            id: ID of the prediction-table generation job.
            project_id: Exact project owning the job. ``None`` uses the active
                project.

        Returns:
            One or more server-authorized references to the generated data.
        """
        resp = self._client._get(f'{self._base_endpoint}/{id}/table_data',
                                 params=self._params(project_id))
        raise_on_error(resp)
        return parse_response(List[str], resp)

    def cancel(self, id: str, project_id: Optional[str] = None) -> None:
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel',
                                      params=self._params(project_id))
        raise_on_error(response)

    def get_job_error(self, id: str,
                      project_id: Optional[str] = None) -> ErrorDetails:
        """Thin API wrapper for fetching errors from the jobs.

        Args:
            id (str): ID of the job whose related errors are expected to be
                queried.
            project_id (Optional[str]): Exact owning project for the job.
        """
        response = self._client._get(f'{self._base_endpoint}/{id}/get_errors',
                                     params=self._params(project_id))
        raise_on_error(response)
        return parse_response(ErrorDetails, response)

    def get_config(
        self,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> GeneratePredictionTableRequest:
        """Load the configuration for a
        prediction table generation job by ID.
        """
        resource = self.get(job_id, project_id=project_id)
        return resource.config


class ArtifactExportJobAPI:
    r"""Typed API definition for artifact export job resource."""

    #: The Kumo service writes model artifacts to a caller's object store; it
    #: does not publish to a model registry, which is a local-backend
    #: capability. Stated rather than left absent so this class remains the
    #: reference implementation of every member the protocol names.
    supports_model_registry = False

    def import_model(self, connector_id: str, *, path: Optional[str] = None,
                     location: Optional[Any] = None,
                     project_id: Optional[str] = None) -> str:
        r"""Refuses: the service has nothing to import a model into.

        It holds every training job itself, so a model trained there is
        reached by its ID rather than fetched.
        """
        del connector_id, path, location, project_id
        raise NotImplementedError(
            "The Kumo service holds every training job, so a model trained "
            "there is loaded by its ID: Trainer.load('trainingjob-...'). "
            "import_model() is not supported.")

    def list_models(self, connector_id: str, *,
                    project_id: Optional[str] = None) -> List[Any]:
        r"""Refuses: the service does not manage a connector's registry."""
        del connector_id, project_id
        raise NotImplementedError(
            "The Kumo service does not manage a connector's model registry. "
            "Use the registry's catalog directly.")

    def published_model(self, id: str,
                        project_id: Optional[str] = None) -> None:
        r"""Always ``None``: the service writes to the path it was given.

        Not a gap. A version is something only a registry assigns, and this
        backend does not publish to one, so there is nothing here a caller did
        not already name.
        """
        del id, project_id
        return None

    def __init__(self, client: Transport) -> None:
        self._client = client
        self._base_endpoint = '/artifact'

    def create(
        self,
        request: ArtifactExportRequest,
        project_id: Optional[str] = None,
    ) -> str:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        response = self._client._post(
            self._base_endpoint,
            params=params,
            json=to_json_dict(request),
        )
        raise_on_error(response)
        return parse_response(ArtifactExportResponse, response).job_id

    # TODO Add an API in artifact export to get
    # JobStatusReport and not just JobStatus
    def get(self, id: str, project_id: Optional[str] = None) -> JobStatus:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        response = self._client._get(f'{self._base_endpoint}/{id}',
                                     params=params)
        raise_on_error(response)
        return parse_response(JobStatus, response)

    def cancel(self, id: str, project_id: Optional[str] = None) -> JobStatus:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        response = self._client._post(f'{self._base_endpoint}/{id}/cancel',
                                      params=params)
        raise_on_error(response)
        return parse_response(JobStatus, response)
