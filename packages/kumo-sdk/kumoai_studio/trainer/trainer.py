# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime
from typing import (
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)

from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.jobs import (
    BatchPredictionOptions,
    BatchPredictionRequest,
    JobStatus,
    MetadataField,
    PredictionOutputConfig,
    TrainingJobRequest,
    TrainingJobResource,
)
from kumoapi.model_plan import ModelPlan

from kumoai_studio.artifact_export.config import OutputConfig
from kumoai_studio.backend import TrainingBackend, cloud_backend
from kumoai_studio.client.jobs import (
    GeneratePredictionTableJobID,
    TrainingJobID,
)
from kumoai_studio.connector.base import Connector
from kumoai_studio.databricks import to_db_table_name
from kumoai_studio.graph import Graph
from kumoai_studio.pquery.prediction_table import (
    PredictionTable,
    PredictionTableJob,
)
from kumoai_studio.pquery.predictive_query import PredictiveQuery
from kumoai_studio.pquery.training_table import (
    TrainingTable,
    TrainingTableJob,
    _recovered_custom_train_table,
)
from kumoai_studio.trainer.job import (
    BatchPredictionJob,
    BatchPredictionJobResult,
    TrainingJob,
    TrainingJobResult,
)
from kumoai_studio.trainer.util import (
    build_prediction_output_config,
    validate_output_arguments,
)

logger = logging.getLogger(__name__)


def _active_generated_table_snapshot(
    graph: Graph,
    table_area: Optional[object],
    job_id: Optional[str],
) -> Optional[GraphSnapshotID]:
    r"""Return a local table job's graph snapshot when it is still registered.

    Local table generation is synchronous but does not update the caller's
    ``Graph`` object. Asking that object to snapshot again can therefore read
    a newer source version between table generation and training or scoring.
    A registered table snapshot is both cheaper and the only graph whose rows
    describe the generated table. A later process has no registered snapshot,
    so it follows the existing restore path instead.
    """
    if job_id is None:
        return None
    backend = cloud_backend()
    if getattr(backend, 'name', None) != 'local':
        return None
    get_snapshot_id = getattr(table_area, 'snapshot_id', None)
    if not callable(get_snapshot_id):
        return None
    snapshot_id = get_snapshot_id(job_id)
    if snapshot_id is None:
        return None
    snapshot_id = GraphSnapshotID(str(snapshot_id))
    try:
        resource = backend.graph.get_snapshot(snapshot_id,
                                              project_id=graph.kumo_project_id)
    except ValueError:
        return None
    expected_graph_id = str(graph.id)
    observed_graph_id = str(resource.graph_name)
    if observed_graph_id != expected_graph_id:
        raise ValueError(f"Generated table job {job_id!s} was built for graph "
                         f"{observed_graph_id}, but this call received graph "
                         f"{expected_graph_id}.")
    return snapshot_id


class Trainer:
    r"""A trainer supports creating a Kumo machine learning model on a
    :class:`~kumoai_studio.pquery.PredictiveQuery`. It is primarily oriented around
    two methods: :meth:`~kumoai_studio.trainer.Trainer.fit`, which accepts a
    :class:`~kumoai_studio.graph.Graph` and :class:`~kumoai_studio.pquery.TrainingTable` and
    produces a :class:`~kumoai_studio.trainer.TrainingJobResult`, and
    :meth:`~kumoai_studio.trainer.Trainer.predict`, which accepts a
    :class:`~kumoai_studio.graph.Graph` and :class:`~kumoai_studio.pquery.PredictionTable`
    and produces a :class:`~kumoai_studio.trainer.BatchPredictionJobResult`.

    A :class:`~kumoai_studio.trainer.Trainer` can also be loaded from a training job,
    with :meth:`~kumoai_studio.trainer.Trainer.load`.

    .. code-block:: python

        import kumoai_studio

        # See `Graph` and `PredictiveQuery` documentation:
        graph = kumoai_studio.Graph(...)
        pquery = kumoai_studio.PredictiveQuery(graph=graph, query=...)

        # Create a `Trainer` object, using a suggested model plan given the
        # predictive query:
        model_plan = pquery.suggest_model_plan()
        trainer = kumoai_studio.Trainer(model_plan)

        # Create a training table from the predictive query:
        training_table = pquery.generate_training_table()

        # Fit a model:
        training_job = trainer.fit(
            graph = graph,
            training_table = training_table,
        )

        # Create a prediction table from the predictive query:
        prediction_table = pquery.generate_prediction_table()

        # Generate predictions:
        prediction_job = trainer.predict(
            graph = graph,
            prediction_table = prediction_table,
            # other arguments here...
        )

        # Load a trained query to generate predictions:
        pquery = kumoai_studio.PredictiveQuery.load_from_training_job("trainingjob-...")
        trainer = kumoai_studio.Trainer.load("trainingjob-...")
        prediction_job = trainer.predict(
            pquery.graph,
            pquery.generate_prediction_table(),
        )

    Args:
        model_plan: A model plan that the trainer should follow when fitting a
            Kumo model to a predictive query. This model plan can either be
            generated from a predictive query, with
            :meth:`~kumoai_studio.pquery.PredictiveQuery.suggest_model_plan`, or can
            be manually specified.
    """  # noqa: E501

    def __init__(self, model_plan: ModelPlan) -> None:
        self._model_plan: Optional[ModelPlan] = model_plan
        self._encoders: Optional[Dict[str, str]] = None

        # Cached from backend:
        self._training_job_id: Optional[TrainingJobID] = None
        self._kumo_project_id: Optional[str] = None

    @property
    def model_plan(self) -> Optional[ModelPlan]:
        return self._model_plan

    @model_plan.setter
    def model_plan(self, model_plan: ModelPlan) -> None:
        self._model_plan = model_plan

    @property
    def encoders(self) -> Optional[Dict[str, str]]:
        return self._encoders

    @encoders.setter
    def encoders(self, encoders: Optional[Dict[str, str]]) -> None:
        self._encoders = encoders

    # Metadata ################################################################

    @property
    def is_trained(self) -> bool:
        r"""Returns ``True`` if this trainer instance has successfully been
        trained (and is therefore ready for prediction); ``False`` otherwise.
        """
        if not self._training_job_id:
            return False
        try:
            api = cloud_backend().training
            res: TrainingJobResource = api.get(
                self._training_job_id, project_id=self._kumo_project_id)
        except Exception as e:  # noqa
            logger.exception(
                "Failed to fetch training status for training job with ID %s",
                self._training_job_id, exc_info=e)
            return False
        return res.job_status_report.status == JobStatus.DONE

    # Fit / predict ###########################################################

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob, None] = None,
        *,
        pquery: Optional[PredictiveQuery] = None,
    ) -> TrainingJobResult:
        pass

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob, None] = None,
        *,
        pquery: Optional[PredictiveQuery] = None,
        non_blocking: Literal[False],
    ) -> TrainingJobResult:
        pass

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob, None] = None,
        *,
        pquery: Optional[PredictiveQuery] = None,
        non_blocking: Literal[True],
    ) -> TrainingJob:
        pass

    @overload
    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob, None] = None,
        *,
        pquery: Optional[PredictiveQuery] = None,
        non_blocking: bool,
    ) -> Union[TrainingJob, TrainingJobResult]:
        pass

    def fit(
        self,
        graph: Graph,
        train_table: Union[TrainingTable, TrainingTableJob, None] = None,
        *,
        pquery: Optional[PredictiveQuery] = None,
        non_blocking: bool = False,
        custom_tags: Mapping[str, str] = {},
        warm_start_job_id: Optional[TrainingJobID] = None,
    ) -> Union[TrainingJob, TrainingJobResult]:
        r"""Fits a model to the specified graph and training table, with the
        strategy defined by this :class:`Trainer`'s :obj:`model_plan`.

        Either :obj:`train_table` or :obj:`pquery` must be provided. If
        :obj:`pquery` is provided and :obj:`train_table` is omitted, Kumo
        generates the training table as an inline child job of the training
        job, sharing the same graph snapshot. This is the recommended path
        when working with live-streaming data to ensure consistency.

        Args:
            graph: The :class:`~kumoai_studio.graph.Graph` object that represents the
                tables and relationships that Kumo will learn from.
            train_table: An optional pre-generated
                :class:`~kumoai_studio.pquery.TrainingTable` or in-progress
                :class:`~kumoai_studio.pquery.TrainingTableJob`. If ``None``, Kumo
                generates the training table as an inline child job of the
                training job, sharing the same graph snapshot. Cannot be
                specified together with :obj:`pquery`.
            pquery: The :class:`~kumoai_studio.pquery.PredictiveQuery` to train on.
                Required when :obj:`train_table` is ``None``. Cannot be
                specified together with :obj:`train_table`.
            non_blocking: Whether this operation should return immediately
                after launching the training job, or await completion of the
                training job.
            custom_tags: Additional, customer defined k-v tags to be associated
                with the job to be launched. Job tags are useful for grouping
                and searching jobs.
            warm_start_job_id: Optional job ID of a completed training job to
                warm start from. Initializes the new model with the best
                weights from the specified job, using its model
                architecture, column processing, and neighbor sampling
                configurations.

        Returns:
            Union[TrainingJobResult, TrainingJob]:
                If ``non_blocking=False``, returns a training job object. If
                ``non_blocking=True``, returns a training job future object.
        """
        if train_table is None and pquery is None:
            raise ValueError(
                "When training a new model, either 'train_table' or "
                "'pquery' must be provided.")
        if train_table is not None and pquery is not None:
            raise ValueError("Cannot specify both 'train_table' and 'pquery'.")

        project_id = graph.kumo_project_id
        if pquery is not None and pquery.kumo_project_id != project_id:
            raise ValueError('Predictive query and graph belong to different '
                             'projects.')
        if (train_table is not None
                and train_table.kumo_project_id != project_id):
            raise ValueError('Training table and graph belong to different '
                             'projects.')
        self._kumo_project_id = project_id

        job_id = None
        custom_table = None
        train_table_job_api = None
        if train_table is None:
            assert pquery is not None
            logger.warning(
                "No training table provided. The training table generation "
                "will be coupled to the training job and share the same graph "
                "snapshot. Pass a pre-generated 'train_table' to decouple "
                "these jobs.")
            pq_id = pquery.save()
        else:
            # TODO(manan, siyang): remove soon:
            job_id = train_table.job_id
            assert job_id is not None
            train_table_job_api = (cloud_backend().train_table)
            resource = train_table_job_api.get(job_id, project_id=project_id)
            pq_id = resource.config.pquery_id
            assert pq_id is not None
            # Off the job record, for both accepted types. `train_table` is
            # typed `Union[TrainingTable, TrainingTableJob]`, and only the
            # former carries `_custom_train_table`; reading it under an
            # `isinstance` check meant a `TrainingTableJob` -- which the
            # signature accepts, and whose docstring invites building one
            # from a job ID off the Jobs page -- fell through to `None` and
            # the run silently generated its own table instead of training on
            # the registered rows. The record is also the authority for a
            # reconstructed `TrainingTable`, so reading it here rather than
            # the in-memory attribute makes both paths agree.
            custom_table = _recovered_custom_train_table(job_id, resource)

        graph_snapshot_id = _active_generated_table_snapshot(
            graph, train_table_job_api, job_id)
        if graph_snapshot_id is None:
            graph_snapshot_id = graph.snapshot(non_blocking=non_blocking)

        # NOTE the backend implementation currently handles sequentialization
        # between a training table future and a training job; that is, if the
        # training table future is still executing, the backend will wait on
        # the job ID completion before executing a training job. This preserves
        # semantics for both futures, ensures that Kumo works as expected if
        # used only via REST API, and allows us to avoid chaining calllbacks
        # in an ugly way here:
        api = cloud_backend().training
        assert self._model_plan is not None
        self._training_job_id = api.create(
            TrainingJobRequest(
                dict(custom_tags),
                pquery_id=pq_id,
                model_plan=self._model_plan,
                graph_snapshot_id=graph_snapshot_id,
                train_table_job_id=job_id,
                custom_train_table=custom_table,
                warm_start_job_id=warm_start_job_id,
            ), project_id=project_id)

        out = TrainingJob(job_id=self._training_job_id, project_id=project_id)
        if non_blocking:
            return out
        return out.attach()

    def predict(
        self,
        graph: Graph,
        prediction_table: Union[PredictionTable, PredictionTableJob,
                                None] = None,
        output_types: Optional[Set[str]] = None,
        output_connector: Optional[Connector] = None,
        output_table_name: Optional[Union[str, Tuple]] = None,
        output_metadata_fields: Optional[List[MetadataField]] = None,
        output_config: Optional[OutputConfig] = None,
        training_job_id: Optional[TrainingJobID] = None,
        binary_classification_threshold: Optional[float] = None,
        num_classes_to_return: Optional[int] = None,
        num_workers: int = 1,
        non_blocking: bool = False,
        custom_tags: Mapping[str, str] = {},
        prediction_time: Optional[datetime] = None,
        explanations: bool = False,
    ) -> Union[BatchPredictionJob, BatchPredictionJobResult]:
        """Using the trained model specified by :obj:`training_job_id` (or
        the model corresponding to the last invocation of
        :meth:`~kumoai_studio.trainer.Trainer.fit`, if not present), predicts the
        future values of the entities in :obj:`graph`.

        If :obj:`prediction_table` is provided, it is used directly as the
        entity set for prediction. If omitted, Kumo generates the prediction
        table automatically as part of this job, using the same graph snapshot
        as the batch prediction job — this is the recommended path when
        working with live-streaming data to ensure consistency.

        Args:
            graph: The :class:`~kumoai_studio.graph.Graph` object that represents the
                tables and relationships that Kumo will use to make
                predictions.
            prediction_table: An optional pre-generated
                :class:`~kumoai_studio.pquery.PredictionTable` or in-progress
                :class:`~kumoai_studio.pquery.PredictionTableJob`. If ``None`` (the
                default), Kumo generates the prediction table as an inline
                child job, sharing the same graph snapshot as this batch
                prediction job. For a custom table, construct a
                :class:`~kumoai_studio.pquery.PredictionTable` from an exact source
                returned by a project-owned S3, GCS, ADLS, or Databricks
                connector. Databricks sources may be physical tables or
                read-only SQL queries. The source connector must belong to the
                graph's project. Cannot be specified together with
                :obj:`prediction_time`.
            output_config: Output configuration defining properties of the
                generated batch prediction outputs. This includes:

                - output_types: The types of outputs that should be produced by
                  the prediction job. Can include either ``'predictions'``,
                  ``'embeddings'``, or both.
                - output_connector: The output data source that Kumo should
                  write batch predictions to, if it is None, produce
                  local download output only.
                - output_table_name: The name of the table in the output data
                  source that Kumo should write batch predictions to. In the
                  case of a Databricks connector, this should be a tuple of
                  two strings, the schema name and the output prediction
                  table name.
                - output_metadata_fields: Any additional metadata fields to
                  include as new columns in the produced ``'predictions'``
                  output. Currently, allowed options are ``JOB_TIMESTAMP``
                  and ``ANCHOR_TIMESTAMP``.
                - connector_specific_config: Custom connector specific output
                  configuration, such as whether to append or overwrite
                  existing tables.

            output_types: *(Deprecated)* The types of outputs that should be
                produced by the prediction job. Can include either
                ``'predictions'``, ``'embeddings'``, or both. Use
                :obj:`output_config` instead.
            output_connector: *(Deprecated)* The output data source that Kumo
                should write batch predictions to, if it is None, produce local
                download output only. Use :obj:`output_config` instead.
            output_table_name: *(Deprecated)* The name of the table in the
                output data source that Kumo should write batch predictions to.
                In the case of a Databricks connector, this should be a tuple
                of two strings: the schema name and the output prediction
                table name. Use :obj:`output_config` instead.
            output_metadata_fields: *(Deprecated)* Any additional metadata
                fields to include as new columns in the produced
                ``'predictions'`` output. Currently, allowed options are
                ``JOB_TIMESTAMP`` and ``ANCHOR_TIMESTAMP``. Use
                :obj:`output_config` instead.
            training_job_id: The job ID of the training job whose model will be
                used for making predictions.
            binary_classification_threshold: If this model corresponds to
                a binary classification task, the threshold for which higher
                values correspond to ``1``, and lower values correspond to
                ``0``.
            num_classes_to_return: If this model corresponds to a ranking task,
                the number of classes to return in the prediction output.
            num_workers: Number of workers to use when generating batch
                predictions. When set to a value greater than 1, the prediction
                table is partitioned into smaller parts and processed in
                parallel. The default is 1, which implies sequential inference
                over the prediction table.
            non_blocking: Whether this operation should return immediately
                after launching the batch prediction job, or await
                completion of the batch prediction job.
            custom_tags: Additional, customer defined k-v tags to be associated
                with the job to be launched. Job tags are useful for grouping
                and searching jobs.
            prediction_time: The point in time at which to generate
                predictions. Only valid when :obj:`prediction_table` is
                ``None``, i.e., when Kumo is generating the prediction table
                inline. If ``None``, the anchor time is inferred from the
                latest available data. Cannot be specified together with
                :obj:`prediction_table`.
            explanations: Whether to enable explanations for the batch
                prediction job. When ``True``, the job will generate
                per-prediction feature attributions in addition to the
                prediction outputs. Defaults to ``False``.

        Returns:
            Union[BatchPredictionJob, BatchPredictionJobResult]:
                If ``non_blocking=False``, returns a training job object. If
                ``non_blocking=True``, returns a training job future object.
        """
        if (output_types is not None or output_connector is not None
                or output_table_name is not None
                or output_metadata_fields is not None):
            raise ValueError(
                'Specifying output_types, output_connector, '
                'output_metadata_fields '
                'and output_table_name as direct inputs to predict() is '
                'deprecated. Please use output_config to specify these '
                'parameters.')
        if output_config is None:
            output_config = OutputConfig(output_types={'predictions'})
        output_table_name = to_db_table_name(output_config.output_table_name)
        validate_output_arguments(
            output_config.output_types,
            output_config.output_connector,
            output_table_name,
            output_paths=output_config.output_paths,
        )

        assert output_config.output_types is not None
        # Create outputs:
        outputs: List[PredictionOutputConfig] = []
        for output_type in output_config.output_types:
            outputs.append(
                build_prediction_output_config(
                    output_type,
                    output_config.output_connector,
                    output_table_name,
                    output_config.output_metadata_fields,
                    output_config,
                ))

        training_job_id = training_job_id or self._training_job_id
        if training_job_id is None:
            raise ValueError(
                "Cannot run batch prediction without a specified or saved "
                "training job ID. Please either call `fit(...)` or pass a "
                "job ID of a completed training job to proceed.")

        if prediction_table is None:
            logger.warning(
                "No prediction table provided. The prediction table "
                "generation will be coupled to the batch prediction job and "
                "share the same graph snapshot. Pass a pre-generated "
                "'prediction_table' to decouple these jobs.")

        project_id = graph.kumo_project_id
        if (self._kumo_project_id is not None
                and self._kumo_project_id != project_id):
            raise ValueError('Trainer and graph belong to different projects.')
        if (prediction_table is not None
                and prediction_table.kumo_project_id != project_id):
            raise ValueError('Prediction table and graph belong to different '
                             'projects.')
        if (output_config.output_connector is not None and
                output_config.output_connector.kumo_project_id != project_id):
            raise ValueError('Output connector and graph belong to different '
                             'projects.')

        pred_table_job_id: Optional[GeneratePredictionTableJobID] = None
        pred_table_source = None
        prediction_table_api = None
        if prediction_table is not None:
            pred_table_job_id = prediction_table.job_id
            if pred_table_job_id is None:
                assert isinstance(prediction_table, PredictionTable)
                pred_table_source = prediction_table._to_api_custom_source()
            else:
                prediction_table_api = cloud_backend().prediction_table

        if prediction_table is not None and prediction_time is not None:
            raise ValueError(
                "Cannot specify both 'prediction_table' and 'prediction_time'."
                " Pass 'prediction_time' only when letting Kumo generate the"
                " prediction table as part of this job.")

        api = cloud_backend().prediction
        graph_snapshot_id = _active_generated_table_snapshot(
            graph, prediction_table_api, pred_table_job_id)
        if graph_snapshot_id is None:
            graph_snapshot_id = graph.snapshot(non_blocking=non_blocking)
        # Remove to resolve https://github.com/kumo-ai/kumo/issues/24250
        # from kumoai_studio.pquery.predictive_query import PredictiveQuery
        # pquery = PredictiveQuery.load_from_training_job(training_job_id)
        # if pquery.get_task_type() == TaskType.BINARY_CLASSIFICATION:
        #     if binary_classification_threshold is None:
        #         logger.warning(
        # "No binary classification threshold provided. "
        # "Using default threshold of 0.5.")
        #         binary_classification_threshold = 0.5
        self._kumo_project_id = project_id
        job_id, response = api.maybe_create(
            BatchPredictionRequest(
                dict(custom_tags),
                model_training_job_id=training_job_id,
                predict_options=BatchPredictionOptions(
                    binary_classification_threshold=(
                        binary_classification_threshold),
                    num_classes_to_return=num_classes_to_return,
                    num_workers=num_workers,
                ),
                outputs=outputs,
                graph_snapshot_id=graph_snapshot_id,
                pred_table_job_id=pred_table_job_id,
                pred_table_source=pred_table_source,
                prediction_time=prediction_time,
                explanations=explanations,
            ), project_id=project_id)

        message = response.message()
        if not response.ok:
            raise RuntimeError(f"Prediction failed. {message}")
        elif not response.empty():
            logger.warning("Prediction produced the following warnings: %s",
                           message)
        assert job_id is not None

        self._batch_prediction_job_id = job_id
        out = BatchPredictionJob(job_id=self._batch_prediction_job_id,
                                 project_id=project_id)
        if non_blocking:
            return out
        return out.attach()

    # Persistence #############################################################

    @classmethod
    def _load_from_job(cls, job: TrainingJobResource) -> 'Trainer':
        trainer = cls(job.config.model_plan)
        trainer._training_job_id = job.job_id
        trainer._kumo_project_id = job.project_id
        trainer.encoders = cls.get_encoders(job.job_id,
                                            project_id=job.project_id)
        return trainer

    @classmethod
    def load(cls, job_id: TrainingJobID,
             project_id: Optional[str] = None) -> 'Trainer':
        r"""Create a trainer from a project-owned training job.

        Args:
            job_id: ID of the training job.
            project_id: Exact project owning the job. ``None`` uses the active
                project. The returned trainer snapshots the job's project.
        """
        api: TrainingBackend = cloud_backend().training
        job = api.get(job_id, project_id=project_id)
        return cls._load_from_job(job)

    # TODO(siyang): load trainer by searching training job via tags.
    @classmethod
    def load_from_tags(
        cls,
        tags: Mapping[str, str],
        project_id: Optional[str] = None,
    ) -> 'Trainer':
        r"""Creates a :class:`~kumoai_studio.trainer.Trainer` instance from a set of
        job tags. If multiple jobs match the list of tags, only one will be
        selected.

        Args:
            tags: Job tags used to select a training job.
            project_id: Exact project to search. ``None`` uses the active
                project. The returned trainer snapshots the selected job's
                project.
        """
        api = cloud_backend().training
        jobs = api.list(limit=1, additional_tags=tags, project_id=project_id)
        if not jobs:
            raise RuntimeError(f'No successful training job found for {tags}')
        assert len(jobs) == 1
        # The area is declared as its protocol, whose `list` answers the base
        # resource; a training area returns training jobs, which is what the
        # tag filter above selected.
        return cls._load_from_job(cast(TrainingJobResource, jobs[0]))

    @classmethod
    def get_encoders(
        cls,
        job_id: TrainingJobID,
        project_id: Optional[str] = None,
    ) -> Dict[str, str] | None:
        r"""Return a training job's encoders, if available.

        Args:
            job_id: ID of the training job.
            project_id: Exact project owning the job. ``None`` uses the active
                project.
        """
        api: TrainingBackend = cloud_backend().training
        encoders = api.get_encoders(job_id, project_id=project_id)
        return encoders
