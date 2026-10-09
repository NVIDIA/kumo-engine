# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from dataclasses import field
from typing import Any, Dict, List, Optional, Union

import kumoapi.jobs as jobs_api
import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
import numpy
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.model_plan import InferredType
from pydantic import Field

from io_utils import kumo_dataclass, logging
from rest.config.jobs import BatchPredictionOptions
from statedb.config.common import ProjectID

log = logging.getLogger(__name__)


@kumo_dataclass
class BatchPredictionFilterOverrides:
    # Entity filter overrides to use if any
    entity_filter: Optional[str] = None
    # Target filter overrides to use if any
    target_filter: Optional[str] = None


@kumo_dataclass
class BatchPredictionInternalRequest:
    # ID of a (successful) model training job.
    model_training_job_id: str

    # Options associated with this Batch Prediction Job.
    predict_options: BatchPredictionOptions

    project_id: ProjectID

    # Configuration for where the Batch Prediction outputs are expected to
    # be exported
    outputs: List[jobs_api.PredictionOutputConfig] = field(default_factory=list)

    # If filter overrides are supplied, this predictive_query should
    # contain the final predictive_query string after all overrides
    # have been applied to the query. This will ensure that the Batch
    # Prediction job regenrates the training table using this new query.
    predictive_query: Optional[str] = None

    # Whether to enable explanations for the Batch Prediction job
    explanations: bool = False

    # The time that a prediction horizon start time indicating the timestamp
    # at which predictions are made. If not set, will be inferred to be the
    # latest timestamp in the target fact table. Note that this value can
    # either be provided as an integer, representing the number of nanoseconds
    # from the Unix epoch, or as a ``datetime.datetime`` object
    # (*default:* ``"inferred"``). Only supported for Temporal queries.
    anchor_time: Union[int, InferredType, datetime.datetime] = Field(
        default=InferredType.VALUE,
        metadata=plan_api.Metadata(
            valid_query_types=[pquery_api.QueryType.TEMPORAL]),
    )

    # Filter overrides supplied from the UI. This interaction is different in
    # UI since users cannot modify the entire predictive query like in SDK
    # but only the specific filters.
    filter_overrides: Optional[BatchPredictionFilterOverrides] = None

    # Graph id to use for this Batch Prediction job.
    graph_id: Optional[FPID] = None

    # Custom tags associated with the UI job.
    tags: Optional[List[str]] = None

    # Custom key-value tags associated with the UI job.
    job_tags: Optional[Dict[str, str]] = None

    # Field to retain information about user that triggered the job.
    # Optional because old jobs don't have this field.
    user: Optional[str] = None


@kumo_dataclass
class PredictionTableSchemaRequest:
    task_type: TaskType
    query_type: pquery_api.QueryType


@kumo_dataclass
class PredictionTableSchemaSamplesRequest:
    task_type: TaskType
    train_table_job_id: str


@kumo_dataclass
class PredictionTableSchemaSampleColumn:
    name: str
    sample_rows: List[Union[float, int, str]]


class SuggestPredictionSamples:
    """
    A class of methods used to generate a preview of the
    final prediction schema for a prediction job.
    """
    def suggest_col_samples_by_task_type(
        self,
        task_type: TaskType,
        samples_rows_dict: Dict[Any, Any],
    ) -> List[PredictionTableSchemaSampleColumn]:
        """
        Depending on the type of task, generate the data for
        the preview of the prediction schema.

        Args:
            - task_type: the training task type (binary
            classification, regression, etc.).
            - samples_rows_dict: The sampled training table in
            a python dictionary format. The format is: {
                column_name: [data]
            }
        Response:
            - List[PredictionTableSchemaSampleColumn] - an array of
            `PredictionTableSchemaSampleColumn`. Each
            item corresponds to one row in the preview schema.
            Each row has a `name` and an array of 10 samples.
        """

        if self.verify_input_data(samples_rows_dict) is False:
            return []

        entity_column_summary = PredictionTableSchemaSampleColumn(
            name="ENTITY",
            sample_rows=samples_rows_dict["ENTITY"],
        )

        response: List[PredictionTableSchemaSampleColumn] = [
            entity_column_summary
        ]

        if task_type == TaskType.REGRESSION or \
            task_type == TaskType.FORECASTING:
            target_column_summary = self._regression_and_forcecast_col_summary(
                samples_rows_dict)
            response.extend(target_column_summary)

        elif task_type == TaskType.BINARY_CLASSIFICATION:
            bc_column_summary = self._binary_classification_col_summary(
                samples_rows_dict)
            response.extend(bc_column_summary)

        elif task_type == TaskType.MULTILABEL_CLASSIFICATION \
            or task_type == TaskType.MULTICLASS_CLASSIFICATION:
            mc_column_summary = self._multiclass_classification_col_summary(
                samples_rows_dict)
            if response[0].name == "ENTITY" and len(
                    response[0].sample_rows) > 10:
                samples_set = response[0].sample_rows
                response[0].sample_rows = list(samples_set)[:10]
            response.extend(mc_column_summary)

        elif task_type.is_ranking:
            rank_column_summary = self._ranking_col_summary(samples_rows_dict)
            response.extend(rank_column_summary)

        else:
            log.error('Invalid task type for Batch Prediction stats: %s',
                      task_type)
            return []

        return response

    def _regression_and_forcecast_col_summary(
        self,
        samples_rows_dict,
    ) -> List[PredictionTableSchemaSampleColumn]:
        min_val = min(samples_rows_dict["TARGET"])
        max_val = max(samples_rows_dict["TARGET"])
        vals = numpy.round(numpy.random.uniform(min_val, max_val, 10),
                           3).tolist()  # type: ignore
        column_summary = PredictionTableSchemaSampleColumn(
            name="TARGET_PRED",
            sample_rows=vals,
        )

        return [column_summary]

    def _binary_classification_col_summary(
        self,
        samples_rows_dict,
    ) -> List[PredictionTableSchemaSampleColumn]:
        target_vals = samples_rows_dict["TARGET"]

        target_column_summary = PredictionTableSchemaSampleColumn(
            name="TARGET_PRED",
            sample_rows=target_vals,
        )

        # Create a random distribution for negative samples
        false_sample_rows = numpy.round(
            numpy.linspace(0.5, 1, num=len(target_vals)), 3).tolist()
        false_column_summary = PredictionTableSchemaSampleColumn(
            name="False_PROB",
            sample_rows=false_sample_rows,
        )

        # values are oppostite of false_prob values
        true_prob_vals = numpy.round([1 - val for val in false_sample_rows],
                                     3).tolist()
        true_column_summary = PredictionTableSchemaSampleColumn(
            name="True_PROB",
            sample_rows=true_prob_vals,
        )

        return [
            target_column_summary, false_column_summary, true_column_summary
        ]

    def _multiclass_classification_col_summary(
        self,
        samples_rows_dict,
    ) -> List[PredictionTableSchemaSampleColumn]:
        target_samples = [
            # sample if sample != "" else "NA"
            sample for sample in samples_rows_dict["TARGET"] if sample != ""
        ]
        # keep only the first 10 valid samples
        num_samples = len(target_samples) if len(target_samples) <= 10 else 10
        target_samples = target_samples[:10] if len(
            target_samples) > 10 else target_samples

        pred_column_summary = PredictionTableSchemaSampleColumn(
            name="PREDICTED",
            sample_rows=numpy.round(
                numpy.random.choice([0, 1], size=num_samples, p=[.8, .2]),
                3).tolist(),
        )

        class_column_summary = PredictionTableSchemaSampleColumn(
            name="CLASS",
            sample_rows=target_samples,  # type: ignore
        )

        score_samples = numpy.round(numpy.random.uniform(0.2, 5.4, num_samples),
                                    3)
        score_column_summary = PredictionTableSchemaSampleColumn(
            name="SCORE",
            sample_rows=score_samples.tolist(),
        )

        return [pred_column_summary, class_column_summary, score_column_summary]

    def _ranking_col_summary(
        self,
        samples_rows_dict,
    ) -> List[PredictionTableSchemaSampleColumn]:

        class_column_summary = PredictionTableSchemaSampleColumn(
            name="CLASS",
            sample_rows=samples_rows_dict["TARGET"],
        )

        score_samples = numpy.round(numpy.random.uniform(0.2, 5.4, 10), 3)
        score_column_summary = PredictionTableSchemaSampleColumn(
            name="SCORE",
            sample_rows=score_samples.tolist(),
        )

        return [class_column_summary, score_column_summary]

    def verify_input_data(self, samples_rows_dict: Dict[Any, Any]) -> bool:
        keys = set(samples_rows_dict.keys())
        if "TARGET" not in keys or "ENTITY" not in keys:
            log.error(
                "Sampled data doesn't contain the necessary columns." +
                f"It should contain both TARGET and ENTITY columns (got {samples_rows_dict})"
            )
            return False
        else:
            return True
