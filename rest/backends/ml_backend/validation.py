# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Optional

from kumo_core.diskgraph.types import InputDataStats
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from io_utils import logging
from io_utils.parquet_util import is_empty
from kumo.pquery import PredictiveQueryV2
from kumo.validation import ValidationError, ValidationResponse
from rest.settings import DataScaleLimits

logger = logging.getLogger(__name__)

# Workflow Validators #########################################################
# Note: the validations defined here should be validations that either cannot
# be performed in the core backend code or are better performed as part of
# workflow execution. A simple example is in validating the number of tables we
# have in an input graph before ingestion is performed.

# Training / Prediction Table Generation ######################################


class TableValidator:
    """
    Base class for Table Validators. This is an abstraction for classes
    implementing validations like training table validation and prediction
    table validation.
    """
    def __init__(self, query_type: QueryType, task_type: TaskType):
        self.query_type = query_type
        self.task_type = task_type

    def pre_generation(self) -> ValidationResponse:
        return ValidationResponse()

    def post_generation(
        self,
        stage: Literal['pred'] | Literal['training'],
        table_path: str,
        predictive_query: Optional[PredictiveQueryV2] = None,
    ) -> ValidationResponse:
        """
        Performs validation after the table is generated.

        Args:
            stage (str): Either 'pred' or 'training'
            table_path (str): Path where the prediction table is materialized.
            predictive_query (Optional[PredictiveQueryV2]): Predictive query
                used to generate the table. Defaults to None.
        """
        # TODO: Make this a more principled enum
        assert stage in ['pred', 'training']
        errors: List[ValidationError] = []

        # Validate that prediction table is not empty:
        logger.info(f"Validating ingested table at path {table_path}")
        if is_empty(table_path):
            # Handle non-intuitive non-empty column case (Batch Predictions
            # using a STATIC query predicting a single column)
            if (stage == 'pred' and self.query_type == QueryType.STATIC
                    and self.task_type.is_node_pred):
                prediction_column = "prediction column"
                if predictive_query is not None:
                    prediction_column = predictive_query.target.to_string()
                errors.append(
                    ValidationError(
                        "Cannot generate predictions - no rows with empty"
                        f" {prediction_column}",
                        f"Your {prediction_column} is completely filled. Our"
                        f" model needs rows with empty {prediction_column} to"
                        " predict values for."
                        f" To fix this: 1. Add {prediction_column}, include "
                        f"rows where your {prediction_column} is empty or "
                        "contains N/A in the graph. 2. Combine your datasets, "
                        "if you have training data and prediction data "
                        "separately, merge them into one dataset before "
                        "running the model."))
            else:
                errors.append(
                    ValidationError(
                        f"Empty {stage} table",
                        f'Generated {stage} table is empty. '
                        'Please modify your query to make filters less '
                        'restrictive or add more data that will fit '
                        'your desired filters.'))

        return ValidationResponse(errors=errors)


class GeneratePredictionTableValidator(TableValidator):
    """
    Contains validations specific to Prediction table.
    """


class GenerateTrainingTableValidator(TableValidator):
    """
    Contains validations specific to Training table.
    """


# Materialization #############################################################


class MaterializeTaskValidator:
    ONE_GB = 1000 * 1000 * 1000  # bytes.
    ONE_TB = 1000 * ONE_GB
    ONE_BILLION = 1 * (10**9)

    def __init__(
        self,
        max_single_table_size: Optional[int] = None,
        max_total_table_size: Optional[int] = None,
        max_total_graph_edges: Optional[int] = None,
    ) -> None:
        r"""Defines limits on task materialization. Can be configured based on
        Kumo deployment type if necessary."""
        config = DataScaleLimits.parse_from_env()

        # Features:
        self.max_single_table_size = (max_single_table_size
                                      or config.max_table_size_bytes)
        self.max_total_table_size = (max_total_table_size
                                     or config.max_graph_total_size_bytes)

        # Graph:
        self.max_total_graph_edges = (max_total_graph_edges
                                      or config.max_graph_total_edges)

    def post_materialization(
        self,
        data_stats: InputDataStats,
    ) -> ValidationResponse:
        r"""Performs validation on data produced post-task materialization.
        This includes:
            * Total raw feature size on disk
            * Total graph size on disk
            * Total number of rows across all tables in the graph
            * Total number of edges in the graph

        It returns a validation response indicating whether the data
        is OK.
        """
        response = ValidationResponse()
        response.warnings
        errors = response.errors

        # 1) Skip size-based validation under DiskGraph — DG-specific
        # size limits are not yet calibrated.

        # 2) Validate no.of rows: the total-node-count cap was removed with the
        # row-based scale limits (kumo#30173); the byte-size caps above and the
        # edge caps below bound graph scale. The empty-table guard stays.
        for table_name, num_rows in data_stats.table_num_nodes.items():
            if num_rows < 1:
                errors.append(
                    ValidationError(
                        "Number of rows should be greater than 0",
                        (f"Table {table_name} has {num_rows} rows.")))

        # 3) Validate no.of edges:
        # We cannot support even with downsampling, so we fail:
        total_downsampled_edges = data_stats.graph_num_downsampled_edges
        total_edges = data_stats.graph_num_edges
        if total_downsampled_edges > self.max_total_graph_edges:
            frac = total_downsampled_edges / self.max_total_graph_edges
            errors.append(
                ValidationError(
                    "Graph is too large",
                    (f"The provided graph exceeds Kumo's graph size limits:"
                     f" it contains {total_edges} edges (computed by "
                     f"[total connections in the graph] x 2). This is "
                     f"{frac:.3f}x more than the maximum number of edges "
                     f"Kumo supports ie. {self.max_total_graph_edges}. "
                     f"Please reduce the number of edges in your "
                     f"graph (e.g. by removing links or reducing the size "
                     f"of your tables) to proceed.")))
        # We can support but will have to downsample, so we warn:
        elif total_edges > self.max_total_graph_edges:
            frac = total_edges / self.max_total_graph_edges
            warn_msg = (
                f"The provided graph exceeds Kumo's graph size limits: it "
                f"contains {total_edges} edges (computed by [total connections"
                f" in the graph x 2]). This is {frac:.3f}x more than "
                f"the maximum number of edges Kumo supports ie. "
                f"{self.max_total_graph_edges}. Relationships that "
                f"produce larger than 2 billion edges may "
                f"be downsampled in the training process.")
            logger.warning(warn_msg)

        # TODO(manan): separately validate fkey tensor since this tensor is
        # currently loaded in GPU node memory...

        # Return only after all validations have been performed:
        return response
