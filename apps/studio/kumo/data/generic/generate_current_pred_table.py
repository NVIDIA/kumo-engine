# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import pandas as pd
from kumo_core.types import PredTime

from io_utils import logging
from kumo.config import PredictiveQueryV2Config
from kumo.pquery.batch_pred_table_generator import PredictionTableGeneratorV2

logger = logging.getLogger(__name__)
TARGET_FILE_NAME = "targets"


def generate_current_pred_table_generic(
    predictive_query_config: PredictiveQueryV2Config,
    output_file_path: str,
    pred_table_validation_response_path: str,
    destination: str,
    job_id: Optional[str] = None,
    training_job_id: Optional[str] = None,
    pred_time: Optional[Union[PredTime, pd.Timestamp]] = None,
    target_file_path: Optional[str] = None,
    diskgraph_index_path: Optional[str] = None,
):
    """
    Generic implementation of generate prediction table for remote execution.
    This deserializes `predictive_query_config` and drives the diskgraph
    pquery producer via ``PredictionTableGeneratorV2.generate_table``.

    Args:
        predictive_query_config: PredictiveQueryV2Config for which pred table
            needs to be generated.
        output_file_path: Output file path where the generated pred table
            needs to be stored.
        pred_table_validation_response_path (str): Output file path where
            the validation response generated during pred table generation
            needs to be stored.
        job_id (str, optional): id of the prediction job
        training_job_id (str, optional): training job id in the training
            stage.
        target_file_path (str, optional): If ALLOW_ARRAY_TARGETS and link
            prediction, valid targets df will be written here.
            (default: :obj:`None`)
        diskgraph_index_path (str, optional): Path to the diskgraph index.
            (default: :obj:`None`)
    """
    from kumo.pquery import PredictiveQuery, PredictiveQueryV2
    from kumo.validation import ValidationResponse
    from kumo.validation.train_table_validation import (
        PRED_TABLE_VALIDATION_RESPONSE_NAME,
        TrainingTableGenerationError,
    )

    table_details = [(table.table_name, table.data_version)
                     for table in predictive_query_config.graph.tables]
    logger.info(f'Table Name and versions : {table_details}')

    predictive_query: PredictiveQuery = PredictiveQueryV2.from_config(
        predictive_query_config)
    table_generator = predictive_query.get_batch_pred_table_generator()

    # Track validation errors and warnings in train table generation due to
    # predictive query parameters entered by customer
    response = ValidationResponse()
    try:
        # ``PredictiveQueryV2.get_batch_pred_table_generator`` always returns
        # a ``PredictionTableGeneratorV2`` on DG; the legacy non-V2 branch
        # was retired in the DG cutover. The diskgraph executor writes the
        # pred table (and, for the separate-lhs-rhs layout, the target
        # parquet) directly, honoring lp_bp_layout.
        assert isinstance(table_generator, PredictionTableGeneratorV2)
        assert isinstance(predictive_query, PredictiveQueryV2)
        lag_timesteps = predictive_query.lag_timesteps
        table_generator.generate_table(
            pred_time=pred_time,
            lag_timesteps=lag_timesteps,
            job_id=job_id,
            use_yoy_labels=predictive_query_config.year_over_year,
            op_id=training_job_id,
            graph_config=predictive_query_config.graph,
            diskgraph_index_path=diskgraph_index_path,
            target_file_path=target_file_path,
            list_distinct_target_is_array=(
                predictive_query.list_distinct_target_is_array),
            lp_bp_layout=predictive_query.lp_bp_layout,
            output_path=output_file_path if diskgraph_index_path else None,
            destination=destination,
        )
        # Diskgraph wrote parquet directly to output_file_path.
    except TrainingTableGenerationError as e:
        response.errors.append(e.as_error())
    finally:
        response.merge(table_generator.get_validate_train_table_response())
        logger.info(f"Save generate pred table validation responses to "
                    f"{pred_table_validation_response_path}.")
        response.save(pred_table_validation_response_path,
                      PRED_TABLE_VALIDATION_RESPONSE_NAME)
