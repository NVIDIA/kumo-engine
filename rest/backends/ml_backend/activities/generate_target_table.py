# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union

import pandas as pd
from temporalio import activity

from common_utils.error_diagnostics import (
    exception_chain_contains_restricted_detail,
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from io_utils import json, logging
from io_utils.exceptions import catch_and_log
from kumo.data import get_active_cache
from kumo.home import HomeDirManager
from kumo.pquery import PredictiveQueryV2
from kumo.pquery.utils import validate_forecasting_parameters_consistency
from kumo.validation import ValidationError, ValidationResponse
from rest.backends.data_backend.utils import get_data_operation_description
from rest.backends.ml_backend.activities.common import (
    save_job_storage_path_to_state_db,
)
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.timing import kumo_timing
from rest.backends.ml_backend.validation import (
    GeneratePredictionTableValidator,
    GenerateTrainingTableValidator,
)
from rest.config.batch_prediction_job import BatchPredictionJobInput
from rest.config.ml_backend import TrainingJobInput
from rest.utils.driver_common import (
    hydrate_predictive_query_for_activity,
    open_connectors,
)
from statedb.config.common import GenerateTrainTableJobID, TrainingJobID
from statedb.config.diskgraph import get_diskgraph_index_path_for_job
from statedb.utils.errors import (
    TrainingTableGenerationError,
    is_error_non_retryable,
    rethrow_or_wrap,
)

log = logging.getLogger(__name__)


@activity.defn
def copy_train_table_stats(generate_train_table_job_id: GenerateTrainTableJobID,
                           training_job_id: TrainingJobID):
    """Temporal Activity to copy train generate stats produced from a
    successfully completed generate-train-table job, to destination dir
    indexed by training job id.
    """
    get_active_cache().copy_pquery_table_stats_cache(
        generate_train_table_job_id, training_job_id)

# We have separate activities for generating training and prediction tables
# for easier debugging and tracking.


def generate_training_table(
    job_input: TrainingJobInput,
    settings: MLJobSettings,
) -> Tuple[str, ValidationResponse, str]:
    return _generate_target_table(
        job_input,
        settings,
        diskgraph_index_path=get_diskgraph_index_path_for_job(job_input),
    )


def generate_prediction_table(
    job_input: BatchPredictionJobInput,
    settings: MLJobSettings,
    home_dir: Optional[str] = None,
    is_child_job: Optional[bool] = False,
) -> Tuple[str, ValidationResponse, str]:
    return _generate_target_table(
        job_input, settings, home_dir, is_child_job=is_child_job,
        diskgraph_index_path=get_diskgraph_index_path_for_job(job_input))


def _generate_target_table(
    job_input: Union[BatchPredictionJobInput, TrainingJobInput],
    settings: MLJobSettings,
    home_dir: Optional[str] = None,
    is_child_job: Optional[bool] = False,
    diskgraph_index_path: Optional[str] = None,
) -> Tuple[str, ValidationResponse, str]:
    r"""
    This helper function performs the following functions:
    * Create Training/Prediction Table as a pandas dataframe
    * Persist the training/prediction table

    Returns:
        Tuple[str, ValidationResponse, str]: A tuple of the path to the
            saved table, the validation response and the path to the train
            table generation home directory.
    """
    response = ValidationResponse()

    job_id = job_input.id
    if home_dir is None:
        home_dir = get_job_dir(get_job_base_dir(), job_id)

    # Set training or prediction stage:
    if isinstance(job_input, BatchPredictionJobInput):
        stage = 'pred'
        pquery_config = job_input.predictive_query

        # Validate consistency in forecasting parameters
        response.merge(
            validate_forecasting_parameters_consistency(
                training_forecast_length=job_input.predictive_query.
                forecast_length,
                training_lag_timesteps=job_input.predictive_query.lag_timesteps,
                training_year_over_year=job_input.predictive_query.
                year_over_year,
                prediction_forecast_length=job_input.config.forecast_length,
                prediction_lag_timesteps=job_input.config.lag_timesteps,
                prediction_year_over_year=job_input.config.year_over_year,
            ))
        response.save(home_dir, job_input.id)
        if not response.ok:
            return "", response, ""

    elif isinstance(job_input, TrainingJobInput):
        stage = 'training'
        pquery_config = job_input.spec.predictive_query
    else:
        raise ValueError(f"Invalid job_input type: {type(job_input)}")

    log.info("Starting generate_%s_table activity for job: %s", stage,
             job_input.id)

    # Update the storage directory for the job in StateDB.
    with catch_and_log():
        save_job_storage_path_to_state_db(job_id, home_dir)

    # No process globals: neither temp_home_dir nor per-job engine setup
    # runs. The diskgraph index path is threaded explicitly into
    # predictive_query.generate_train_table / generate_current_pred_table
    # below (via the diskgraph_index_path param), and home_dir is threaded into
    # predictive_query.save / generate_* (-> get_table_path + the table
    # generators) so the output table, intermediate staging, and saved query
    # all land in this job's home_dir. Nothing reads process-global
    # engine/home state, so concurrent jobs on the pool worker's shared
    # thread executor can't collide.
    # Create predictive query
    predictive_query_config, credentials = (
        hydrate_predictive_query_for_activity(
            pquery_config,
            job_input.graph_data_compute_input.connectors,
        ))
    with kumo_timing("target_table.pquery_from_config", stage=stage):
        predictive_query = PredictiveQueryV2.from_config(
            predictive_query_config)

    assert isinstance(predictive_query, PredictiveQueryV2)

    # Define the validators now that we have the task and query types
    if isinstance(job_input, BatchPredictionJobInput):
        validator = GeneratePredictionTableValidator(
            task_type=predictive_query.get_task_type(),
            query_type=predictive_query.query_type)
    elif isinstance(job_input, TrainingJobInput):
        validator = GenerateTrainingTableValidator(
            task_type=predictive_query.get_task_type(),
            query_type=predictive_query.query_type)

    # Set table data versions to the one decided at the beginning
    # of the training workflow
    predictive_query.graph.set_table_data_versions(job_input.id)

    # Open the connectors
    log.debug("Opening Connectors '%s' for job: %s",
              predictive_query.graph.data_sources, job_input.id)
    with kumo_timing("target_table.open_connectors", stage=stage):
        open_connectors(predictive_query.graph, credentials)

    # Data backend cache contains materialization stage, stats for
    # all the tables, etc. Load it to avoid reingesting the tables
    # from scratch during training table generation and task
    # materialization.
    with kumo_timing("target_table.load_table_caches", stage=stage):
        get_active_cache().load_all_table_caches(
            predictive_query.graph.config())

    # Track the predictive query syntax corresponding to the
    # current workflow.
    predictive_query.save(destination=home_dir)

    # Track a validation response for scale if the job ends up failing
    scale_validation_response: Optional[ValidationResponse] = None

    try:
        # Save the table ###########
        # TODO(zeyuan): Path `train_table/<task_name>/train_table`
        # could be further simplified.
        with kumo_timing("target_table.pre_generation_validation",
                         stage=stage):
            response.merge(validator.pre_generation())
        if not response.ok:
            return "", response, ""

        if stage == 'training':
            assert isinstance(job_input, TrainingJobInput)

            # Perform Pre train table generation validation checks
            if isinstance(predictive_query, PredictiveQueryV2):
                table_generator = predictive_query.table_generator
                with kumo_timing(
                    "target_table.pre_train_validation",
                    stage=stage,
                ):
                    response.merge(
                        table_generator.validate_pre_train_table_generate(
                            skip_timestamp_validation=False))

                # Since we potentially return if response.ok is not True,
                # we persist the result of ValidationResponse
                response.save(home_dir, job_id)
                if not response.ok:
                    return "", response, ""

                # Check if the data is very large
                with kumo_timing("target_table.validate_scale",
                                 stage=stage):
                    scale_validation_response = (
                        table_generator.validate_scale())

            index_path = diskgraph_index_path
            with kumo_timing("target_table.generate_train_table",
                             stage=stage):
                table_path, _, train_response = (
                    predictive_query.generate_train_table(
                        job_id=job_input.id,
                        diskgraph_index_path=index_path,
                        destination=home_dir))
            # Now that we decouple generate-train-table job from the
            # training job, we need to make sure the train table stats are
            # persisted under the generate-train-table job id; so that
            # another training job can reuse the train table & stats
            # produced from an earlier generate-train-table job.
            response.merge(train_response)
            response.save(home_dir, job_id)

            if train_response.ok:
                with kumo_timing("target_table.save_train_stats",
                                 stage=stage):
                    train_table_stats = (
                        predictive_query.get_train_table_stats(
                            job_id=job_input.id))
                    assert train_table_stats is not None
                    train_table_stats_path = HomeDirManager(
                        home_dir).train_table_stats
                    json.save_json(train_table_stats,
                                   train_table_stats_path)

            # Mirror DiskGraph's native split stats for split-bearing PQL
            # generation. Old/no-split bundles without the sidecar are a no-op.
            if train_response.ok:
                from kumo.task.split import Split
                from rest.backends.ml_backend.activities.materialize_training_task import (  # noqa: E501
                    _mirror_ttng_split_stats_to_job_dir,
                )
                split_fpid = (
                    Split.get_fpid_from_config(job_input.spec.split)
                    if job_input.spec.split is not None else None)
                _mirror_ttng_split_stats_to_job_dir(
                    train_table_path=table_path,
                    job_dir=home_dir,
                    split_fpid=split_fpid,
                    task_type=predictive_query.get_task_type(),
                )

            if (train_response.ok and job_input.id
                    != job_input.generate_train_table_job_id):
                get_active_cache().copy_pquery_table_stats_cache(
                    job_input.id, job_input.generate_train_table_job_id)
        else:
            assert isinstance(job_input, BatchPredictionJobInput)
            prediction_time: Optional[pd.Timestamp] = (
                pd.Timestamp(job_input.prediction_time)
                if job_input.prediction_time is not None else None)

            # Perform Pre pred table generation validation checks
            pred_table_generator = None
            if isinstance(predictive_query, PredictiveQueryV2):
                pred_table_generator = (
                    predictive_query.get_batch_pred_table_generator())
                if not is_child_job:
                    is_child_job = False
                with kumo_timing("target_table.pre_pred_validation",
                                 stage=stage):
                    response.merge(
                        pred_table_generator.
                        validate_pre_pred_table_generate(
                            timestamp=prediction_time,
                            lag_timestep=predictive_query.lag_timesteps,
                            is_child_job=is_child_job,
                        ))
                if not response.ok:
                    return "", response, ""
                response.save(home_dir, job_id)

            index_path = diskgraph_index_path
            with kumo_timing("target_table.generate_pred_table",
                             stage=stage):
                table_path, train_response = (
                    predictive_query.generate_current_pred_table(
                        job_id=(job_input.generate_pred_table_job_id
                                or job_input.id),
                        training_job_id=job_input.training_job_id,
                        pred_time=prediction_time,
                        diskgraph_index_path=index_path,
                        destination=home_dir,
                    ))

            # NOTE: We have already moved all of the customer to PQLv2.
            # Thus it is safe to only support bp stats per job directory
            # persistence for PQLv2.
            if train_response.ok and isinstance(predictive_query,
                                                PredictiveQueryV2):
                assert pred_table_generator is not None
                batch_pred_table_stats_config = (
                    pred_table_generator.get_batch_pred_table_stats_config(
                        job_id=job_input.id))
                if batch_pred_table_stats_config is not None:
                    batch_pred_table_stats_path = HomeDirManager(
                        home_dir).pred_table_stats
                    json.save_json(batch_pred_table_stats_config,
                                   batch_pred_table_stats_path)
                    log.info(f"Saved batch_pred_table_stats to "
                             f"{batch_pred_table_stats_path}")

        response.merge(train_response)
        if not response.ok:
            return "", response, ""
        response.save(home_dir, job_id)

        # TT-NG writes no canonical table at `table_path`, so its validated
        # producer-manifest row count is authoritative for emptiness. Keep the
        # file-based check only for historical non-DiskGraph training tables.
        if stage == 'training' and diskgraph_index_path is None:
            with kumo_timing("target_table.post_generation_validation",
                             stage=stage):
                response.merge(
                    validator.post_generation(stage, table_path,
                                              predictive_query))
            if not response.ok:
                return "", response, ""
            response.save(home_dir, job_id)
        log.info("Generated %s table at %s", stage, table_path)
    except Exception as e:
        message = (f'Generate {stage} table failed due to {type(e)}: {e}, '
                   f'context: {get_data_operation_description()}')
        message_is_visible = exception_message_should_be_logged_at_error(e)
        has_restricted_detail = exception_chain_contains_restricted_detail(e)
        if message_is_visible:
            if has_restricted_detail:
                log.error(message)
            else:
                log.error(message, exc_info=(type(e), e, e.__traceback__))
        else:
            safe_message = safe_exception_message_for_log(e)
            log.error(
                'Generate %s table failed before public error conversion '
                '(type=%s%s)',
                stage,
                type(e).__name__,
                (f', detail={safe_message}'
                 if safe_message is not None else ''),
            )
        if has_restricted_detail:
            log.debug(message, exc_info=(type(e), e, e.__traceback__))

        # If the job failed and we had scale warnings, it was likely
        # the issue, so we can return the scale warnings as a validation
        # error.
        if scale_validation_response is not None:
            log.error("Scale validation response: %s",
                      scale_validation_response)
            # Scale warnings are useful context, but warnings alone are a
            # successful ValidationResponse. Attach a sanitized error so the
            # workflow cannot continue with the empty table path below. The
            # original exception remains confined to server logs.
            scale_validation_response.errors.append(
                ValidationError(
                    title="Training table generation failed",
                    message=(
                        "Kumo encountered an error while generating the "
                        "training table."),
                ))
            return "", scale_validation_response, ""

        public_message = (
            f"Kumo encountered an error while generating the {stage} table.")
        if not message_is_visible:
            raise TrainingTableGenerationError(
                message=public_message,
                non_retryable=is_error_non_retryable(e),
            ) from None
        rethrow_or_wrap(
            e,
            TrainingTableGenerationError,
            public_message,
        )

    return table_path, response, home_dir
