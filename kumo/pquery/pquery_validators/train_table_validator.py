# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional

import pandas as pd
from kumo_core.training.options import allow_array_targets
from kumoapi.pquery.AST import DateOffsetRange
from kumoapi.typing import ProblemType

from io_utils import logging
from kumo.data import MAX_NUM_CLASSES
from kumo.graph.column import EncoderGenerator

# isort: off
from kumo.pquery.pquery_validators.generation_validator import (
    GenerationValidator, )
# isort: on
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from kumo.pquery.stats_manager import StatsManager
from kumo.pquery.utils import get_max_train_rows, get_rhs_lp_table, table_name
from kumo.task.split import (
    ApproxDateOffsetSplit,
    BacktestingSplit,
    DateOffsetSplit,
    Split,
)
from kumo.validation import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from service_framework.env_var import EnvVar

logger = logging.getLogger(__name__)

MIN_ABSOLUTE_ENTITIES = 100  # Minimum entities for warning.
MIN_TRAIN_ROWS = 1000  # Minimum training examples for warning.
MIN_NA_FRACTION = 0.05
# The limit of the training table rows should be defined in data
# workflow worker for training table materialization.
# Maximum training examples for error.
MAX_TRAIN_ROWS = EnvVar('MAX_TRAIN_ROWS', int, required=False,
                        default_value=200_000_000).parse_value()
# Maximum number of rhs entities in link prediction for error
MAX_RHS_ROWS = EnvVar('MAX_RHS_ROWS', int, required=False,
                      default_value=10_000_000).parse_value()


class TrainTableValidator(GenerationValidator):
    """Runs validation using the stats of the generated training table.

    Args:
        stats_manager (StatsManager): `StatsManager` that has been used to
            cache stats.
        split (Split): `Split` used by the predictive query.
        timeframe_step (int, optional): Timeframe step, if it has been used.
            (default: :obj:`None`)

    """
    def __init__(
        self,
        stats_manager: StatsManager,
        split: Split,
        timeframe_step: Optional[pd.DateOffset] = None,
    ):
        super(TrainTableValidator, self).__init__(stats_manager)
        self.stats_manager = stats_manager
        self.query = stats_manager.query
        self.task_type = stats_manager.task_type
        self.split = split
        self.timeframe_step = timeframe_step
        stats_manager_value = getattr(stats_manager,
                                      'list_distinct_target_is_array', None)
        self.list_distinct_target_is_array = (stats_manager_value if isinstance(
            stats_manager_value, bool) else allow_array_targets())

    def validate_task_type(self, stats: Any) -> ValidationResponse:
        r"""Validates and issues a warning for multilabel task type when the
        cardinality of label column is 2, to prompt the user to consider
        changing the task to binary classification.

        Args:
            stats (Any): This is usually TemporalNodeTimeFrameStats or similar
                based on the problem type. Check stats_manager for available
                APIs.
        """
        response = ValidationResponse()
        task_type = self.task_type
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            if stats.num_targets == 2:
                response.warnings.append(
                    ValidationWarning(
                        title='Multiclass classification has only two targets.',
                        message=('Query is multiclass but only two labels are '
                                 'present. Use a binary label instead '
                                 '(e.g., `PREDICT target.column = "A"`).')))
        return response

    def validate_post_train_table_generate(self) -> ValidationResponse:
        """Validates the contents of the generated training table based on
        generated stats.

        Returns:
            ValidationResponse: The encountered warnings and errors.

        """
        validation_response = ValidationResponse()
        stats = self.stats_manager.get_train_table_stats()
        validation_response.merge(self.validate_task_type(stats))

        validation_response.merge(self.validate_table_size(stats))
        if (hasattr(stats, 'num_timeframes')
                and not isinstance(self.split, BacktestingSplit)):
            assert hasattr(stats, 'num_timeframes_post_filter')
            validation_response.merge(self.check_for_empty_timeframes(stats))
            if isinstance(self.split, (ApproxDateOffsetSplit, DateOffsetSplit)):
                validation_response.merge(
                    self.validate_timeframes_post_generation(stats))
        skip_scalar_target_check = (self.task_type == TaskType.FORECASTING)
        if (self.task_type in {TaskType.REGRESSION, TaskType.FORECASTING}
                and not skip_scalar_target_check):
            validation_response.merge(
                self.validate_num_unique_target_values(stats))
        if self.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION
        ]:
            validation_response.merge(self.validate_class_number(stats))
        if self.query.top_k is not None:
            validation_response.merge(
                self.validate_topk(stats, self.query.top_k))
        if self.task_type.is_link_pred:
            validation_response.merge(
                self.validate_link_prediction_table(stats))
        return validation_response

    def validate_link_prediction_table(self, stats: Any) -> ValidationResponse:
        r"""All validations connected to link prediction."""
        validation_response = ValidationResponse()
        # NOTE we currently need this check to avoid OOMing when generating
        # RHS embeddings for link prediction:
        num_rhs_entities = stats.num_targets

        rhs_table = get_rhs_lp_table(
            self.query,
            self.stats_manager.graph,
        )
        if num_rhs_entities > MAX_RHS_ROWS:
            aggr = self.query.get_final_target_aggregation()
            assert aggr is not None
            rhs_too_large_error = (
                f"The maximum supported number of unique values in the "
                f"link prediction target column is {MAX_RHS_ROWS}, "
                f"while {rhs_table} contains "
                f"{num_rhs_entities} unique values. Consider manually "
                f"removing some the possible targets.")
            logger.error(rhs_too_large_error)
            validation_response.errors.append(
                ValidationError(
                    "Target column contains too many unique values",
                    rhs_too_large_error,
                ))

        return validation_response

    def validate_num_unique_target_values(self,
                                          stats: Any) -> ValidationResponse:
        r"""Validate that the number of unique target values is more than one
        for regression problems.

        Args:
            stats (TrainTableStats): A `TrainTableStats` object.

        Returns:
            (ValidationResponse): A validation response.
        """
        validation_response = ValidationResponse()
        num_unique_targets = stats.num_targets
        assert num_unique_targets > 0
        if num_unique_targets == 1:
            unique_target_value = stats.target_distribution.get('target_mode')
            assert unique_target_value is not None
            one_unique_targets_error = (
                f"Training table (before split) only has one target value "
                f"({unique_target_value}), so the prediction task is "
                f"ill-defined. Adjust the query to produce more than one "
                f"target value.")
            logger.error(one_unique_targets_error)
            validation_response.errors.append(
                ValidationError(
                    "Too few unique target values to predict",
                    one_unique_targets_error,
                ))
        return validation_response

    def validate_class_number(self, stats: Any) -> ValidationResponse:
        r"""Validate that the number of classes is not too large/small."""
        validation_response = ValidationResponse()
        num_classes = stats.num_targets
        if num_classes > MAX_NUM_CLASSES:
            many_num_classes_error = (f'Number of different target classes '
                                      f'exceeds {MAX_NUM_CLASSES} '
                                      f'(got {num_classes})')
            logger.error(many_num_classes_error)
            validation_response.errors.append(
                ValidationError(
                    "Too many classes to predict",
                    (f'{many_num_classes_error}. Consider using RANK instead '
                     f'of CLASSIFY.') if self.query.problem_type
                    == ProblemType.CLASSIFY else many_num_classes_error,
                ))

        classes = list(stats.target_distribution.get('percentage', {}).keys())
        if num_classes == 1:
            # If the entity or target filters are too restrictive sometimes
            # we can get no entity, timeframe x target combinations that are
            # valid.
            all_negative_samples = len(classes) == 0
            if all_negative_samples:
                all_negative_samples_error = (
                    "The training table (before split) only contains negative "
                    "samples. Loosen filters or widen the prediction window "
                    "to include positive examples.")
                logger.error(all_negative_samples_error)
                validation_response.errors.append(
                    ValidationError(
                        "No positive samples",
                        all_negative_samples_error,
                    ))
            else:
                target_key = classes[0]
                one_num_classes_error = (
                    f"Training table (before split) has only one target value "
                    f"({target_key}), so the prediction task is ill-defined. "
                    f"Adjust the query to produce multiple target values.")
                logger.error(one_num_classes_error)
                validation_response.errors.append(
                    ValidationError(
                        "Too few classes to predict",
                        one_num_classes_error,
                    ))
        return validation_response

    def validate_timeframes_post_generation(self,
                                            stats: Any) -> ValidationResponse:
        r"""Validates that the generated timeframes are sufficient for
        data split creation. If there are fewer than 2 timeframes, an error
        is returned. If fewer than 3, a warning.

        Args:
            stats (Stats): Training data stats.

        Returns:
            ValidationResponse: The errors and warnings encountered.
        """
        assert self.query.query_type == QueryType.TEMPORAL
        response = ValidationResponse()
        if isinstance(self.split, BacktestingSplit):
            return response
        num_splits = self.split.num_splits  # type: ignore
        assert num_splits in {2, 3}

        if stats.num_timeframes_post_filter < num_splits:
            response.errors.append(
                ValidationError(
                    title=("Insufficient train table timeframes to create "
                           "train, validation and test sets"),
                    message=(
                        f"Training table has {stats.num_timeframes_post_filter} "
                        f"timeframe(s), fewer than the {num_splits} needed for "
                        f"train/val/test. Extend the time range, loosen "
                        f"filters, or pick a split compatible with fewer "
                        f"timeframes (e.g., RandomSplit or a 2-way "
                        f"DateOffsetSplit).")))
        return response

    def validate_topk(self, stats: Any, topk: int) -> ValidationResponse:
        r"""Checks if the topk is larger than the number of targets"""
        validation_response = ValidationResponse()
        num_targets = stats.num_targets
        if topk > num_targets:
            topk_too_large_error = (
                f"TOP k ({topk}) exceeds the number of target values "
                f"({num_targets}). Set TOP k below {num_targets}.")
            logger.error(topk_too_large_error)
            validation_response.errors.append(
                ValidationError(
                    "Top k is too large",
                    topk_too_large_error,
                ))
        return validation_response

    def check_for_empty_timeframes(self, stats: Any) -> ValidationResponse:
        r"""Checks if any timeframes are empty due to restrictive filters."""
        validation_response = ValidationResponse()
        if stats.num_timeframes <= stats.num_timeframes_post_filter:
            # Nothing to do
            return validation_response

        time_range = self.query.get_combined_date_offset_range()
        assert time_range is not None
        target_aggregation_timeframe = (DateOffsetRange.date_offset_subtract(
            time_range.end_date_offset,
            time_range.start_date_offset))  # type: ignore
        train_table_timeframe_expected = (DateOffsetRange.date_offset_multiply(
            target_aggregation_timeframe, stats.num_timeframes))
        train_table_timeframe_expected_repr = (
            DateOffsetRange.date_offset_repr(train_table_timeframe_expected))

        train_table_timeframe_post_filter = (
            DateOffsetRange.date_offset_multiply(
                target_aggregation_timeframe, stats.num_timeframes_post_filter))
        train_table_timeframe_post_filter_repr = (
            DateOffsetRange.date_offset_repr(train_table_timeframe_post_filter))

        target_tables = list(
            set([
                table_name(aggr._get_target_column_name())
                for aggr in self.query.get_all_target_aggregations()
            ]))

        timeframes_filtered_warning = (
            f"Most training timeframes were filtered out. Expected "
            f"{train_table_timeframe_expected_repr} from targets "
            f"{target_tables}, but only "
            f"{train_table_timeframe_post_filter_repr} remains after filters.")
        logger.warning(timeframes_filtered_warning)
        validation_response.warnings.append(
            ValidationWarning(
                "Some training table timeframes are being filtered out",
                f"{timeframes_filtered_warning} Loosen entity filters or "
                f"ASSUMING conditions to keep more timeframes."))
        return validation_response

    def validate_table_size(self, stats: Any) -> ValidationResponse:
        r"""Contains validations related to table size, such as the table
        being too large, too small, or containing too few entities."""
        validation_response = ValidationResponse()
        if isinstance(self.split, BacktestingSplit):
            return validation_response
        if stats.num_entities < MIN_ABSOLUTE_ENTITIES:
            few_absolute_entities_warning = (
                f"The number of entities in the training table is very "
                f"low (there are {stats.num_entities} entities), which may "
                f"lead to degraded performance. For improved performance, "
                f"consider modifying your query to include more training "
                f"data.")
            logger.warning(few_absolute_entities_warning)

        num_examples = stats.num_train_rows
        if (self.list_distinct_target_is_array and self.task_type.is_link_pred):
            num_examples = stats.num_links

        if num_examples < MIN_TRAIN_ROWS:
            few_num_train_rows_warning = (
                f"The training table generated by the query has only "
                f"{num_examples} training examples, which could "
                f"result in degraded performance. For improved performance, "
                f"consider modifying your query to include more training data.")
            logger.warning(few_num_train_rows_warning)

        max_permitted_samples = get_max_train_rows()
        if (self.list_distinct_target_is_array and self.task_type.is_link_pred):
            max_permitted_samples *= (
                EncoderGenerator.MAX_MULTI_LABEL_CATEGORIES)
        if num_examples > max_permitted_samples:
            suggest_msg_list: List[str] = []
            suggest_msg_list.append(
                "Manually reduce the size of your entity table.")

            suggest_msg_list.append(
                "Add more entity filters or make them "
                "more restrictive to remove "
                "entities that are irrelevant to your task.")

            suggest_msg_list.append(
                "Manually reduce the size of your entity or target table.")

            if self.query.target_timeframe is not None:
                suggest_msg_list.append(
                    "Decrease the size of your target table by setting your "
                    "train_start_offset to a smaller time range.")

            if self.timeframe_step is not None:
                time_range = self.query.get_combined_date_offset_range()
                assert time_range is not None
                target_aggregation_timeframe = (
                    DateOffsetRange.date_offset_subtract(
                        time_range.start_date_offset,
                        time_range.end_date_offset))  # type: ignore

                # `DateOffset` does not allow comparison.
                now = pd.Timestamp.now()
                if (now + target_aggregation_timeframe
                        > now + self.timeframe_step):
                    suggest_msg_list.append(
                        f"Since your 'timeframe_step' {self.timeframe_step} "
                        f"is smaller than the time frame in the target "
                        f"aggregations, consider increasing "
                        f"the size of your 'timeframe_step' or removing it "
                        f"from the predictive query.")
            suggest_msg = "\n".join([
                f"{i}. {message}"
                for i, message in enumerate(suggest_msg_list, 1)
            ])
            # Note: We assume that data backend downsampling will already have
            # been merged by the time the user can see this.
            # A better solution with automatic sampling adjustment is a TODO
            many_num_train_rows_warning = (
                f"The generated training table contains "
                f"{num_examples} examples, which is larger than our "
                f"current allowed limit of {max_permitted_samples} examples. "
                f"Only a subset of these examples will be used during the "
                f"training. "
                f"To remove this warning, consider the following suggestions: "
                f"{suggest_msg}")

            logger.warning(many_num_train_rows_warning)

        if (self.task_type != TaskType.STATIC_LINK_PREDICTION
                and self.query.query_type == QueryType.STATIC):
            assert hasattr(stats, 'num_total_rows')
            num_total_rows = int(stats.num_total_rows)
            if num_total_rows <= 0:
                invalid_rows_error = (
                    "Training table has an invalid total row count "
                    f"({num_total_rows}). Please refresh data and retry.")
                logger.error(invalid_rows_error)
                validation_response.errors.append(
                    ValidationError(
                        "Invalid training table row count",
                        invalid_rows_error,
                    ))
                return validation_response
            na_fraction = 1 - int(stats.num_train_rows) / num_total_rows
            if na_fraction == 0.0:
                zero_na_warning = (
                    "Target column has no missing values. "
                    "This means there will be no values to predict "
                    "and batch prediction will fail.")
                logger.warning(zero_na_warning)
            elif na_fraction < MIN_NA_FRACTION:
                few_na_warning = (
                    f"Target column has ~{100 * na_fraction:.1f}% missing "
                    f"values which is less than our recommended minimum of "
                    f"{100 * MIN_NA_FRACTION:.0f}%. This means there won't "
                    f"be many prediction outputs in Batch Prediction.")
                logger.warning(few_na_warning)
        return validation_response
