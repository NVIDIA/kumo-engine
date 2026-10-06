# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK authoring adapters for portable model-plan records."""

from kumo_core.model_plan import (
    ModelPlan,
    PredictiveQueryAdvancedOptions,
    inferred_or_missing_type_to_false,
    inferred_or_missing_type_to_none,
)

from kumo.task.splits.parsing import get_split


def to_predictive_query_kwargs(
    plan: ModelPlan, ) -> PredictiveQueryAdvancedOptions:
    split_string = inferred_or_missing_type_to_none(
        plan.training_table_generation.split)
    parsed_split = get_split(split_string)
    split_config = parsed_split.config() if parsed_split is not None else None
    section = plan.training_table_generation
    return PredictiveQueryAdvancedOptions(
        timeframe_step=inferred_or_missing_type_to_none(section.timeframe_step),
        train_start_offset=inferred_or_missing_type_to_none(
            section.train_start_offset),
        train_end_offset=inferred_or_missing_type_to_none(
            section.train_end_offset),
        start_time=inferred_or_missing_type_to_none(section.start_time),
        end_time=inferred_or_missing_type_to_none(section.end_time),
        forecast_length=inferred_or_missing_type_to_none(
            section.forecast_length),
        lag_timesteps=inferred_or_missing_type_to_none(section.lag_timesteps),
        year_over_year=inferred_or_missing_type_to_false(
            section.year_over_year),
        entity_candidate=section.entity_candidate,
        entity_candidate_aggregation=section.entity_candidate_aggregation,
        weight_col=section.weight_col,
        split=split_config,
    )


__all__ = ['to_predictive_query_kwargs']
