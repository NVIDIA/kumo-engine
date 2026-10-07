# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit environment capabilities consumed by portable AutoTrainer."""

from __future__ import annotations

import copy
import dataclasses
import posixpath
import warnings
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any, Protocol, get_args, get_type_hints

from kumo_core.io.json import load_json
from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumo_core.resolved_task import ResolvedEncoderInput, ResolvedNumericalTask
from kumoapi.encoder import EncoderType, NAStrategy
from kumoapi.typing import ColStatType, Stype
from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig
from pydantic import TypeAdapter


def _with_encoder_stats(
    encoder: EncoderType,
    value: ResolvedEncoderInput,
) -> EncoderType:
    encoder = copy.deepcopy(encoder)
    encoder._stats = {
        ColStatType(name): stat_value
        for name, stat_value in value.stats.items()
    }
    missing = set(encoder.required_stats) - set(encoder._stats)
    if missing:
        names = sorted(stat.value for stat in missing)
        raise ValueError(
            f"encoder stats missing for {value.column_name!r}: {names!r}")
    return encoder


def _materialize_encoder(value: ResolvedEncoderInput) -> EncoderType:
    encoder = TypeAdapter(EncoderType).validate_python(value.encoder)
    return _with_encoder_stats(encoder, value)


def collect_resolved_encoders(task: ResolvedNumericalTask) -> EncodersConfig:
    encoders = EncodersConfig()
    for table_name, values in task.encoder_inputs.items():
        encoders[table_name] = {
            value.column_name: _materialize_encoder(value)
            for value in values
        }
    target = task.target_encoder_input
    if target is not None:
        encoders.target = TargetEncoderConfig(
            target.column_name,
            _materialize_encoder(target),
        )
    return encoders


def allowed_na_strategies(encoder: Any) -> tuple[NAStrategy, ...]:
    if not hasattr(encoder, 'na_strategy'):
        return ()
    return get_args(get_type_hints(type(encoder))['na_strategy'])


def with_na_strategy(
    stype: str,
    encoder: Any,
    na_strategy: Mapping[Stype, NAStrategy] | None,
) -> EncoderType | None:
    strategy = (na_strategy or {}).get(stype)
    if strategy is None or strategy not in allowed_na_strategies(encoder):
        return None
    return dataclasses.replace(encoder, na_strategy=strategy)


def apply_na_strategy(
    task: ResolvedNumericalTask,
    encoders: EncodersConfig,
    na_strategy: Mapping[Stype, NAStrategy] | None,
    overridden: Collection[str] = (),
) -> EncodersConfig:
    if not na_strategy:
        return encoders
    skipped: dict[str, list[str]] = defaultdict(list)
    for table_name, values in task.encoder_inputs.items():
        table = encoders.encoders.get(table_name, {})
        for value in values:
            name = f"{table_name}.{value.column_name}"
            encoder = table.get(value.column_name)
            if name in overridden:
                continue
            updated = with_na_strategy(value.stype, encoder, na_strategy)
            if updated is not None:
                table[value.column_name] = _with_encoder_stats(updated, value)
            elif (value.stype in na_strategy
                  and (allowed := allowed_na_strategies(encoder))):
                skipped[value.stype].append(
                    f"'{name}' ({type(encoder).__name__}, allowed: "
                    f"{', '.join(allowed)})")
    for stype, columns in skipped.items():
        warnings.warn(f"N/A strategy '{na_strategy[stype]}' for semantic "
                      f"type '{stype}' is not supported by "
                      f"{'; '.join(columns)}. These columns keep their "
                      "default N/A strategy.")
    return encoders


def apply_encoder_overrides(
    task: ResolvedNumericalTask,
    encoders: EncodersConfig,
    overrides: Mapping[str, EncoderType],
) -> EncodersConfig:
    """Apply a resolved ModelPlan's encoders to materialized task encoders."""
    for qualified_name, configured in overrides.items():
        table_name, separator, column_name = qualified_name.rpartition(".")
        if not separator:
            warnings.warn(f"'{qualified_name}' is not a valid column name. "
                          "Skipping encoder override.")
            continue

        target = encoders.target
        target_input = task.target_encoder_input
        # ModelPlan has historically exposed the generated target through the
        # literal ``train_table.TARGET`` qualifier as well as its resolved
        # train-table name.
        if (target is not None and column_name == target.col_name
                and target_input is not None
                and table_name in {"train_table", task.train_table.name}):
            encoder_input = target_input
            destination = target
        else:
            table_inputs = task.encoder_inputs.get(table_name)
            if table_inputs is None:
                warnings.warn(f"'{table_name}' is not a valid table name. "
                              "Skipping encoder override.")
                continue
            encoder_input = next(
                (value
                 for value in table_inputs if value.column_name == column_name),
                None,
            )
            if encoder_input is None:
                warnings.warn(f"'{column_name}' is not a valid column name "
                              f"for table '{table_name}'. Skipping encoder "
                              "override.")
                continue
            destination = encoders.encoders[table_name]

        encoder = _with_encoder_stats(configured, encoder_input)
        if destination is target:
            destination.encoder = encoder
        else:
            destination[column_name] = encoder
    return encoders


def validate_warm_start_task_type(
    current: TaskType,
    source: TaskType | None = None,
) -> None:
    """Hosted warm-start compatibility policy, shared with local training."""
    if current == TaskType.FORECASTING:
        raise ValueError("Warm start is not supported for forecasting tasks")
    if source is not None and source != current:
        raise ValueError(f"Warm start task type {source} "
                         f"is different from the current task type {current}")


def inherit_warm_start_model_plan(space: ModelPlan, source: ModelPlan) -> None:
    """Keep destination training controls and inherit fitted model structure."""
    space.column_processing = source.column_processing
    space.model_architecture = source.model_architecture
    space.neighbor_sampling = source.neighbor_sampling


def resolve_warm_start(
    space: ModelPlan,
    task: ResolvedNumericalTask,
    runtime: AutoTrainingRuntime,
    *,
    trainer_dir: str | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve a hosted trainer or a portable bundle into checkpoint inputs.

    Bundles already name the selected (possibly refitted) checkpoint and fitted
    encoders; they do not retain hosted AutoML directory layouts.
    """
    if artifacts is not None:
        source_schema = load_json(artifacts["prediction_schema"], dict)
        validate_warm_start_task_type(task.task_type,
                                      TaskType(source_schema["task_type"]))
        checkpoint = artifacts["selected_checkpoint"]
        plan_path = artifacts["resolved_model_plan"]
        encoder_dir = posixpath.dirname(artifacts["encoders"])
    elif trainer_dir:
        validate_warm_start_task_type(task.task_type)
        from kumo_core.train.trainer import Trainer
        trainer = Trainer.load(task.to_task_config(), trainer_dir, version=None,
                               infer_config_path=False,
                               **runtime.trainer_runtime_kwargs())
        trainer._set_fit_mode_to_use()
        checkpoint = trainer.best_ckpt
        plan_path = posixpath.join(trainer_dir, "model_plan.json")
        # Retain the existing hosted encoder-directory convention.
        encoder_dir = None
    else:
        return None, None
    inherit_warm_start_model_plan(space, load_json(plan_path, ModelPlan))
    return checkpoint, encoder_dir


class AutoTrainingRuntime(Protocol):
    def default_home_dir(self) -> str:
        ...

    def metrics_job_name(self) -> str:
        ...

    def collect_encoders(
        self,
        task: ResolvedNumericalTask,
    ) -> EncodersConfig:
        ...

    def load_task(self, task_name: str, home_dir: str) -> Any:
        ...

    def load_automl_config(self, job_config: Any) -> Any:
        ...

    def trainer_runtime_kwargs(self) -> dict[str, Any]:
        ...


__all__ = [
    'AutoTrainingRuntime',
    'apply_encoder_overrides',
    'apply_na_strategy',
    'with_na_strategy',
    'collect_resolved_encoders',
    'inherit_warm_start_model_plan',
    'resolve_warm_start',
    'validate_warm_start_task_type',
]
