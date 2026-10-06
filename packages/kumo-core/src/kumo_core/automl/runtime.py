# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit environment capabilities consumed by portable AutoTrainer."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from typing import Any, Protocol

from kumo_core.resolved_task import ResolvedEncoderInput, ResolvedNumericalTask
from kumoapi.encoder import EncoderType
from kumoapi.typing import ColStatType
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
    'collect_resolved_encoders',
]
