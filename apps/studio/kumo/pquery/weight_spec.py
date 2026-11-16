# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from kumo_core.types import AggregationType

from kumo.validation import ValidationException

DEFAULT_WEIGHT_COL_NAME = 'WEIGHT'


@dataclass(frozen=True)
class WeightSpec:
    """Parsed representation of a weight column specification."""
    column: Optional[str]
    aggregation: Optional[AggregationType]
    raw: str
    is_expression: bool


def validate_weight_spec_format(weight_spec: WeightSpec) -> None:
    """Validate that weight_spec uses strict table.column format."""
    if weight_spec.column is None:
        raise ValidationException(
            message="Weight column must be in table.column form.")
    parts = weight_spec.column.split('.')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValidationException(
            message="Weight column must be in table.column form.")


def normalize_weight_col(weight_col: Optional[str]) -> Optional[str]:
    """Normalize weight_col input."""
    if weight_col is None:
        return None
    return weight_col.strip()


def parse_weight_spec(weight_col: Optional[str]) -> Optional[WeightSpec]:
    """Parse a weight_col string into a WeightSpec.

    Args:
        weight_col: Raw weight column string from the plan/config.

    Returns:
        Parsed WeightSpec if provided; otherwise None.

    Raises:
        ValidationException: If the input is empty or not in table.column format.
    """
    expr = normalize_weight_col(weight_col)
    if expr is None:
        return None

    if not expr:
        raise ValidationException(message="Weight column cannot be empty.")

    spec = WeightSpec(column=expr, aggregation=None, raw=expr,
                      is_expression=False)
    validate_weight_spec_format(spec)
    return spec
