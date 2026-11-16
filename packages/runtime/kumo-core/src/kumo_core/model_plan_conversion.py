# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-way conversion between internal model plans and public contracts."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import fields
from typing import Any, TypeVar

import kumoapi.model_plan as api
from kumo_core.io.json import from_json_dict, to_json_dict
from kumo_core.model_plan import ModelPlan

#: Internal plan sections that are never exposed on the public contract.
INTERNAL_ONLY_FIELDS = ('training_table_generation', )

InternalT = TypeVar('InternalT')
PublicT = TypeVar('PublicT')


def _field_names(cls: type) -> frozenset[str]:
    return frozenset(field.name for field in fields(cls))


def _check_fields(
    actual: Collection[str],
    expected: Collection[str],
    *,
    context: str,
) -> None:
    actual_fields = frozenset(actual)
    expected_fields = frozenset(expected)
    if actual_fields != expected_fields:
        raise ValueError(
            f'{context} fields differ from the reviewed public projection: '
            f'missing={sorted(expected_fields - actual_fields)}, '
            f'unexpected={sorted(actual_fields - expected_fields)}')


def project_public_dataclass(
    value: Any,
    public_type: type[PublicT],
    *,
    internal_only_fields: Collection[str],
) -> PublicT:
    """Project an internal dataclass through an exact, reviewed public field
    set, dropping ``internal_only_fields``.
    """
    payload = to_json_dict(value)
    _check_fields(
        payload,
        _field_names(public_type) | frozenset(internal_only_fields),
        context=f'Internal {type(value).__name__}',
    )

    for field_name in internal_only_fields:
        del payload[field_name]
    return from_json_dict(payload, public_type)


def lift_public_dataclass(
    value: Any,
    internal_type: type[InternalT],
    *,
    internal_only_fields: Collection[str],
    base: InternalT | None = None,
) -> InternalT:
    """Lift a public dataclass back into its internal counterpart.

    ``internal_only_fields`` are taken from ``base`` when given, and otherwise
    default to a freshly constructed ``internal_type``.
    """
    payload = to_json_dict(value)
    public_fields = _field_names(type(value))
    _check_fields(payload, public_fields,
                  context=f'Public {type(value).__name__}')
    _check_fields(
        _field_names(internal_type),
        public_fields | frozenset(internal_only_fields),
        context=f'Internal {internal_type.__name__}',
    )

    internal_defaults = to_json_dict(
        base if base is not None else internal_type())
    for field_name in internal_only_fields:
        payload[field_name] = internal_defaults[field_name]
    return from_json_dict(payload, internal_type)


def public_model_plan_from_internal(model_plan: ModelPlan) -> api.ModelPlan:
    """Converts an internal model plan into its public contract."""
    return project_public_dataclass(
        model_plan,
        api.ModelPlan,
        internal_only_fields=INTERNAL_ONLY_FIELDS,
    )


def internal_model_plan_from_public(
    model_plan: api.ModelPlan,
    *,
    base: ModelPlan | None = None,
) -> ModelPlan:
    """Converts a public model plan into its internal counterpart.

    Internal-only sections are copied from ``base`` when given, so that an
    edit round-trip through the public contract preserves them.
    """
    return lift_public_dataclass(
        model_plan,
        ModelPlan,
        internal_only_fields=INTERNAL_ONLY_FIELDS,
        base=base,
    )
