# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import kumoapi.model_plan as api
import pytest
from kumo_core.io.json import to_json_dict
from kumo_core.model_plan import ModelPlan
from kumo_core.model_plan_conversion import (
    internal_model_plan_from_public,
    public_model_plan_from_internal,
)


class Case(NamedTuple):
    internal_type: type
    public_type: type
    to_public: Callable[[Any], Any]
    to_internal: Callable[..., Any]

    def internal_plan(self) -> Any:
        plan = self.internal_type()
        plan.training_table_generation.weight_col = "events.weight"
        return plan


_CASES = (
    Case(ModelPlan, api.ModelPlan, public_model_plan_from_internal,
         internal_model_plan_from_public),
)
_CASE_IDS = ("model_plan", )


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_projection_removes_only_internal_section(case: Case) -> None:
    projected = case.to_public(case.internal_plan())

    assert isinstance(projected, case.public_type)
    assert "training_table_generation" not in to_json_dict(projected)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_round_trip_defaults_internal_only_sections(case: Case) -> None:
    internal = case.internal_plan()
    internal.training_job.num_experiments = 3

    restored = case.to_internal(case.to_public(internal))

    assert isinstance(restored, case.internal_type)
    assert restored.training_job.num_experiments == 3
    # The internal-only section never reaches the wire, so it falls back to
    # the default instead of the source plan's value.
    assert to_json_dict(restored.training_table_generation) == to_json_dict(
        case.internal_type().training_table_generation)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_round_trip_with_base_preserves_internal_only_sections(
        case: Case) -> None:
    internal = case.internal_plan()

    restored = case.to_internal(case.to_public(internal), base=internal)

    assert to_json_dict(restored) == to_json_dict(internal)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda payload: payload.update({"unexpected_internal_field": True}),
         "unexpected_internal_field"),
        (lambda payload: payload.pop("training_job"), "training_job"),
    ),
)
def test_projection_fails_closed_on_field_drift(
    monkeypatch: pytest.MonkeyPatch,
    case: Case,
    mutation: Callable[[dict], Any],
    expected: str,
) -> None:
    from kumo_core import model_plan_conversion

    internal = case.internal_plan()
    payload = to_json_dict(internal)
    mutation(payload)
    monkeypatch.setattr(model_plan_conversion, "to_json_dict",
                        lambda _: payload)

    with pytest.raises(ValueError, match=expected):
        case.to_public(internal)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_lift_fails_closed_on_public_field_drift(
    monkeypatch: pytest.MonkeyPatch,
    case: Case,
) -> None:
    from kumo_core import model_plan_conversion

    public = case.to_public(case.internal_plan())
    payload = to_json_dict(public)
    payload["unexpected_public_field"] = True
    monkeypatch.setattr(model_plan_conversion, "to_json_dict",
                        lambda _: payload)

    with pytest.raises(ValueError, match="unexpected_public_field"):
        case.to_internal(public)


@pytest.mark.parametrize('task_type,forecast_length,top_k', [
    ('regression', None, None),
    ('forecasting', 3, None),
    ('static_link_prediction', None, 3),
    ('regression', None, 3),
])
@pytest.mark.parametrize('split',
                         [None, 'TimeSplit(fractions=[0.8, 0.1, 0.1])'])
def test_query_defaults_preserve_the_public_boundary(task_type, forecast_length,
                                                     top_k, split):
    from kumo_core.types import TaskType
    from kumoapi.pquery import QueryType

    facts = dict(task_type=TaskType(task_type), query_type=QueryType.STATIC,
                 has_train_table_time_col=False, run_mode=api.RunMode.FAST)
    original = ModelPlan.from_default(**facts)
    suggested = ModelPlan.from_default(**facts, split=split,
                                       forecast_length=forecast_length,
                                       top_k=top_k)
    generation = suggested.training_table_generation
    assert generation.split == (split if split is not None else
                                original.training_table_generation.split)
    assert generation.forecast_length == (
        forecast_length if forecast_length is not None else
        original.training_table_generation.forecast_length)
    metrics = set(original.training_job.metrics)
    if facts['task_type'].is_ranking and top_k is not None:
        metrics.update(['map@3', 'precision@3', 'recall@3', 'f1@3'])
    assert set(suggested.training_job.metrics) == metrics
    public = public_model_plan_from_internal(suggested)
    assert 'training_table_generation' not in to_json_dict(public)
    restored = internal_model_plan_from_public(public)
    assert to_json_dict(restored.training_table_generation) == to_json_dict(
        ModelPlan().training_table_generation)
    assert set(restored.training_job.metrics) == metrics
