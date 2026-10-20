# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Generic, Literal, Protocol, TypeVar

from kumoapi.model_plan import MissingType, PlanMixin
from kumo_core.evaluation.suite import MetricStore


class TrainingJobSearchPlan(Protocol):
    """Training-plan fields consumed by hyperparameter search."""

    tune_metric: str


class SearchPlan(Protocol):
    """Structural contract required by the portable searchers.

    Model-plan ownership remains outside the search implementation. Any
    dataclass plan whose sections expose ``items()`` metadata and whose
    training section defines ``tune_metric`` satisfies this contract.
    """

    training_job: TrainingJobSearchPlan


PlanT = TypeVar("PlanT", bound=SearchPlan)


class Searcher(ABC, Generic[PlanT]):
    """Suggest experiments from a model-plan search space."""
    def __init__(self, space: PlanT, max_attempts: int = 50) -> None:
        self.space = space
        self.max_attempts = max_attempts
        self.state: list[tuple[PlanT, list[float]]] = []

    @property
    def tune_metric(self) -> str:
        return f"val_{self.space.training_job.tune_metric}"

    @property
    def tune_mode(self) -> Literal["min", "max"]:
        return MetricStore.tune_mode(self.tune_metric)

    def update(self, cfg: PlanT, performance: list[float]) -> None:
        self.state.append((cfg, performance))

    def suggest(self) -> PlanT:
        for _ in range(self.max_attempts):
            cfg = self.sample_instance()
            if cfg not in [prev_cfg for prev_cfg, _ in self.state]:
                break
        return cfg

    @abstractmethod
    def analyze(self, path: str) -> None:
        """Write analysis of the experimental results to ``path``."""

    @abstractmethod
    def sample_value(self, key: str, value: list[Any]) -> Any:
        """Sample one value from a tunable field."""

    def sample_instance(self) -> PlanT:
        instance = copy.deepcopy(self.space)

        for field in fields(instance):
            section = getattr(instance, field.name)
            for key, value, metadata in section.items():
                if not metadata.tunable or isinstance(value, MissingType):
                    continue

                assert isinstance(value, list)
                if isinstance(value[0], PlanMixin):
                    index_of_plan = self.sample_value(key,
                                                      list(range(len(value))))
                    plan = value[index_of_plan]
                    for subkey, subvalue, submetadata in plan.items():
                        if not submetadata.tunable or isinstance(
                                subvalue, MissingType):
                            continue
                        assert isinstance(subvalue, list)
                        unique_key = f"{key}{index_of_plan}_{subkey}"
                        setattr(
                            plan,
                            subkey,
                            [self.sample_value(unique_key, subvalue)],
                        )
                    setattr(section, key, [plan])
                else:
                    setattr(section, key, [self.sample_value(key, value)])

        return instance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def searched_params(space: SearchPlan, cfg: SearchPlan) -> dict[str, Any]:
    """Returns the values ``cfg`` took for fields ``space`` searches over."""
    def is_searched(value: Any, metadata: Any) -> bool:
        if not metadata.tunable or isinstance(value, MissingType):
            return False
        return len(value) > 1 or any(
            is_searched(subvalue, submetadata) for plan in value
            if isinstance(plan, PlanMixin)
            for _, subvalue, submetadata in plan.items())

    params: dict[str, Any] = {}
    for field in fields(space):
        for key, value, metadata in getattr(space, field.name).items():
            if is_searched(value, metadata):
                chosen = getattr(getattr(cfg, field.name), key)[0]
                params[f"{field.name}.{key}"] = (
                    chosen if chosen is None
                    or isinstance(chosen, (bool, int, float, str)) else
                    str(chosen))
    return params
