# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from pydantic import Field

from io_utils import kumo_dataclass
from statedb.config.common import TrainingJobID


@kumo_dataclass
class JobScalarTags:
    r"""All scalar tags within a job. The field `trial_tags` contains a
    dictionary from `trial_d` to a list of tags.
    """
    trial_tags: Dict[str, List[str]]


@kumo_dataclass
class ScalarEvent:
    r"""Scalar event in the form of [wall_time, step, value].
    """
    step: int
    wall_time: float
    value: float


@kumo_dataclass
class ScalarsResponse:
    r"""The scalar response that contains a list of scalar events for a given
    tag and trial.
    """
    scalars: List[ScalarEvent] = Field(default_factory=list)


@kumo_dataclass
class GetManyScalarsRequest:
    r"""Request to get scalars for a list of tags for a given trial."""
    job_id: TrainingJobID
    trial: str
    tags: List[str] = Field(default_factory=list)
