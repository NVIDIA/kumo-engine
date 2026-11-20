# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact retained prediction status, independent of producer artifact formats."""

from pathlib import PurePosixPath
from typing import Literal

from kumo_core.io import filesystem
from kumo_core.prediction_explanation_result import PredictionExplanationResult
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MANIFEST_PATH = 'xai/manifest.json'


class PredictionExplanationManifest(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)

    schema_version: Literal[1]
    kind: Literal['prediction_explanations']
    run_id: str
    enabled: bool
    state: Literal['running', 'complete']
    result: PredictionExplanationResult | None = None

    @field_validator('schema_version', mode='before')
    @classmethod
    def validate_version(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError('Explanation schema version must be an integer')
        return value

    @model_validator(mode='after')
    def validate_record(self) -> 'PredictionExplanationManifest':
        if not self.run_id.strip():
            raise ValueError('Prediction run ID must not be empty')
        if not self.enabled:
            if self.state != 'complete' or self.result is not None:
                raise ValueError('Disabled explanations cannot have producers')
            return self
        if self.result is None:
            raise ValueError('Enabled explanations require producer outcomes')
        for outcome in (self.result.cohorts, self.result.samples):
            if outcome.status == 'failed':
                if not outcome.error:
                    raise ValueError('Failed producers require an error')
            elif outcome.error is not None:
                raise ValueError('Successful or unsupported producers cannot fail')
            if self.state == 'running' and outcome.status == 'complete':
                raise ValueError('Running producers cannot be complete')
        coverage = self.result.coverage
        for count in (coverage.cohort_rows_processed, coverage.sampled_graphs):
            if count is not None and count < 0:
                raise ValueError('Explanation coverage must be nonnegative')
        checkpoint = coverage.checkpoint_path
        if checkpoint is not None:
            path = PurePosixPath(checkpoint)
            if (not checkpoint or '\\' in checkpoint or ':' in checkpoint
                    or path.is_absolute()
                    or any(part in {'', '.', '..'}
                           for part in checkpoint.split('/'))):
                raise ValueError('Expected a relative published checkpoint path')
        if checkpoint is None and any(
                outcome.status == 'complete'
                for outcome in (self.result.cohorts, self.result.samples)):
            raise ValueError('Completed producers require checkpoint identity')
        return self


def write_record(root: str, record: PredictionExplanationManifest) -> None:
    record.validate_record()
    filesystem.makedirs(f'{root}/xai', exist_ok=True)
    filesystem.atomic_write_bytes(
        f'{root}/{MANIFEST_PATH}',
        (record.model_dump_json(indent=2) + '\n').encode())


def read_record(root: str) -> PredictionExplanationManifest:
    with filesystem.open_file(f'{root}/{MANIFEST_PATH}', 'rt') as source:
        return PredictionExplanationManifest.model_validate_json(source.read())
