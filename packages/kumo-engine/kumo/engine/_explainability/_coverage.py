# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect when Kumo evaluated or explained only the head of the test split.

Kumo stops test evaluation after ``max_test_steps`` batches, and its
explanation pass replays at most a fixed number of batches at a quarter of the
evaluation batch size. When either cap leaves rows out, the affected cards
carry one fixed line; when no cap applies, they carry nothing.
"""

import posixpath
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from kumo.engine._explainability import _files
from kumo.engine._explainability._format import count


@dataclass(frozen=True)
class Coverage:
    """Test rows behind each family of cards, when fewer than the split."""

    test_rows: Optional[int]
    metrics: Optional[int] = None
    holdout: Optional[int] = None
    explanation: Optional[int] = None
    holdout_truncated: bool = False

    def line(self, covered: Optional[int]) -> Optional[str]:
        if covered is None or self.test_rows is None:
            return None
        if covered >= self.test_rows:
            return None
        return (f'based on the first {count(covered)} of '
                f'{count(self.test_rows)} test rows')

    def holdout_line(self) -> Optional[str]:
        line = self.line(self.holdout)
        if self.holdout_truncated:
            return (
                line + '; retained holdout is truncated' if line else
                'Based on a truncated retained holdout; full coverage is unavailable.'
            )
        return line


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def eval_batch_size(files: _files.RunFiles) -> Optional[int]:
    """The published model's evaluation batch size, as the trainer resolves it."""
    config = _files.read_json(
        posixpath.join(files.bundle, 'model', 'config.json'))
    if not isinstance(config, dict):
        return None
    for key in ('eval_loader_kwargs', 'loader_kwargs'):
        kwargs = config.get(key)
        if isinstance(kwargs, dict) and _int(kwargs.get('batch_size')):
            return _int(kwargs['batch_size'])
    return None


def explanation_batches() -> int:
    from kumo_core.training.runtime import LocalExplainRuntime

    return int(LocalExplainRuntime.max_batches)


def holdout_seeds(frame: pd.DataFrame) -> int:
    """Test-table rows behind a holdout frame.

    A forecasting holdout carries one row per horizon step, so its test rows
    are its distinct entity and anchor pairs.
    """
    if 'ANCHOR_TIMESTAMP' in frame.columns and 'ENTITY' in frame.columns:
        return int(len(frame[['ENTITY', 'ANCHOR_TIMESTAMP']].drop_duplicates()))
    return int(len(frame))


def detect(files: _files.RunFiles,
           holdout: Optional[pd.DataFrame] = None) -> Coverage:
    from kumo.engine.trainer._spec import HOLDOUT_META_ROLE

    stats = _files.split_stats(files)
    test = (stats or {}).get('test_table_stats') or {}
    test_rows = _int(test.get('num_train_rows'))
    seeds = None if holdout is None else holdout_seeds(holdout)
    meta = _files.read_json(_files.bundle_file(files, HOLDOUT_META_ROLE))
    truncated = (files.holdout_truncated is True
                 or isinstance(meta, dict) and meta.get('truncated') is True)
    if test_rows is None:
        return Coverage(None, holdout=seeds, holdout_truncated=truncated)
    batch = eval_batch_size(files)
    optimization = _files.plan(files).get('optimization') or {}
    steps = _int(optimization.get('max_test_steps'))
    metrics = None
    if batch is not None and steps is not None and steps > 0:
        metrics = min(test_rows, steps * batch)
    explanation = None
    if batch is not None:
        explanation = min(test_rows, explanation_batches() * max(batch // 4, 1))
    return Coverage(test_rows, metrics, seeds, explanation, truncated)
