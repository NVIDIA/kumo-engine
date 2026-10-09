# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation checks shared by hosted and in-process execution."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
from kumoapi.common import KumoValidationError
from kumoapi.pquery.AST import DateOffsetRange

MAX_NUM_CLASSES = int(os.environ.get('MAX_NUM_CLASSES', '6000'))


class GenerationValidationError(ValueError):
    """A structured data-quality failure portable across execution modes."""
    def __init__(self, error: KumoValidationError) -> None:
        self.error = error
        super().__init__(f"{error.title}. {error.message}")


def validate_train_offsets(
    train_start_offset: pd.DateOffset | None,
    train_end_offset: pd.DateOffset | None,
) -> KumoValidationError | None:
    """Reject a training end offset at or beyond its start offset."""
    if train_start_offset is None or train_end_offset is None:
        return None
    now = datetime.now()
    if now + train_end_offset < now + train_start_offset:
        return None
    start = DateOffsetRange.date_offset_repr(train_start_offset)
    end = DateOffsetRange.date_offset_repr(train_end_offset)
    return KumoValidationError(
        'Train start/end offsets error',
        'Argument train_end_offset should be smaller than '
        'train_start_offset, but it is not: '
        f'train_start_offset={start}, train_end_offset={end}',
    )


def validate_anchor_time(
    anchor_date: pd.Timestamp,
    min_date: pd.Timestamp | None,
    target_table_name: str,
) -> KumoValidationError | None:
    """Reject anchors before source data begins; future dates are allowed."""
    if min_date is None or anchor_date >= min_date:
        return None
    return KumoValidationError(
        'Invalid Prediction Anchor Time',
        f'The prediction anchor time {anchor_date} is '
        'earlier than the start of your data range in the table '
        f'{target_table_name} ({min_date}). '
        'Please select an anchor time within the range of your '
        'data to ensure accurate predictions.',
    )


def validate_target_classes(
    num_classes: int,
    classes: Sequence[Any],
    *,
    classify: bool,
) -> list[KumoValidationError]:
    """Reject unsupported classification target cardinality."""
    if num_classes > MAX_NUM_CLASSES:
        message = ('Number of different target classes '
                   f'exceeds {MAX_NUM_CLASSES} (got {num_classes})')
        if classify:
            message += '. Consider using RANK instead of CLASSIFY.'
        return [KumoValidationError('Too many classes to predict', message)]
    if num_classes != 1:
        return []
    if not classes:
        return [
            KumoValidationError(
                'No positive samples',
                'The training table (before split) only contains negative '
                'samples. Loosen filters or widen the prediction window '
                'to include positive examples.',
            )
        ]
    return [
        KumoValidationError(
            'Too few classes to predict',
            'Training table (before split) has only one target value '
            f'({classes[0]}), so the prediction task is ill-defined. '
            'Adjust the query to produce multiple target values.',
        )
    ]
