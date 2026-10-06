# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from typing import List, Optional, Tuple

from pydantic.dataclasses import dataclass

from kumoapi.common import StrEnum
from kumoapi.typing import _FORBID_EXTRA_CONFIG


class TrainingStage(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    PRED = 'pred'

    @staticmethod
    def list_stages() -> List['TrainingStage']:
        return [TrainingStage.TRAIN, TrainingStage.VAL, TrainingStage.TEST]

    def add_stage_prefix(self, name: str) -> str:
        return f'{self.value}_{name}'

    @classmethod
    def split_staged_name(cls, name: str) -> Tuple['TrainingStage', str]:
        stage, suffix = name.split('_', maxsplit=1)
        return cls(stage), suffix


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TrainingTableSpec:
    """Specifies the modifications made
    to a training table generated from a predictive query.

    Args:
        weight_col: Weight column used for loss computation.
        split_col: Column holding each row's split assignment, valued
            ``0`` for train, ``1`` for validation and ``2`` for test. The
            column is read out of the supplied table by name, routes the row,
            and is dropped from the seed output. ``None`` leaves the split to
            the generation plan.
    """
    weight_col: Optional[str] = None
    split_col: Optional[str] = None
