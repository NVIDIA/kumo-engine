# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.types import ColStatType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.typing import Dtype, Stype

from kumo.stats.stats import Stats
from kumo.stats.utils import col_stat_dict_complete


def test_col_stat_dict_complete_returns_false_for_missing_stat():
    col = BaseColumnConfig('embedding', dtype=Dtype.floatlist,
                           stype=Stype.sequence)
    result = {
        col: {
            ColStatType.SEQUENCE_MIN_LENGTH:
            Stats(value=3, approx=False, data_version='test')
        }
    }
    request = {
        col: [
            ColStatType.SEQUENCE_MIN_LENGTH,
            ColStatType.SEQUENCE_MAX_LENGTH,
        ]
    }

    assert not col_stat_dict_complete(result, request)
