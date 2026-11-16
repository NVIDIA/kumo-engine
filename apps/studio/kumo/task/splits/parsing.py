# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from io_utils.yaml import repr_to_object
from kumo.task.split import (
    ApproxDateOffsetSplit,
    DateOffsetSplit,
    ManualSplit,
    RandomLinkSplit,
    RandomSplit,
    Split,
    TemporalSplit,
    TimeRangeSplit,
)
from kumo.task.splits.time_split import TimeSplit

# This map is used to register the split classes in the model planner
classes = {
    'RandomSplit': RandomSplit,
    'RandomLinkSplit': RandomLinkSplit,
    'TemporalSplit': TemporalSplit,
    'DateOffsetSplit': DateOffsetSplit,
    'TimeRangeSplit': TimeRangeSplit,
    'ApproxDateOffsetSplit': ApproxDateOffsetSplit,
    'TimeSplit': TimeSplit,
    'ManualSplit': ManualSplit,
}


def get_split(value: str | None) -> Optional[Split]:
    """Convert a string representation of a split configuration into a
    Split object.
    Args:
        value (str | None): A string representation of a split
            configuration. If None, returns None.
    Returns:
        Optional[Split]: A Split object if the string can be parsed
            successfully, None if the input is None.
    Raises:
        ValueError: If the string cannot be parsed into a valid Split
            object, with details about the parsing error.
    """
    if value is None:
        return None

    try:
        obj = repr_to_object(value, classes)
        assert isinstance(obj, Split)
        return obj
    except Exception as e:
        raise ValueError(f'"{value}" is not a valid split. Error: {e}')
