# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Union

import numpy as np
import pandas as pd


def to_python_scalar(obj: Any) -> Any:
    r"""Convert a numpy object to Python scalar values."""
    return obj.item() if isinstance(obj, np.generic) else obj


def to_python_list(obj: List[Any]) -> List[Any]:
    r"""Convert a list of numpy objects to a list of Python scalar values."""
    if len(obj) == 0:
        return []
    test_item = obj[0]
    return [o.item() if isinstance(test_item, np.generic) else o for o in obj]


def to_timestamp_int(time: Union[str, pd.Timestamp]) -> int:
    r"""Convert a datetime string or pandas Timestamp to int."""
    if isinstance(time, pd.Timestamp):
        return time.value
    try:
        return pd.Timestamp(time).value
    except ValueError:
        raise ValueError(f'Could not convert {time} to timestamp')


def normalize_timestamp_to_utc_local(ts):
    """Convert a timestamp to UTC, handling both timezone-aware and naive timestamps."""
    if ts.tz is None:
        # Assume naive timestamps are UTC
        return ts.tz_localize('UTC')
    else:
        # Convert to UTC
        return ts.tz_convert('UTC')
