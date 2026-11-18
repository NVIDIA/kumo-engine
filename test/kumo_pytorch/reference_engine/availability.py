# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inclusive root availability shared by history, graph and length execution."""

import pyarrow as pa
import pyarrow.compute as pc

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.data.records import timestamp_to_epoch

from .typed import typed_array


def check_root_availability(spec, fetched, cutoffs):
    if spec.entity.available_from:
        available = fetched.column(spec.entity.available_from)
        if pa.types.is_timestamp(available.type):
            available = pc.cast(
                typed_array(available, DType("timestamp", unit="s", timezone="UTC")),
                pa.int64(),
            )
        for value, cutoff in zip(available.to_pylist(), cutoffs, strict=True):
            if value is None:
                raise ValueError("root availability timestamp is missing")
            if timestamp_to_epoch(value) > cutoff:
                raise ValueError(
                    "entity did not exist at or before the requested sample time"
                )
