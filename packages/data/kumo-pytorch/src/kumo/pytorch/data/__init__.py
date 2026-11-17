# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime record types: the values an engine produces, not the contract.

`BuiltSample` and everything the contract declares live in
`kumo.pytorch.contract`. This package holds what those declarations are made
*of* at run time: typed field values, events, and entity histories.
"""

from kumo.pytorch.data.records import (
    DatedValue,
    EntityHistory,
    EntityState,
    Event,
    FieldValue,
    Scalar,
    cutoff_to_epoch,
    timestamp_to_epoch,
)

__all__ = [
    "DatedValue",
    "EntityHistory",
    "EntityState",
    "Event",
    "FieldValue",
    "Scalar",
    "cutoff_to_epoch",
    "timestamp_to_epoch",
]
