# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io_utils import kumo_dataclass


@kumo_dataclass
class FormatterMetadata:
    """Serves as the Metadata base class for kumo's YAML Formatter.
    These fields help specify the name of the key and whether or not this
    field is required in the YAML file for it to be valid"""
    name: str
    required: bool


@kumo_dataclass
class FieldRequirement:
    """Whether a property of a field (has aggregation, has condition)
    is allowed and required"""
    required: bool
    allowed: bool
