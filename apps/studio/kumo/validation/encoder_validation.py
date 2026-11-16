# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.validation.validation import ValidationException


class TableNotInGraphError(ValidationException):
    r"""
    Thrown when a table is queried for in a Graph object but it doesn't exist.
    """


class ColumnNotInTableError(ValidationException):
    r"""
    Thrown when a column is queried for in a Table object but it doesn't exist.
    """
