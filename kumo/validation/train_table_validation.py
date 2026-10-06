# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.validation.validation import ValidationException

TRAIN_TABLE_VALIDATION_RESPONSE_NAME = 'validation_response'
PRED_TABLE_VALIDATION_RESPONSE_NAME = 'validation_response'


class TrainingTableGenerationError(ValidationException):
    r"""Thrown when a valid training table failed to be generated due to
    predictive query parameters. e.g. A generated training table is empty
    due to target filter being too strict, etc.
    """


class TrainingTableSplitError(ValidationException):
    r"""Thrown when train table, validation table or test table is invalid
    after splitting the training table."""
