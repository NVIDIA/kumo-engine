# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed errors for schema, leakage, and composition failures."""


class TfmError(Exception):
    """Base error for the tfm package."""


class SchemaError(TfmError):
    """Event or batch schema is invalid."""


class TemporalLeakageError(TfmError):
    """A split, label, or feature would leak future information."""


class IncompatibleCompositionError(TfmError):
    """Public composition components cannot be combined."""


class MissingLabelError(TfmError):
    """Required labels are absent or not yet available."""


class InsufficientLabelSupportError(MissingLabelError):
    """Too few labeled entities to split a probe without reusing one.

    Raised instead of overlapping the train and evaluation sides, which would
    score a probe on rows it was fit on (ADR 0008).
    """


class UnsupportedCombinationError(TfmError):
    """Requested configuration is not supported in this release."""


class ExtraUnavailableError(TfmError):
    """An optional extra (RAPIDS, HuggingFace, ...) is not installed."""
