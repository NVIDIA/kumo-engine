# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Errors raised while opening retained training explanations."""


class ExplanationUnavailableError(FileNotFoundError):
    """The requested explanation artifact was not retained or produced."""


class InvalidExplanationError(ValueError):
    """A retained explanation artifact violates its data contract."""


class InvalidRunError(ValueError):
    """A path does not identify one supported retained training run."""
