# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-facing training errors without platform exception dependencies."""

from __future__ import annotations

from kumo_core.errors import PublicError


class PublicTrainingError(PublicError):
    """User-facing validation failure raised by numerical training."""
