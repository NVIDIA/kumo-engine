# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monolith-owned trainer and data-module integration."""

from kumo_core.io.filesystem import register_fsspec_fallback

from io_utils.fsutil.compat_fsspec import fsspec_mangle

register_fsspec_fallback(fsspec_mangle)
