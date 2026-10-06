# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers.

Modules in this package are *opt-in* utilities for tests; the autouse
conftest fixture in :mod:`test.conftest` does NOT pull these in. Tests
that need them request the corresponding fixture explicitly.
"""
