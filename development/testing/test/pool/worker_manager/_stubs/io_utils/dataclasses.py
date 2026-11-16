# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub: kumo_dataclass → dataclasses.dataclass."""
from dataclasses import dataclass as _stdlib_dataclass


def kumo_dataclass(cls=None, **kwargs):
    """Drop-in replacement for io_utils.kumo_dataclass.

    Real version is pydantic.dataclasses.dataclass with monkey-patches
    for underscored fields. For runtime tests we just want field access
    + repr; stdlib dataclass is enough.
    """
    if cls is None:
        return lambda c: _stdlib_dataclass(c, **kwargs)
    return _stdlib_dataclass(cls, **kwargs)
