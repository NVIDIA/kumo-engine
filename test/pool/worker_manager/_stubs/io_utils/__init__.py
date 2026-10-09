# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only stub for kumo's io_utils package.

The real io_utils transitively depends on logging_loki, pydantic, and
several other kumo-internal libraries. For our laptop-driven Block C
integration tests we only need the two symbols K8sPoolWorkerManager
imports — kumo_dataclass + logging.getLogger — so we provide minimal
stdlib-backed equivalents and skip the dep tree.
"""
from . import logging
from .dataclasses import kumo_dataclass

__all__ = ["kumo_dataclass", "logging"]
