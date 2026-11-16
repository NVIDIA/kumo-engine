# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch declarations and fitting, also available through :mod:`kumo.pytorch.api`."""

from kumo.pytorch.contract.batch import *  # noqa: F403
from kumo.pytorch.contract.batch import __all__ as _batch_api


def prepare_batch_spec(*args, **kwargs):
    """Fit a batch specification against explicitly supplied sampled values."""
    from kumo.pytorch.assembly import prepare_batch_spec as prepare

    return prepare(*args, **kwargs)


__all__ = [*_batch_api, "prepare_batch_spec"]
