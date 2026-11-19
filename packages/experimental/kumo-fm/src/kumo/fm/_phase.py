# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small phase boundary for shared assembly code."""
from contextlib import nullcontext


def phase(*_args, **_kwargs):
    return nullcontext()
