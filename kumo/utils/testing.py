# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def pytest_only(func):
    def wrapper(*args, **kwargs):
        if 'PYTEST_CURRENT_TEST' not in os.environ:
            raise Exception("This function can only be run under pytest")
        return func(*args, **kwargs)

    return wrapper
