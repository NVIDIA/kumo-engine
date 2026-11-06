# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse the maintainer approval and feature-branch protection checks."""
import importlib.util
from pathlib import Path

from config import BRANCH, ENVIRONMENT


def main():
    path = Path(
        __file__).parent.parent / 'snowflake_notebook/check_environment.py'
    spec = importlib.util.spec_from_file_location('protected_environment', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.main(environment=ENVIRONMENT, branch=BRANCH)


if __name__ == '__main__':
    main()
