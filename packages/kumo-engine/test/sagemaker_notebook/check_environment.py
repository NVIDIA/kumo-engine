# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check maintainer approval and branch restrictions for SageMaker tests."""
import importlib.util
from pathlib import Path

from config import ENVIRONMENT


def main():
    path = Path(
        __file__).parent.parent / 'snowflake_notebook/check_environment.py'
    spec = importlib.util.spec_from_file_location('protected_environment', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.main(environment=ENVIRONMENT,
               branch='egu/sagemaker-platform-validation')


if __name__ == '__main__':
    main()
