# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require approved-branch dispatch and independent maintainer approval."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowflake_notebook import check_environment  # noqa: E402


def main():
    check_environment.main(environment='sagemaker-mlops-test')


if __name__ == '__main__':
    main()
