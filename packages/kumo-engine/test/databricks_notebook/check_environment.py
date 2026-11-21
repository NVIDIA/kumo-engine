# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require manual dispatch and the configured Databricks test policy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowflake_notebook.check_environment import main  # noqa: E402

if __name__ == '__main__':
    main(environment='databricks-mlops-test')
