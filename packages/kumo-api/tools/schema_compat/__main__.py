# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point for running as a module: python -m tools.schema_compat"""

import sys

from tools.schema_compat.cli import main

if __name__ == "__main__":
    sys.exit(main())
