# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the `kumo.pytorch.datasets.retail` generator.

python -m e2e.datasets.hm --scale S --seed 0 --out /tmp/hm-s
"""

from kumo.pytorch.datasets.retail import *  # noqa: F403
from kumo.pytorch.datasets.retail import main

if __name__ == "__main__":
    main()
