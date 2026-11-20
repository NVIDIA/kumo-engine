# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an engine-owned benchmark adapter and write readable/structured reports."""

from .benchmarking import run
from .benchmarking.cli import main as run_cli


def main(argv=None):
    return run_cli(run, argv, scope="worker", description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
