# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic restaurant files locally; retained filename supports existing commands."""
import argparse
import runpy
from pathlib import Path


def download(destination):
    generator = Path(__file__).resolve(
    ).parents[4] / 'synthetic-data/engine_hospitality/generate.py'
    return runpy.run_path(str(generator))['generate']('restaurant', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    download(args.destination)
