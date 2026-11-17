# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One reproducible local command for the approved examples and generated properties."""

from __future__ import annotations

import argparse
import os

import pytest

from conformance.cases import ENGINES, REPOSITORY
from conformance.properties import seed_batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("in_memory", "diskgraph", "all"), default="in_memory")
    parser.add_argument(
        "--seeds",
        type=int,
        help="also run generated seed batches SEED_START .. SEED_START+SEEDS-1 per engine, on "
        "top of the required ones (nightly); default: the required batches only",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first extra seed batch; the nightly run derives it from the date",
    )
    args = parser.parse_args()
    if args.seeds is not None:
        for engine in ENGINES if args.engine == "all" else (args.engine,):
            try:
                seed_batches(engine, args.seeds, args.seed_start)
            except ValueError as error:
                parser.error(str(error))
    os.chdir(REPOSITORY)
    print(f"Checking goldens, generated properties and corpus on: {args.engine}", flush=True)
    if args.engine == "in_memory":
        print("DiskGraph is not executed by this command; use --engine all for both.", flush=True)
    return pytest.main(
        [
            "-q",
            "-rA",
            "tests/conformance",
            f"--conformance-engine={args.engine}",
            "--conformance-complete",
            *([f"--conformance-seeds={args.seeds}"] if args.seeds is not None else []),
            f"--conformance-seed-start={args.seed_start}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
