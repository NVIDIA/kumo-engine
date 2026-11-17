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
    parser.add_argument("--engine", choices=("diskgraph", "all"), default="diskgraph")
    parser.add_argument(
        "--no-in-memory",
        action="store_true",
        help="Run behavior obligations without harness self-tests",
    )
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
    if args.no_in_memory and args.engine != "diskgraph":
        parser.error("--no-in-memory requires --engine diskgraph")
    if args.seeds is not None:
        for engine in ENGINES if args.engine == "all" else (args.engine,):
            try:
                seed_batches(engine, args.seeds, args.seed_start)
            except ValueError as error:
                parser.error(str(error))
    os.chdir(REPOSITORY)
    print(f"Checking goldens, generated properties and corpus on: {args.engine}", flush=True)
    if args.no_in_memory:
        print(
            "InMemory execution is forbidden; harness self-tests "
            "are outside this run. DiskGraph behavior obligations remain required.",
            flush=True,
        )
    return pytest.main(
        [
            "-q",
            "-rA",
            "tests/conformance",
            f"--conformance-engine={args.engine}",
            "--conformance-complete",
            *(["--conformance-no-in-memory"] if args.no_in_memory else []),
            *([f"--conformance-seeds={args.seeds}"] if args.seeds is not None else []),
            f"--conformance-seed-start={args.seed_start}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
