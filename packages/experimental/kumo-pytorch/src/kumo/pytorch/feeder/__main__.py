# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve batches for `DistributedEngine` drivers; the contract arrives from the driver as data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from kumo.pytorch.engines.distributed import wire
from kumo.pytorch.engines.distributed.server import FeederServer, watch_parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m kumo.pytorch.feeder")
    parser.add_argument("--listen", default="0.0.0.0:8815", help="HOST:PORT; port 0 picks one")
    parser.add_argument("--parent-pid", type=int, help="exit when this process is gone")
    args = parser.parse_args(argv)
    host, _, port = args.listen.rpartition(":")
    if args.parent_pid is not None:
        watch_parent(args.parent_pid)
    server = FeederServer(f"grpc://{host}:{port}")
    print(f"{wire.LISTENING} grpc://{host}:{server.port}", flush=True)
    sys.stdout = sys.stderr
    server.serve()


if __name__ == "__main__":
    main()
