#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve relative benchmark paths against a chosen local or cloud data root."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the source diskgraph config JSON")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Path to write rewritten config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    for table in cfg.get("tables", []):
        src = table.get("source", {})
        paths = src.get("paths", [])
        src["paths"] = [
            args.data_root.rstrip("/") + "/" + p for p in paths
        ]

    with open(args.output, "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
