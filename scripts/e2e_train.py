#!/usr/bin/env python3
"""E2E training scaffold for diskgraph sampling + Kumo trainer.

This script intentionally starts as a schema/bootstrap scaffold. Subsequent
commits fill in feature-store construction and trainer execution.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import pyarrow.flight as flight


EdgeType = Tuple[str, str, str]


def fetch_schema_mapping(server_address: str) -> dict:
    client = flight.connect(server_address)
    result = list(client.do_action(flight.Action("get_schema_mapping", b"")))
    if not result:
        raise RuntimeError("Gateway returned no schema mapping payload")
    return json.loads(result[0].body.to_pybytes())


def edge_types_from_mapping(mapping: dict) -> List[EdgeType]:
    return [
        (rel["src_type"], rel["name"], rel["dst_type"])
        for rel in mapping.get("edge_relations", [])
        if rel.get("name")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diskgraph E2E training script")
    parser.add_argument(
        "--server",
        type=str,
        default="grpc://localhost:8815",
        help="Arrow Flight server address",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory with social_sampling_50m parquet tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = fetch_schema_mapping(args.server)
    node_types = mapping.get("node_types", [])
    edge_types = edge_types_from_mapping(mapping)

    print(f"Gateway node types: {node_types}")
    print(f"Gateway edge types ({len(edge_types)}):")
    for edge_type in edge_types:
        print(f"  - {edge_type}")

    print("Schema bootstrap complete. Training pipeline wiring follows in next commit.")


if __name__ == "__main__":
    main()
