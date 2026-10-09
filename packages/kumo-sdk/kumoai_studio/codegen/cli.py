# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI interface for Kumo SDK code generation utility."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kumoai_studio.codegen import generate_code

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for kumo-codegen command."""
    parser = argparse.ArgumentParser(
        description="Generate Python SDK code from Kumo entities",
        epilog="""
Examples:
  kumo-codegen --id myconnector --entity-class S3Connector
  kumo-codegen --id trainingjob-abc123
  kumo-codegen --id myconnector --entity-class S3Connector -o output.py
  kumo-codegen --json config.json -o output.py
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--id", help="Entity ID to generate code for")
    input_group.add_argument("--json", type=Path,
                             help="JSON file with entity specification")

    parser.add_argument(
        "--entity-class",
        help="Entity class for ID mode (e.g., S3Connector, TrainingJob)",
        type=str,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    # Build input_spec based on mode
    if args.id:
        input_spec = {"id": args.id}
        if args.entity_class:
            input_spec["entity_class"] = args.entity_class

        if args.verbose:
            entity_info = f"ID: {args.id}"
            if args.entity_class:
                entity_info += f", Class: {args.entity_class}"
            logger.info(f"Generating code for {entity_info}")

    else:
        if args.verbose:
            logger.info(f"Using JSON mode with file: {args.json}")

        try:
            with open(args.json, "r") as f:
                json_data = json.load(f)
            input_spec = {"json": json_data}
        except Exception as e:
            logger.error(f"Error reading JSON file {args.json}: {e}")
            sys.exit(1)

    try:
        output_path = str(args.output) if args.output else None
        code = generate_code(input_spec, output_path=output_path)

        if args.verbose and args.output:
            logger.info(f"Code written to {args.output}")
        elif not args.output:
            print(code, end="")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
