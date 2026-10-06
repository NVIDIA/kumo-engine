# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""Command-line interface for schema compatibility checker.

Usage:
    python -m tools.schema_compat dump [--output FILE]
    python -m tools.schema_compat compare --old FILE --new FILE
    python -m tools.schema_compat check-release --version VERSION
    python -m tools.schema_compat upload --version VERSION
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from tools.schema_compat.comparator import (
    compare_multiple_versions,
    compare_schemas,
    format_multi_version_report,
)
from tools.schema_compat.config import PYDANTIC_VERSION
from tools.schema_compat.extractor import extract_schema
from tools.schema_compat.models import FullSchema
from tools.schema_compat.storage import (
    fetch_schema,
    get_baseline_versions,
    upload_schema,
)


def cmd_dump(args: argparse.Namespace) -> int:
    """Dump current schema to JSON."""
    version = args.version or "0.0.0"  # Placeholder if not specified

    print(f"Extracting schema for kumoapi (Pydantic v{PYDANTIC_VERSION})...",
          file=sys.stderr)

    schema = extract_schema(version)

    output = json.dumps(schema.to_dict(), indent=2)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Schema written to {args.output}", file=sys.stderr)
    else:
        print(output)

    print(
        f"Extracted {len(schema.dataclasses)} dataclasses and "
        f"{len(schema.enums)} enums", file=sys.stderr)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two schema files."""
    old_path = Path(args.old)
    new_path = Path(args.new)

    if not old_path.exists():
        print(f"Error: Old schema file not found: {old_path}", file=sys.stderr)
        return 1

    if not new_path.exists():
        print(f"Error: New schema file not found: {new_path}", file=sys.stderr)
        return 1

    old_data = json.loads(old_path.read_text())
    new_data = json.loads(new_path.read_text())

    old_schema = FullSchema.from_dict(old_data)
    new_schema = FullSchema.from_dict(new_data)

    result = compare_schemas(old_schema, new_schema)

    print(result.format_report())

    return 1 if result.has_breaking_changes else 0


def cmd_check_release(args: argparse.Namespace) -> int:
    """Check if a new release is backward compatible."""
    version = args.version

    print(
        f"Checking schema compatibility for v{version} "
        f"(Pydantic v{PYDANTIC_VERSION})", file=sys.stderr)
    print("", file=sys.stderr)

    # Extract current schema
    print("Extracting current schema...", file=sys.stderr)
    current_schema = extract_schema(version)
    print(
        f"  Found {len(current_schema.dataclasses)} dataclasses and "
        f"{len(current_schema.enums)} enums", file=sys.stderr)

    # Get baseline versions to compare against
    print("", file=sys.stderr)
    print("Fetching baseline versions from S3...", file=sys.stderr)

    baselines = get_baseline_versions(version, PYDANTIC_VERSION)

    if not baselines:
        print(
            "  No baseline versions found. This appears to be the first "
            "version with schema tracking.", file=sys.stderr)
        print("", file=sys.stderr)
        print("RESULT: PASSED (no baselines to compare against)",
              file=sys.stderr)
        return 0

    versions_str = ', '.join(baselines)
    print(f"  Found {len(baselines)} baseline version(s): {versions_str}",
          file=sys.stderr)

    # Fetch baseline schemas
    baseline_schemas: List[FullSchema] = []
    for baseline_version in baselines:
        print(f"  Fetching v{baseline_version}...", file=sys.stderr)
        schema = fetch_schema(baseline_version, PYDANTIC_VERSION)
        if schema:
            baseline_schemas.append(schema)
        else:
            print(f"    Warning: Schema not found for v{baseline_version}",
                  file=sys.stderr)

    if not baseline_schemas:
        print("", file=sys.stderr)
        print("Warning: No baseline schemas could be fetched.",
              file=sys.stderr)
        print("RESULT: PASSED (no baselines to compare against)",
              file=sys.stderr)
        return 0

    # Compare against all baselines
    print("", file=sys.stderr)
    print("Comparing schemas...", file=sys.stderr)

    results = compare_multiple_versions(current_schema, baseline_schemas)

    # Print report
    print("", file=sys.stderr)
    report = format_multi_version_report(version, results)
    print(report)

    # Return non-zero if any breaking changes
    has_breaking = any(r.has_breaking_changes for r in results)
    return 1 if has_breaking else 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload current schema to S3."""
    version = args.version

    print(
        f"Extracting and uploading schema for v{version} "
        f"(Pydantic v{PYDANTIC_VERSION})...", file=sys.stderr)

    schema = extract_schema(version)

    print(
        f"  Extracted {len(schema.dataclasses)} dataclasses and "
        f"{len(schema.enums)} enums", file=sys.stderr)

    s3_uri = upload_schema(schema)

    print(f"  Uploaded to {s3_uri}", file=sys.stderr)
    print("", file=sys.stderr)
    print("DONE", file=sys.stderr)

    return 0


def cmd_list_versions(args: argparse.Namespace) -> int:
    """List available schema versions in S3."""
    from tools.schema_compat.storage import list_available_versions

    pydantic_ver = args.pydantic_version or PYDANTIC_VERSION

    print(f"Available schema versions (Pydantic v{pydantic_ver}):",
          file=sys.stderr)

    versions = list_available_versions(pydantic_ver)

    if not versions:
        print("  (none)", file=sys.stderr)
    else:
        for v in versions:
            print(f"  v{v}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="schema_compat",
        description="Schema backward compatibility checker for kumo-api",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # dump command
    dump_parser = subparsers.add_parser(
        "dump",
        help="Dump current schema to JSON",
    )
    dump_parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )
    dump_parser.add_argument(
        "--version",
        "-v",
        help="Version string to embed in schema",
        default="0.0.0",
    )

    # compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two schema files",
    )
    compare_parser.add_argument(
        "--old",
        required=True,
        help="Path to old/baseline schema JSON",
    )
    compare_parser.add_argument(
        "--new",
        required=True,
        help="Path to new schema JSON",
    )

    # check-release command
    check_parser = subparsers.add_parser(
        "check-release",
        help="Check if a release is backward compatible",
    )
    check_parser.add_argument(
        "--version",
        "-v",
        required=True,
        help="Version to check (e.g., 0.52.3)",
    )

    # upload command
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload current schema to S3",
    )
    upload_parser.add_argument(
        "--version",
        "-v",
        required=True,
        help="Version string (e.g., 0.52.3)",
    )

    # list-versions command
    list_parser = subparsers.add_parser(
        "list-versions",
        help="List available schema versions in S3",
    )
    list_parser.add_argument(
        "--pydantic-version",
        type=int,
        help="Pydantic version to filter by (1 or 2)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "dump": cmd_dump,
        "compare": cmd_compare,
        "check-release": cmd_check_release,
        "upload": cmd_upload,
        "list-versions": cmd_list_versions,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
