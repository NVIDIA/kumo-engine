# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""Schema comparison logic.

This module compares two FullSchema objects and produces a ComparisonResult
with all detected changes.
"""

from typing import List

from tools.schema_compat.models import (
    ChangeType,
    ComparisonResult,
    FullSchema,
    SchemaChange,
)
from tools.schema_compat.rules import (
    compare_dataclass_schemas,
    compare_enum_schemas,
)


def compare_schemas(
    old_schema: FullSchema,
    new_schema: FullSchema,
) -> ComparisonResult:
    """Compare two schema versions and detect all changes.

    Args:
        old_schema: The baseline (older) schema
        new_schema: The new schema to compare against baseline

    Returns:
        ComparisonResult with all detected changes
    """
    changes: List[SchemaChange] = []

    # Compare dataclasses
    old_dataclasses = old_schema.dataclasses
    new_dataclasses = new_schema.dataclasses

    # Check for removed dataclasses (breaking)
    for name in old_dataclasses:
        if name not in new_dataclasses:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.MODEL_REMOVED,
                    model_name=name,
                ))

    # Check for added dataclasses (non-breaking)
    for name in new_dataclasses:
        if name not in old_dataclasses:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.MODEL_ADDED,
                    model_name=name,
                ))

    # Compare existing dataclasses
    for name in old_dataclasses:
        if name in new_dataclasses:
            old_dc = old_dataclasses[name]
            new_dc = new_dataclasses[name]
            changes.extend(compare_dataclass_schemas(old_dc, new_dc))

    # Compare enums
    old_enums = old_schema.enums
    new_enums = new_schema.enums

    # Check for removed enums (breaking)
    for name in old_enums:
        if name not in new_enums:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.MODEL_REMOVED,
                    model_name=name,
                ))

    # Check for added enums (non-breaking)
    for name in new_enums:
        if name not in old_enums:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.MODEL_ADDED,
                    model_name=name,
                ))

    # Compare existing enums
    for name in old_enums:
        if name in new_enums:
            old_enum = old_enums[name]
            new_enum = new_enums[name]
            changes.extend(compare_enum_schemas(old_enum, new_enum))

    return ComparisonResult(
        old_version=old_schema.version,
        new_version=new_schema.version,
        changes=changes,
    )


def compare_multiple_versions(
    new_schema: FullSchema,
    baseline_schemas: List[FullSchema],
) -> List[ComparisonResult]:
    """Compare a new schema against multiple baseline versions.

    Args:
        new_schema: The new schema to validate
        baseline_schemas: List of baseline schemas to compare against

    Returns:
        List of ComparisonResult, one per baseline version
    """
    results = []
    for baseline in baseline_schemas:
        result = compare_schemas(baseline, new_schema)
        results.append(result)
    return results


def aggregate_breaking_changes(
        results: List[ComparisonResult]) -> List[SchemaChange]:
    """Aggregate all breaking changes from multiple comparisons.

    Returns unique breaking changes (some changes may appear in multiple
    comparisons if they break compatibility with multiple versions).
    """
    seen = set()
    breaking_changes = []

    for result in results:
        for change in result.breaking_changes:
            # Create a unique key for deduplication
            key = (
                change.change_type,
                change.model_name,
                change.field_name,
                change.old_value,
                change.new_value,
            )
            if key not in seen:
                seen.add(key)
                breaking_changes.append(change)

    return breaking_changes


def format_multi_version_report(
    new_version: str,
    results: List[ComparisonResult],
) -> str:
    """Format a human-readable report for multiple version comparisons.

    Args:
        new_version: The version being validated
        results: List of comparison results against baseline versions

    Returns:
        Formatted string report
    """
    lines = []
    lines.append(f"Schema Compatibility Report for v{new_version}")
    lines.append("=" * 70)

    has_any_breaking = any(r.has_breaking_changes for r in results)
    has_any_warnings = any(r.has_warnings for r in results)

    if not has_any_breaking and not has_any_warnings:
        lines.append("")
        lines.append("All compatibility checks passed!")
        lines.append("")
        for result in results:
            lines.append(f"  [PASS] Compatible with v{result.old_version}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("RESULT: PASSED")
        return "\n".join(lines)

    # Report breaking changes per version
    if has_any_breaking:
        lines.append("")
        lines.append("BREAKING CHANGES DETECTED")
        lines.append("-" * 70)

        for result in results:
            if result.has_breaking_changes:
                lines.append("")
                lines.append(f"Incompatible with v{result.old_version}:")
                for change in result.breaking_changes:
                    lines.append(f"  [BREAKING] {change.format_message()}")

    # Report warnings
    if has_any_warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.append("-" * 70)

        # Deduplicate warnings across versions
        seen_warnings = set()
        for result in results:
            for change in result.warnings:
                msg = change.format_message()
                if msg not in seen_warnings:
                    seen_warnings.add(msg)
                    lines.append(f"  [WARNING] {msg}")

    lines.append("")
    lines.append("=" * 70)

    if has_any_breaking:
        lines.append("RESULT: FAILED - Breaking changes detected")
        lines.append("")
        lines.append("To proceed, either:")
        lines.append("  1. Fix the breaking changes to maintain compatibility")
        lines.append("  2. Bump the major version number for breaking release")
    else:
        lines.append("RESULT: PASSED with warnings")

    return "\n".join(lines)
