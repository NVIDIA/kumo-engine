# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Parquet discovery and anchor-binding validation for arena engines.

Path discovery has no Kumo import side effects. Anchor validation imports the
selected Kumo runtime only when explicitly requested; no engine is constructed.
"""

from __future__ import annotations

from glob import glob, has_magic
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from kumo.pytorch.anchors import AnchorTable
    from kumo.pytorch.spec import ValidationIssue


def local_parquet_files(value: str) -> tuple[str, ...]:
    """Resolve a local file, recursive directory or file glob in canonical order.

    Direct/glob entries must be Parquet files. Directory discovery ignores
    unrelated files. File contents are checked by the selected reader, not here.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a local Parquet path must be a nonempty string")
    if urlsplit(value).scheme or "://" in value:
        raise ValueError(f"only local Parquet paths are supported, got {value!r}")
    path = Path(value).expanduser()
    if has_magic(str(path)):
        candidates = [Path(item) for item in glob(str(path), recursive=True)]
    elif path.is_dir():
        candidates = [
            item for item in path.rglob("*") if item.suffix.lower() == ".parquet"
        ]
    else:
        candidates = [path]
    if not candidates:
        raise FileNotFoundError(f"no local Parquet files match {value!r}")
    resolved = []
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"local Parquet file does not exist: {candidate}")
        if not candidate.is_file() or candidate.suffix.lower() != ".parquet":
            raise ValueError(f"expected a local .parquet file, got {str(candidate)!r}")
        resolved.append(str(candidate.resolve()))
    return tuple(sorted(set(resolved)))


def anchor_columns(table: AnchorTable) -> dict[str, str]:
    """Map required logical anchor names to physical Parquet columns."""
    physical = {logical: name for name, logical in table.source.columns.items()}
    return {name: physical.get(name, name) for name in table.columns}


def check_parquet_anchors(table: AnchorTable) -> tuple[ValidationIssue, ...]:
    """Check source kind, candidate exclusion and every file's required columns."""
    from kumo.pytorch.dataset import Parquet
    from kumo.pytorch.spec import ValidationIssue

    if not isinstance(table.source, Parquet):
        return (
            ValidationIssue(
                "anchors.source",
                "arena engines read anchor tables from local Parquet only",
                "write the anchors as local Parquet or supply explicit SampleRequest values",
            ),
        )
    if table.candidates:
        return (
            ValidationIssue(
                "anchors.candidates",
                "candidate anchor columns are outside the DiskGraph scope",
                "remove candidate columns for this engine comparison",
            ),
        )
    try:
        files = local_parquet_files(table.source.path)
    except (OSError, ValueError) as error:
        return (
            ValidationIssue(
                "anchors.source.path", str(error), "point at local Parquet files"
            ),
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    required = anchor_columns(table)
    issues = []
    for file in files:
        try:
            schema = pq.read_schema(file)
        except (OSError, pa.ArrowException) as error:
            issues.append(
                ValidationIssue(
                    "anchors.source.path",
                    f"cannot read anchor Parquet metadata for {file!r}: {error}",
                    "provide a readable, valid Parquet file",
                )
            )
            continue
        missing = sorted(
            logical
            for logical, physical in required.items()
            if physical not in schema.names
        )
        if missing:
            issues.append(
                ValidationIssue(
                    "anchors.columns",
                    f"anchor Parquet file {file!r} lacks columns {missing}",
                    "write the required columns or map physical names with Parquet(columns=...)",
                )
            )
        ambiguous = sorted(
            set(required.values())
            & {name for name in schema.names if schema.names.count(name) > 1}
        )
        if ambiguous:
            issues.append(
                ValidationIssue(
                    "anchors.columns",
                    f"anchor Parquet file {file!r} repeats required column names {ambiguous}",
                    "write one physical column for each required name",
                )
            )
    return tuple(issues)
