"""Shared Kumo factory seam for arena-owned DiskGraph-scope engines."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.dataset import Dataset, TableSource
from kumo.pytorch.engine import AnchorTask
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue

from .validation import contract_issues


def positive_threads(value: int | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError("num_threads must be a positive integer or None")
    return value


class EntityEngine:
    """Native factories retain Kumo ownership of fitting, encoding and assembly."""

    name: str
    implementation_stage = "diskgraph_scope"

    def configure(self, *, index, ingest):
        if ingest not in {"if_missing", "never"}:
            raise ValueError("ingest must be 'if_missing' or 'never'")
        if index is None and ingest == "never":
            raise ValueError("ingest='never' requires an index directory")
        self.index = None if index is None else Path(index)
        self.ingest = ingest

    def check_sources(
        self, sources: Mapping[str, TableSource], path: str
    ) -> tuple[ValidationIssue, ...]:
        """Refuse unsupported source bindings before Kumo reads metadata."""
        issues = []
        for name, source in sources.items():
            location = f"{path}.{name}" if name else path
            if source.kind != "parquet":
                issues.append(
                    ValidationIssue(
                        location,
                        f"engine {self.name!r} requires local Parquet, got {source.kind!r}",
                        "write this source as local Parquet",
                    )
                )
            elif urlsplit(source.path).scheme or "://" in source.path:
                issues.append(
                    ValidationIssue(
                        location + ".path",
                        f"local Parquet paths are required, got {source.path!r}",
                        "stage the source as local Parquet files",
                    )
                )
        return tuple(issues)

    def check_contract(self, contract: DataLoaderSpec) -> tuple[ValidationIssue, ...]:
        return contract_issues(contract)

    @staticmethod
    def check_anchors(table: AnchorTable) -> tuple[ValidationIssue, ...]:
        from .parquet import check_parquet_anchors

        return check_parquet_anchors(table)

    def materialize_anchors(self, table: AnchorTable) -> AnchorTask:
        from .anchors import ParquetAnchors

        return ParquetAnchors(table)

    @staticmethod
    def validate_binding(dataset: Dataset, schema: DataSchema) -> None:
        declared = {table.name for table in schema.tables}
        if set(dataset.tables) != declared:
            raise ValueError("dataset tables must exactly match the declared schema")
        for name, source in dataset.tables.items():
            if source.kind != "parquet":
                raise TypeError(
                    f"table {name!r} requires local Parquet, got {source.kind!r}"
                )
            if source.columns:
                raise ValueError(
                    f"table {name!r}: dataset column maps are outside the selected engine scope"
                )
