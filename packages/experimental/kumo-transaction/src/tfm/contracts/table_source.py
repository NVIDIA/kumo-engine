# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-backed keyed tables, independently of model and retrieval backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tfm.contracts.relational import RelationalForeignKey, RelationalTable


@dataclass(frozen=True)
class TableSource:
    """One schema over source Parquet files; never a list of prediction windows.

    Tables use the same explicit keys, feature allowlists and observation clocks
    as ``RelationalDataset``. A dataset adapter may provide this declaration
    without constructing event objects. Only metadata is retained here. Paths
    are resolved locally; reading/validation belongs to the selected backend.

    Example::

        TableSource(
            tables=(RelationalTable("users", ("id",), immutable=True),),
            relationships=(),
            files={"users": Path("users.parquet")},
        )
    """

    tables: tuple[RelationalTable, ...]
    relationships: tuple[RelationalForeignKey, ...]
    files: Mapping[str, Path]
    graph_numeric: Mapping[str, Mapping[str, str]] | None = None
    """Optional graph-only allowlists of already encoded finite scalar columns.

    Unset uses numeric value/presence features. A supplied mapping explicitly
    chooses participating graph tables as well as their feature columns.
    """
    profile_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Optional point-in-time profile streams, separate from event histories."""
    event_labels: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    """Table -> (binary value, availability-us column); withheld task metadata."""
    future_numeric: Mapping[str, Mapping[str, str]] | None = None
    """Optional future-statistic name -> observed numeric alias per graph table."""
    observed_through: datetime | None = None
    ownership_versions: Path | None = None
    """Optional versioned-ownership facts; unsupported native rules fail explicitly."""

    def __post_init__(self) -> None:
        tables, roles = tuple(self.tables), tuple(self.relationships)
        names = tuple(table.name for table in tables)
        if not names or len(set(names)) != len(names):
            raise ValueError("Table sources require nonempty unique table names")
        if set(self.files) != set(names):
            raise ValueError("Source files must exactly match declared tables")
        if len({role.name for role in roles}) != len(roles):
            raise ValueError("Source relationship names must be unique")
        for role in roles:
            if role.parent not in names or role.child not in names:
                raise ValueError(f"Relationship {role.name!r} has an undeclared endpoint")
            parent = next(table for table in tables if table.name == role.parent)
            if len(role.columns) != len(parent.key):
                raise ValueError(f"Relationship {role.name!r} does not match its parent key")
        if self.graph_numeric is not None:
            if set(self.graph_numeric) - set(names):
                raise ValueError("Graph feature allowlists reference undeclared tables")
            object.__setattr__(
                self,
                "graph_numeric",
                MappingProxyType(
                    {
                        name: MappingProxyType(dict(fields))
                        for name, fields in self.graph_numeric.items()
                    }
                ),
            )
        if set(self.event_labels) - set(names):
            raise ValueError("Event label bindings reference undeclared tables")
        if self.future_numeric is not None:
            for name, fields in self.future_numeric.items():
                table = next((table for table in tables if table.name == name), None)
                if (
                    table is None
                    or table.role != "input"
                    or set(fields.values()) - set(table.numeric)
                ):
                    raise ValueError(
                        "Future statistics must reference declared observed numeric fields"
                    )
            object.__setattr__(
                self,
                "future_numeric",
                MappingProxyType(
                    {
                        name: MappingProxyType(dict(fields))
                        for name, fields in self.future_numeric.items()
                    }
                ),
            )
        if self.ownership_versions is not None:
            object.__setattr__(self, "ownership_versions", Path(self.ownership_versions).resolve())
        if self.observed_through is not None and self.observed_through.tzinfo is None:
            raise ValueError("Source observation coverage requires a timezone")
        object.__setattr__(self, "event_labels", MappingProxyType(dict(self.event_labels)))
        object.__setattr__(
            self,
            "profile_paths",
            MappingProxyType({name: tuple(path) for name, path in self.profile_paths.items()}),
        )
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "relationships", roles)
        object.__setattr__(
            self,
            "files",
            MappingProxyType({name: Path(path).resolve() for name, path in self.files.items()}),
        )

    def manifest(self) -> dict[str, Any]:
        """Serializable source semantics; file contents are fingerprinted by ingestion."""

        def record(value: Any) -> dict[str, Any]:
            return {
                item.name: dict(raw)
                if isinstance(raw := getattr(value, item.name), Mapping)
                else list(raw)
                if isinstance(raw, tuple)
                else raw
                for item in fields(value)
            }

        return {
            "tables": [record(table) for table in self.tables],
            "relationships": [record(role) for role in self.relationships],
            "feature_order": {
                table.name: {
                    kind: list(getattr(table, kind)) for kind in ("numeric", "categorical", "text")
                }
                for table in self.tables
            },
            "graph_feature_order": None
            if self.graph_numeric is None
            else {name: list(value) for name, value in self.graph_numeric.items()},
            "future_target_order": None
            if self.future_numeric is None
            else {name: list(value) for name, value in self.future_numeric.items()},
            "has_ownership_versions": self.ownership_versions is not None,
            "graph_numeric": None
            if self.graph_numeric is None
            else {name: dict(value) for name, value in self.graph_numeric.items()},
            "profile_paths": {name: list(path) for name, path in self.profile_paths.items()},
            "event_labels": {name: list(value) for name, value in self.event_labels.items()},
            "future_numeric": None
            if self.future_numeric is None
            else {name: dict(value) for name, value in self.future_numeric.items()},
            "observed_through": None
            if self.observed_through is None
            else self.observed_through.isoformat(),
        }
