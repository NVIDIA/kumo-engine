# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for schema compatibility checker."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChangeType(Enum):
    """Types of schema changes."""
    # Breaking changes
    FIELD_REMOVED = "field_removed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    FIELD_TYPE_NARROWED = "field_type_narrowed"
    FIELD_DEFAULT_REMOVED = "field_default_removed"
    FIELD_ORDER_CHANGED = "field_order_changed"
    ENUM_VALUE_REMOVED = "enum_value_removed"
    MODEL_REMOVED = "model_removed"

    # Warnings (not blocking)
    FIELD_DEFAULT_VALUE_CHANGED = "field_default_value_changed"

    # Non-breaking changes (informational)
    FIELD_ADDED = "field_added"
    FIELD_TYPE_WIDENED = "field_type_widened"
    FIELD_DEFAULT_ADDED = "field_default_added"
    ENUM_VALUE_ADDED = "enum_value_added"
    MODEL_ADDED = "model_added"


# Changes that should block a release
BREAKING_CHANGES = {
    ChangeType.FIELD_REMOVED,
    ChangeType.FIELD_TYPE_CHANGED,
    ChangeType.FIELD_TYPE_NARROWED,
    ChangeType.FIELD_DEFAULT_REMOVED,
    ChangeType.FIELD_ORDER_CHANGED,
    ChangeType.ENUM_VALUE_REMOVED,
    ChangeType.MODEL_REMOVED,
}

# Changes that should produce a warning but not block
WARNING_CHANGES = {
    ChangeType.FIELD_DEFAULT_VALUE_CHANGED,
}


@dataclass
class SchemaField:
    """Represents a field in a dataclass schema."""
    name: str
    type_str: str  # Original type string
    type_normalized: str  # Normalized (Optional[X] -> Union[X, None])
    required: bool
    has_default: bool
    default_value: Optional[Any] = None
    default_factory: Optional[str] = None  # String repr of factory
    constraints: Dict[str, Any] = field(default_factory=dict)  # ge, le, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type_str": self.type_str,
            "type_normalized": self.type_normalized,
            "required": self.required,
            "has_default": self.has_default,
            "default_value": self.default_value,
            "default_factory": self.default_factory,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaField":
        return cls(
            name=data["name"],
            type_str=data["type_str"],
            type_normalized=data["type_normalized"],
            required=data["required"],
            has_default=data["has_default"],
            default_value=data.get("default_value"),
            default_factory=data.get("default_factory"),
            constraints=data.get("constraints", {}),
        )


@dataclass
class EnumSchema:
    """Represents an enum schema."""
    name: str
    values: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "enum",
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "EnumSchema":
        return cls(
            name=name,
            values=data["values"],
        )


@dataclass
class DataclassSchema:
    """Represents a dataclass schema."""
    name: str
    fields: List[SchemaField]
    field_order: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "dataclass",
            "fields": [f.to_dict() for f in self.fields],
            "field_order": self.field_order,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "DataclassSchema":
        return cls(
            name=name,
            fields=[SchemaField.from_dict(f) for f in data["fields"]],
            field_order=data["field_order"],
        )


@dataclass
class FullSchema:
    """Complete schema dump for a version."""
    version: str
    pydantic_version: int
    generated_at: str
    dataclasses: Dict[str, DataclassSchema]
    enums: Dict[str, EnumSchema]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "pydantic_version": self.pydantic_version,
            "generated_at": self.generated_at,
            "models": {
                **{
                    n: dc.to_dict()
                    for n, dc in self.dataclasses.items()
                },
                **{
                    n: e.to_dict()
                    for n, e in self.enums.items()
                },
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FullSchema":
        dataclasses = {}
        enums = {}

        for name, model_data in data.get("models", {}).items():
            if model_data.get("type") == "enum":
                enums[name] = EnumSchema.from_dict(name, model_data)
            else:
                dataclasses[name] = DataclassSchema.from_dict(name, model_data)

        return cls(
            version=data["version"],
            pydantic_version=data["pydantic_version"],
            generated_at=data["generated_at"],
            dataclasses=dataclasses,
            enums=enums,
        )


@dataclass
class SchemaChange:
    """Represents a single schema change."""
    change_type: ChangeType
    model_name: str
    field_name: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    message: str = ""

    @property
    def is_breaking(self) -> bool:
        return self.change_type in BREAKING_CHANGES

    @property
    def is_warning(self) -> bool:
        return self.change_type in WARNING_CHANGES

    def format_message(self) -> str:
        """Format a human-readable message for this change."""
        if self.message:
            return self.message

        location = self.model_name
        if self.field_name:
            location = f"{self.model_name}.{self.field_name}"

        old_new = f"({self.old_value} -> {self.new_value})"
        change_descriptions = {
            ChangeType.FIELD_REMOVED: f"Field removed: {location}",
            ChangeType.FIELD_TYPE_CHANGED:
            f"Type changed: {location} {old_new}",
            ChangeType.FIELD_TYPE_NARROWED:
            f"Type narrowed: {location} {old_new}",
            ChangeType.FIELD_DEFAULT_REMOVED: f"Default removed: {location}",
            ChangeType.FIELD_ORDER_CHANGED:
            f"Field order changed in {self.model_name}: {old_new}",
            ChangeType.ENUM_VALUE_REMOVED:
            f"Enum removed: {location} (removed: {self.old_value})",
            ChangeType.MODEL_REMOVED: f"Model removed: {self.model_name}",
            ChangeType.FIELD_DEFAULT_VALUE_CHANGED:
            f"Default changed: {location} {old_new}",
            ChangeType.FIELD_ADDED: f"Field added: {location}",
            ChangeType.FIELD_TYPE_WIDENED:
            f"Type widened: {location} {old_new}",
            ChangeType.FIELD_DEFAULT_ADDED: f"Default added: {location}",
            ChangeType.ENUM_VALUE_ADDED:
            f"Enum added: {location} (added: {self.new_value})",
            ChangeType.MODEL_ADDED: f"Model added: {self.model_name}",
        }

        return change_descriptions.get(self.change_type,
                                       f"Unknown change: {self.change_type}")


@dataclass
class ComparisonResult:
    """Result of comparing two schemas."""
    old_version: str
    new_version: str
    changes: List[SchemaChange] = field(default_factory=list)

    @property
    def has_breaking_changes(self) -> bool:
        return any(c.is_breaking for c in self.changes)

    @property
    def has_warnings(self) -> bool:
        return any(c.is_warning for c in self.changes)

    @property
    def breaking_changes(self) -> List[SchemaChange]:
        return [c for c in self.changes if c.is_breaking]

    @property
    def warnings(self) -> List[SchemaChange]:
        return [c for c in self.changes if c.is_warning]

    @property
    def non_breaking_changes(self) -> List[SchemaChange]:
        return [
            c for c in self.changes if not c.is_breaking and not c.is_warning
        ]

    def format_report(self) -> str:
        """Format a human-readable report of the comparison."""
        lines = []
        lines.append(
            f"Schema Comparison: {self.old_version} -> {self.new_version}")
        lines.append("=" * 60)

        breaking = self.breaking_changes
        warnings = self.warnings
        non_breaking = self.non_breaking_changes

        if breaking:
            lines.append("")
            lines.append(f"BREAKING CHANGES ({len(breaking)}):")
            lines.append("-" * 40)
            for change in breaking:
                lines.append(f"  [BREAKING] {change.format_message()}")

        if warnings:
            lines.append("")
            lines.append(f"WARNINGS ({len(warnings)}):")
            lines.append("-" * 40)
            for change in warnings:
                lines.append(f"  [WARNING] {change.format_message()}")

        if non_breaking:
            lines.append("")
            lines.append(f"Non-breaking changes ({len(non_breaking)}):")
            lines.append("-" * 40)
            for change in non_breaking:
                lines.append(f"  [INFO] {change.format_message()}")

        if not self.changes:
            lines.append("")
            lines.append("No changes detected.")

        lines.append("")
        lines.append("=" * 60)
        if self.has_breaking_changes:
            lines.append("RESULT: FAILED - Breaking changes detected")
        elif self.has_warnings:
            lines.append("RESULT: PASSED with warnings")
        else:
            lines.append("RESULT: PASSED")

        return "\n".join(lines)
