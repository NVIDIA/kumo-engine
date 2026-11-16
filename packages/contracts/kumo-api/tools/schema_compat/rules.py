# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Breaking change detection rules.

This module defines the rules for detecting breaking changes between
two schema versions.
"""

import re
from typing import List, Optional, Set, Tuple

from tools.schema_compat.models import (
    ChangeType,
    DataclassSchema,
    EnumSchema,
    SchemaChange,
    SchemaField,
)


def parse_union_types(type_str: str) -> Set[str]:
    """Parse a Union type string into its constituent types.

    Examples:
        "Union[str, int, None]" -> {"str", "int", "None"}
        "str" -> {"str"}
    """
    union_match = re.match(r'^Union\[(.+)\]$', type_str)
    if not union_match:
        return {type_str}

    inner = union_match.group(1)
    members = _parse_type_members(inner)
    return set(members)


def _parse_type_members(inner: str) -> List[str]:
    """Parse comma-separated type members handling nested brackets."""
    members = []
    current = ""
    depth = 0

    for char in inner:
        if char == '[':
            depth += 1
            current += char
        elif char == ']':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            members.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        members.append(current.strip())

    return members


def is_type_narrowing(old_type: str, new_type: str) -> bool:
    """Check if the new type is a narrowing of the old type.

    Type narrowing means the new type accepts fewer values than the old type.

    Examples:
        Union[str, int] -> str  # narrowing (int no longer accepted)
        Optional[str] -> str    # narrowing (None no longer accepted)
        str -> str              # not narrowing
    """
    old_types = parse_union_types(old_type)
    new_types = parse_union_types(new_type)

    # If new type has fewer members than old, it's narrowing
    if new_types < old_types:  # proper subset
        return True

    return False


def is_type_widening(old_type: str, new_type: str) -> bool:
    """Check if the new type is a widening of the old type.

    Type widening means the new type accepts more values than the old type.

    Examples:
        str -> Union[str, int]  # widening
        str -> Optional[str]    # widening (None now accepted)
    """
    old_types = parse_union_types(old_type)
    new_types = parse_union_types(new_type)

    # If new type has more members and includes all old members, it's widening
    if old_types < new_types:  # proper subset
        return True

    return False


def parse_literal_values(type_str: str) -> Optional[Set[str]]:
    """Parse a Literal type string into its constituent values.

    Returns None if the type is not a Literal, otherwise the set of values.

    Examples:
        "Literal[zero, mean, raise]" -> {"zero", "mean", "raise"}
        "str" -> None
    """
    literal_match = re.match(r'^Literal\[(.+)\]$', type_str)
    if not literal_match:
        return None

    inner = literal_match.group(1)
    members = _parse_type_members(inner)
    return set(members)


def is_literal_widening(old_type: str, new_type: str) -> bool:
    """Check if the new Literal type accepts more values than the old.

    Examples:
        Literal[a, b] -> Literal[a, b, c]      # widening
        Literal[a, b] -> Literal[a, b]          # not widening
        Literal[a, b, c] -> Literal[a, b]       # not widening (narrowing)
    """
    old_vals = parse_literal_values(old_type)
    new_vals = parse_literal_values(new_type)

    if old_vals is None or new_vals is None:
        return False

    return old_vals < new_vals  # proper subset


def is_literal_narrowing(old_type: str, new_type: str) -> bool:
    """Check if the new Literal type accepts fewer values than the old.

    Examples:
        Literal[a, b, c] -> Literal[a, b]       # narrowing
        Literal[a, b] -> Literal[a, b]           # not narrowing
        Literal[a, b] -> Literal[a, b, c]        # not narrowing (widening)
    """
    old_vals = parse_literal_values(old_type)
    new_vals = parse_literal_values(new_type)

    if old_vals is None or new_vals is None:
        return False

    return new_vals < old_vals  # proper subset


def is_container_type(type_str: str) -> bool:
    """Check if type is a container (List, Set, Dict, Sequence, etc.)."""
    container_patterns = [
        r'^List\[',
        r'^Set\[',
        r'^Dict\[',
        r'^Sequence\[',
        r'^Tuple\[',
        r'^FrozenSet\[',
        r'^Iterable\[',
        r'^Collection\[',
    ]
    for pattern in container_patterns:
        if re.match(pattern, type_str):
            return True
    return False


def get_container_info(type_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract container type and element type from a type string.

    Returns:
        Tuple of (container_name, element_type) or (None, None)
    """
    match = re.match(r'^(\w+)\[(.+)\]$', type_str)
    if match:
        return match.group(1), match.group(2)
    return None, None


def is_container_widening(old_type: str, new_type: str) -> bool:
    """Check if new container type is a widening of old.

    Example: List[X] -> Sequence[X] is widening (more general)
    """
    old_container, old_elem = get_container_info(old_type)
    new_container, new_elem = get_container_info(new_type)

    if not old_container or not new_container:
        return False

    # Element types must match exactly
    if old_elem != new_elem:
        return False

    # Sequence is more general than List
    widening_pairs = [
        ('List', 'Sequence'),
        ('List', 'Iterable'),
        ('List', 'Collection'),
        ('Set', 'Collection'),
        ('Set', 'Iterable'),
    ]

    return (old_container, new_container) in widening_pairs


def _is_element_widening(old_elem: str, new_elem: str) -> bool:
    """Check if an element type was widened (Union or Literal)."""
    if is_type_widening(old_elem, new_elem):
        return True
    if is_literal_widening(old_elem, new_elem):
        return True
    return False


def _is_element_narrowing(old_elem: str, new_elem: str) -> bool:
    """Check if an element type was narrowed (Union or Literal)."""
    if is_type_narrowing(old_elem, new_elem):
        return True
    if is_literal_narrowing(old_elem, new_elem):
        return True
    return False


def _compare_container_elements(
    old_type: str,
    new_type: str,
) -> Optional[bool]:
    """Compare element types of two containers.

    Returns True if widened, False if narrowed, None if not comparable.
    """
    old_container, old_elem = get_container_info(old_type)
    new_container, new_elem = get_container_info(new_type)

    if not old_container or not new_container:
        return None
    if not old_elem or not new_elem:
        return None
    if old_container != new_container:
        return None

    # For multi-param containers (e.g. Dict[K, V]), split and check each
    old_parts = _parse_type_members(old_elem)
    new_parts = _parse_type_members(new_elem)

    if len(old_parts) != len(new_parts):
        return None

    has_widening = False
    for old_part, new_part in zip(old_parts, new_parts):
        if old_part == new_part:
            continue
        if _is_element_narrowing(old_part, new_part):
            return False
        if _is_element_widening(old_part, new_part):
            has_widening = True
        else:
            return None  # Incompatible change

    return True if has_widening else None


def is_container_element_widening(old_type: str, new_type: str) -> bool:
    """Check if a container's element type is widened.

    This handles cases like:
        List[Union[A, B]] -> List[Union[A, B, C]]      # element type widened
        Dict[Literal[a, b], Any] -> Dict[Literal[a, b, c], Any]  # widened

    Adding new values to a Union or Literal inside a container is non-breaking
    because existing code that produces the old values will still be valid.
    """
    return _compare_container_elements(old_type, new_type) is True


def is_container_element_narrowing(old_type: str, new_type: str) -> bool:
    """Check if a container's element type is narrowed.

    This handles cases like:
        List[Union[A, B, C]] -> List[Union[A, B]]         # narrowed
        Dict[Literal[a, b, c], Any] -> Dict[Literal[a], Any]  # narrowed
    """
    return _compare_container_elements(old_type, new_type) is False


def compare_field_types(
    old_field: SchemaField,
    new_field: SchemaField,
    model_name: str,
) -> List[SchemaChange]:
    """Compare types between old and new field versions."""
    changes = []
    old_type = old_field.type_normalized
    new_type = new_field.type_normalized

    if old_type == new_type:
        return changes

    # Check for type narrowing (breaking)
    if is_type_narrowing(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_NARROWED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Check for type widening (non-breaking)
    if is_type_widening(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_WIDENED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Check for container widening (non-breaking)
    if is_container_widening(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_WIDENED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Check for container element narrowing (breaking)
    # e.g., Dict[Literal[a, b, c], Any] -> Dict[Literal[a], Any]
    if is_container_element_narrowing(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_NARROWED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Check for container element widening (non-breaking)
    # e.g., List[Union[A, B]] -> List[Union[A, B, C]]
    # e.g., Dict[Literal[a, b], Any] -> Dict[Literal[a, b, c], Any]
    if is_container_element_widening(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_WIDENED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Check for Literal narrowing (breaking)
    # e.g., Literal[a, b, c] -> Literal[a, b]
    if is_literal_narrowing(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_NARROWED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Check for Literal widening (non-breaking)
    # e.g., Literal[a, b] -> Literal[a, b, c]
    if is_literal_widening(old_type, new_type):
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_TYPE_WIDENED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=old_type,
                new_value=new_type,
            ))
        return changes

    # Any other type change is breaking
    changes.append(
        SchemaChange(
            change_type=ChangeType.FIELD_TYPE_CHANGED,
            model_name=model_name,
            field_name=old_field.name,
            old_value=old_type,
            new_value=new_type,
        ))

    return changes


def compare_field_defaults(
    old_field: SchemaField,
    new_field: SchemaField,
    model_name: str,
) -> List[SchemaChange]:
    """Compare default values between old and new field versions."""
    changes = []

    # Check if default was removed (breaking)
    if old_field.has_default and not new_field.has_default:
        old_val = old_field.default_value or old_field.default_factory
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_DEFAULT_REMOVED,
                model_name=model_name,
                field_name=old_field.name,
                old_value=str(old_val),
            ))
        return changes

    # Check if default was added (non-breaking)
    if not old_field.has_default and new_field.has_default:
        new_val = new_field.default_value or new_field.default_factory
        changes.append(
            SchemaChange(
                change_type=ChangeType.FIELD_DEFAULT_ADDED,
                model_name=model_name,
                field_name=new_field.name,
                new_value=str(new_val),
            ))
        return changes

    # Check if default value changed (warning)
    if old_field.has_default and new_field.has_default:
        old_default = old_field.default_value or old_field.default_factory
        new_default = new_field.default_value or new_field.default_factory

        if old_default != new_default:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.FIELD_DEFAULT_VALUE_CHANGED,
                    model_name=model_name,
                    field_name=old_field.name,
                    old_value=str(old_default),
                    new_value=str(new_default),
                ))

    return changes


def compare_dataclass_schemas(
    old_schema: DataclassSchema,
    new_schema: DataclassSchema,
) -> List[SchemaChange]:
    """Compare two dataclass schemas and detect changes."""
    changes = []
    model_name = old_schema.name

    old_fields = {f.name: f for f in old_schema.fields}
    new_fields = {f.name: f for f in new_schema.fields}

    # Check for removed fields (breaking)
    for field_name in old_fields:
        if field_name not in new_fields:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.FIELD_REMOVED,
                    model_name=model_name,
                    field_name=field_name,
                ))

    # Check for added fields
    for field_name in new_fields:
        if field_name not in old_fields:
            new_field = new_fields[field_name]
            # Adding a field without default is breaking if existing code
            # doesn't pass it. However, we consider it non-breaking since
            # the caller must update anyway. The key rule is: if you add
            # a required field, existing calls break.
            # But we'll treat field addition as non-breaking (info only)
            # since the typical pattern is to add with a default.
            changes.append(
                SchemaChange(
                    change_type=ChangeType.FIELD_ADDED,
                    model_name=model_name,
                    field_name=field_name,
                    new_value=new_field.type_normalized,
                ))

    # Check for field changes (type, default)
    for field_name in old_fields:
        if field_name in new_fields:
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]

            # Compare types
            changes.extend(
                compare_field_types(old_field, new_field, model_name))

            # Compare defaults
            changes.extend(
                compare_field_defaults(old_field, new_field, model_name))

    # Check for field order changes (breaking)
    if old_schema.field_order != new_schema.field_order:
        # Only report if the sets are the same (pure reordering)
        old_set = set(old_schema.field_order)
        new_set = set(new_schema.field_order)

        if old_set == new_set:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.FIELD_ORDER_CHANGED,
                    model_name=model_name,
                    old_value=str(old_schema.field_order),
                    new_value=str(new_schema.field_order),
                ))

    return changes


def compare_enum_schemas(
    old_schema: EnumSchema,
    new_schema: EnumSchema,
) -> List[SchemaChange]:
    """Compare two enum schemas and detect changes."""
    changes = []
    model_name = old_schema.name

    old_values = set(old_schema.values)
    new_values = set(new_schema.values)

    # Check for removed values (breaking)
    removed = old_values - new_values
    for value in removed:
        changes.append(
            SchemaChange(
                change_type=ChangeType.ENUM_VALUE_REMOVED,
                model_name=model_name,
                old_value=value,
            ))

    # Check for added values (non-breaking)
    added = new_values - old_values
    for value in added:
        changes.append(
            SchemaChange(
                change_type=ChangeType.ENUM_VALUE_ADDED,
                model_name=model_name,
                new_value=value,
            ))

    return changes
