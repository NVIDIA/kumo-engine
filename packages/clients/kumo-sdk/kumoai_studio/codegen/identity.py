# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration-based identity system for codegen deduplication."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kumoai_studio.connector import (
    ADLSConnector,
    BigQueryConnector,
    DatabricksConnector,
    FileUploadConnector,
    GCSConnector,
    S3Connector,
    SnowflakeConnector,
)


def get_config_id(obj: Any) -> str:
    """Return configuration-based identity for codegen deduplication.

    Objects with identical configurations get the same config_id, enabling
    variable reuse during code generation. Uses get_editable_attributes() to
    automatically capture all configurable properties.

    This configuration-based identity is used for deduplication only of
    object which are SAFE to be shared
    when they have the same configuration always.
    For example, a S3Connector
    with the same root_dir and name can always be shared,
    it does not matter if some other objects wants to copy it.

    But for certain objects like Graphs,
    some object may want 2 copies of the same graph,
    and there we should not use this config_id
    and always use the memory address.
    For example, a Graph with the same tables and edges can always be shared,
    it does not matter if some other objects wants to copy it.

    Args:
        obj: Object to get configuration ID for

    Returns:
        Configuration-based identity string

    Note:
        - For deduplication only, not cycle detection (use id() for cycles)
        - Only applied to connector types for now; other objects use memory ID
    """
    generic_object_types = (S3Connector, GCSConnector, ADLSConnector,
                            BigQueryConnector, SnowflakeConnector,
                            DatabricksConnector, FileUploadConnector)
    if isinstance(obj, generic_object_types):
        return _get_generic_config_id(obj)
    else:
        return f"id_{id(obj)}"


def _get_generic_config_id(obj: Any) -> str:
    """Generate config ID by hashing object type and editable attributes.

    Uses get_editable_attributes() to capture all configurable properties,
    then creates a SHA256 hash for consistent identity.
    """
    try:
        connector_id = getattr(obj, 'id', None)
        if connector_id:
            return f'{type(obj).__name__}_{connector_id}'
        # Import here to avoid circular imports
        from kumoai_studio.codegen.edits import get_editable_attributes

        # Get object type name
        obj_type = type(obj).__name__

        # Get all editable attributes
        editable_attrs = get_editable_attributes(obj)

        # Build configuration dict
        config: dict[str, Any] = {'type': obj_type, 'attributes': {}}

        # Extract values for all editable attributes
        for attr_name in sorted(editable_attrs):  # Sort for consistent hashing
            try:
                attr_value = getattr(obj, attr_name)
                config['attributes'][attr_name] = _serialize_value(attr_value)
            except (AttributeError, RuntimeError, TypeError):
                # Skip attributes that can't be accessed
                continue

        # Create hash from configuration
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return f"{obj_type}_{config_hash}"

    except Exception:
        # Fallback to memory address if hashing fails
        return f"{type(obj).__name__}_{id(obj)}"


def _serialize_value(value: Any) -> Any:
    """Convert value to JSON-serializable format for consistent hashing.

    Handles nested objects by recursively applying config-based identity.
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif hasattr(value, '__dict__'):
        # For objects with __dict__, recurse into their config_id
        return get_config_id(value)
    else:
        # For other types, convert to string
        return str(value)
