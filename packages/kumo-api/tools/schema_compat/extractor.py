# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""Schema extraction from pydantic dataclasses.

This module handles extracting schema information from pydantic dataclasses
in a way that works consistently across both Pydantic v1 and v2.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Set, Tuple, Type, Union, get_type_hints

from tools.schema_compat.config import PYDANTIC_VERSION
from tools.schema_compat.models import (
    DataclassSchema,
    EnumSchema,
    FullSchema,
    SchemaField,
)

# Type origin/args helpers - handle different Python versions
try:
    from typing import get_args, get_origin
except ImportError:
    # Python 3.7 fallback
    def get_origin(tp):
        return getattr(tp, '__origin__', None)

    def get_args(tp):
        return getattr(tp, '__args__', ())


def normalize_type_string(type_str: str) -> str:
    """Normalize a type string for consistent comparison.

    - Optional[X] -> Union[X, None]
    - Sort Union members alphabetically for consistent comparison
    - Remove whitespace inconsistencies
    """
    # Remove extra whitespace
    type_str = re.sub(r'\s+', ' ', type_str).strip()

    # Convert Optional[X] to Union[X, None]
    optional_match = re.match(r'^Optional\[(.+)\]$', type_str)
    if optional_match:
        inner = optional_match.group(1)
        type_str = f"Union[{inner}, None]"

    # Sort Union members for consistent comparison
    union_match = re.match(r'^Union\[(.+)\]$', type_str)
    if union_match:
        inner = union_match.group(1)
        # Parse union members (handle nested generics)
        members = _parse_union_members(inner)
        # Sort members, but keep None at the end
        non_none = sorted([m for m in members if m != 'None'])
        has_none = 'None' in members
        if has_none:
            non_none.append('None')
        type_str = f"Union[{', '.join(non_none)}]"

    return type_str


def _parse_union_members(inner: str) -> List[str]:
    """Parse union members handling nested brackets."""
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


def type_to_string(tp: Any) -> str:
    """Convert a type annotation to a normalized string representation."""
    if tp is None:
        return "None"

    if tp is type(None):
        return "None"

    # Handle string annotations
    if isinstance(tp, str):
        return tp

    # Handle ForwardRef
    if hasattr(tp, '__forward_arg__'):
        return tp.__forward_arg__

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is None:
        # Simple type
        if hasattr(tp, '__name__'):
            return tp.__name__
        return str(tp)

    # Handle Union types
    if origin is Union:
        args_str = ', '.join(type_to_string(arg) for arg in args)
        return f"Union[{args_str}]"

    # Handle other generic types (List, Dict, etc.)
    origin_name = getattr(origin, '__name__', str(origin))

    # Clean up origin name
    if origin_name == 'list':
        origin_name = 'List'
    elif origin_name == 'dict':
        origin_name = 'Dict'
    elif origin_name == 'set':
        origin_name = 'Set'
    elif origin_name == 'tuple':
        origin_name = 'Tuple'
    elif origin_name == 'type':
        origin_name = 'Type'

    if args:
        args_str = ', '.join(type_to_string(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    return origin_name


def is_pydantic_dataclass(cls: Type) -> bool:
    """Check if a class is a pydantic dataclass."""
    if not dataclasses.is_dataclass(cls):
        return False

    return hasattr(cls, '__pydantic_complete__')


def is_regular_dataclass(cls: Type) -> bool:
    """Check if a class is a regular (non-pydantic) dataclass."""
    return dataclasses.is_dataclass(cls) and not is_pydantic_dataclass(cls)


def is_str_enum(cls: Type) -> bool:
    """Check if a class is a string enum."""
    return (inspect.isclass(cls) and issubclass(cls, Enum)
            and issubclass(cls, str))


def extract_field_info(
    cls: Type,
    field_obj: dataclasses.Field,
    type_hints: Dict[str, Any],
) -> SchemaField:
    """Extract schema information from a dataclass field."""
    field_name = field_obj.name
    field_type = type_hints.get(field_name, field_obj.type)

    # Convert type to string
    type_str = type_to_string(field_type)
    type_normalized = normalize_type_string(type_str)

    # Determine if field has a default
    has_default = (field_obj.default is not dataclasses.MISSING
                   or field_obj.default_factory is not dataclasses.MISSING)

    # Get default value
    default_value = None
    default_factory = None

    if field_obj.default is not dataclasses.MISSING:
        default_value = _serialize_default(field_obj.default)
    elif field_obj.default_factory is not dataclasses.MISSING:
        default_factory = _get_factory_repr(field_obj.default_factory)

    # Extract constraints from pydantic Field
    constraints = _extract_constraints(cls, field_obj)

    return SchemaField(
        name=field_name,
        type_str=type_str,
        type_normalized=type_normalized,
        required=not has_default,
        has_default=has_default,
        default_value=default_value,
        default_factory=default_factory,
        constraints=constraints,
    )


def _serialize_default(value: Any) -> Any:
    """Serialize a default value to JSON-compatible format."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_serialize_default(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_default(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    # For complex objects, return string representation
    return repr(value)


def _get_factory_repr(factory: Any) -> str:
    """Get a string representation of a default factory."""
    if factory is list:
        return "list"
    if factory is dict:
        return "dict"
    if factory is set:
        return "set"
    if hasattr(factory, '__name__'):
        return factory.__name__
    return repr(factory)


def _extract_constraints(
    cls: Type,
    field_obj: dataclasses.Field,
) -> Dict[str, Any]:
    """Extract field constraints from pydantic field metadata."""
    constraints = {}

    # Try to get constraints from field metadata
    if field_obj.metadata:
        for key in ('ge', 'le', 'gt', 'lt', 'min_length', 'max_length'):
            if key in field_obj.metadata:
                constraints[key] = field_obj.metadata[key]

    # Constraints might be in the field info
    try:
        pydantic_fields = getattr(cls, '__pydantic_fields__', {})
        if field_obj.name in pydantic_fields:
            field_info = pydantic_fields[field_obj.name]
            if hasattr(field_info, 'metadata'):
                for meta in field_info.metadata:
                    if hasattr(meta, 'ge'):
                        constraints['ge'] = meta.ge
                    if hasattr(meta, 'le'):
                        constraints['le'] = meta.le
                    if hasattr(meta, 'gt'):
                        constraints['gt'] = meta.gt
                    if hasattr(meta, 'lt'):
                        constraints['lt'] = meta.lt
    except Exception:
        pass

    return constraints


def extract_dataclass_schema(cls: Type) -> DataclassSchema:
    """Extract schema from a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f'{cls!r} is not a dataclass')

    try:
        type_hints = get_type_hints(cls)
    except Exception:
        # Fall back to __annotations__ if get_type_hints fails
        type_hints = getattr(cls, '__annotations__', {})

    fields = []
    field_order = []

    for field_obj in dataclasses.fields(cls):
        # Skip private fields
        if field_obj.name.startswith('_'):
            continue

        field_schema = extract_field_info(cls, field_obj, type_hints)
        fields.append(field_schema)
        field_order.append(field_obj.name)

    return DataclassSchema(
        name=_get_qualified_name(cls),
        fields=fields,
        field_order=field_order,
    )


def extract_enum_schema(cls: Type[Enum]) -> EnumSchema:
    """Extract schema from an enum class."""
    values = [
        member.value if hasattr(member, 'value') else member.name
        for member in cls
    ]
    return EnumSchema(
        name=_get_qualified_name(cls),
        values=values,
    )


def _get_qualified_name(cls: Type) -> str:
    """Get the fully qualified name of a class."""
    module = cls.__module__
    name = cls.__qualname__
    return f"{module}.{name}"


def discover_models(
        package_name: str = "kumoapi",  #
) -> Tuple[List[Type], List[Type[Enum]]]:
    """Discover all dataclasses and enums in a package.

    Returns:
        Tuple of (dataclass_list, enum_list)
    """
    dataclasses_found: List[Type] = []
    enums_found: List[Type[Enum]] = []
    seen: Set[int] = set()

    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(f"Cannot import package {package_name}: {e}")

    def process_module(module):
        for name, obj in inspect.getmembers(module):
            # Skip private members
            if name.startswith('_'):
                continue

            # Skip if already processed
            if id(obj) in seen:
                continue

            # Check if it's from our package
            if not hasattr(obj, '__module__'):
                continue
            if not obj.__module__.startswith(package_name):
                continue

            seen.add(id(obj))

            # Check for dataclasses
            if inspect.isclass(obj):
                if is_pydantic_dataclass(obj) or is_regular_dataclass(obj):
                    dataclasses_found.append(obj)
                elif is_str_enum(obj):
                    enums_found.append(obj)

    # Process the main package
    process_module(package)

    # Process all submodules
    if hasattr(package, '__path__'):
        for _importer, modname, _ispkg in pkgutil.walk_packages(
                package.__path__,
                prefix=package_name + ".",
        ):
            try:
                module = importlib.import_module(modname)
                process_module(module)
            except ImportError:
                # Skip modules that can't be imported
                continue

    return dataclasses_found, enums_found


def extract_schema(
    version: str,
    package_name: str = "kumoapi",
) -> FullSchema:
    """Extract the full schema for a package.

    Args:
        version: Version string (e.g., "0.52.0")
        package_name: Python package to extract from

    Returns:
        FullSchema object containing all discovered models
    """
    dataclasses_list, enums_list = discover_models(package_name)

    dataclasses_dict = {}
    enums_dict = {}

    for cls in dataclasses_list:
        try:
            schema = extract_dataclass_schema(cls)
            dataclasses_dict[schema.name] = schema
        except Exception as e:
            # Log but continue on extraction errors
            print(f"Warning: Failed to extract schema for {cls}: {e}")

    for cls in enums_list:
        try:
            schema = extract_enum_schema(cls)
            enums_dict[schema.name] = schema
        except Exception as e:
            print(f"Warning: Failed to extract schema for {cls}: {e}")

    return FullSchema(
        version=version,
        pydantic_version=PYDANTIC_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        dataclasses=dataclasses_dict,
        enums=enums_dict,
    )
