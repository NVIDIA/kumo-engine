# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import kumoai_studio as kumo
from kumoai_studio.exceptions import HTTPException


@runtime_checkable
class _ByName(Protocol):
    r"""An entity class that resolves a name through its own lookup.

    Declared as a callable attribute rather than a method because the object
    matched against it is the *class*, on which ``get_by_name`` is an already
    bound classmethod taking just the name.
    """
    get_by_name: Callable[[str], object]


@runtime_checkable
class _Loadable(Protocol):
    r"""An entity class that resolves an id through its own loader."""
    load: Callable[[str], object]


def _get_supported_entities() -> dict[str, type]:
    """Get all supported entity types from registered handlers."""
    from kumoai_studio.codegen.registry import REG

    return {cls.__name__: cls for cls in REG.keys()}


# This map is for converting string names to Python types.
# The keys should be the string a user would provide.
# Auto-generated from handlers registry while serving as supported
# entities filter.
ENTITY_TYPE_MAP = _get_supported_entities()

# This map contains prefixes for type inference from their ID.
ENTITY_PREFIX_MAP = {
    # Job Types
    "gen-traintable-job": kumo.TrainingTable,
    "trainingjob": kumo.TrainingJob,
    "bp-job": kumo.BatchPredictionJob,
    # Query Types
    "pquery": kumo.PredictiveQuery,
    # Note: Table IDs don't have prefixes, so they require explicit
    # --entity-class
}


def _load_with_class(entity_id: str, entity_class: type) -> object:
    """Helper to load an entity when the class is known."""
    # Order of attempts: get_by_name, load, constructor
    if isinstance(entity_class, _ByName):
        return entity_class.get_by_name(entity_id)
    elif isinstance(entity_class, _Loadable):
        return entity_class.load(entity_id)

    try:
        # For jobs like TrainingJob, BatchPredictionJob,
        # FileUploadConnector
        return entity_class(entity_id)
    except (TypeError, AttributeError):
        pass  # Fall through to the error

    raise NotImplementedError(
        f"Don't know how to load object of type {entity_class.__name__}")


def load_from_id(
    entity_id: str,
    entity_class_str: str | None = None,
) -> object:
    """Load a Kumo object as an SDK object.
    - If entity_class_str is provided, it's used to find the type.
    - If not, the type is inferred from the ID prefix.
    """
    try:
        # Scenario A: Explicit class string provided
        if entity_class_str:
            if entity_class_str not in ENTITY_TYPE_MAP:
                raise ValueError(f"Unknown entity_class '{entity_class_str}'. "
                                 f"Supported types are: "
                                 f"{', '.join(ENTITY_TYPE_MAP.keys())}")
            entity_class = ENTITY_TYPE_MAP[entity_class_str]
            return _load_with_class(entity_id, entity_class)

        # Scenario B: No class string provided, so infer from ID
        # prefix
        prefix = entity_id.split("-", 1)[0].lower()
        if prefix in ENTITY_PREFIX_MAP:
            inferred_class = ENTITY_PREFIX_MAP[prefix]
            return _load_with_class(entity_id, inferred_class)
        else:
            raise ValueError(
                f"Could not infer entity type from ID '{entity_id}'. "
                "For an entity with a non-prefixed ID"
                "(like a Connector, Graph, or Table), "
                "please provide the 'entity_class' parameter. "
                "Supported prefixes are: " +
                ", ".join(ENTITY_PREFIX_MAP.keys()) +
                "\n and supported classes are: " +
                ", ".join(ENTITY_TYPE_MAP.keys()))
    except (HTTPException, ValueError) as e:
        class_name = (entity_class_str
                      if entity_class_str else "inferred type")
        raise ValueError(
            f"Failed to load entity '{entity_id}' of type {class_name}") from e
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while"
                         f"loading entity '{entity_id}'") from e
