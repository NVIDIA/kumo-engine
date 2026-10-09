# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import keyword
from collections import defaultdict
from typing import Any

from kumoai_studio.codegen.identity import get_config_id


def _sanitize_identifier(name: str) -> str:
    """Sanitize a name to be a valid Python identifier."""
    if not name:
        return "obj"

    sanitized = "".join(char if char.isalnum() else "_"
                        for char in name.lower())
    sanitized = "_".join(filter(None, sanitized.split("_")))

    if not sanitized:
        return "obj"

    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"

    if keyword.iskeyword(sanitized) or sanitized in dir(__builtins__):
        sanitized = f"{sanitized}_"

    return sanitized


class NameManager:
    def __init__(self) -> None:
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._names: dict[str, str] = {}  # config_id -> variable_name

    def assign_entity_variable(self, obj: Any) -> str:
        config_id = get_config_id(obj)
        if config_id in self._names:
            return self._names[config_id]

        entity_name = ""
        if hasattr(obj, "name") and obj.name:
            entity_name = str(obj.name)
        elif hasattr(obj, "source_name") and obj.source_name:
            entity_name = str(obj.source_name)

        type_name = obj.__class__.__name__.lower()

        if entity_name:
            sanitized_name = _sanitize_identifier(entity_name)
            base_name = (sanitized_name if sanitized_name.replace("_", "")
                         == type_name else f"{sanitized_name}_{type_name}")
        else:
            base_name = type_name

        self._counts[base_name] += 1
        name = f"{base_name}_{self._counts[base_name]}"
        self._names[config_id] = name
        return name

    def assign_temp_variable(self, path: str, value: Any) -> str:
        base_name = self._get_base_name_for_temp(path, value)
        self._counts[base_name] += 1
        return f"{base_name}_{self._counts[base_name]}"

    def _get_base_name_for_temp(self, path: str, value: Any) -> str:
        if path:
            if "." in path:
                parts = path.split(".")
                for part in reversed(parts):
                    if part and not part.startswith("["):
                        return part.split("[")[0]
            if "[" in path:
                return path.split("[")[0]
            if not path.startswith("["):
                return path

        primitives = (type(None), str, int, float, bool, list, dict, set,
                      tuple)
        if not isinstance(value, primitives):
            import re
            class_name = type(value).__name__
            return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()

        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "dict"
        if isinstance(value, set):
            return "set"
        if isinstance(value, tuple):
            return "tuple"

        return "temp_obj"
