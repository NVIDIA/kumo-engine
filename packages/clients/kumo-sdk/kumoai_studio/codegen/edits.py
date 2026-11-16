# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from typing import Any, List, NamedTuple, Set

from kumoai_studio.codegen.naming import NameManager

logger = logging.getLogger(__name__)


class EditResult(NamedTuple):
    edits: List["UniversalReplacementEdit"]
    imports: List[str]


def _is_primitive(obj: object) -> bool:
    return obj is None or isinstance(obj, (str, int, float, bool))


def _is_collection(obj: object) -> bool:
    return isinstance(obj, (list, dict, set, tuple))


def _collect_required_imports(obj: object) -> List[str]:
    if _is_primitive(obj):
        return []

    from kumoai_studio.codegen.handlers.utils import _get_canonical_import_path

    obj_type = type(obj)
    if hasattr(obj_type, "__module__") and hasattr(obj_type, "__name__"):
        canonical_module = _get_canonical_import_path(obj_type)
        if canonical_module:
            return [f"from {canonical_module} import {obj_type.__name__}"]

    return []


_TYPE_DEFAULTS = {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    list: [],
    dict: {},
}


def _get_constructor_requirements(obj_type: type) -> dict[str, Any]:
    """Analyze constructor to determine required
    parameters and their default values.
    """
    try:
        sig = inspect.signature(obj_type.__init__)
        required_params = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            if param.kind == inspect.Parameter.VAR_POSITIONAL or \
                    param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            if param.default is param.empty:
                if param.annotation != param.empty:
                    required_params[name] = _TYPE_DEFAULTS.get(
                        param.annotation, None)
                else:
                    required_params[name] = None

        return required_params

    except (ValueError, TypeError):
        return {}


def _get_value_repr(value: object) -> str:
    """Get proper string representation for a value
    , handling enums specially.
    """
    if hasattr(value, "value") and hasattr(value, "name"):
        try:
            enum_class = type(value)
            string_value = str(value)
            reconstructed = getattr(enum_class, string_value, None)
            if reconstructed == value:
                return repr(string_value)
        except (ValueError, TypeError, AttributeError):
            pass

        enum_class_name = type(value).__name__
        return f"{enum_class_name}('{str(value)}')"
    elif hasattr(value, "__str__") and not _is_primitive(value):
        return repr(str(value))
    else:
        return repr(value)


def get_editable_attributes(obj: object) -> List[str]:
    """Extract editable attributes from an object using __dict__."""
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is None:
        return []

    editable_attrs = []
    for key, value in obj_dict.items():
        if callable(value):
            continue
        if key.startswith("__"):
            continue
        if key.startswith("_"):
            public_key = key[1:]
            if hasattr(obj, public_key):
                editable_attrs.append(key)
        else:
            editable_attrs.append(key)

    return editable_attrs


class UniversalReplacementEdit:
    """Represents a single edit operation for an object's attribute or element.

    This class generates Python code lines to update
    an object's property, collection element, or assign a new value.

    It handles primitives, collections, and complex objects,
    producing the necessary assignment
    or construction code to perform the edit programmatically.

    Example usage:
        # Primitive attribute edit
        nm = NameManager()
        edit = UniversalReplacementEdit("name", "Alice", nm)
        lines = edit.emit_lines("person")
        # lines == ["person.name = 'Alice'"]

        # Collection attribute edit
        edit = UniversalReplacementEdit("items", [1, 2, 3], nm)
        lines = edit.emit_lines("container")
        # lines == ["items_1 = [1, 2, 3]", "container.items = items_1"]

        # Complex object attribute edit
        address = Address("123 Main St", "NYC")
        edit = UniversalReplacementEdit("address", address, nm)
        lines = edit.emit_lines("person")
        # lines contains something like:
        # ["address_1 = Address(street='123 Main St', city='NYC')",
        #  "person.address = address_1"]
    """
    def __init__(self, path: str, value: object, name_manager: NameManager):
        if path is None:
            raise TypeError("path cannot be None")
        self.path = path
        self.value = value
        self.temp_var_name = name_manager.assign_temp_variable(path, value)
        self.required_imports = _collect_required_imports(value)
        self.name_manager = name_manager

    def emit_lines(self, var_name: str) -> List[str]:
        """Generate Python code lines for this edit.

        This method handles three cases:
        1. Primitives: Direct assignment (obj.name = "value")
        2. Collections: temp_var = [1, 2, 3]; target = temp_var
        3. Complex objects: Construct object + set properties + assign

        Args:
            var_name: The target variable name to assign to

        Returns:
            List of Python code lines to execute this edit

        Examples:
            For primitive:
            'obj.name = "new_value"'

            For object:
            'user_1 = User(id=123)',
            'user_1.active = True',
            'obj.user = user_1'
        """
        lines = []

        if _is_primitive(self.value):
            # For primitives, assign directly without temp variable
            value_repr = _get_value_repr(self.value)
            if self.path:
                target = (f"{var_name}{self.path}" if self.path.startswith("[")
                          or self.path.startswith(".") else
                          f"{var_name}.{self.path}")
                lines.append(f"{target} = {value_repr}")
            else:
                lines.append(f"{var_name} = {value_repr}")
            return lines

        elif _is_collection(self.value):
            lines.append(f"{self.temp_var_name} = {repr(self.value)}")

        else:
            # Complex object: construction + property decomposition
            obj_type = type(self.value)
            required_params = _get_constructor_requirements(obj_type)
            # For required parameters, use actual values from the object
            # instead of defaults
            for param_name, param_value in required_params.items():
                if hasattr(self.value, param_name):
                    required_params[param_name] = getattr(
                        self.value, param_name)
                else:
                    required_params[param_name] = param_value

            try:
                if required_params:
                    baseline_obj = obj_type(**required_params)
                else:
                    baseline_obj = obj_type()

                property_changes = detect_edits_recursive(
                    self.value, baseline_obj, "")
                self.required_imports = self.required_imports + \
                    property_changes.imports
                constructor_params = (required_params.copy()
                                      if required_params else {})
                remaining_edits = []

                for edit in property_changes.edits:
                    remaining_edits.append(edit)

                if constructor_params:
                    params_str = ", ".join(
                        f"{name}={_get_value_repr(val)}"
                        for name, val in constructor_params.items())
                    lines.append(f"{self.temp_var_name} = "
                                 f"{obj_type.__name__}({params_str})")
                else:
                    lines.append(
                        f"{self.temp_var_name} = {obj_type.__name__}()")

                for edit in remaining_edits:
                    prop_lines = edit._emit_lines_for_target(
                        self.temp_var_name)
                    lines.extend(prop_lines)

            except Exception:
                lines = [f"{self.temp_var_name} = {repr(self.value)}"]

        # For collections and complex objects, add the final
        # assignment
        if self.path:
            target = (f"{var_name}{self.path}" if self.path.startswith("[") or
                      self.path.startswith(".") else f"{var_name}.{self.path}")
            lines.append(f"{target} = {self.temp_var_name}")
        else:
            lines.append(f"{var_name} = {self.temp_var_name}")

        return lines

    def _emit_lines_for_target(self, target_var: str) -> List[str]:
        """Helper method to generate edits targeting a specific variable."""
        lines = []

        if _is_primitive(self.value):
            if self.path.startswith("[") or self.path.startswith("."):
                full_path = f"{target_var}{self.path}"
            else:
                full_path = (f"{target_var}.{self.path}"
                             if self.path else target_var)

            value_repr = _get_value_repr(self.value)
            lines.append(f"{full_path} = {value_repr}")
        else:
            lines.extend(self.emit_lines(target_var))

        return lines

    def __repr__(self) -> str:
        return (f"UniversalReplacementEdit(path={self.path}, "
                f"value_type={type(self.value).__name__})")


def get_element_replacement(path: str, value: object,
                            name_manager: NameManager) -> EditResult:
    """Mode 1: Generate edit to replace any value at
    any path using universal temp pattern.
    """
    edit = UniversalReplacementEdit(path, value, name_manager)
    return EditResult([edit], edit.required_imports)


def get_collection_element_replacement(target: object, baseline: object,
                                       path: str, name_manager: NameManager,
                                       visited: Set[int]) -> EditResult:
    """Mode 2: Generate edits for changes within collections
    (lists, dicts, sets, tuples).
    """
    edits = []
    all_imports = []

    if isinstance(target, list) and isinstance(baseline, list):
        max_len = max(len(target), len(baseline))

        for i in range(max_len):
            target_item = target[i] if i < len(target) else None
            base_item = baseline[i] if i < len(baseline) else None

            if target_item != base_item:
                element_path = f"{path}[{i}]"
                result = get_element_replacement(element_path, target_item,
                                                 name_manager)
                edits.extend(result.edits)
                all_imports.extend(result.imports)

    elif isinstance(target, dict) and isinstance(baseline, dict):
        all_keys = set(target.keys()) | set(baseline.keys())

        for key in sorted(all_keys):
            target_val = target.get(key)
            base_val = baseline.get(key)

            if target_val != base_val:
                key_path = f"{path}['{key}']"

                if (not _is_primitive(target_val)
                        and not _is_collection(target_val)
                        and target_val is not None):
                    result = detect_edits_recursive(target_val, base_val,
                                                    key_path, name_manager,
                                                    visited)
                    edits.extend(result.edits)
                    all_imports.extend(result.imports)
                elif _is_collection(target_val) and _is_collection(base_val):
                    result = get_collection_element_replacement(
                        target_val, base_val, key_path, name_manager, visited)
                    edits.extend(result.edits)
                    all_imports.extend(result.imports)
                else:
                    result = get_element_replacement(key_path, target_val,
                                                     name_manager)
                    edits.extend(result.edits)
                    all_imports.extend(result.imports)

    elif isinstance(target, (set, tuple)):
        if target != baseline:
            result = get_element_replacement(path, target, name_manager)
            edits.extend(result.edits)
            all_imports.extend(result.imports)

    return EditResult(edits, list(set(all_imports)))


def get_property_recursion(target: object, baseline: object, path: str,
                           name_manager: NameManager,
                           visited: Set[int]) -> EditResult:
    """Mode 3: Generate edits by recursively detecting
    property-level changes in objects.
    """
    edits = []
    all_imports = []

    attrs = get_editable_attributes(target)

    for attr in sorted(attrs):
        try:
            target_val = getattr(target, attr)
            base_val = getattr(baseline, attr, None)

            attr_path = f"{path}.{attr}" if path else attr

            result = detect_edits_recursive(target_val, base_val, attr_path,
                                            name_manager, visited)
            edits.extend(result.edits)
            all_imports.extend(result.imports)

        except (AttributeError, RuntimeError, TypeError):
            continue

    return EditResult(edits, list(set(all_imports)))


def _determine_edit_strategy(target: object, baseline: object) -> str:
    """Determine which edit strategy to use based
    on object types and values.
    """
    # Type mismatch -> Element replacement
    if not isinstance(target, type(baseline)):
        return "element_replacement"

    # Equality check -> No edits needed
    if target == baseline:
        return "no_edit"

    # Primitives -> Element replacement
    if _is_primitive(target):
        return "element_replacement"

    # Collections -> Collection element replacement
    if _is_collection(target):
        return "collection_element_replacement"

    # Complex objects -> Property recursion
    return "property_recursion"


# Strategy dispatch table: maps strategy names to handler functions
_EDIT_STRATEGIES = {
    "no_edit":
    lambda target, base, path, nm, visited: EditResult([], []),
    "element_replacement":
    lambda target, base, path, nm, visited:
    (get_element_replacement(path, target, nm)),
    "collection_element_replacement":
    get_collection_element_replacement,
    "property_recursion":
    get_property_recursion,
}


def detect_edits_recursive(
    target: object,
    baseline: object,
    path: str = "",
    name_manager: NameManager | None = None,
    visited: Set[int] | None = None,
) -> EditResult:
    """Generate edits to transform baseline
     into target using recursive analysis.

    This is the main entry point for edit detection.
     It analyzes two objects and
    determines what changes are needed, then routes
     to the appropriate mode:

    Decision:
    1. Type mismatch or primitives -> Element replacement mode
    2. Collections -> Collection element replacement mode
    3. Complex objects -> Property recursion mode

    Args:
        target: Target object state to achieve
        baseline: Starting object state
        path: Current path in object hierarchy (e.g., "user.profile.name")
        name_manager: The NameManager instance for variable naming.
        visited: Set of visited object IDs for cycle detection

    Returns:
        EditResult containing list of edits and required imports

    Examples:
        Simple change: obj.name = "new" ->
        [UniversalReplacementEdit("name", "new")]
        Nested change: obj.user.active = True ->
        [UniversalReplacementEdit("user.active", True)]
        Collection: obj.items[0] = val ->
        [UniversalReplacementEdit("items[0]", val)]
    """
    if visited is None:
        visited = set()

    if name_manager is None:
        name_manager = NameManager()

    target_id, base_id = id(target), id(baseline)
    if target_id in visited:
        return EditResult([], [])

    visited_extended = visited | {target_id, base_id}

    strategy = _determine_edit_strategy(target, baseline)
    handler = _EDIT_STRATEGIES[strategy]

    return handler(target, baseline, path, name_manager, visited_extended)
