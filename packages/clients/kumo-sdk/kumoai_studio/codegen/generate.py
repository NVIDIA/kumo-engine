# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Optional

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.exceptions import (
    CyclicDependencyError,
    UnsupportedEntityError,
)
from kumoai_studio.codegen.identity import get_config_id
from kumoai_studio.codegen.loader import load_from_id
from kumoai_studio.codegen.naming import NameManager
from kumoai_studio.codegen.registry import (
    REG,
    Handler,
    execute_in_env,
    get_from_env,
    init_execution_env,
    register_shared_parents,
    store_object_var,
)

logger = logging.getLogger(__name__)


def _get_handler(obj_type: type) -> Handler[Any]:
    if obj_type not in REG:
        raise UnsupportedEntityError(
            f"No handler registered for type: {obj_type.__name__}")
    return REG[obj_type]


def get_kumo_id(obj: object) -> str:
    if hasattr(obj, 'id'):
        return str(obj.id)
    elif hasattr(obj, 'name'):
        return str(obj.name)
    elif hasattr(obj, 'source_name'):
        return str(obj.source_name)
    else:
        return ''


def _generate(
    obj: object,
    name_manager: NameManager,
    config_to_var: dict[str, str],
    stack: set[int],
    codegen_ctx: CodegenContext,
    context: Optional[dict[str, Any]] = None,
    id_to_var: Optional[dict[int, str]] = None,
) -> tuple[list[str], list[str]]:
    """Generate code for an object and its parents.

    Args:
        obj: The object to generate code for.
        name_manager: The name manager to use for variable names.
        config_to_var: A dictionary mapping config IDs to variable names.
        stack: A set of object IDs to detect cycles.
        codegen_ctx: The codegen context.
        context: A dictionary of context information.
        id_to_var: A dictionary mapping object IDs to variable names.

    Returns:
        A tuple containing a list of imports and a list of lines of code.
    """
    if id_to_var is None:
        id_to_var = {}

    obj_id = id(obj)
    config_id = get_config_id(obj)

    # Check for configuration-based deduplication first
    if config_id in config_to_var:
        # Reuse existing variable for this configuration
        id_to_var[obj_id] = config_to_var[config_id]
        return [], []

    # Check for exact object reuse (faster path)
    if obj_id in id_to_var:
        return [], []

    # Cycle detection using real object IDs
    if obj_id in stack:
        raise CyclicDependencyError(
            f"Cyclic dependency detected for object ID: {obj_id}")

    stack.add(obj_id)
    handler = _get_handler(type(obj))
    all_imports, all_lines = [], []

    for parent in handler.parents(obj, codegen_ctx):
        parent_imports, parent_lines = _generate(parent, name_manager,
                                                 config_to_var, stack,
                                                 codegen_ctx, context,
                                                 id_to_var)
        all_imports.extend(parent_imports)
        all_lines.extend(parent_lines)

    # Register shared parents if handler supports it
    register_shared_parents(codegen_ctx, obj, handler)

    var_name = name_manager.assign_entity_variable(obj)
    # Store both config-based and id-based mappings
    config_to_var[config_id] = var_name
    id_to_var[obj_id] = var_name

    # Store in codegen context for handlers to access
    store_object_var(codegen_ctx, obj, var_name)

    all_imports.extend(handler.required_imports(obj))
    context = context or {}
    context['target_id'] = get_kumo_id(obj)
    creation_lines = handler.emit_lines(obj, var_name, context, codegen_ctx)

    # Execute creation lines immediately in context environment
    execute_in_env(codegen_ctx, creation_lines, handler.required_imports(obj))

    all_lines.extend(creation_lines)

    if handler.detect_edits:
        # Get baseline object from context environment
        baseline_obj = get_from_env(codegen_ctx, var_name)
        if baseline_obj is not None:
            edits = handler.detect_edits(obj, baseline_obj, name_manager)
            for edit in edits:
                edit_lines = edit.emit_lines(var_name)
                all_lines.extend(edit_lines)
                all_imports.extend(edit.required_imports)
                # Execute edit lines immediately in context environment
                execute_in_env(codegen_ctx, edit_lines, edit.required_imports)

    stack.remove(obj_id)
    return all_imports, all_lines


def _load_entity_from_spec(input_spec: dict[str, Any]) -> object:
    """Load entity from input specification."""
    if "id" in input_spec:
        entity_class = input_spec.get("entity_class")
        return load_from_id(input_spec["id"], entity_class)
    elif "json" in input_spec:
        raise NotImplementedError("JSON loading not yet implemented")
    elif "object" in input_spec:
        return input_spec["object"]
    else:
        raise ValueError("input_spec must contain 'id', 'json', or 'object'")


def _write_script(code: str, output_path: str) -> None:
    """Write generated code to file."""
    with open(output_path, "w") as f:
        f.write(code)


def _assemble_code(imports: list[str], lines: list[str]) -> str:
    """Assemble final code from components."""
    from kumoai_studio import __version__

    header = [
        f"# Generated with Kumo SDK version: {__version__}",
        "import kumoai_studio as kumo",
        "import os",
        "",
        'kumo.init(url=os.getenv("KUMO_API_ENDPOINT"), '
        'api_key=os.getenv("KUMO_API_KEY"))',
        "",
    ]

    unique_imports = list(OrderedDict.fromkeys(imports))
    code = header + unique_imports + [""] + lines
    return "\n".join(code) + "\n"


def _init_kumo() -> None:
    """Initialize Kumo SDK for this python session."""
    import kumoai_studio as kumo
    if os.getenv("KUMO_API_ENDPOINT") is None:
        logger.warning("KUMO_API_ENDPOINT env variable is not set, "
                       "assuming kumo.init has already been called")
        return
    if os.getenv("KUMO_API_KEY") is None:
        logger.warning("KUMO_API_KEY env variable is not set, "
                       "assuming kumo.init has already been called")
        return
    kumo.init(url=os.getenv("KUMO_API_ENDPOINT"),
              api_key=os.getenv("KUMO_API_KEY"))


def generate_code(input_spec: dict[str, Any],
                  output_path: Optional[str] = None) -> str:
    """Generate Python SDK code from Kumo entity specification."""
    # Create codegen context for this generation session
    codegen_ctx = CodegenContext()

    # Initialize execution environment in context
    init_execution_env(codegen_ctx)

    _init_kumo()
    entity = _load_entity_from_spec(input_spec)

    context = {}
    if "id" in input_spec:
        context["input_method"] = "id"
        context["target_id"] = input_spec["id"]
    elif "json" in input_spec:
        context["input_method"] = "json"
    else:
        context["input_method"] = "object"

    name_manager = NameManager()
    imports, lines = _generate(entity, name_manager, config_to_var={},
                               stack=set(), codegen_ctx=codegen_ctx,
                               context=context, id_to_var={})

    code = _assemble_code(imports, lines)
    if output_path:
        _write_script(code, output_path)
    return code
