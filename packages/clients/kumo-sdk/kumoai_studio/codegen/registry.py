# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.edits import UniversalReplacementEdit
from kumoai_studio.codegen.identity import get_config_id
from kumoai_studio.codegen.naming import NameManager

_T = TypeVar('_T')


@dataclass(frozen=True)
class Handler(Generic[_T]):
    parents: Callable[[_T, CodegenContext], List[Any]]
    required_imports: Callable[[_T], List[str]]
    emit_lines: Callable[[_T, str, dict, CodegenContext], List[str]]
    detect_edits: Optional[Callable[[_T, _T, NameManager],
                                    Sequence[UniversalReplacementEdit]]]
    get_parent_map: Optional[Callable[[_T], dict[str, dict[str, Any]]]] = None


REG: dict[type, Handler[Any]] = {}


def register_shared_parents(ctx: CodegenContext, obj: object,
                            handler: Handler[Any]) -> None:
    """Register parents that this handler wants
    to share with other handlers.
    """
    if handler.get_parent_map:
        parent_map = handler.get_parent_map(obj)
        # parent_map format: {object_id: {key: parent_obj}}
        for obj_id, shared_data in parent_map.items():
            ctx.shared_parents[obj_id] = shared_data


def lookup_shared_parent(ctx: CodegenContext, obj: object, key: str) -> Any:
    """Look up a shared parent by key from another handler using config ID."""
    config_id = get_config_id(obj)
    return ctx.shared_parents.get(config_id, {}).get(key)


def store_shared_parent(ctx: CodegenContext, obj: object, key: str,
                        parent_obj: object) -> None:
    """Store a shared parent for an object."""
    config_id = get_config_id(obj)
    if config_id not in ctx.shared_parents:
        ctx.shared_parents[config_id] = {}
    ctx.shared_parents[config_id][key] = parent_obj


def store_object_var(ctx: CodegenContext, obj: object, var_name: str) -> None:
    """Store the variable name for an object using config_id."""
    config_id = get_config_id(obj)
    ctx.object_to_var[config_id] = var_name


def get_object_var(ctx: CodegenContext, obj: object) -> str:
    """Get the variable name for an object using config_id."""
    config_id = get_config_id(obj)
    var_name = ctx.object_to_var.get(config_id)
    if not var_name:
        raise ValueError(
            f"No variable name found for object {type(obj).__name__} "
            f"with config_id {config_id}")
    return var_name


def init_execution_env(ctx: CodegenContext) -> None:
    """Initialize the execution environment in context."""
    import kumoai_studio as kumo
    ctx.execution_env = {"kumo": kumo}


def execute_in_env(ctx: CodegenContext, lines: list[str],
                   imports: Optional[list[str]] = None) -> None:
    """Execute lines in the context's execution environment."""
    if imports:
        for import_line in imports:
            exec(import_line, ctx.execution_env)

    for line in lines:
        if line.strip() and not line.strip().startswith("#"):
            exec(line, ctx.execution_env)


def get_from_env(ctx: CodegenContext, var_name: str) -> Any:
    """Get an object from the context's execution environment."""
    return ctx.execution_env.get(var_name)


def _discover_and_register_handlers() -> None:
    """Dynamically discover and import all modules in the 'handlers' folders,
    call their `get_handlers` function, and register the returned handlers.
    """
    from . import handlers

    handlers_dir = handlers.__path__
    prefix = f"{handlers.__name__}."

    for _, module_name, _ in pkgutil.iter_modules(handlers_dir, prefix):
        module = importlib.import_module(module_name)
        if hasattr(module, "get_handlers"):
            handlers_to_register: Dict[type,
                                       Handler[Any]] = (module.get_handlers())
            for cls, handler in handlers_to_register.items():
                REG[cls] = handler


_discover_and_register_handlers()
