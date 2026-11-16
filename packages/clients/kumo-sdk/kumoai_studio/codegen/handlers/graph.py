# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.utils import _get_canonical_import_path
from kumoai_studio.codegen.registry import Handler, get_object_var
from kumoai_studio.graph import Edge, Graph

logger = logging.getLogger(__name__)


def get_handlers() -> Dict[type, Handler[Any]]:
    """Returns a dictionary of handlers for Graph types."""
    handlers: Dict[type, Handler[Any]] = {}

    def _graph_parents(obj: object,
                       codegen_ctx: CodegenContext) -> list[object]:
        """Graph depends on its tables and edges."""
        assert isinstance(obj, Graph)
        return [table for table in obj.tables.values()]

    def _graph_imports(obj: object) -> list[str]:
        """Get import statements needed for Graph."""
        imports_needed = [Graph, Edge]
        imports = []
        for obj_type in imports_needed:
            canonical_module = _get_canonical_import_path(obj_type)
            imports.append(
                f"from {canonical_module} import {obj_type.__name__}")
        return imports

    def _graph_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        """Generate code lines to recreate a Graph using Graph()."""
        assert isinstance(obj, Graph)

        tables_vars = {
            table_name: get_object_var(codegen_ctx, table)
            for table_name, table in obj.tables.items()
        }
        tables_format_inner = ", ".join(f"{key!r}: {value}"
                                        for key, value in tables_vars.items())
        all_tables_var_name = f"{var_name}_tables"
        all_edges_var_name = f"{var_name}_edges"

        note = (f"# Note: This could also be done with Graph.load("
                f"'{context['target_id']}') for simpler code")

        lines = [
            note, f"{all_tables_var_name} = {{{tables_format_inner}}}",
            f"{all_edges_var_name} = {obj.edges}",
            f"{var_name} = Graph(tables={all_tables_var_name}, "
            f"edges={all_edges_var_name})", f"{var_name}.validate()",
            f"# Optionally, you can save the graph to backend using "
            f"{var_name}.save({context['target_id']}) and then load it"
        ]

        return lines

    handlers[Graph] = Handler[Graph](
        parents=_graph_parents,
        required_imports=_graph_imports,
        emit_lines=_graph_emit_lines,
        detect_edits=None,
    )

    return handlers
