# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.utils import _get_canonical_import_path
from kumoai_studio.codegen.registry import Handler, get_object_var
from kumoai_studio.pquery import PredictiveQuery

logger = logging.getLogger(__name__)


def get_handlers() -> Dict[type, Handler[Any]]:
    """Returns a dictionary of handlers for PredictiveQuery types."""
    handlers: Dict[type, Handler[Any]] = {}

    def _pquery_parents(obj: object,
                        codegen_ctx: CodegenContext) -> list[object]:
        """PredictiveQuery depends on its graph."""
        assert isinstance(obj, PredictiveQuery)
        return [obj.graph]

    def _pquery_imports(obj: object) -> list[str]:
        """Get import statements needed for PredictiveQuery."""
        imports_needed = [PredictiveQuery]
        imports = []
        for obj_type in imports_needed:
            canonical_module = _get_canonical_import_path(obj_type)
            imports.append(
                f"from {canonical_module} import {obj_type.__name__}")
        return imports

    def _pquery_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        """Generate code lines to recreate a PredictiveQuery."""
        assert isinstance(obj, PredictiveQuery)

        graph_var = get_object_var(codegen_ctx, obj.graph)

        if '\n' in obj.query:
            formatted_query = f'"""{obj.query}"""'
        else:
            formatted_query = repr(obj.query)

        lines = [
            f"{var_name} = PredictiveQuery(" + f"graph={graph_var}," +
            f" query={formatted_query}," + ")", f"{var_name}.validate()"
        ]

        return lines

    handlers[PredictiveQuery] = Handler[PredictiveQuery](
        parents=_pquery_parents,
        required_imports=_pquery_imports,
        emit_lines=_pquery_emit_lines,
        detect_edits=None,
    )

    return handlers
