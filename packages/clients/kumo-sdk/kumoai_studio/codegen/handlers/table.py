# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, Sequence

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.edits import (
    UniversalReplacementEdit,
    detect_edits_recursive,
)
from kumoai_studio.codegen.handlers.utils import _get_canonical_import_path
from kumoai_studio.codegen.naming import NameManager
from kumoai_studio.codegen.registry import Handler, get_object_var
from kumoai_studio.connector import SQLSourceTable
from kumoai_studio.graph.table import Table

logger = logging.getLogger(__name__)


def get_handlers() -> Dict[type, Handler[Any]]:
    """Returns a dictionary of handlers for Table types."""
    handlers: Dict[type, Handler[Any]] = {}

    def _table_parents(obj: object,
                       codegen_ctx: CodegenContext) -> list[object]:
        """Table depends on its source_table's connector."""
        assert isinstance(obj, Table)
        return [obj.source_table.connector]

    def _table_imports(obj: object) -> list[str]:
        """Get import statements needed for Table."""
        canonical_module = _get_canonical_import_path(Table)
        return [f"from {canonical_module} import Table"]

    def _build_args(obj: Table, connector_var: str,
                    table_name: str) -> list[str]:
        """Build arguments for Table.from_source_table() call."""
        args = [f"source_table={connector_var}['{table_name}']"]

        if obj.primary_key is not None:
            args.append(f"primary_key='{obj.primary_key.name}'")

        if obj.time_column is not None:
            args.append(f"time_column='{obj.time_column.name}'")

        if obj.end_time_column is not None:
            args.append(f"end_time_column='{obj.end_time_column.name}'")

        source_table_cols = {col.name: col for col in obj.source_table.columns}
        table_cols = {col.name: col for col in obj.columns}

        if len(source_table_cols) != len(table_cols):
            assert len(source_table_cols) > len(table_cols)
            table_col_names = [col_name for col_name in table_cols.keys()]
            args.append(f"column_names={table_col_names}")

        return args

    def _build_sql_args(obj: Table, connector_var: str) -> list[str]:
        """Build arguments for a Table.from_sql() call.

        A custom-SQL table has no physical source, so the full output schema
        is emitted inline rather than re-derived from a source table name.
        """
        assert isinstance(obj.source_table, SQLSourceTable)
        args = [
            f"connector={connector_var}",
            f"sql={obj.source_table.sql!r}",
        ]
        if obj.primary_key is not None:
            args.append(f"primary_key={obj.primary_key.name!r}")
        if obj.time_column is not None:
            args.append(f"time_column={obj.time_column.name!r}")
        if obj.end_time_column is not None:
            args.append(f"end_time_column={obj.end_time_column.name!r}")

        col_dicts = []
        for col in obj.columns:
            stype = f"'{col.stype}'" if col.stype is not None else "None"
            dtype = f"'{col.dtype}'" if col.dtype is not None else "None"
            parts = [f"name={col.name!r}", f"stype={stype}", f"dtype={dtype}"]
            if col.timestamp_format is not None:
                timestamp_format = getattr(col.timestamp_format, 'value',
                                           col.timestamp_format)
                parts.append(f"timestamp_format={timestamp_format!r}")
            # A multicategorical column is only valid with its delimiter, so
            # it must ride along or the regenerated from_sql(...) raises a
            # ValueError.
            if col.multicategorical_delimiter is not None:
                parts.append("multicategorical_delimiter="
                             f"{col.multicategorical_delimiter!r}")
            col_dicts.append(f"dict({', '.join(parts)})")
        if not obj._discover_sql_columns:
            args.append(f"columns=[{', '.join(col_dicts)}]")

        return args

    def _table_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        """Generate code lines to recreate a Table using from_source_table."""
        assert isinstance(obj, Table)

        connector_var = get_object_var(codegen_ctx, obj.source_table.connector)

        lines = []

        if context.get("input_method") == "id" and context.get("target_id"):
            note = (f"# Note: This could also be done with Table.load("
                    f"'{context['target_id']}') for simpler code")
            lines.append(note)

        # A custom-SQL table carries the query itself; regenerate it with
        # Table.from_sql rather than the physical, name-keyed source table.
        if isinstance(obj.source_table, SQLSourceTable):
            factory, args = "from_sql", _build_sql_args(obj, connector_var)
        else:
            factory = "from_source_table"
            args = _build_args(obj, connector_var, obj.source_table.name)

        if len(args) == 1:
            lines.append(f"{var_name} = Table.{factory}({args[0]})")
        else:
            args_str = ",\n    ".join(args)
            lines.append(f"{var_name} = Table.{factory}(\n    {args_str},\n)")

        return lines

    def _table_detect_edits(
            target: object, baseline: object,
            name_manager: NameManager) -> Sequence[UniversalReplacementEdit]:
        """Detect edits needed to make baseline match target table."""
        assert isinstance(target, Table)
        assert isinstance(baseline, Table)

        try:
            result = detect_edits_recursive(target, baseline, "", name_manager)
            logger.debug(f"Found for table {len(result.edits)} edits with "
                         f"{len(result.imports)} imports")
            return result.edits
        except Exception as e:
            logger.error(f"Error during table edit detection: {e}")
            return []

    handlers[Table] = Handler[Table](
        parents=_table_parents,
        required_imports=_table_imports,
        emit_lines=_table_emit_lines,
        detect_edits=_table_detect_edits,
    )

    return handlers
