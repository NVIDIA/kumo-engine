# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.utils import _get_canonical_import_path
from kumoai_studio.codegen.registry import Handler
from kumoai_studio.connector import (
    ADLSConnector,
    BigQueryConnector,
    DatabricksConnector,
    FileUploadConnector,
    GCSConnector,
    S3Connector,
    SnowflakeConnector,
)


def _get_by_name_handler_factory(cls: type) -> Handler[Any]:
    """Factory for creating a Handler for any connector that uses the
    get_by_name pattern.
    """
    def get_imports(obj: object, bound_cls: type = cls) -> list[str]:
        canonical_module = _get_canonical_import_path(bound_cls)
        return [f"from {canonical_module} import {bound_cls.__name__}"]

    def get_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        assert isinstance(obj, cls)
        assert hasattr(obj, "name")
        obj_name = getattr(obj, "name")
        return [f"{var_name} = {cls.__name__}.get_by_name('{obj_name}')"]

    return Handler(
        parents=lambda e, ctx: [],
        required_imports=get_imports,
        emit_lines=get_lines,
        detect_edits=None,
    )


def get_handlers() -> Dict[type, Handler[Any]]:
    """Returns a dictionary of handlers for all connector types."""
    handlers: Dict[type, Handler[Any]] = {}

    # All identity-backed connectors are durable project resources. Generated
    # code retrieves the existing resource by its project-local name instead
    # of embedding identity or target configuration.
    persistent_connectors = [
        S3Connector,
        GCSConnector,
        ADLSConnector,
        BigQueryConnector,
        SnowflakeConnector,
        DatabricksConnector,
    ]
    for connector_cls in persistent_connectors:
        handlers[connector_cls] = _get_by_name_handler_factory(connector_cls)

    def _file_upload_parents(obj: object,
                             codegen_ctx: CodegenContext) -> list[object]:
        return []

    def _file_upload_imports(obj: object) -> list[str]:
        canonical_module = _get_canonical_import_path(type(obj))
        return [f"from {canonical_module} import {type(obj).__name__}"]

    def _file_upload_emit_lines(
        obj: object,
        var_name: str,
        context: dict,
        codegen_ctx: CodegenContext,
    ) -> list[str]:
        assert isinstance(obj, FileUploadConnector)
        return [
            f"{var_name} = {type(obj).__name__}"
            f"(file_type='{obj.file_type}')"
        ]

    handlers[FileUploadConnector] = Handler[FileUploadConnector](
        parents=_file_upload_parents,
        required_imports=_file_upload_imports,
        emit_lines=_file_upload_emit_lines,
        detect_edits=None,
    )

    return handlers
