# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This script generates and saves the OpenAPI schema for the FastAPI app.

Output Location: ui/src/gen/spec/openapi.json

Usage:
Run this module from the repository root:
    DEPLOY_STAGE=dev_local python -m rest.scripts.download_schema

`DEPLOY_STAGE` can be `dev` or `dev_local`.
"""

import json
import os
import re
from http import HTTPStatus
from typing import Any, Dict

from fastapi.openapi.utils import get_openapi

from rest.scripts.openapi_routes import routes_for_prefix

# Schema generation imports rest.main, which builds the app from env vars.
# Keep auth enabled for accurate security metadata, but avoid CI/dev shells
# inheriting the wildcard CORS default that is invalid for cookie auth.
if os.environ.get("REST_CORS_ALLOWED_ORIGINS", "*").strip() == "*":
    os.environ["REST_CORS_ALLOWED_ORIGINS"] = "http://localhost:3000"

from rest.main import app_builder

# Configuration
SPEC_PATH = 'ui/src/gen/spec'
UI_DIR = os.path.join(os.path.dirname(__file__), '../..', SPEC_PATH)
FILE_NAME = 'openapi.json'

app = app_builder.app

# ============================================================================
# Schema Processing
# ============================================================================


def fix_response_status_keys(openapi_schema: Dict[str, Any]) -> None:
    # FastAPI on Python <3.11 serializes HTTPStatus.NOT_FOUND as
    # "HTTPSTATUS.NOT_FOUND" via str().upper(). Replace with numeric codes.
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for key in list(responses):
                m = re.fullmatch(r"HTTPSTATUS\.([A-Z_]+)", str(key))
                if m:
                    try:
                        code = str(HTTPStatus[m.group(1)].value)
                        responses[code] = responses.pop(key)
                    except KeyError:
                        pass


def normalize_nullable_keyword(obj: Any) -> None:
    # OpenAPI 3.0 'nullable: true' is not a valid keyword in OpenAPI 3.1
    # (JSON Schema 2020-12). App code (e.g. nullable_response() in
    # rest/utils/routers.py) injects it into response schemas, so re-express it
    # in 3.1 form: append {"type": "null"} to an existing anyOf/oneOf, or wrap
    # the schema in an anyOf with null.
    if isinstance(obj, list):
        for item in obj:
            normalize_nullable_keyword(item)
        return
    if not isinstance(obj, dict):
        return
    for value in obj.values():
        normalize_nullable_keyword(value)
    if obj.get("nullable") is not True:
        return
    obj.pop("nullable")
    null_schema = {"type": "null"}
    if isinstance(obj.get("anyOf"), list):
        if null_schema not in obj["anyOf"]:
            obj["anyOf"].append(null_schema)
    elif isinstance(obj.get("oneOf"), list):
        if null_schema not in obj["oneOf"]:
            obj["oneOf"].append(null_schema)
    else:
        base = {k: obj.pop(k) for k in list(obj)}
        obj["anyOf"] = [base, null_schema]


def strip_metadata_keyword(obj: Any, _is_properties_dict: bool = False) -> None:
    # Pydantic v2 adds a non-standard "metadata" keyword to some field schema
    # objects. Traverse the entire spec and remove it from schema nodes, but
    # NOT from a model's "properties" dict (where keys are field names, not
    # schema keywords — removing "metadata" there would delete the field).
    if isinstance(obj, list):
        for item in obj:
            strip_metadata_keyword(item)
        return
    if not isinstance(obj, dict):
        return
    if not _is_properties_dict:
        obj.pop("metadata", None)
    for k, v in obj.items():
        strip_metadata_keyword(v, _is_properties_dict=(k == "properties"))


# ============================================================================
# Main Functions
# ============================================================================


def save_openapi_schema(prefix: str) -> None:
    """
    Generate and save OpenAPI schema for routes with the given prefix.

    Args:
        prefix: URL prefix to filter routes (e.g., "/v0", "/v1")
    """
    # Prepare file paths
    file_name = f"{prefix.replace('/', '')}_{FILE_NAME}"
    output_path = os.path.join(UI_DIR, file_name)

    # Filter routes by prefix
    filtered_routes = routes_for_prefix(app.routes, prefix)
    if not filtered_routes:
        raise RuntimeError(
            f"Refusing to overwrite {output_path}: no {prefix} routes were "
            "registered in the FastAPI app")

    # Generate schema with custom settings. We emit OpenAPI 3.1.0 natively
    # (Pydantic v2 / FastAPI default) and keep a single schema per model
    # (separate_input_output_schemas=False) to preserve stable schema names.
    def custom_openapi():
        return get_openapi(
            title=f"FastAPI - {prefix} API's",
            version="1.0.0",
            openapi_version="3.1.0",
            description=f"OpenAPI schema for {prefix}",
            routes=filtered_routes,
            separate_input_output_schemas=False,
        )

    app.openapi = custom_openapi
    openapi_schema = app.openapi()
    if not openapi_schema.get("paths"):
        raise RuntimeError(
            f"Refusing to overwrite {output_path}: generated {prefix} "
            "OpenAPI schema has no paths")

    # Post-process: fixes that are independent of the OpenAPI version.
    fix_response_status_keys(openapi_schema)
    strip_metadata_keyword(openapi_schema)
    normalize_nullable_keyword(openapi_schema)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    file_path = f"{SPEC_PATH}/{file_name}"
    print(f"✅ {prefix} OpenAPI schema saved to → kumo/{file_path}")


if __name__ == "__main__":
    save_openapi_schema("/v0")
    save_openapi_schema("/v1")
