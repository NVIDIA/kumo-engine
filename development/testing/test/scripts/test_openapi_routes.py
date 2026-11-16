# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, FastAPI
from fastapi.openapi.utils import get_openapi

from rest.scripts.openapi_routes import routes_for_prefix


def test_routes_for_prefix_traverses_nested_routers() -> None:
    app = FastAPI()
    router = APIRouter()

    @router.get('/v0/items')
    def get_v0_items() -> None:
        pass

    @router.get('/v1/items')
    def get_v1_items() -> None:
        pass

    app.include_router(router)

    routes = routes_for_prefix(app.routes, '/v0')
    schema = get_openapi(title='test', version='1', routes=routes)

    assert set(schema['paths']) == {'/v0/items'}
