# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

from fastapi import APIRouter, Depends
from kumo_core.version import get_kumo_version

from rest.config.constants import DebugInfo, FeatureFlags
from rest.deps.authentication import Authentication
from rest.utils.routers import enable_api_caching


class ConstantsResource:
    def __init__(self, auth: Authentication) -> None:
        self.router = APIRouter(
            prefix="/v0/constants",
            tags=["Constants"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description": "Not found (Constants)"
                }
            },
        )

        self.router.add_api_route(
            "/flags",
            self.flags,
            response_model=DebugInfo,
            methods=['GET'],
        )
        # Enable caching for ALL GET apis.
        enable_api_caching(self.router, method='GET')

    def flags(self) -> DebugInfo:
        r"""Return a struct with flags needed to render the UI"""

        # Return data
        return DebugInfo(
            db_path='',
            release_version=get_kumo_version(),
            temporal_url='',
            feature_flags=FeatureFlags.parse_from_env_vars(),
        )
