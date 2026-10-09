# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Callable

from fastapi import APIRouter, Depends, Response


class AuthenticationRouter:
    """Group-neutral authenticated probe used to bootstrap SDK sessions."""

    def __init__(self, api_key_auth: Callable) -> None:
        self.router = APIRouter(
            prefix='/authentication',
            tags=['API v1: Authentication'],
            dependencies=[Depends(api_key_auth)],
        )
        self.router.add_api_route(
            '',
            self.authenticate,
            methods=['GET'],
            status_code=HTTPStatus.NO_CONTENT,
            response_class=Response,
            summary='Validate the current Kumo authentication credentials.',
        )

    async def authenticate(self) -> Response:
        return Response(status_code=HTTPStatus.NO_CONTENT)
