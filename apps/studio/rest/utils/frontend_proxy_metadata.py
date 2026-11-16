# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import Request, Response
from starlette.types import ASGIApp, Receive, Scope, Send


class ExternalHostMiddleware:
    """Restore the configured public host after the mesh-routable proxy hop."""
    def __init__(self, app: ASGIApp, external_host: str):
        self._app = app
        self._host = external_host.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive,
                       send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        forwarded_scope = dict(scope)
        forwarded_scope["headers"] = [
            (b"host", self._host),
            *((name, value)
              for name, value in scope["headers"] if name != b"host"),
        ]
        await self._app(forwarded_scope, receive, send)


def is_secure_request(request: Request) -> bool:
    return request.url.scheme == "https"


def client_ip(request: Request) -> str:
    if request.client is not None:
        return request.client.host or ""
    return ""


def redirect_uri(request: Request) -> str:
    return str(request.base_url).rstrip("/") + "/v0/auth/callback"


def set_cookie(response: Response, request: Request, *, name: str, value: str,
               max_age: int, path: str = "/") -> None:
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=True,
        secure=is_secure_request(request),
        samesite="lax",
        path=path,
    )


def clear_cookie(response: Response, request: Request, *, name: str,
                 path: str = "/") -> None:
    response.delete_cookie(
        key=name,
        httponly=True,
        secure=is_secure_request(request),
        samesite="lax",
        path=path,
    )
