# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from fastapi import Request, Response
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from rest.utils.frontend_proxy_metadata import (
    ExternalHostMiddleware,
    client_ip,
    is_secure_request,
    redirect_uri,
    set_cookie,
)


def _request(*, scheme: str, client: tuple[str, int] | None,
             forwarded_proto: str = "http",
             forwarded_for: str = "198.51.100.99") -> Request:
    host = "kumo.customer.example"
    return Request({
        "type": "http",
        "asgi": {
            "version": "3.0"
        },
        "http_version": "1.1",
        "method": "GET",
        "scheme": scheme,
        "path": "/v0/auth/login",
        "raw_path": b"/v0/auth/login",
        "query_string": b"",
        "headers": [
            (b"host", host.encode()),
            (b"x-forwarded-proto", forwarded_proto.encode()),
            (b"x-forwarded-for", forwarded_for.encode()),
        ],
        "client": client,
        "server": (host, 443 if scheme == "https" else 80),
        "root_path": "",
    })


@pytest.mark.parametrize(
    ("scheme", "raw_forwarded_proto", "expected_secure"),
    [("https", "http", True), ("http", "https", False)],
)
def test_auth_metadata_uses_normalized_scope(
        scheme, raw_forwarded_proto, expected_secure):
    request = _request(
        scheme=scheme,
        client=("10.42.0.7", 43123),
        forwarded_proto=raw_forwarded_proto,
        forwarded_for="198.51.100.99, 203.0.113.8",
    )

    assert is_secure_request(request) is expected_secure
    assert redirect_uri(request) == (
        f"{scheme}://kumo.customer.example/v0/auth/callback")
    assert client_ip(request) == "10.42.0.7"

    response = Response()
    set_cookie(
        response,
        request,
        name="kumo_direct_connect_login",
        value="opaque",
        max_age=60,
    )
    assert ("; Secure" in response.headers["set-cookie"]) is expected_secure


@pytest.mark.asyncio
async def test_pinned_uvicorn_proxy_middleware_normalizes_trusted_headers():
    observed = {}

    async def app(scope, _receive, _send):
        request = Request(scope)
        observed.update(
            scheme=scope["scheme"],
            client=scope["client"],
            redirect_uri=redirect_uri(request),
        )

    middleware = ProxyHeadersMiddleware(
        ExternalHostMiddleware(app, "kumo.customer.example"),
        trusted_hosts="*",
    )
    scope = _request(
        scheme="http",
        client=("10.42.0.8", 43123),
        forwarded_proto="https",
        forwarded_for="10.42.0.7",
    ).scope
    scope["headers"][0] = (b"host", b"kumo-rest:80")
    scope["headers"].append((b"x-forwarded-host", b"hostile.invalid"))

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        return None

    await middleware(scope, receive, send)

    assert observed == {
        "scheme": "https",
        "client": ("10.42.0.7", 0),
        "redirect_uri": "https://kumo.customer.example/v0/auth/callback",
    }
