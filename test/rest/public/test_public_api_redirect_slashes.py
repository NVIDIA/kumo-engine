# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #29812: a trailing-slash request to a public-API
collection route must return a clean 404, never a 307 redirect.

The public app is built with ``FastAPI(redirect_slashes=False)`` (see
``rest/public/app.py``). Without it, ``GET /v1/projects/`` 307-redirects, and
behind the reverse proxy the redirect ``Location`` is reconstructed from the
app's own ASGI scope (``http://<host>:8080/...``): an https->http scheme
downgrade, an internal-port disclosure, and a target unreachable from the
public origin, so clients that follow the redirect hang.

The flag must live on the app router: every public router is attached with
``include_router``, which flattens sub-router routes into the app router, so a
per-sub-router ``redirect_slashes`` flag does not take effect. These tests
exercise the real assembled public app via the ``public_api_client`` fixture,
so they fail if the app-level flag is dropped.
"""
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("path", [
    "/v1/projects/",
    "/v1/graphs/",
    "/v1/tables/",
])
def test_trailing_slash_returns_404_not_internal_redirect(
    public_api_client: TestClient,
    path: str,
) -> None:
    resp = public_api_client.get(path, follow_redirects=False)
    assert resp.status_code == HTTPStatus.NOT_FOUND, (
        f"{path} returned {resp.status_code}; a 307 here redirects to the "
        "internal http://<host>:8080 origin (issue #29812)")
    assert "location" not in {k.lower() for k in resp.headers}


def test_explicit_slash_route_unaffected(
    public_api_client: TestClient, ) -> None:
    """The predictive-queries router registers an explicit ``"/"`` route, so
    the trailing-slash form is a real match. Disabling redirects must not turn
    that into a 404."""
    resp = public_api_client.get("/v1/predictive_queries/",
                                 follow_redirects=False)
    assert resp.status_code != HTTPStatus.NOT_FOUND
    assert resp.status_code != HTTPStatus.TEMPORARY_REDIRECT


def test_no_slash_collection_route_still_served(
    public_api_client: TestClient, ) -> None:
    """The canonical no-slash form is unchanged by the fix (control)."""
    resp = public_api_client.get("/v1/projects", follow_redirects=False)
    assert resp.status_code != HTTPStatus.TEMPORARY_REDIRECT
    assert resp.status_code != HTTPStatus.NOT_FOUND
