# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient


def test_security_header_middleware(client: TestClient):
    response = client.get("/")

    # Assert that the custom header is added by the middleware
    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "0"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Content-Type"] == "application/json"
    assert response.headers[
        "Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert response.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert response.headers[
        "Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers[
        "Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
