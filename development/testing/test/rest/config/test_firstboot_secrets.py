# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from rest.config import firstboot_secrets


def test_load_firstboot_secret_from_explicit_path(tmp_path, monkeypatch):
    secret_path = tmp_path / "direct-connect.json"
    secret_path.write_text(json.dumps({
        "credentials_encryption_key": "key",
        "oidc_issuer": "https://idp.example.com",
    }))
    monkeypatch.setenv("KUMO_FIRSTBOOT_SECRET_PATH", str(secret_path))

    assert firstboot_secrets.load_firstboot_secret("ignored-name") == {
        "credentials_encryption_key": "key",
        "oidc_issuer": "https://idp.example.com",
    }


def test_load_firstboot_secret_from_secret_dir(tmp_path, monkeypatch):
    secret_path = tmp_path / "kumo-test-secret"
    secret_path.write_text(json.dumps({"api_key_pepper": "pepper"}))
    monkeypatch.delenv("KUMO_FIRSTBOOT_SECRET_PATH", raising=False)
    monkeypatch.setenv("KUMO_FIRSTBOOT_SECRET_DIR", str(tmp_path))

    assert firstboot_secrets.load_firstboot_secret("kumo-test-secret") == {
        "api_key_pepper": "pepper"
    }


def test_load_firstboot_secret_requires_json_object(tmp_path, monkeypatch):
    secret_path = tmp_path / "secret.json"
    secret_path.write_text(json.dumps(["not", "an", "object"]))
    monkeypatch.setenv("KUMO_FIRSTBOOT_SECRET_PATH", str(secret_path))

    with pytest.raises(RuntimeError, match="must contain a JSON object"):
        firstboot_secrets.load_firstboot_secret("ignored-name")


def test_get_secret_prefer_firstboot_does_not_call_remote_secrets(
        tmp_path, monkeypatch):
    secret_path = tmp_path / "secret.json"
    secret_path.write_text(json.dumps({"credentials_encryption_key": "local"}))
    monkeypatch.setenv("KUMO_FIRSTBOOT_SECRET_PATH", str(secret_path))

    def fail_remote_lookup(_secret_name):
        raise AssertionError("remote secrets should not be called")

    monkeypatch.setattr(firstboot_secrets, "get_secrets", fail_remote_lookup)

    assert firstboot_secrets.get_secret_prefer_firstboot("kumo-test-secret") == {
        "credentials_encryption_key": "local"
    }
