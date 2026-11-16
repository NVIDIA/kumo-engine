# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the administrator identity lifecycle on the SDK client.

These cover the customer-admin `/v1/identities` surface added so the SDK can
create and manage identities end-to-end, not just discover them. The gate
itself (admin-only) is enforced server-side; here we assert the client targets
the correct method + route and (de)serializes safely.
"""
from unittest.mock import MagicMock, patch

from kumoapi.data_source import (
    CreateIdentityArgs,
    ReplaceIdentityGroupsArgs,
    RotateIdentityCredentialsArgs,
    S3AssumeRoleAuthentication,
    S3IdentityConfig,
    UpdateIdentityArgs,
)

import kumoai_studio
from kumoai_studio.client.client import KumoClient

BASE_URL = "https://kumo.ai"
_MINIO_CA = """-----BEGIN CERTIFICATE-----
MIIBSjCB/aADAgECAhRiPITwqOT0gz4XKMTFMkS41Z2f7jAFBgMrZXAwGzEZMBcG
A1UEAwwQa3Vtby1hcGktdGVzdC1jYTAeFw0yNjA4MTIxOTM3MDNaFw0yNjA4MTMx
OTM3MDNaMBsxGTAXBgNVBAMMEGt1bW8tYXBpLXRlc3QtY2EwKjAFBgMrZXADIQBv
iURRF3Kk370JkN7pUmmxmRXKOToEyBp1t/09UPGNsaNTMFEwHQYDVR0OBBYEFCkf
DW84FLpOgv59tsdFJSoUlLnjMB8GA1UdIwQYMBaAFCkfDW84FLpOgv59tsdFJSoU
lLnjMA8GA1UdEwEB/wQFMAMBAf8wBQYDK2VwA0EAVDph44XtYxnNfHE+ISJET136
I1mq8yFZ7bQkQ9MEleKKvXO13frbEvWZeBaX34PR+ZymWlEmuxiDRsIYTxA+DQ==
-----END CERTIFICATE-----"""


def _identity_json(identity_id: str = "identity-aws") -> dict:
    return {
        "id": identity_id,
        "slug": "prod-aws",
        "display_name": "Production AWS",
        "description": "",
        "timestamps": {
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        },
        "credential_status": {
            "configured": True,
            "last_rotated_at": "2026-01-01T00:00:00Z",
        },
        "config": {
            "type": "s3",
            "authentication": {
                "mode": "assume_role",
                "partition": "aws",
                "role_arn": "arn:aws:iam::123456789012:role/production",
            },
        },
    }


def _s3_config() -> S3IdentityConfig:
    return S3IdentityConfig(authentication=S3AssumeRoleAuthentication(
        partition="aws",
        role_arn="arn:aws:iam::123456789012:role/production",
    ))


def _minio_identity_json() -> dict:
    response = _identity_json("identity-minio")
    response["slug"] = "private-minio"
    response["display_name"] = "Private MinIO"
    response["config"] = {
        "type": "s3",
        "authentication": {
            "mode": "access_key",
        },
        "connection": {
            "endpoint_url": "https://minio.example.com:9000",
            "region": "us-east-1",
            "addressing_style": "path",
        },
    }
    return response


def test_create_posts_to_admin_identities_route(requests_mock):
    matcher = requests_mock.post(f"{BASE_URL}/v1/identities",
                                 json=_identity_json())
    client = KumoClient(BASE_URL, "api-key")

    identity = client.identity_api.create(
        CreateIdentityArgs(display_name="Production AWS", config=_s3_config()))

    assert identity.id == "identity-aws"
    assert identity.type.value == "s3"
    assert not hasattr(identity, "credentials")
    body = matcher.last_request.json()
    assert body["display_name"] == "Production AWS"
    assert body["config"]["type"] == "s3"


def test_public_sdk_creates_minio_access_key_identity(requests_mock):
    matcher = requests_mock.post(f"{BASE_URL}/v1/identities",
                                 json=_minio_identity_json())
    client = KumoClient(BASE_URL, "api-key")
    args = kumoai_studio.CreateIdentityArgs(
        display_name="Private MinIO",
        group_ids=["group-analytics"],
        config=kumoai_studio.S3IdentityConfig(
            authentication=kumoai_studio.S3AccessKeyAuthentication(),
            connection=kumoai_studio.S3ConnectionConfig(
                endpoint_url="https://minio.example.com:9000",
                region="us-east-1",
                addressing_style=kumoai_studio.S3AddressingStyle.PATH,
            ),
        ),
        credentials=kumoai_studio.s3_access_key_credentials(
            access_key_id="minio-access-key",
            secret_access_key="minio-secret-key",
            ca_certificate=_MINIO_CA,
        ),
    )

    identity = client.identity_api.create(args)

    assert identity.id == "identity-minio"
    assert isinstance(identity.config, kumoai_studio.S3IdentityConfig)
    assert isinstance(args.config, kumoai_studio.S3IdentityConfig)
    assert identity.config.connection == args.config.connection
    body = matcher.last_request.json()
    assert body["config"]["connection"] == {
        "endpoint_url": "https://minio.example.com:9000",
        "region": "us-east-1",
        "addressing_style": "path",
    }
    assert body["credentials"]["kind"] == "s3_access_key"
    assert body["credentials"]["access_key_id"] == "minio-access-key"
    assert body["credentials"]["secret_access_key"] == "minio-secret-key"
    assert body["credentials"]["ca_certificate"] == _MINIO_CA
    assert "minio-secret-key" not in repr(args)


def test_public_sdk_rotates_minio_credentials_without_losing_pem(
        requests_mock):
    matcher = requests_mock.put(
        f"{BASE_URL}/v1/identities/identity-minio/credentials",
        json=_minio_identity_json())
    client = KumoClient(BASE_URL, "api-key")
    args = kumoai_studio.RotateIdentityCredentialsArgs(
        credentials=kumoai_studio.s3_access_key_credentials(
            access_key_id="replacement-access-key",
            secret_access_key="replacement-secret-key",
            ca_certificate=_MINIO_CA,
        ))

    client.identity_api.rotate_credentials("identity-minio", args)

    body = matcher.last_request.json()
    assert body["credentials"]["access_key_id"] == "replacement-access-key"
    assert body["credentials"]["secret_access_key"] == (
        "replacement-secret-key")
    assert body["credentials"]["ca_certificate"] == _MINIO_CA


def test_s3_compatible_identity_models_are_public_sdk_exports():
    names = {
        "CreateIdentityArgs",
        "RotateIdentityCredentialsArgs",
        "S3AccessKeyAuthentication",
        "S3AccessKeyCredentials",
        "S3AddressingStyle",
        "S3ConnectionConfig",
        "S3IdentityConfig",
        "s3_access_key_credentials",
    }

    assert names <= set(kumoai_studio.__all__)
    assert all(hasattr(kumoai_studio, name) for name in names)


def test_list_admin_targets_tenant_wide_route(requests_mock):
    requests_mock.get(f"{BASE_URL}/v1/identities", json=[_identity_json()])
    client = KumoClient(BASE_URL, "api-key")

    identities = client.identity_api.list_admin()

    assert [identity.id for identity in identities] == ["identity-aws"]
    assert requests_mock.last_request.qs == {}


def test_update_patches_identity_metadata(requests_mock):
    matcher = requests_mock.patch(f"{BASE_URL}/v1/identities/identity-aws",
                                  json=_identity_json())
    client = KumoClient(BASE_URL, "api-key")

    client.identity_api.update("identity-aws",
                               UpdateIdentityArgs(display_name="Renamed"))

    assert matcher.last_request.method == "PATCH"
    assert matcher.last_request.json()["display_name"] == "Renamed"


def test_delete_targets_identity_item(requests_mock):
    matcher = requests_mock.delete(f"{BASE_URL}/v1/identities/identity-aws",
                                   json=_identity_json())
    client = KumoClient(BASE_URL, "api-key")

    identity = client.identity_api.delete("identity-aws")

    assert identity.id == "identity-aws"
    assert matcher.last_request.method == "DELETE"


def test_rotate_credentials_uses_put(requests_mock):
    matcher = requests_mock.put(
        f"{BASE_URL}/v1/identities/identity-aws/credentials",
        json=_identity_json())
    client = KumoClient(BASE_URL, "api-key")

    client.identity_api.rotate_credentials("identity-aws",
                                           RotateIdentityCredentialsArgs())

    assert matcher.last_request.method == "PUT"


def test_test_credentials_posts_to_test_route(requests_mock):
    matcher = requests_mock.post(f"{BASE_URL}/v1/identities/identity-aws/test",
                                 json=_identity_json())
    client = KumoClient(BASE_URL, "api-key")

    identity = client.identity_api.test_credentials("identity-aws")

    assert identity.id == "identity-aws"
    assert matcher.last_request.method == "POST"


def test_list_groups_returns_binding_set(requests_mock):
    requests_mock.get(
        f"{BASE_URL}/v1/identities/identity-aws/groups", json={
            "identity_id": "identity-aws",
            "group_ids": ["group-risk"]
        })
    client = KumoClient(BASE_URL, "api-key")

    groups = client.identity_api.list_groups("identity-aws")

    assert groups.identity_id == "identity-aws"
    assert groups.group_ids == ["group-risk"]


def test_replace_groups_uses_put(requests_mock):
    matcher = requests_mock.put(
        f"{BASE_URL}/v1/identities/identity-aws/groups", json={
            "identity_id": "identity-aws",
            "group_ids": ["group-risk"]
        })
    client = KumoClient(BASE_URL, "api-key")

    groups = client.identity_api.replace_groups(
        "identity-aws", ReplaceIdentityGroupsArgs(group_ids=["group-risk"]))

    assert groups.group_ids == ["group-risk"]
    assert matcher.last_request.method == "PUT"
    assert matcher.last_request.json()["group_ids"] == ["group-risk"]


# -- Module-level wrappers (kumoai_studio.*) delegate to the client API -----------


def test_create_identity_delegates_to_client_api():
    state = MagicMock()
    args = CreateIdentityArgs(display_name="prod-aws", config=_s3_config())
    with patch("kumoai_studio.global_state", state):
        kumoai_studio.create_identity(args)
    state.client.identity_api.create.assert_called_once_with(args)


def test_list_all_identities_delegates_to_admin_list():
    state = MagicMock()
    with patch("kumoai_studio.global_state", state):
        kumoai_studio.list_all_identities()
    state.client.identity_api.list_admin.assert_called_once_with()


def test_update_identity_builds_args_from_kwargs():
    state = MagicMock()
    with patch("kumoai_studio.global_state", state):
        kumoai_studio.update_identity("identity-aws", display_name="Renamed")
    identity_id, args = state.client.identity_api.update.call_args.args
    assert identity_id == "identity-aws"
    assert isinstance(args, UpdateIdentityArgs)
    assert args.display_name == "Renamed"
    assert args.description is None


def test_delete_identity_delegates_to_client_api():
    state = MagicMock()
    with patch("kumoai_studio.global_state", state):
        kumoai_studio.delete_identity("identity-aws")
    state.client.identity_api.delete.assert_called_once_with("identity-aws")


def test_test_identity_credentials_delegates_to_client_api():
    state = MagicMock()
    with patch("kumoai_studio.global_state", state):
        kumoai_studio.test_identity_credentials("identity-aws")
    state.client.identity_api.test_credentials.assert_called_once_with(
        "identity-aws")


def test_set_identity_groups_builds_replace_args():
    state = MagicMock()
    with patch("kumoai_studio.global_state", state):
        kumoai_studio.set_identity_groups("identity-aws", ["group-risk"])
    identity_id, args = state.client.identity_api.replace_groups.call_args.args
    assert identity_id == "identity-aws"
    assert isinstance(args, ReplaceIdentityGroupsArgs)
    assert args.group_ids == ["group-risk"]


def test_a_container_token_identity_is_refused_before_the_round_trip(
        requests_mock):
    """A Snowflake ``OAUTH_TOKEN_FILE`` identity authenticates with a token
    mounted into a running SPCS container. The platform is not that container,
    so the identity could never connect. The server refuses it too; failing
    here saves a round trip.
    """
    import pytest
    from kumoapi.data_source import (
        SnowflakeAuthenticationKind,
        SnowflakeIdentityConfig,
    )

    matcher = requests_mock.post(f"{BASE_URL}/v1/identities",
                                 json=_identity_json())
    client = KumoClient(BASE_URL, "api-key")

    with pytest.raises(ValueError, match="does not accept this kind"):
        client.identity_api.create(
            CreateIdentityArgs(
                display_name="Snowflake SPCS", config=SnowflakeIdentityConfig(
                    account="xy12345", authentication=(
                        SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE))))

    assert not matcher.called
