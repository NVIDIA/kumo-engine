# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-MinIO S3 API objects.

These payloads were captured from kumo-api commit
5588e0dc3e2bc04178082d26afba62abc8288222.  They intentionally contain no
S3-compatible connection metadata and protect the existing AWS defaults while
the S3 identity contract grows new authentication and transport variants.
"""

from pydantic import SecretStr

from kumoapi.data_source import (
    CreateConnectorArgs,
    CreateIdentityArgs,
    RotateIdentityCredentialsArgs,
    S3AmbientAuthentication,
    S3AssumeRoleAuthentication,
    S3AssumeRoleCredentials,
    S3ConnectorResourceConfig,
    S3IdentityConfig,
    S3RootedConnectorTarget,
)
from kumoapi.json_serde import from_json, to_json_dict

_LEGACY_ASSUME_ROLE = """{
  "display_name": "Legacy role",
  "config": {
    "authentication": {
      "partition": "aws",
      "role_arn": "arn:aws:iam::123456789012:role/kumo-reader",
      "mode": "assume_role"
    },
    "type": "s3"
  },
  "group_ids": [],
  "credentials": {
    "external_id": "legacy-external",
    "kind": "s3_assume_role"
  },
  "description": "",
  "slug": null
}"""

_LEGACY_AMBIENT = """{
  "display_name": "Legacy ambient",
  "config": {
    "authentication": {"mode": "ambient"},
    "type": "s3"
  },
  "group_ids": [],
  "credentials": null,
  "description": "",
  "slug": null
}"""

_LEGACY_ROOTED_CONNECTOR = """{
  "name": "Legacy S3",
  "config": {
    "target": {
      "bucket": "company-data",
      "prefix": "customer",
      "region": "us-west-2",
      "mode": "rooted"
    },
    "type": "s3"
  },
  "identity_id": null,
  "description": ""
}"""

_LEGACY_ROTATION = """{
  "credentials": {
    "external_id": "replacement",
    "kind": "s3_assume_role"
  },
  "authentication": null
}"""


def test_pre_minio_assume_role_identity_remains_unchanged() -> None:
    payload = from_json(_LEGACY_ASSUME_ROLE, CreateIdentityArgs)

    assert payload == CreateIdentityArgs(
        display_name='Legacy role',
        config=S3IdentityConfig(
            authentication=S3AssumeRoleAuthentication(
                partition='aws',
                role_arn='arn:aws:iam::123456789012:role/kumo-reader',
            ), ),
        credentials=S3AssumeRoleCredentials(
            external_id=SecretStr('legacy-external')),
    )
    assert payload.config.connection is None
    assert to_json_dict(payload)['config']['connection'] is None


def test_pre_minio_ambient_identity_remains_unchanged() -> None:
    payload = from_json(_LEGACY_AMBIENT, CreateIdentityArgs)

    assert payload == CreateIdentityArgs(
        display_name='Legacy ambient',
        config=S3IdentityConfig(authentication=S3AmbientAuthentication()),
    )
    assert payload.config.connection is None
    assert to_json_dict(payload)['config']['connection'] is None


def test_pre_minio_rooted_connector_remains_unchanged() -> None:
    payload = from_json(_LEGACY_ROOTED_CONNECTOR, CreateConnectorArgs)

    assert payload == CreateConnectorArgs(
        name='Legacy S3',
        config=S3ConnectorResourceConfig(
            target=S3RootedConnectorTarget(
                bucket='company-data',
                prefix='customer',
                region='us-west-2',
            ), ),
    )
    assert to_json_dict(payload)['config'] == {
        'target': {
            'bucket': 'company-data',
            'prefix': 'customer',
            'region': 'us-west-2',
            'mode': 'rooted',
        },
        'type': 's3',
    }


def test_pre_minio_assume_role_rotation_remains_unchanged() -> None:
    payload = from_json(_LEGACY_ROTATION, RotateIdentityCredentialsArgs)

    assert payload == RotateIdentityCredentialsArgs(
        credentials=S3AssumeRoleCredentials(
            external_id=SecretStr('replacement')))
    assert to_json_dict(payload, insecure=True) == {
        'credentials': {
            'external_id': 'replacement',
            'kind': 's3_assume_role',
        },
        'authentication': None,
    }
