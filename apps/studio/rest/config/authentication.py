# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List, Literal, Optional

from io_utils import kumo_dataclass

PYTEST_TOKEN = 'PYTEST_NOT_NECESSARY'
PYTEST_USER_ID = 'direct_connect|pytest'
PYTEST_API_KEY = 'NOT_NECESSARY'
# TODO(frg100): Switch this to 'kumo'


@kumo_dataclass
class AuthenticationSettings:
    # If true, bypasses authentication and allows all requests to go through.
    disable: bool = False

    # If true, allows basic READ/WRITE requests to go through.
    proxy: bool = False

    # AWS IAM role for the customer. This needs to be supplied if the customer
    # data is in S3.
    kumo_external_shared_iam_role: Optional[str] = None
    is_unit_test: bool = False

    direct_connect_login_method: Literal['oidc', 'password'] = 'oidc'
    direct_connect_allowed_return_origins: List[str] = field(
        default_factory=list)
    # A source-IP-wide password-login bucket is safe only when the deployment
    # can supply a distinct, trustworthy client address. Frontend-gateway
    # deployments behind an arbitrary customer edge disable this bucket while
    # retaining the per-email password-login limit.
    direct_connect_password_login_ip_rate_limit_enabled: bool = True


@kumo_dataclass
class ClientAuthConfigResponse:
    auth_disabled: bool
    direct_connect_login_method: Optional[Literal['oidc', 'password']] = None


@kumo_dataclass
class AdminInformation:
    api_enabled: bool


@kumo_dataclass
class AdminResponse:
    organization_name: str
    user_is_customer_admin: bool
    customer_admin_emails: List[str]
    # If user_is_admin
    customer_admin_info: Optional[AdminInformation]


@kumo_dataclass
class OrganizationMember:
    name: str
    email: str
    uid: str
    roles: List[Literal["Admin", "User"]]


RequestsResponse = Dict[str, Any]
