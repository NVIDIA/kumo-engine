# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development-only ambient credential policy helpers."""

import os

from service_framework.deploy_stage import deploy_stage

ENV_DEV_GCP_USER_CREDENTIALS_ENABLED = (
    'KUMO_DEV_GCP_USER_CREDENTIALS_ENABLED')


def dev_gcp_user_credentials_enabled() -> bool:
    """Whether this local process may use Google developer credentials.

    The explicit environment opt-in is necessary but not sufficient: even if
    it is accidentally copied into a deployment, non-development stages stay
    workload-credential-only.
    """
    value = os.environ.get(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED,
                           '').strip().lower()
    if value in {'', '0', 'false', 'no', 'off'}:
        return False
    if value not in {'1', 'true', 'yes', 'on'}:
        raise ValueError(
            f'{ENV_DEV_GCP_USER_CREDENTIALS_ENABLED} must be an explicit '
            'true/false value')
    return deploy_stage().is_dev
