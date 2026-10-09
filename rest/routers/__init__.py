# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: List the exported variables here
from service_framework.env_var import EnvVar

DISABLE_UI_WARNINGS = EnvVar(
    'DISABLE_UI_WARNINGS',
    type=bool,
    default_value=False,
    required=False,
    description='If enabled, warnings will be disabled in the UI.',
).parse_value()

__all__ = []
