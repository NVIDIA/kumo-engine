# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from service_framework.env_var import EnvVar

AWS_REGION = EnvVar(
    name="AWS_REGION",
    type=str,
    default_value='us-west-2',
    description="The aws region we are deploying.",
).parse_value()

AWS_ACCOUNT_ID = EnvVar(
    name="AWS_ACCOUNT_ID",
    type=str,
    default_value='926922431314',
    description="Kumo AWS account ID",
).parse_value()
