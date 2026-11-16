# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from service_framework.env_var import EnvVar

ALLOW_ARRAY_TARGETS = EnvVar('ALLOW_ARRAY_TARGETS', bool, required=False,
                             default_value=False).parse_value()
ENABLE_INDIVIDUAL_SPLIT_TABLES = EnvVar('ENABLE_INDIVIDUAL_SPLIT_TABLES', bool,
                                        default_value=False).parse_value()


def allow_individual_split_tables() -> bool:
    return ENABLE_INDIVIDUAL_SPLIT_TABLES
