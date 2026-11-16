# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def parse_failure_warning(table: str, column: str, count: int) -> str:
    return (f"{count} invalid values found in {table}.{column}. "
            "Invalid values are parsed as null. "
            "Please correct and retrain if needed.")
