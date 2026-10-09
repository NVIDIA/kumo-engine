# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unicodedata


def normalize_resource_alias(value: str) -> str:
    """Return the canonical project-local table or graph alias key."""
    return unicodedata.normalize('NFKC', value).strip().casefold()


def graph_alias_conflict_message(name: str) -> str:
    return f'Graph name {name} is already used in this project.'
