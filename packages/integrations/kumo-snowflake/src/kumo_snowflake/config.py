# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Snowflake developer-configuration helpers."""

from typing import Dict, Optional


def connection_toml_entry(
        connection_name: Optional[str] = None) -> Dict[str, str]:
    r"""Returns one ``connections.toml`` entry, or says what the file holds.

    The Snowflake connector's own config manager resolves the file's location,
    so ``SNOWFLAKE_HOME`` and the default directory behave exactly as they do
    for ``snowflake.connector`` itself.
    """
    from snowflake.connector.config_manager import CONFIG_MANAGER

    CONFIG_MANAGER.read_config()
    cache = CONFIG_MANAGER.conf_file_cache or {}
    connections = cache.get('connections', {})
    wanted = connection_name or cache.get('default_connection_name')
    entry = connections.get(wanted) if wanted else None
    if entry is None:
        known = ', '.join(repr(key) for key in sorted(connections)) or 'none'
        raise ValueError(
            f"connections.toml has no connection {wanted!r}. It defines: "
            f"{known}.")
    return dict(entry)
