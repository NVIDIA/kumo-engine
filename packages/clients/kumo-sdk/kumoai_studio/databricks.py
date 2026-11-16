# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DB_SEP = '__kumo__'


def to_db_table_name(table_name: str | tuple[str, ...] | None) -> str | None:
    r"""For Databricks connectors, return the table name whichs is
    a Tuple as a string with the format `f"{schema}__kumo__{table}"`.
    """
    if isinstance(table_name, tuple):
        return f"{table_name[0]}{DB_SEP}{table_name[1]}"
    return table_name
