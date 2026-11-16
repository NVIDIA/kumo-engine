# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Union

from kumoapi.pquery import ParsedPredictiveQuery, ValidatedPredictiveQuery

from kumo.graph.graph import Graph
from kumo.pquery.utils import table_name


def _set_pq_credentials(
    parsed_query: Union[ParsedPredictiveQuery, ValidatedPredictiveQuery],
    graph: Graph,
    credentials: Dict[str, Dict],
):
    # Assumption: all used tables will appear in the query.
    # This currently holds because all inferred keys and time cols indeed
    # come from tables that must have appeared in the query.
    # This could potentially break with multi-hop if we hop through a
    # table that does not appear in the query by name (depends on future
    # syntax choices)
    all_tables = set(
        table_name(fqn_col) for fqn_col in parsed_query.all_query_columns)
    all_connectors = [graph.table(t_name).connector for t_name in all_tables]
    # Filter duplicates:
    all_connectors = list({c.name: c for c in all_connectors}.values())
    for c in all_connectors:
        c.set_credentials(credentials.get(c.name))
