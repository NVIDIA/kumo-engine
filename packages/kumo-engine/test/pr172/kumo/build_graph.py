# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind the prepared synthetic tables to their fixed public graph metadata."""

import json
from dataclasses import replace
from pathlib import Path


def kumo_graph(name, files, variant='full', task=None):
    """Build one admitted graph without a source checkout or registry."""
    graphs = json.loads((Path(__file__).resolve().parents[1] /
                         'graphs.json').read_text())
    if name not in graphs or variant != 'full':
        raise ValueError('Unsupported prepared graph or variant')
    spec = graphs[name]
    if (task or spec['default_task']) not in spec['tasks']:
        raise ValueError('Unsupported prepared graph task')

    from kumoapi.graph import Edge
    from kumoapi.table import Column

    from kumo.engine import Graph, Table

    tables = {}
    for table_name, definition in spec['tables'].items():
        columns = [Column(**column) for column in definition['cols']]
        table = Table.from_source_table(
            files[table_name], [column.name for column in columns],
            primary_key=definition.get('pkey'),
            time_column=definition.get('time_col'),
            end_time_column=definition.get('end_time_col'),
            dtypes={column.name: column.dtype for column in columns},
        )
        tables[table_name] = replace(
            table, definition=replace(table.definition, cols=columns))
    return Graph(tables, [Edge(**edge) for edge in spec['edges']]), None
