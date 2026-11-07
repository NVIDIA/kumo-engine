# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preview interactive inspection of a small synthetic shop graph.
Synthetic data setup, starting in the NVIDIA/kumo-engine checkout:
  cd examples/kumo-example-notebooks
  python synthetic-data/engine_other/generate.py --dataset modules --output-dir ./example-data/modules
Use ./example-data/modules as SOURCE_ROOT for connector examples.
For prompts, pass prompts.parquet as --input; for collections, read interests.parquet.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install kumo-engine ipython
Run in IPython with a local SSD configured through KUMO_SCRATCH_DIR.
Graph visualization is a preview and requires a release exposing visualize.
"""
import os
from pathlib import Path

from kumo.engine import Edge, FileConnector, Graph, Table, configure

configure(scratch_dir=Path(os.environ['KUMO_SCRATCH_DIR']))
data = os.environ[
    'SOURCE_ROOT']  # Local directory generated with --dataset modules.
source = FileConnector(root_dir=data)
graph = Graph(
    {
        'users': Table.from_source_table(source['users'],
                                         primary_key='user_id'),
        'orders': Table.from_source_table(source['orders'],
                                          primary_key='order_id')
    }, [Edge('orders', 'user_id', 'users')])
graph.validate()
# Interactive graph preview:
# from IPython.display import display
# display(graph.visualize(show_cols=True).as_widget())
