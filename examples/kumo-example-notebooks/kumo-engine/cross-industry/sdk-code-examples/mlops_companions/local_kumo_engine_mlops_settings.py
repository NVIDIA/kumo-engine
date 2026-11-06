# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared local fit/score settings for the users repeat-purchase workflow.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset mlops --output-dir ./example-data/mlops
Pass ./example-data/mlops as --source-root where applicable.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.


"""
import argparse
from pathlib import Path

from kumo.engine import (
    Edge,
    FileConnector,
    Graph,
    S3Connector,
    Table,
    configure,
)

QUERY = 'PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.USER_ID'


def parse_args(score=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-root', required=True,
        help='Local or S3 users/items/orders parquet directory.')
    parser.add_argument('--scratch-dir', type=Path, required=True,
                        help='Local SSD working directory.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New retained output directory.')
    if score:
        parser.add_argument('--model-path', type=Path, required=True,
                            help='Exact trusted local model bundle.')
    args = parser.parse_args()
    scratch, output = args.scratch_dir.resolve(), args.output_dir.resolve()
    if scratch.is_relative_to(output) or output.is_relative_to(scratch):
        parser.error('Scratch and output directories must be separate.')
    output.mkdir(parents=True, exist_ok=False)
    configure(scratch_dir=scratch)
    return args


def build_graph(root):
    source = S3Connector(
        root_dir=root) if root.startswith('s3://') else FileConnector(
            root_dir=root)
    return Graph(
        {
            'users':
            Table.from_source_table(source['users.parquet'],
                                    primary_key='USER_ID'),
            'items':
            Table.from_source_table(source['items.parquet'],
                                    primary_key='ITEM_ID'),
            'orders':
            Table.from_source_table(source['orders.parquet'],
                                    primary_key='ORDER_ID',
                                    time_column='ORDER_TIME'),
        }, [
            Edge('orders', 'USER_ID', 'users'),
            Edge('orders', 'ITEM_ID', 'items')
        ])
