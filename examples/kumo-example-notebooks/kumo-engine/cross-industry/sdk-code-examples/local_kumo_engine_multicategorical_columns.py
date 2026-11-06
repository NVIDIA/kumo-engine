# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure delimited strings and native arrays as categorical collections.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset modules --output-dir ./example-data/modules
Use ./example-data/modules as SOURCE_ROOT for connector examples.
For prompts, pass prompts.parquet as --input; for collections, read interests.parquet.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install kumo-engine

"""
from kumo.engine import SourceTable, Table
from kumoapi.typing import Dtype, Stype


def configure_multicategorical_table(source_table: SourceTable) -> Table:
    return (Table.from_source_table(source_table)
            .with_column('interests', stype=Stype.multicategorical,
                         dtype=Dtype.string, multicategorical_delimiter='|')
            .with_column('product_ids', stype=Stype.multicategorical,
                         dtype=Dtype.intlist))
