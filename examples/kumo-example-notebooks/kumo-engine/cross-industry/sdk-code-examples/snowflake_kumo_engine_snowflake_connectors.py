# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect a Snowflake schema with a configured local connector identity.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset modules --output-dir ./example-data/modules
Use ./example-data/modules as SOURCE_ROOT for connector examples.
For prompts, pass prompts.parquet as --input; for collections, read interests.parquet.
The generator writes local files only. For remote connectors, load those files
into your own source first, preserving file stems as table names and column types.
Fill the connection configuration with your own account, database and schema.
Synthetic evaluation metrics are demonstration only.
Install: pip install "kumo-engine[snowflake]"
Set SNOWFLAKE_CONNECTION_NAME to a connections.toml entry with password or
key-pair authentication, warehouse, database and schema. Set SOURCE_TABLE.
"""
import os
from kumo.engine import SnowflakeConnector

source = SnowflakeConnector.from_connection_toml(connection_name=os.environ['SNOWFLAKE_CONNECTION_NAME'])
print('Visible tables:', source.table_names()[:10])
table = source[os.environ['SOURCE_TABLE']]
print('Source schema:', table.columns)
