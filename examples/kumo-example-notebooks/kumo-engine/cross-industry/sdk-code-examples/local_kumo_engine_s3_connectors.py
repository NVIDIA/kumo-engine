# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect rooted and any-URI Amazon S3 connectors using ambient AWS credentials.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset modules --output-dir ./example-data/modules
Use ./example-data/modules as SOURCE_ROOT for connector examples.
For prompts, pass prompts.parquet as --input; for collections, read interests.parquet.
The generator writes local files only. For remote connectors, load those files
into your own source first, preserving file stems as table names and column types.
Fill the connection configuration with your own account, database and schema.
Synthetic evaluation metrics are demonstration only.
Install: pip install "kumo-engine[s3]"
Set SOURCE_ROOT to an authorized S3 prefix, SOURCE_TABLE to a table below it,
and SOURCE_URI to a complete S3 table URI. These checks perform source reads.
"""
import os

from kumo.engine import S3Connector

rooted = S3Connector(root_dir=os.environ['SOURCE_ROOT'])
print('Visible tables:', rooted.table_names()[:10])
print('Rooted schema:', rooted[os.environ['SOURCE_TABLE']].columns)
any_uri = S3Connector()
print('Any-URI schema:', any_uri[os.environ['SOURCE_URI']].columns)
# Discovery requires a root; any-URI mode addresses each table explicitly.
