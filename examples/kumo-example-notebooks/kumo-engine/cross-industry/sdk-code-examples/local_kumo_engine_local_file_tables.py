# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read synthetic CSV and Parquet tables directly and snapshot their graph.
Synthetic data setup, starting in the NVIDIA/kumo-engine checkout:
  cd examples/kumo-example-notebooks
  python synthetic-data/engine_other/generate.py --dataset modules --output-dir ./example-data/modules
Use ./example-data/modules as SOURCE_ROOT for connector examples.
For prompts, pass prompts.parquet as --input; for collections, read interests.parquet.
The generator writes local files only. Point FileConnector at that directory.
Synthetic evaluation metrics are demonstration only.
Install: pip install kumo-engine
Set KUMO_SCRATCH_DIR to local SSD storage. Temporary source files remain alive
"""
import os
import tempfile
from pathlib import Path

import pandas as pd

from kumo.engine import FileConnector, Graph, Table, configure


def main():
    configure(scratch_dir=Path(os.environ['KUMO_SCRATCH_DIR']))
    with tempfile.TemporaryDirectory() as directory:
        rows = pd.read_parquet(
            Path(os.environ['SOURCE_ROOT']) / 'sensors.parquet')
        rows.to_parquet(Path(directory) / 'sensors.parquet', index=False)
        rows.to_csv(Path(directory) / 'sensors.csv', index=False)
        source = FileConnector(root_dir=directory)
        graph = Graph(
            {
                name: Table.from_source_table(source[file],
                                              primary_key='sensor_id')
                for name, file in [(
                    'parquet_sensors',
                    'sensors.parquet'), ('csv_sensors', 'sensors.csv')]
            }, [])
        graph.validate()
        snapshot = graph.snapshot()
        print('Snapshot:', snapshot.id)


if __name__ == '__main__':
    main()
