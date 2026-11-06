# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule daily local Engine batch predictions on an Airflow worker.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset mlops --output-dir ./example-data/mlops
Pass ./example-data/mlops as --source-root where applicable.
The generator writes local files only. For remote connectors, load those files
into your own source first, preserving file stems as table names and column types.
Fill the connection configuration with your own account, database and schema.
Synthetic evaluation metrics are demonstration only.
Install: pip install apache-airflow "kumo-engine[s3,text]"

"""
import os
from datetime import datetime, timezone

from airflow.sdk import dag, task
from airflow_companions.local_kumo_engine_batch_prediction_helpers import (
    predict,
)


@dag(schedule='0 0 * * *', start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
     catchup=False)
def daily_engine_predictions():
    @task
    def score():
        from pathlib import Path
        from uuid import uuid4

        # Unique attempt paths avoid overwriting a prior successful run.
        attempt = uuid4().hex
        return predict(os.environ['KUMO_SOURCE_ROOT'],
                       os.environ['KUMO_MODEL_PATH'],
                       str(Path(os.environ['KUMO_SCRATCH_ROOT']) / attempt),
                       str(Path(os.environ['KUMO_OUTPUT_ROOT']) / attempt))

    score()


daily_engine_predictions()
