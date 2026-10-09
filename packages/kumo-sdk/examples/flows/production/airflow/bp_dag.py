# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param
from bp_utils import init, predict

# Define the default arguments for the DAG
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 6),  # example
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'params': {
        'kumo_url':
        Param("", type="string", description="Kumo API endpoint"),
        'kumo_api_key':
        Param("", type="string", description="Kumo API key"),
        'snowflake_warehouse':
        Param("", type="string", description="Snowflake warehouse"),
        'snowflake_database':
        Param("", type="string", description="Snowflake database"),
        'snowflake_schema':
        Param("", type="string", description="Snowflake schema"),
        'snowflake_identity_id':
        Param("", type="string", description="Snowflake identity ID"),
        'kumo_project_id':
        Param("", type="string", description="Kumo project"),
        'kumo_pq_name':
        Param("", type="string", description="Kumo predictive query name"),
        'kumo_training_job_id':
        Param("", type="string", description="Kumo training job ID")
    }
}


# Initialize the DAG with the defined arguments and schedule interval
@dag(
    'daily_kumo_batch_prediction',
    default_args=default_args,
    description='An Airflow DAG to run a Kumo daily BP job',
    schedule_interval="0 0 * * *"  # Run daily schedule
)
def KumoBatchPrediction():
    # Trigger a BP job using the latest model
    @task
    def trigger_bp(params):
        init(params)
        try:
            return predict(params)
        except Exception as e:
            print(f"Exception encountered: {e}")
            raise

    trigger_bp()


KumoBatchPrediction()
