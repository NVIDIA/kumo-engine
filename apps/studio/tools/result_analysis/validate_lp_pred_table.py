# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from kumo_core.job_config import JobConfig

from io_utils import fsutil, parquet_util
from io_utils.json import load_json
from rest.config.batch_prediction_job import BatchPredictionJobInput

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ls(dirpath: str):
    """
    List all files in the specified directory.

    Args:
        dirpath (str): Path to the directory to list files from.

    Logs:
        - Directory path
        - Names of all files in the directory
    """
    files = fsutil.listdir(dirpath)
    logger.info(f"Dirpath: {dirpath}")
    for f in files:
        logger.info(f' - {os.path.basename(f)}')


def validate_lp_pred_table(
    pred_table_path: str,
    pred_config_path: str,
    job_config_path: str,
    rhs_table_name: str,
    rhs_pkey_col_name: str,
    verbose: int = 1,
) -> None:
    """
    Validates a Link Prediction (LP) prediction table by ensuring all foreign
    keys exist in the corresponding RHS (right-hand side) table.

    Args:
        pred_table_path (str): Path to the prediction table parquet file.
        pred_config_path (str): Path to the prediction job configuration JSON
            file.
        job_config_path (str): Path to the main job configuration JSON file.
        rhs_table_name (str): Name of the RHS table to validate against.
        rhs_pkey_col_name (str): Name of the primary key column in the RHS
            table.
        verbose (int, optional): Verbosity level for logging. Defaults to 1.
            0: No logging
            1: Basic validation info
            2: Detailed config information

    Raises:
        ValueError: If any of the following conditions are met:
            - Duplicate TARGET values found in prediction table
            - RHS table not found in job config
            - TARGET values in prediction table don't exist in RHS table

    Notes:
        - The function checks for uniqueness of TARGET values in the
            prediction table
        - Validates that all TARGET values exist as primary keys in the RHS
            table
        - Provides detailed logging of missing foreign keys if count is less
            than 100
        - Opens a debugger (pdb) if validation fails
    """
    # Load prediction configuration
    pred_config = load_json(pred_config_path, BatchPredictionJobInput)
    if verbose > 1:
        print(pred_config)

    # Load job configuration
    job_config = load_json(job_config_path, JobConfig)
    if verbose > 1:
        print(job_config)

    # Read and validate prediction table
    if verbose > 0:
        logger.info(f"Validating LP pred table at {pred_table_path}")
    pred_table = parquet_util.read_parquet(pred_table_path)
    if verbose > 0:
        logger.info(f"Pred table columns: {pred_table.columns}")

    # Validate uniqueness of TARGET values
    fkeys = pred_table['TARGET']
    if not (fkeys.value_counts() == 1).all():
        raise ValueError("Some TARGET values appear more than once")

    # ``JobConfig.table_names_with_paths`` is unavailable on DG (the DG
    # short-circuit discards it before any reader runs). This debug tool
    # globbed ``features.db`` parquet under those paths to look up the
    # RHS table by name; the equivalent data on DG lives in the diskgraph
    # manifest, which needs a different reader. Until this tool is
    # rewritten to consume the DG manifest, the RHS lookup is
    # unavailable: raise instead of pretending to succeed.
    del rhs_pkey_col_name  # Unused; see raise below.
    raise NotImplementedError(
        "validate_lp_pred_table does not have access to table persist "
        "paths on DiskGraph; rewrite to consume the diskgraph manifest "
        "before re-enabling.")


# Example usage for Databricks environment
dirpath = (
    "dbfs:/Volumes/kumo_test_catalogue/test_output/"
    "test_volume/testing-uw2-test/"
    "nightly-test-hnm_10k_transactions_customer_id_list_temporal_target_filter"
    "/72e2915e-cd8d-4e46-95aa-0608b0dcfd63/nightly-905272be")
pred_table_subpath = ("/jobs/bp-job-11b46070ebce41b6be12ed050d259988/"
                      "pred_table/test_release_test_v1_42_3/pred_table")

pred_config_subpath = ("/jobs/bp-job-11b46070ebce41b6be12ed050d259988/"
                       "inputs/prediction_job_input.json")
job_config_subpath = ("/jobs/bp-job-11b46070ebce41b6be12ed050d259988/"
                      "jobs/bp-job-11b46070ebce41b6be12ed050d259988-0-3.json")

pred_table_path = (dirpath + pred_table_subpath)
pred_config_path = (dirpath + pred_config_subpath)
job_config_path = (dirpath + job_config_subpath)

# Example validation call
validate_lp_pred_table(
    pred_table_path=pred_table_path,
    pred_config_path=pred_config_path,
    job_config_path=job_config_path,
    rhs_table_name='CUSTOMERS',
    rhs_pkey_col_name='CUSTOMER_ID',
)
