# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Copy a directory from Databricks, S3, or Snowflake to the local filesystem.

Example usage:

python scripts/debug/data_copy.py --s3_base_path \
    "s3://<bucket>/<path>"

python scripts/debug/data_copy.py --databricks_base_path \
    "dbfs:/Volumes/<path>"

python scripts/debug/data_copy.py --snowflake_relative_path \
    data-persist-dir --snowflake_app_name DM_TRIAL_1 \
        --snowflake_schema_name KUMO_APP_SCHEMA \
        --snowflake_stage_name LOCAL_STAGE \
        --snowflake_account_name TLB22710
"""
import argparse
import hashlib
import json
import logging
import os
from typing import Optional

from io_utils import fsutil
from io_utils.snowflake.session import setup_fsutil_session

kumo_repo_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(kumo_repo_base, "debug")


def copydir(base_path: str, target_path: str):
    os.makedirs(target_path, exist_ok=True)
    for subpath in fsutil.listdir(base_path):
        target_subpath = os.path.join(target_path, os.path.basename(subpath))
        print(f"Copying {subpath} to {target_subpath}")
        if fsutil.isdir(subpath):
            copydir(subpath, target_subpath)
        else:
            if not os.path.exists(target_subpath):
                try:
                    fsutil.copy(subpath, target_subpath)
                except Exception as e:
                    logger.error(
                        f"Error copying {subpath} to {target_subpath}: {e}")


def validate_permissions(
    databricks_base_path: Optional[str],
    snowflake_relative_path: Optional[str],
    s3_base_path: Optional[str],
) -> str:
    # Only one of the three arguments should be provided.
    if sum(x is not None for x in
           [databricks_base_path, snowflake_relative_path, s3_base_path]) != 1:
        raise ValueError("Only one of the three arguments should be provided.")

    # We are not allowed to copy customer data to the local filesystem for some
    # snowflake and databricks paths.
    # This script is only allowed to be run for internal tests jobs.
    if (databricks_base_path is not None
            and fsutil.is_ucv(databricks_base_path)):
        if "kumo_test_catalogue" not in databricks_base_path:
            raise ValueError(
                "You are not allowed to copy customer data to the local "
                "filesystem. If this is not customer data and involves a new "
                "internal path, you can submit a pull request to edit this "
                "script with approval from our security team.")
        return databricks_base_path
    elif (s3_base_path is not None and fsutil.is_s3(s3_base_path)
          and "kumo-prod-dataplane-bucket" in s3_base_path):
        print("You are copying data from the kumo-prod-dataplane-bucket. "
              "You may temporarily copy this data locally for debugging but "
              "you must remove it after you are done debugging.")
        return s3_base_path
    elif (snowflake_relative_path is not None):
        print(
            "You are copying data from Snowflake. You may temporarily copy "
            "this data locally for debugging but you must remove it after you "
            "are done debugging.")
        return snowflake_relative_path
    else:
        raise ValueError("Invalid path type.")


def authenticate_snowflake(
    account_name: str,
    app_name: str,
    schema_name: str,
    stage_name: str,
    relative_path: str,
) -> str:

    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")

    if user is None:
        raise ValueError("SNOWFLAKE_USER is not set. Set export "
                         "SNOWFLAKE_USER=kumo_snowservices")
    if password is None:
        raise ValueError(
            "SNOWFLAKE_PASSWORD is not set. Set export SNOWFLAKE_PASSWORD="
            "<from https://start.1password.com/open/i?a=BOTF4QWPWFAUNIAYMWX"
            "ORJSXYI&v=uxnits3zcd3v4ucjewoatj6lty&i=wh6vhyzwblohzmt6abeodnya"
            "im&h=team-kumo.1password.com>")

    print(f"Authenticating with Snowflake with account {account_name}, "
          f"user {user}, app {app_name}, schema {schema_name}, "
          f"stage {stage_name}, path {relative_path}")

    connection_parameters = {
        "account": account_name,
        "user": user,
        "password": password,
    }
    setup_fsutil_session(connection_parameters)

    os.environ['SNOWFLAKE_DATABASE'] = app_name
    os.environ['SNOWFLAKE_SCHEMA'] = schema_name
    os.environ['VOLUME_STAGE_MAP'] = json.dumps(
        {"/tmp/": f"{schema_name}.{stage_name}"})

    stage_base_path = f'/{account_name}_{app_name}_USER_SCHEMA/'
    stage_path = os.path.join(stage_base_path, relative_path)

    return f'@{app_name}.{schema_name}.{stage_name}{stage_path}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--databricks_base_path", type=str, required=False)
    parser.add_argument("--snowflake_relative_path", type=str, required=False)
    parser.add_argument("--s3_base_path", type=str, required=False)

    parser.add_argument("--snowflake_app_name", type=str, required=False)
    parser.add_argument("--snowflake_schema_name", type=str, required=False,
                        default="KUMO_APP_SCHEMA")
    parser.add_argument("--snowflake_stage_name", type=str, required=False,
                        default="LOCAL_STAGE")
    parser.add_argument("--snowflake_account_name", type=str, required=False,
                        default="TLB22710")

    args = parser.parse_args()

    databricks_base_path = args.databricks_base_path
    snowflake_relative_path = args.snowflake_relative_path
    s3_base_path = args.s3_base_path
    base_path = validate_permissions(
        databricks_base_path,
        snowflake_relative_path,
        s3_base_path,
    )

    if snowflake_relative_path is not None:
        base_path = authenticate_snowflake(
            args.snowflake_account_name,
            args.snowflake_app_name,
            args.snowflake_schema_name,
            args.snowflake_stage_name,
            args.snowflake_relative_path,
        )
    base_hash = hashlib.sha256(base_path.encode()).hexdigest()
    target_path = os.path.join(DATA_PATH, base_hash)
    print(f"Copying {base_path} to {target_path}")

    copydir(base_path, target_path)
