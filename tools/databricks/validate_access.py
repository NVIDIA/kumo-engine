# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script to validate Databricks connectivity and UC volume permissions.

Example usage:
python tools/databricks/validate_access.py \
    --uc_volume_path \
    "dbfs:/Volumes/<catalog_name>/<schema_name>/<volume_name>/<path>" \
    --expected_runtime "14.3"
"""

import argparse
import os
import traceback

from databricks.sdk import WorkspaceClient  # noqa: E402

from io_utils import fsutil, logging
from io_utils.databricks import DatabricksClient, DatabricksConnectionInfo
from io_utils.fsutil.uri import UCURI, ParseURI  # noqa: E402

logging.initialize_logging(mode=logging.LoggingMode.PRODUCTION)
logger = logging.getLogger(__name__)


def validate_sql_connection(connection_info: DatabricksConnectionInfo) -> bool:
    """Validate SQL connectivity by attempting to create a connection."""
    connection = None
    try:
        client = DatabricksClient(connection_info)
        connection = client.create_sql_connection()
        logger.info("Successfully validated SQL connection")
        return True
    except Exception as e:
        logger.error(f"Failed to validate SQL connection: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False
    finally:
        if connection:
            connection.close()


def validate_databricks_runtime_version(
        connection_info: DatabricksConnectionInfo,
        expected_runtime: str | None = None) -> bool:
    """Validate Databricks runtime version by checking cluster information."""
    try:

        client_id = connection_info.credentials['client_id']
        client_secret = connection_info.credentials['client_secret']

        w = WorkspaceClient(host=connection_info.host, auth_type="oauth-m2m",
                            client_id=client_id.get_secret_value(),
                            client_secret=client_secret.get_secret_value())

        # Get cluster information
        cluster = w.clusters.get(connection_info.cluster_id)

        runtime_version = cluster.spark_version
        if not runtime_version:
            logger.error("spark_version is not available")
            return False

        logger.info(f"Customer Databricks runtime version: {runtime_version}")

        if expected_runtime:
            if runtime_version.startswith(expected_runtime):
                logger.info(
                    f"Runtime version matches expected: {expected_runtime}")
                logger.info("Successfully validated runtime version")
                return True
            else:
                logger.error(
                    f"Runtime version mismatch. Expected: {expected_runtime}, "
                    f"Got: {runtime_version}")
                return False

        logger.info("Successfully validated runtime version")
        return True
    except Exception as e:
        logger.error(f"Failed to validate runtime version: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def validate_uc_volume_access(uc_volume_path: str) -> bool:
    """Validate access to UC volume path."""
    try:
        if not isinstance(ParseURI(uc_volume_path), UCURI):
            logger.error(
                f"Path is not a valid UC volume path: {uc_volume_path}")
            return False

        # Check if path exists
        if not fsutil.exists(uc_volume_path):
            logger.error(f"UC volume path does not exist: {uc_volume_path}")
            return False

        # Try to list contents (read permission)
        try:
            contents = fsutil.listdir(uc_volume_path)
            logger.info(f"Successfully listed contents of volume. "
                        f"Found {len(contents)} items")
        except Exception as e:
            logger.error(f"Failed to list contents of UC volume "
                         f"(READ permission issue): {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

        # Try to create a test file (write permission)
        test_file = os.path.join(uc_volume_path, "test_write_permission.txt")
        try:
            with fsutil.get_handle(test_file, 'w') as f:
                f.write("test")
            logger.info("Successfully wrote test file")
            fsutil.rm(test_file)
            logger.info("Successfully removed test file")
        except Exception as e:
            logger.error(f"Failed to write/remove test file "
                         f"(WRITE permission issue): {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

        logger.info(
            f"Successfully validated access to UC volume: {uc_volume_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to validate UC volume access: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def main():
    # Parse command line arguments when run directly
    parser = argparse.ArgumentParser(
        description="Validate Databricks connectivity and UC volume access")
    parser.add_argument("--uc_volume_path", type=str, required=True,
                        help="UC volume path to validate access")
    parser.add_argument(
        "--expected_runtime", type=str, required=False, default='14.3',
        help="Expected Databricks runtime version (e.g., '14.3.x-scala2.12')")
    args = parser.parse_args()

    connection_info = DatabricksConnectionInfo.get()
    if not connection_info:
        raise ValueError("Failed to get Databricks connection info")

    if not validate_sql_connection(connection_info):
        raise ValueError("Failed to validate SQL connection")

    if not validate_databricks_runtime_version(connection_info,
                                               args.expected_runtime):
        raise ValueError("Failed to validate runtime version")

    if not validate_uc_volume_access(args.uc_volume_path):
        raise ValueError(f"Failed to validate access to UC volume: "
                         f"{args.uc_volume_path}")


if __name__ == "__main__":
    main()
