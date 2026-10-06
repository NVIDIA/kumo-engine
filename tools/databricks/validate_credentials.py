# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script to validate Databricks credential formats.

Example usage:
python tools/databricks/validate_credentials.py \
    --client_id <client_id> \
    --client_secret <client_secret> \
    --cluster_id <cluster_id> \
    --warehouse_id <warehouse_id> \
    --host_url <host_url>

"""

import argparse
import logging
import re
from urllib.parse import urlparse

from io_utils.databricks import validate_host

logger = logging.getLogger(__name__)


def validate_client_id(client_id: str) -> tuple[bool, str | None]:
    """
    Validate Databricks client ID format: alphanumeric, can contain dashes.
    """
    if not client_id or not isinstance(client_id, str):
        return False, "Client ID cannot be empty."
    if not re.match(r'^[a-zA-Z0-9-]+$', client_id):
        return False, (f"Client ID '{client_id}' contains invalid characters. "
                       "Only alphanumeric characters and dashes are allowed.")
    return True, None


def validate_client_secret(client_secret: str) -> tuple[bool, str | None]:
    """
    Validate Databricks client secret: alphanumeric, must start with 'dose'.
    """
    if not client_secret or not isinstance(client_secret, str):
        return False, "Client secret cannot be empty."
    if not client_secret.startswith('dose'):
        return False, "Client secret must start with the prefix 'dose'."
    if not re.match(r'^dose[a-zA-Z0-9]+$', client_secret):
        return False, ("Client secret contains invalid characters. "
                       "It must be alphanumeric and start with 'dose'. "
                       "A client secret is not a personal access token.")
    return True, None


def validate_cluster_id(cluster_id: str) -> tuple[bool, str | None]:
    """
    Validate Databricks cluster ID: alphanumeric, can contain dashes.
    """
    if not cluster_id or not isinstance(cluster_id, str):
        return False, "Cluster ID cannot be empty."
    if not re.match(r'^[a-zA-Z0-9-]+$', cluster_id):
        return False, (
            f"Cluster ID '{cluster_id}' contains invalid characters. "
            "Only alphanumeric characters and dashes are allowed.")
    return True, None


def validate_warehouse_id(warehouse_id: str) -> tuple[bool, str | None]:
    """
    Validate Databricks warehouse ID: strictly alphanumeric.
    """
    if not warehouse_id or not isinstance(warehouse_id, str):
        return False, "Warehouse ID cannot be empty."
    if not re.match(r'^[a-zA-Z0-9]+$', warehouse_id):
        return False, (
            f"Warehouse ID '{warehouse_id}' contains invalid characters. "
            "Only alphanumeric characters are allowed.")
    return True, None


def validate_host_url(host_url: str) -> tuple[bool, str | None]:
    """Sanitize and validate Databricks host URL. Clean with validate_host,
       then check domain.
    """
    if not host_url or not isinstance(host_url, str):
        return False, "Host URL cannot be empty."
    try:
        cleaned_url = validate_host(host_url)
        parsed = urlparse(cleaned_url)
        hostname = parsed.netloc.lower()
        if not (hostname.endswith('.net') or hostname.endswith('.com')):
            return False, (
                f"Host URL '{cleaned_url}' does not end with .net or .com. "
                "Databricks workspace URLs should end with these domains.")
        return True, None
    except Exception as e:
        return False, (f"Failed to validate host URL '{host_url}': {str(e)}")


def main():
    """Function to validate Databricks credential formats."""

    parser = argparse.ArgumentParser(
        description="Validate Databricks credential formats")
    parser.add_argument("--client_id", type=str, required=True,
                        help="Databricks client ID")
    parser.add_argument("--client_secret", type=str, required=True,
                        help="Databricks client secret")
    parser.add_argument("--cluster_id", type=str, required=True,
                        help="Databricks cluster ID")
    parser.add_argument("--warehouse_id", type=str, required=True,
                        help="Databricks warehouse ID")
    parser.add_argument("--host_url", type=str, required=True,
                        help="Databricks host URL")

    args = parser.parse_args()

    logger.info("Starting Databricks credential format validation...")

    credentials = [
        ("client_id", args.client_id, validate_client_id),
        ("client_secret", args.client_secret, validate_client_secret),
        ("cluster_id", args.cluster_id, validate_cluster_id),
        ("warehouse_id", args.warehouse_id, validate_warehouse_id),
        ("host_url", args.host_url, validate_host_url),
    ]

    failed = []
    error_messages = []
    for name, value, validator in credentials:
        valid, msg = validator(value)
        if not valid:
            logger.error(msg)
            failed.append(name)
            error_messages.append(msg)

    if failed:
        error_summary = f"Validation failed for: {', '.join(failed)}"
        logger.error(error_summary)
        logger.error("Please check the error messages above and correct the "
                     "credential formats.")

        # Raise an error with all validation messages
        full_error_message = f"{error_summary}\n" + "\n".join(error_messages)
        raise ValueError(full_error_message)

    else:
        logger.info("All Databricks credential formats are valid!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
