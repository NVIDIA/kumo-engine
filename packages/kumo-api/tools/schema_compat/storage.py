# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""S3 storage operations for schema files.

This module handles reading and writing schema JSON files to/from S3.
"""

import json
import re
import urllib.request
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

from tools.schema_compat.config import (
    ENFORCEMENT_BASELINE_VERSION,
    S3_SCHEMA_BASE_PATH,
    get_pydantic_schema_key,
)
from tools.schema_compat.models import FullSchema


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse an S3 path into bucket and key.

    Args:
        s3_path: S3 path like "s3://bucket/prefix/"

    Returns:
        Tuple of (bucket, key_prefix)
    """
    parsed = urlparse(s3_path)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 path: {s3_path}")

    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    return bucket, key


def get_schema_s3_key(version: str,
                      pydantic_version: Optional[int] = None) -> str:
    """Get the S3 key for a schema file.

    Args:
        version: Version string (e.g., "0.52.0")
        pydantic_version: Pydantic major version (1 or 2). If None, current.

    Returns:
        S3 key like "schemas/v0.52.0/pydantic-v2.json"
    """
    _, prefix = parse_s3_path(S3_SCHEMA_BASE_PATH)

    if pydantic_version is None:
        schema_file = get_pydantic_schema_key()
    else:
        schema_file = f"pydantic-v{pydantic_version}.json"

    return f"{prefix}v{version}/{schema_file}"


def fetch_schema(
    version: str,
    pydantic_version: Optional[int] = None,
) -> Optional[FullSchema]:
    """Fetch a schema from S3.

    Args:
        version: Version string (e.g., "0.52.0")
        pydantic_version: Pydantic major version. If None, uses current.

    Returns:
        FullSchema if found, None if not found
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. "
                          "Install with: pip install boto3")

    bucket, _ = parse_s3_path(S3_SCHEMA_BASE_PATH)
    key = get_schema_s3_key(version, pydantic_version)

    s3 = boto3.client('s3')

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        return FullSchema.from_dict(data)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise


def upload_schema(schema: FullSchema) -> str:
    """Upload a schema to S3.

    Args:
        schema: The schema to upload

    Returns:
        The S3 URI where the schema was uploaded
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. "
                          "Install with: pip install boto3")

    bucket, _ = parse_s3_path(S3_SCHEMA_BASE_PATH)
    key = get_schema_s3_key(schema.version, schema.pydantic_version)

    s3 = boto3.client('s3')

    content = json.dumps(schema.to_dict(), indent=2)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode('utf-8'),
        ContentType='application/json',
    )

    return f"s3://{bucket}/{key}"


def list_available_versions(
        pydantic_version: Optional[int] = None) -> List[str]:
    """List all available schema versions in S3.

    Args:
        pydantic_version: Filter by pydantic version. If None, uses current.

    Returns:
        List of version strings, sorted by version number
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. "
                          "Install with: pip install boto3")

    bucket, prefix = parse_s3_path(S3_SCHEMA_BASE_PATH)

    if pydantic_version is None:
        schema_file = get_pydantic_schema_key()
    else:
        schema_file = f"pydantic-v{pydantic_version}.json"

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    versions = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Extract version from key like "schemas/v0.52.0/pydantic-v2.json"
            if key.endswith(schema_file):
                match = re.search(r'/v(\d+\.\d+\.\d+)/', key)
                if match:
                    versions.append(match.group(1))

    # Sort by version number
    versions.sort(key=_parse_version)
    return versions


def _parse_version(version: str) -> Tuple[int, int, int]:
    """Parse a version string into a tuple for sorting."""
    parts = version.split('.')
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def get_pypi_available_versions(package_name: str = "kumo-api") -> Set[str]:
    """Get all available (non-yanked) versions from PyPI.

    Args:
        package_name: The package name on PyPI

    Returns:
        Set of version strings that are available (not yanked)
    """
    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        # If PyPI is unreachable, return empty set (fail open)
        print(f"Warning: Could not fetch PyPI versions: {e}")
        return set()

    available = set()
    releases = data.get("releases", {})

    for version, files in releases.items():
        # A version is yanked if all its files are yanked (or no files exist)
        if not files:
            continue

        # Check if any file is not yanked
        has_available_file = any(not f.get("yanked", False) for f in files)
        if has_available_file:
            available.add(version)

    return available


def get_baseline_versions(
    current_version: str,
    pydantic_version: Optional[int] = None,
) -> List[str]:
    """Get baseline versions that current version must be compatible with.

    For a version like 0.60.3, this returns all minor version baselines
    from 0.52.x (enforcement start) up to and including 0.60.x.

    Only includes versions that are:
    - Available in S3 (schema exists)
    - Available on PyPI (not yanked)

    Args:
        current_version: The version being checked (e.g., "0.60.3")
        pydantic_version: Pydantic version to filter by

    Returns:
        List of version strings to compare against
    """
    available_in_s3 = list_available_versions(pydantic_version)

    if not available_in_s3:
        return []

    # Get versions available on PyPI (not yanked)
    pypi_versions = get_pypi_available_versions()

    # If PyPI check failed, use all S3 versions (fail open)
    if pypi_versions:
        available = [v for v in available_in_s3 if v in pypi_versions]
    else:
        available = available_in_s3

    current = _parse_version(current_version)
    baseline = _parse_version(ENFORCEMENT_BASELINE_VERSION)

    # Only include versions:
    # 1. Same major version
    # 2. >= baseline version
    # 3. < current version (don't compare against self)
    baselines = []

    # Group by minor version and get highest patch for each
    minor_versions = {}
    for v in available:
        parsed = _parse_version(v)

        # Skip if different major version
        if parsed[0] != current[0]:
            continue

        # Skip if below baseline
        if parsed < baseline:
            continue

        # Skip if >= current version
        if parsed >= current:
            continue

        minor_key = (parsed[0], parsed[1])
        if minor_key not in minor_versions:
            minor_versions[minor_key] = v
        elif parsed > _parse_version(minor_versions[minor_key]):
            minor_versions[minor_key] = v

    # Return sorted list of highest patches per minor
    baselines = sorted(minor_versions.values(), key=_parse_version)
    return baselines


def schema_exists(version: str,
                  pydantic_version: Optional[int] = None) -> bool:
    """Check if a schema exists in S3 for a given version.

    Args:
        version: Version string
        pydantic_version: Pydantic version

    Returns:
        True if schema exists, False otherwise
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. "
                          "Install with: pip install boto3")

    bucket, _ = parse_s3_path(S3_SCHEMA_BASE_PATH)
    key = get_schema_s3_key(version, pydantic_version)

    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
