# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

storage_account = os.environ['AZURE_STORAGE_ACCOUNT']
container = os.environ['AZURE_CONTAINER_NAME']

# DefaultAzureCredential uses the Azure CLI session established by
# azure/login in CI and supports managed identity in Azure-hosted runners.
credential = DefaultAzureCredential()

account_url = f"https://{storage_account}.dfs.core.windows.net"
service_client = DataLakeServiceClient(account_url=account_url,
                                       credential=credential)
fs_client = service_client.get_file_system_client(container)

print("Creating test fixtures...")

for directory in (
        "test_fixtures",
        "test_fsutil",
        "test_fsutil_with_suffix",
):
    try:
        fs_client.create_directory(directory)
    except ResourceExistsError:
        pass

# 1. Create test_fixtures/hnm_10k_churn.json (exactly 7037 bytes)
print("Creating test_fixtures/hnm_10k_churn.json...")
test_json = '{"test": "data"}' * 500
data = test_json[:7037].encode()
file_client = fs_client.get_file_client("test_fixtures/hnm_10k_churn.json")
file_client.upload_data(data, overwrite=True)
print(f"  ✅ Created (size: {len(data)} bytes)")

# 2. Create existing_file
print("Creating existing_file...")
file_client = fs_client.get_file_client("existing_file")
file_client.upload_data(b"existing content", overwrite=True)
print("  ✅ Created")

# 3. Create test_fsutil directory (with a file to make it exist)
print("Creating test_fsutil directory...")
file_client = fs_client.get_file_client("test_fsutil/.keep")
file_client.upload_data(b"keep", overwrite=True)
print("  ✅ Created")

# 4. Create test_fsutil_with_suffix directory
print("Creating test_fsutil_with_suffix directory...")
file_client = fs_client.get_file_client("test_fsutil_with_suffix/.keep")
file_client.upload_data(b"keep", overwrite=True)
print("  ✅ Created")

print("\n🎉 All test fixtures created!")
print("\nVerifying...")
paths = list(fs_client.get_paths())
print(f"Total files in container: {len(paths)}")
