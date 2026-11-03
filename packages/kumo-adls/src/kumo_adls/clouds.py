# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cloud endpoints shared by SDK clients and the engine's credential adapters."""

from kumoapi.data_source import AzureCloud

# Storage uses the generic audience in all clouds; login authorities differ.
CLOUDS = {
    AzureCloud.PUBLIC: ('dfs.core.windows.net', 'https://storage.azure.com',
                        'login.microsoftonline.com'),
    AzureCloud.US_GOVERNMENT:
    ('dfs.core.usgovcloudapi.net', 'https://storage.azure.com',
     'login.microsoftonline.us'),
    AzureCloud.CHINA: ('dfs.core.chinacloudapi.cn', 'https://storage.azure.com',
                       'login.chinacloudapi.cn'),
}
