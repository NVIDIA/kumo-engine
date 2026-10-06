// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const validResponse = {
    "files_selected": [
        {
            "table_name": "ORDERS_2",
            "is_dir": false,
            "table_files": [
                {
                    "file_name":
                        "s3://kumo-public-datasets/test_validation/test_dataset/ORDERS_2.csv",
                    "details": ""
                }
            ],
            "file_type": "csv"
        },
        {
            "table_name": "STORES",
            "is_dir": false,
            "table_files": [
                {
                    "file_name":
                        "s3://kumo-public-datasets/test_validation/test_dataset/STORES.csv",
                    "details": ""
                }
            ],
            "file_type": "csv"
        },
        {
            "table_name": "USERS",
            "is_dir": false,
            "table_files": [
                {
                    "file_name":
                        "s3://kumo-public-datasets/test_validation/test_dataset/USERS.csv",
                    "details": ""
                }
            ],
            "file_type": "csv"
        }
    ],
    "files_ignored": [],
    "count_total": 3,
    "count_selected": 3,
    "count_ignored": 0,
    "warnings": ["Kumo can read this connector, but cannot write to it."]
};

export const invalidS3PathResponse = {
    "detail":
        "S3 path `s3://kumo-public-datasets/test_validation/test_dataset/` either empty or you have insufficient permissions to access this resource. First, make sure that the directory exists. After that, refer to our user guide to give Kumo access to your S3 directory (https://test.kumoai.cloud/docs/using_kumo.connector#aws-s3)."
};
