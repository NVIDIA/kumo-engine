#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Read the json file on s3 without copy to local.
#
# Usage:
#
#   ./tools/probe_json.py s3://example-bucket/TENANT/ \
#     prod/data-persist-dir/data_backend_cache/tables/<connector_id>/ \
#     <table_name>/<data_version>/ \
#     materialization_stage
#

import sys

import pandas

COMPRESSION_FORMAT = 'gzip'


def get_handle(
    path: str,
    mode: str,
):
    out = pandas.io.common.get_handle(path, mode)  # type: ignore
    return out.handle if hasattr(out, 'handle') else out[0]  # type: ignore


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with get_handle(file_path, 'r') as f:
            json_data = f.read()
            print(json_data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


if __name__ == "__main__":
    main()
