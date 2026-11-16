#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Quickly copy files recursively from one directly on s3 to another.
# Much faster than aws s3 cp --recursive.
#
# Usage:
#
#   ./tools/fastcopy.py SRC_PATH DST_PATH
#
# Example: Copy the entire demo dataplane to your devserver.
#
#   ./tools/fastcopy.py \
#     s3://example-source-bucket/demo/prod \
#     s3://example-dev-bucket/test/example-instance
#

import argparse
import logging

from io_utils.fsutil import s3_filesystem


def run(src_path: str, dst_path: str) -> None:
    if 'kumo-prod-dataplane-bucket' in dst_path:
        print("DST_PATH cannot be a production path:", dst_path)
        exit(1)
    s3_filesystem.copydir(src_path, dst_path)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("io_utils.fsutil").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('SRC_PATH')
    parser.add_argument('DST_PATH')
    args = parser.parse_args()

    run(args.SRC_PATH, args.DST_PATH)
