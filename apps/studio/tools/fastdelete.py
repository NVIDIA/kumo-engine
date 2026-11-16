#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Quickly delete files recursively on s3.
# Much faster than aws s3 rm --recursive.
#
# Usage:
#
#   ./tools/fastdelete.py PATH
#
# Example: Delete your devserver dataplane
#
#   ./tools/fastdelete.py \
#     s3://example-dev-bucket/test/example-instance
#

import argparse
import logging

from io_utils.fsutil import s3_filesystem


def run(path: str) -> None:
    if 'kumo-prod-dataplane-bucket' in path:
        print("PATH cannot be a production path:", path)
        exit(1)
    s3_filesystem.rmdir(path)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("io_utils.fsutil").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('PATH')
    args = parser.parse_args()

    run(args.PATH)
