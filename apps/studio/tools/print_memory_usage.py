#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Prints memory usage in an easy-to-read format.
#
import argparse
import math
import time
from datetime import datetime

import psutil


def make_usage_bar(percentage: float, bins: int) -> str:
    parts = []
    parts.append('[')
    for i in range(bins):
        if i <= math.ceil(percentage * bins):
            parts.append('#')
        else:
            parts.append(' ')
    parts.append(']')
    return ''.join(parts)


def format_usage(stats) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    used_pct = stats.used / stats.total
    usage_bar = make_usage_bar(used_pct, 80)
    gb = 1 << 30

    msg = (f"{now}    Mem Used: "
           f"{stats.used/gb:0.1f}/{stats.total/gb:0.1f} GiB")
    msg += ' ' * (55 - len(msg))
    msg += usage_bar
    return msg


def print_usage(min_used_pct: float, force: bool = False) -> None:
    stats = psutil.virtual_memory()
    used_pct = stats.used / stats.total
    if force or used_pct >= min_used_pct:
        print(format_usage(stats), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--interval',
        type=float,
        default=5,
        help='Seconds between memory samples.',
    )
    parser.add_argument(
        '--min-used-pct',
        type=float,
        default=0,
        help='Only print samples at or above this used-memory fraction.',
    )
    parser.add_argument(
        '--always-first',
        action='store_true',
        help='Always print the first sample, even below --min-used-pct.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    first = True
    while True:
        print_usage(args.min_used_pct, force=args.always_first and first)
        first = False
        time.sleep(args.interval)
