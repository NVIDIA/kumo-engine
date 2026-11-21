# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny runner control: compare a known integer result, never claim a speedup."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    answers = [sum(range(size)) for size in [0, 1, 10, 1000]]
    expected = [0, 0, 45, 499500]
    if answers != expected:
        raise ValueError(
            'Integer sums differ from the independent expected answers')
    result = {
        'cases': 4,
        'output_sha256':
        hashlib.sha256(json.dumps(answers).encode()).hexdigest()
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(result) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
