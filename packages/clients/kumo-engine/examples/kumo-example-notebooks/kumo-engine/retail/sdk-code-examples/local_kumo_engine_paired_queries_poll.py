# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run paired H&M model scripts as local processes and retry only failed fits.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_retail/generate.py --dataset fashion --output-dir ./synthetic-fashion
Use that directory as --source-root or --input-dir. Metrics are demonstration only.

"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-root', 'scratch-dir', 'output-dir']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name(
        'local_kumo_engine_paired_queries_train.py')
    conversion_population = None
    for task in ['P_conversion', 'E_amount']:
        for attempt in range(2):
            destination = root / f'{task}_{attempt}'
            command = [
                sys.executable,
                str(script), '--source-root', args.source_root, '--scratch-dir',
                str(Path(args.scratch_dir) / task / str(attempt)),
                '--output-dir',
                str(destination), '--problem', task
            ]
            if task == 'E_amount':
                command += ['--scoring-file', str(conversion_population)]
            with (root / f'{task}_{attempt}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log,
                                           stderr=subprocess.STDOUT)
                status = process.wait()
            print(task, 'exit status', status)
            if status == 0:
                if task == 'P_conversion':
                    conversion_population = destination / 'scoring.parquet'
                break
        else:
            raise RuntimeError(
                f'{task} failed twice; inspect its logs before retrying.')


if __name__ == '__main__':
    main()
