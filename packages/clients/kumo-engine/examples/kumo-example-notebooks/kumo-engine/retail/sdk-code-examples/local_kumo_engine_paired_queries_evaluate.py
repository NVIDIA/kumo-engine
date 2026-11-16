# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine two-part spend predictions with explicit row alignment.
Pass the conversion model's predictions as --conversion, the amount model's
predictions as --amount, and realized spend as --actual, all in Parquet format.
Each artifact must contain matching ENTITY and TIMESTAMP rows; the respective
value columns are True_PROB, TARGET_PRED and TARGET.
Write the aligned results to a new --output path.
Metrics from synthetic inputs are demonstration only.
Install: pip install pandas pyarrow
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['conversion', 'amount', 'actual', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    keys = ['ENTITY', 'TIMESTAMP']
    conversion = pd.read_parquet(args.conversion)[keys + ['True_PROB']]
    amount = pd.read_parquet(args.amount)[keys + ['TARGET_PRED']]
    actual = pd.read_parquet(args.actual)[keys + ['TARGET']]
    combined = conversion.merge(amount, on=keys, how='outer',
                                validate='one_to_one', indicator=True)
    if not combined['_merge'].eq('both').all():
        raise ValueError(
            'Score both models on exactly the same entity/time rows.')
    combined = combined.drop(columns='_merge').merge(actual, on=keys,
                                                     how='outer',
                                                     validate='one_to_one',
                                                     indicator=True)
    if not combined['_merge'].eq('both').all() or combined['TARGET'].isna().any(
    ):
        raise ValueError(
            'Actual spend must cover every scoring row exactly once.')
    combined['EXPECTED_SPEND'] = combined['True_PROB'] * combined['TARGET_PRED']
    print('MAE:',
          (combined['EXPECTED_SPEND'] - combined['TARGET']).abs().mean())
    print(combined.head(10).to_string(index=False))
    if args.output.exists():
        raise FileExistsError(args.output)
    combined.drop(columns='_merge').to_parquet(args.output, index=False)


if __name__ == '__main__':
    main()
