# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate local synthetic account-network files used by both money-muling notebooks.
Requires numpy, pandas and pyarrow. No service connection or upload is performed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Synthetic data and the frozen observation date.
SEED = 42
N_ACCOUNTS = 240
N_TRANSFERS = 2400
CUTOFF = "2026-10-01"
HISTORY_DAYS = 180
HORIZON_DAYS = 30
REPORT_BASE_RATE = 0.04
REPORT_SIGNAL = 0.65
SPLIT_FRACTIONS = (0.60, 0.15, 0.10, 0.15)

rng = np.random.default_rng(SEED)
cutoff = pd.Timestamp(CUTOFF)
account_ids = np.array(
    [f"synthetic-account-{i:04d}" for i in range(N_ACCOUNTS)])
# This hidden tendency adds noisy signal; it is never supplied as a feature.
propensity = rng.beta(1.2, 5.0, N_ACCOUNTS)
activity_weights = (0.5 + 8 * propensity) / (0.5 + 8 * propensity).sum()
accounts = pd.DataFrame({
    "account_id":
    account_ids,
    "account_age_days":
    rng.integers(HISTORY_DAYS + 1, 2400, N_ACCOUNTS),
    "account_type":
    rng.choice(["checking", "savings"], N_ACCOUNTS, p=[0.8, 0.2]),
    "synthetic_region":
    rng.choice(["region_a", "region_b", "region_c"], N_ACCOUNTS),
    "mule_label":
    np.zeros(N_ACCOUNTS, dtype=np.int64),
})

senders = rng.choice(N_ACCOUNTS, N_TRANSFERS, p=activity_weights)
receivers = rng.choice(N_ACCOUNTS, N_TRANSFERS, p=activity_weights)
receivers = np.where(receivers == senders, (receivers + 1) % N_ACCOUNTS,
                     receivers)
event_times = cutoff - pd.to_timedelta(
    rng.integers(1, HISTORY_DAYS * 86400, N_TRANSFERS), unit="s")
transactions = pd.DataFrame({
    "transaction_id":
    [f"synthetic-transfer-{i:05d}" for i in range(N_TRANSFERS)],
    "timestamp":
    event_times,
    "amount":
    np.round(rng.lognormal(4.2, 1.0, N_TRANSFERS), 2),
    "channel":
    rng.choice(["online", "mobile", "branch"], N_TRANSFERS, p=[0.45, 0.5,
                                                               0.05]),
})
inputs = pd.DataFrame({
    "input_id": [f"synthetic-input-{i:05d}" for i in range(N_TRANSFERS)],
    "transaction_id":
    transactions.transaction_id,
    "account_id":
    account_ids[senders],
    "timestamp":
    event_times,
})
outputs = pd.DataFrame({
    "output_id": [f"synthetic-output-{i:05d}" for i in range(N_TRANSFERS)],
    "transaction_id":
    transactions.transaction_id,
    "account_id":
    account_ids[receivers],
    "timestamp":
    event_times,
})

report_flags = rng.random(
    N_ACCOUNTS) < REPORT_BASE_RATE + REPORT_SIGNAL * propensity
reports = pd.DataFrame({
    "report_id":
    [f"synthetic-report-{i:04d}" for i in range(report_flags.sum())],
    "account_id":
    account_ids[report_flags],
    "timestamp":
    cutoff + pd.to_timedelta(
        rng.integers(1, HORIZON_DAYS * 86400 + 1, report_flags.sum()),
        unit="s"),
})
frames = {
    "accounts": accounts,
    "transactions": transactions,
    "inputs": inputs,
    "outputs": outputs,
    "reports": reports,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_parquet(args.output_dir / f"{name}.parquet", index=False,
                         compression=None)
    print(sum(p.stat().st_size for p in args.output_dir.iterdir()), "bytes")
