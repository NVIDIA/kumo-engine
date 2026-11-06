"""Generate local account-closure rows with fee, engagement and support signal."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rng = np.random.default_rng(1203)
N_CONTEXT, N_TEST, N_SCORE = 2500, 1000, 500
CONTEXT_TIME, TEST_TIME, SCORE_TIME = "2026-01-01", "2026-02-15", "2026-04-01"
OBSERVATION_DAYS = HORIZON_DAYS = 30
cohort_sizes = (N_CONTEXT, N_TEST, N_SCORE)
snapshot_times = pd.to_datetime(
    np.repeat([CONTEXT_TIME, TEST_TIME, SCORE_TIME], cohort_sizes)
)
cohort = np.repeat(["context", "test", "score"], cohort_sizes)
account_count = sum(cohort_sizes)
accounts = pd.DataFrame(
    {
        "account_id": [f"synthetic_account_{i:05d}" for i in range(account_count)],
        "cohort": cohort,
        "snapshot_time": snapshot_times,
        "observation_start": snapshot_times - pd.Timedelta(days=OBSERVATION_DAYS),
        "last_observation_at": snapshot_times
        - pd.to_timedelta(rng.integers(1, 24, account_count), unit="h"),
    }
)


accounts["product_type"] = rng.choice(["checking", "savings"], account_count)
accounts["segment"] = rng.choice(["standard", "plus", "student"], account_count)
accounts["region"] = rng.choice(["region_a", "region_b", "region_c"], account_count)
accounts["balance"] = rng.lognormal(7.4, 1.0, account_count).round(2)
accounts["inflow_30d"] = rng.lognormal(7.6, 0.7, account_count).round(2)
accounts["outflow_30d"] = rng.lognormal(7.4, 0.8, account_count).round(2)
accounts["transaction_count_30d"] = rng.poisson(20, account_count)
accounts["digital_login_count_30d"] = rng.poisson(5, account_count)
accounts["support_contacts_30d"] = rng.poisson(1.2, account_count)
accounts["monthly_fee"] = rng.choice([0.0, 5.0, 12.0, 20.0], account_count)
accounts["tenure_days"] = rng.integers(90, 3000, account_count)


assert accounts.account_id.notna().all() and accounts.account_id.is_unique
assert (accounts.observation_start < accounts.last_observation_at).all()
assert (accounts.last_observation_at < accounts.snapshot_time).all()
assert accounts.tenure_days.gt(OBSERVATION_DAYS).all()
feature_columns = [
    "product_type",
    "segment",
    "region",
    "balance",
    "inflow_30d",
    "outflow_30d",
    "transaction_count_30d",
    "digital_login_count_30d",
    "support_contacts_30d",
    "monthly_fee",
    "tenure_days",
]
features = accounts[feature_columns].copy()
assert not features.isna().any().any()
assert np.isfinite(features.select_dtypes(include="number").to_numpy()).all()
print(f"Prepared {len(features)} accounts with {len(feature_columns)} features each.")


known_outcomes = accounts.cohort.ne("score")
horizon_end = accounts.snapshot_time + pd.Timedelta(days=HORIZON_DAYS)
accounts["observation_end"] = horizon_end.where(known_outcomes)
accounts["closed_at"] = pd.NaT
known_accounts = accounts.loc[known_outcomes]
log_odds = (
    -1.3
    + 0.07 * known_accounts.monthly_fee
    - 0.25 * known_accounts.digital_login_count_30d
    + 0.55 * known_accounts.support_contacts_30d
    + rng.normal(0, 0.6, known_outcomes.sum())
)
closes = rng.binomial(1, 1 / (1 + np.exp(-log_odds))).astype(bool)
closure_offsets = pd.to_timedelta(
    rng.integers(1, HORIZON_DAYS + 1, known_outcomes.sum()), unit="D"
)
closure_dates = (known_accounts.snapshot_time + closure_offsets).where(closes)
accounts.loc[known_outcomes, "closed_at"] = closure_dates
mature = accounts.observation_end.notna() & (accounts.observation_end >= horizon_end)
closed_in_horizon = accounts.closed_at.gt(
    accounts.snapshot_time
) & accounts.closed_at.le(horizon_end)
labels = closed_in_horizon.astype(float).where(mature)
accounts["closure_in_horizon"] = labels
assert labels[cohort == "score"].isna().all()
assert set(labels[cohort == "context"]) == set(labels[cohort == "test"]) == {0, 1}
print(
    f"Known outcomes: {labels.notna().sum()}; prospective outcomes: {labels.isna().sum()}."
)

args.output.mkdir(parents=True, exist_ok=True)
accounts.to_csv(args.output / "accounts.csv", index=False)
