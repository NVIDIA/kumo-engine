# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How much of a fraud label one transaction gives away, on a pragmatiq book.

Fits the same gradient-boosted classifier on three feature sets, on a
user-disjoint split:

- `event`: the labelled transaction's own fields (amount, hour, weekday, MCC,
  channel, country, currency, type).
- `event_profile`: plus the user's home country and whether the transaction's
  country and currency match it.
- `event_profile_history`: plus features that need the user's own past: time
  since the previous transaction, transactions in the previous hour and day,
  amount against the user's prior mean, prior payments at the same merchant,
  and whether a device first seen in the last two hours, or an app session
  from abroad, precedes the transaction.

A task whose fraud needs history scores well below 1 on `event` and gains on
`event_profile_history`. History features use only events strictly before the
labelled transaction.

Two tasks:

- `--task native`: the book's own `labels/fraud.parquet`.
- `--task splice`: the spliced identity-theft task of dataset strategy. Each victim
  receives one day of a donor user's real transactions and app sessions at
  their original timestamps; the inserted transactions are fraud, and the
  victim's own transactions in the preceding week are the negatives.
  `--donors random` draws any donor; `--donors matched` requires the same home
  country and a median amount within a factor of 1.5. Users with the book's
  own fraud are excluded from both roles.

    uv run python benchmarks/fraud_separability.py --book ~/data/pragmatiq/data/<book> \
        --task splice --donors matched --out benchmarks/results/<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

TXN_KEYS = ("amount", "currency", "mcc", "merchant", "txn_type", "channel", "country")
APP_KEYS = ("device_id", "geo_country")
CATEGORICAL = ("mcc", "channel", "country", "currency", "txn_type", "home_country")
EVENT = ["log_amount", "hour", "weekday", "mcc", "channel", "country", "currency", "txn_type"]
PROFILE = ["home_country", "country_is_home", "currency_is_home"]
HISTORY = [
    "log_gap_s",
    "txn_prev_1h",
    "txn_prev_24h",
    "amount_vs_prior_mean",
    "merchant_prior_count",
    "new_device_prev_2h",
    "foreign_session_prev_2h",
]
HOME_CCY = {"GB": "GBP", "PL": "PLN"}
# A card thief spends on the card; the donor's salary, rent, and transfers stay out.
CARD_TYPES = ("card_payment", "atm_withdrawal")
HOUR_US = 3_600_000_000
DAY_US = 24 * HOUR_US


def _lookup(fields: pa.ChunkedArray, key: str) -> np.ndarray:
    return pc.map_lookup(fields, key, "first").to_numpy(zero_copy_only=False)


def _events(book: Path, users: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = ds.dataset(book / "events.parquet")
    wanted = pa.array(sorted(users))
    table = dataset.to_table(
        filter=pc.field("user_id").isin(wanted) & pc.field("source").isin(["transaction", "app"])
    )
    source = table.column("source").to_numpy(zero_copy_only=False)
    base = pd.DataFrame(
        {
            "user_id": table.column("user_id").to_numpy(zero_copy_only=False),
            "ts": table.column("ts").cast(pa.int64()).to_numpy(),
        }
    )
    fields = table.column("fields")
    for key in TXN_KEYS + APP_KEYS:
        base[key] = _lookup(fields, key)
    txn = base[source == "transaction"].drop(columns=list(APP_KEYS))
    app = base[source == "app"][["user_id", "ts", *APP_KEYS]]
    return _sorted(txn), _sorted(app)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)


def _home(book: Path) -> pd.Series:
    profiles = pq.read_table(book / "profiles.parquet", columns=["user_id", "attributes"])
    return pd.Series(
        _lookup(profiles.column("attributes"), "country"),
        index=profiles.column("user_id").to_numpy(zero_copy_only=False),
    )


def _history(txn: pd.DataFrame, app: pd.DataFrame, home: pd.Series) -> pd.DataFrame:
    txn = txn.copy()
    txn["amount_f"] = pd.to_numeric(txn["amount"], errors="coerce").fillna(0.0)
    g = txn.groupby("user_id", sort=False)
    gap = txn["ts"] - g["ts"].shift(1)
    txn["log_gap_s"] = np.log1p(gap.fillna(-1).clip(lower=0) / 1e6).where(gap.notna(), -1.0)
    prior_sum = g["amount_f"].cumsum() - txn["amount_f"]
    prior_n = g.cumcount()
    prior_mean = (prior_sum / prior_n.replace(0, np.nan)).fillna(0.0)
    txn["amount_vs_prior_mean"] = np.log1p(txn["amount_f"]) - np.log1p(prior_mean)
    txn["merchant_prior_count"] = np.log1p(txn.groupby(["user_id", "merchant"]).cumcount())
    ts_all = txn["ts"].to_numpy()
    for name, width in (("txn_prev_1h", HOUR_US), ("txn_prev_24h", DAY_US)):
        counts = np.zeros(len(txn), dtype=np.int64)
        for _, idx in g.indices.items():
            ts = ts_all[idx]
            counts[idx] = np.arange(len(ts)) - np.searchsorted(ts, ts - width, side="left")
        txn[name] = counts

    app = app.copy()
    first_seen = ~app.duplicated(["user_id", "device_id"])
    # A device's appearance at the start of a user's history is account setup, not a new device.
    first_seen &= app["ts"] > app.groupby("user_id")["ts"].transform("min")
    foreign = app["geo_country"].notna() & (app["geo_country"] != app["user_id"].map(home))
    for name, mask in (("new_device_prev_2h", first_seen), ("foreign_session_prev_2h", foreign)):
        marks = app.loc[mask, ["user_id", "ts"]].rename(columns={"ts": "mark_ts"})
        joined = pd.merge_asof(
            txn[["ts", "user_id"]].reset_index().sort_values("ts"),
            marks.sort_values("mark_ts"),
            left_on="ts",
            right_on="mark_ts",
            by="user_id",
            allow_exact_matches=False,
        ).set_index("index")
        recent = (joined["ts"] - joined["mark_ts"]) <= 2 * HOUR_US
        txn[name] = recent.reindex(txn.index).fillna(False).astype(np.int8)
    return txn


def _features(rows: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(rows["ts"], unit="us", utc=True)
    out = pd.DataFrame(index=rows.index)
    out["log_amount"] = np.log1p(rows["amount_f"])
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    for key in ("mcc", "channel", "country", "currency", "txn_type", "home_country"):
        out[key] = rows[key]
    out["country_is_home"] = (rows["country"] == rows["home_country"]).astype(np.int8)
    home_ccy = rows["home_country"].map(lambda c: HOME_CCY.get(c, "EUR"))
    out["currency_is_home"] = (rows["currency"] == home_ccy).astype(np.int8)
    for key in HISTORY:
        out[key] = rows[key]
    for key in CATEGORICAL:
        out[key] = out[key].astype("category")
    return out


def _score(x: pd.DataFrame, y: np.ndarray, test: np.ndarray, seed: int) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier  # type: ignore[import-not-found]

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=8,
    )
    clf.fit(x[~test], y[~test])
    p = clf.predict_proba(x[test])[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y[test], p)),
        "pr_auc": float(average_precision_score(y[test], p)),
    }


def _report(rows: pd.DataFrame, y: np.ndarray, seed: int, extra: dict[str, Any]) -> dict[str, Any]:
    x = _features(rows)
    rng = np.random.default_rng(seed)
    users = np.array(sorted(rows["user_id"].unique()), dtype=object)
    test_users = set(rng.choice(users, size=int(0.3 * len(users)), replace=False).tolist())
    test = rows["user_id"].isin(test_users).to_numpy()
    positives = y == 1
    sets = {
        "event": EVENT,
        "event_profile": EVENT + PROFILE,
        "event_profile_history": EVENT + PROFILE + HISTORY,
    }
    return {
        **extra,
        "seed": seed,
        "rows": int(len(rows)),
        "positives": int(positives.sum()),
        "test_rows": int(test.sum()),
        "test_positives": int((positives & test).sum()),
        "positive_share": {
            "online": float((rows.loc[positives, "channel"] == "online").mean()),
            "abroad": float((x.loc[positives, "country_is_home"] == 0).mean()),
            "first_time_merchant": float((rows.loc[positives, "merchant_prior_count"] == 0).mean()),
        },
        "negative_share": {
            "online": float((rows.loc[~positives, "channel"] == "online").mean()),
            "abroad": float((x.loc[~positives, "country_is_home"] == 0).mean()),
            "first_time_merchant": float(
                (rows.loc[~positives, "merchant_prior_count"] == 0).mean()
            ),
        },
        "scores": {name: _score(x[cols], y, test, seed) for name, cols in sets.items()},
    }


def measure_native(book: Path, *, n_users: int, seed: int) -> dict[str, Any]:
    labels = pq.read_table(book / "labels" / "fraud.parquet")
    users = labels.column("user_id").to_numpy(zero_copy_only=False)
    flags = labels.column("label").to_numpy()
    positive = set(users[flags == 1].tolist())
    rest = np.array(sorted(set(users.tolist()) - positive), dtype=object)
    rng = np.random.default_rng(seed)
    sampled = positive | set(rng.choice(rest, size=min(n_users, len(rest)), replace=False).tolist())

    home = _home(book)
    txn, app = _events(book, sampled)
    txn = _history(txn, app, home)
    txn["home_country"] = txn["user_id"].map(home)
    lab = labels.to_pandas()
    lab = lab[lab["user_id"].isin(sampled)]
    lab["ts"] = lab["ts"].astype("datetime64[us]").astype(np.int64)
    rows = lab.merge(txn.drop_duplicates(["user_id", "ts"]), on=["user_id", "ts"], how="inner")
    y = rows["label"].to_numpy().astype(int)
    return _report(rows, y, seed, {"book": str(book), "task": "native"})


def _pick_donor(
    victim: str,
    day0: int,
    candidates: np.ndarray,
    txn_by_user: dict[str, np.ndarray],
    ok: Any,
    rng: np.random.Generator,
) -> str | None:
    for donor in rng.permutation(candidates)[:200]:
        if donor == victim or not ok(victim, donor):
            continue
        ts = txn_by_user.get(donor, np.empty(0, dtype=np.int64))
        if np.any((ts >= day0) & (ts < day0 + DAY_US)):
            return str(donor)
    return None


def _card_ts(txn: pd.DataFrame) -> dict[str, np.ndarray]:
    card = txn[txn["txn_type"].isin(CARD_TYPES)]
    return {u: g["ts"].to_numpy() for u, g in card.groupby("user_id", sort=False)}


def measure_splice(book: Path, *, n_users: int, seed: int, donors: str) -> dict[str, Any]:
    labels = pq.read_table(book / "labels" / "fraud.parquet", columns=["user_id", "label"])
    flags = labels.column("label").to_numpy()
    native = set(labels.column("user_id").to_numpy(zero_copy_only=False)[flags == 1].tolist())
    home = _home(book)
    pool_all = np.array(sorted(set(home.index) - native), dtype=object)
    rng = np.random.default_rng(seed)
    pool = set(rng.choice(pool_all, size=min(n_users, len(pool_all)), replace=False).tolist())

    txn, app = _events(book, pool)
    txn_by_user = {u: g["ts"].to_numpy() for u, g in txn.groupby("user_id", sort=False)}
    card_by_user = _card_ts(txn)
    amount = pd.to_numeric(txn["amount"], errors="coerce")
    median = amount.groupby(txn["user_id"]).median()
    active = [u for u, ts in txn_by_user.items() if len(ts) >= 50]
    order = rng.permutation(np.array(sorted(active), dtype=object))
    victims, donor_pool = order[: len(order) // 2], order[len(order) // 2 :]

    def ok(victim: str, donor: str) -> bool:
        if donors == "random":
            return True
        ratio = float(median[donor]) / max(float(median[victim]), 1e-6)
        return home[donor] == home[victim] and 1 / 1.5 <= ratio <= 1.5

    by_user = {
        "txn": dict(tuple(txn.groupby("user_id", sort=False))),
        "app": dict(tuple(app.groupby("user_id", sort=False))),
    }
    inserted_txn, inserted_app, windows = [], [], []
    for victim in victims:
        ts = txn_by_user[victim]
        lo, hi = int(ts[0]) + 60 * DAY_US, int(ts[-1]) - DAY_US
        if hi <= lo:
            continue
        day0 = lo + int(rng.integers(0, (hi - lo) // DAY_US)) * DAY_US
        day0 -= day0 % DAY_US
        donor = _pick_donor(victim, day0, donor_pool, card_by_user, ok, rng)
        if donor is None:
            continue
        for source, out in (("txn", inserted_txn), ("app", inserted_app)):
            frame = by_user[source].get(donor)
            if frame is not None:
                part = frame[frame["ts"].between(day0, day0 + DAY_US - 1)]
                if source == "txn":
                    part = part[part["txn_type"].isin(CARD_TYPES)]
                out.append(part.assign(user_id=victim))
        windows.append((victim, day0))

    spliced_txn = pd.concat(inserted_txn).assign(is_fraud=1)
    victim_set = {v for v, _ in windows}
    own_txn = txn[txn["user_id"].isin(victim_set)].assign(is_fraud=0)
    own_app = app[app["user_id"].isin(victim_set)]
    all_txn = _sorted(pd.concat([own_txn, spliced_txn]))
    all_app = _sorted(pd.concat([own_app, *inserted_app]))
    all_txn = _history(all_txn, all_app, home)
    all_txn["home_country"] = all_txn["user_id"].map(home)

    start = pd.Series({v: d for v, d in windows})
    lo = all_txn["user_id"].map(start) - 7 * DAY_US
    hi = all_txn["user_id"].map(start) + DAY_US
    in_window = (all_txn["ts"] >= lo) & (all_txn["ts"] < hi)
    rows = all_txn[in_window & all_txn["txn_type"].isin(CARD_TYPES)].reset_index(drop=True)
    y = rows["is_fraud"].to_numpy().astype(int)
    return _report(
        rows,
        y,
        seed,
        {"book": str(book), "task": "splice", "donors": donors, "victims": len(windows)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--task", choices=["native", "splice"], default="native")
    parser.add_argument("--donors", choices=["random", "matched"], default="matched")
    parser.add_argument("--n-users", type=int, default=8000, help="users to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    book = args.book.expanduser()
    if args.task == "native":
        result = measure_native(book, n_users=args.n_users, seed=args.seed)
    else:
        result = measure_splice(book, n_users=args.n_users, seed=args.seed, donors=args.donors)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
