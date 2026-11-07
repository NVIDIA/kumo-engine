# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from e2e.datasets.hm import (
    DAY,
    FREQUENCY_WINDOW_DAYS,
    HORIZON_DAYS,
    PRODUCT_TYPES,
    SCALES,
    SESSION_HOURS,
    START_EPOCH,
    HMDataset,
    fingerprint_mismatches,
    generate,
    load,
    sample_anchors,
)
from e2e.metrics import auroc, auroc_standard_error

REPOSITORY = Path(__file__).resolve().parents[2]
PARQUET_FILES = (
    "tables/customers.parquet",
    "tables/articles.parquet",
    "tables/transactions.parquet",
    "anchors.parquet",
    "oracle/labels.parquet",
    "oracle/latent.parquet",
)


@pytest.fixture(scope="module")
def xs(tmp_path_factory: pytest.TempPathFactory) -> HMDataset:
    return generate("XS", 7, tmp_path_factory.mktemp("hm-xs"))


@pytest.fixture(scope="module")
def s(tmp_path_factory: pytest.TempPathFactory) -> HMDataset:
    return generate("S", 0, tmp_path_factory.mktemp("hm-s"))


def id_column(path: Path, name: str) -> npt.NDArray[np.int64]:
    return pq.read_table(path, columns=[name])[name].to_numpy()


def seconds(values: pa.ChunkedArray) -> npt.NDArray[np.int64]:
    """Seconds after START; the safe cast fails on a sub-second timestamp."""
    return values.cast(pa.timestamp("s")).cast(pa.int64()).to_numpy() - START_EPOCH


def test_auroc_matches_hand_counted_pairs() -> None:
    labels = np.array([True, False, True, False])
    # Positive/negative pairs: 0.9>0.1, 0.9>0.4, 0.4>0.1, 0.4=0.4 (half) -> 3.5 / 4.
    assert auroc(labels, np.array([0.9, 0.1, 0.4, 0.4])) == 0.875
    assert auroc(labels, np.array([1.0, 0.0, 1.0, 0.0])) == 1.0
    assert auroc(labels, np.array([0.0, 1.0, 0.0, 1.0])) == 0.0
    assert auroc(labels, np.full(4, 0.5)) == 0.5
    with pytest.raises(ValueError, match="one positive and one negative"):
        auroc(np.array([True, True]), np.array([0.1, 0.2]))


def test_auroc_equals_brute_force_with_ties() -> None:
    rng = np.random.default_rng(3)
    labels = rng.random(300) < 0.3
    scores = rng.integers(0, 12, 300).astype(np.float64)
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p, n in product(scores[labels], scores[~labels])
    )
    assert math.isclose(auroc(labels, scores), wins / (labels.sum() * (~labels).sum()))


def test_generation_is_deterministic(s: HMDataset, tmp_path: Path) -> None:
    again = generate("S", 0, tmp_path / "again")
    assert again.metadata == s.metadata
    assert (tmp_path / "again" / "metadata.json").read_text() == (
        s.root / "metadata.json"
    ).read_text()
    for name in PARQUET_FILES:
        assert pq.read_table(again.root / name).equals(pq.read_table(s.root / name)), name
    assert load(s.root).metadata == s.metadata

    other = generate("S", 1, tmp_path / "other")
    assert other.metadata.table_sha256["transactions"] != s.metadata.table_sha256["transactions"]


def test_fingerprint_ignores_last_bit_float_differences_but_not_data_changes(s: HMDataset) -> None:
    # The baseline is recorded on arm64 and checked on x86-64, where vectorized exp/sin can differ.
    meta = s.metadata
    assert fingerprint_mismatches(meta, meta.table_sha256, meta.float_sums) == []
    nudged = {
        t: {c: float(np.nextafter(v, np.inf)) for c, v in cols.items()}
        for t, cols in meta.float_sums.items()
    }
    assert fingerprint_mismatches(meta, meta.table_sha256, nudged) == []
    changed = {
        **meta.float_sums,
        "oracle/latent": {
            **meta.float_sums["oracle/latent"],
            "mu": meta.float_sums["oracle/latent"]["mu"] + 0.01,
        },
    }
    assert fingerprint_mismatches(meta, meta.table_sha256, changed) == ["oracle/latent"]
    rehashed = {**meta.table_sha256, "transactions": "0" * 64}
    assert fingerprint_mismatches(meta, rehashed, meta.float_sums) == ["transactions"]


def test_s_scale_matches_the_brief(s: HMDataset) -> None:
    sizes = s.metadata.sizes
    assert sizes["customers"] == 2_000
    assert sizes["articles"] == 500
    assert 40_000 <= sizes["transactions"] <= 60_000
    scale = SCALES["S"]
    for split in ("train", "val", "test"):
        assert len(s.metadata.cutoffs[split]) == len(scale.cutoffs(split))
    assert sizes["anchors"] == 2_000 * sum(len(cuts) for cuts in s.metadata.cutoffs.values())


@pytest.mark.parametrize("scale", sorted(SCALES))
def test_splits_are_ordered_and_label_windows_never_cross_a_split(scale: str) -> None:
    size = SCALES[scale]
    for earlier, later in (("train", "val"), ("val", "test")):
        assert max(size.cutoffs(earlier)) + HORIZON_DAYS <= min(size.cutoffs(later))
    assert max(size.test_cutoffs) + HORIZON_DAYS <= size.days
    for split in ("train", "val", "test"):
        cutoffs = size.cutoffs(split)
        assert all(b - a >= HORIZON_DAYS for a, b in zip(cutoffs, cutoffs[1:], strict=False))


def test_tables_are_well_formed(s: HMDataset) -> None:
    transactions = pq.read_table(s.table("transactions"))
    times = seconds(transactions["t_dat"])
    second_of_day = times % DAY
    assert second_of_day.min() >= SESSION_HOURS[0] * 3600
    assert second_of_day.max() < SESSION_HOURS[1] * 3600
    assert times.min() >= 0
    assert times.max() < SCALES["S"].days * DAY
    assert set(id_column(s.table("transactions"), "customer_id")) <= set(
        id_column(s.table("customers"), "customer_id")
    )
    assert set(id_column(s.table("transactions"), "article_id")) <= set(
        id_column(s.table("articles"), "article_id")
    )
    ids = id_column(s.table("transactions"), "transaction_id")
    assert np.array_equal(ids, np.arange(ids.size))
    cutoffs = seconds(pq.read_table(s.anchors)["cutoff"])
    assert (cutoffs % DAY == 0).all()
    # Every article is listed before any purchase, so the article hop drops none.
    assert seconds(pq.read_table(s.table("articles"))["listed_at"]).max() < 0


def sessions_by_customer(dataset: HMDataset) -> dict[int, dict[int, list[int]]]:
    """customer -> session day -> article ids, from the written tables only."""
    table = pq.read_table(dataset.table("transactions"))
    days = seconds(table["t_dat"]) // DAY
    sessions: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    customers = table["customer_id"].to_pylist()
    articles = table["article_id"].to_pylist()
    for customer, day, article in zip(customers, days.tolist(), articles, strict=True):
        sessions[customer][day].append(article)
    return sessions


def anchors_with_oracle(dataset: HMDataset) -> list[tuple[int, int, str, bool, float]]:
    anchors = pq.read_table(dataset.anchors)
    oracle = pq.read_table(dataset.labels)
    assert anchors["sample_id"].equals(oracle["sample_id"])
    days = seconds(anchors["cutoff"]) // DAY
    return list(
        zip(
            anchors["customer_id"].to_pylist(),
            days.tolist(),
            anchors["split"].to_pylist(),
            oracle["label"].to_pylist(),
            oracle["true_prob"].to_pylist(),
            strict=True,
        )
    )


def test_labels_are_any_purchase_in_the_next_seven_days(xs: HMDataset) -> None:
    sessions = sessions_by_customer(xs)
    for customer, cutoff, _, label, _ in anchors_with_oracle(xs):
        expected = any(cutoff <= day < cutoff + HORIZON_DAYS for day in sessions[customer])
        assert label == expected, (customer, cutoff)


def test_true_prob_is_recomputed_from_past_rows_and_latent_rate(xs: HMDataset) -> None:
    """An independent, row-by-row reading of the signal documented in ``hm.py``."""
    signal = xs.metadata.signal
    sessions = sessions_by_customer(xs)
    replenish_by_type = {name: score for name, _, score in PRODUCT_TYPES}
    articles = pq.read_table(xs.table("articles"))
    replenish = dict(
        zip(
            articles["article_id"].to_pylist(),
            [replenish_by_type[name] for name in articles["product_type_name"].to_pylist()],
            strict=True,
        )
    )
    latent_table = pq.read_table(xs.latent).to_pylist()
    latent = {row["customer_id"]: row for row in latent_table}

    def log_rate(customer: int, day: int) -> float:
        row = latent[customer]
        drift = row["drift_amplitude"] * math.sin(
            2 * math.pi * day / row["drift_period_days"] + row["drift_phase"]
        )
        churned = signal.churn_offset if day >= row["churn_day"] else 0.0
        return row["mu"] + drift + churned

    checked = 0
    for customer, cutoff, _, _, recorded in anchors_with_oracle(xs):
        past = sorted(day for day in sessions[customer] if day < cutoff)
        affinity = 0.0
        for day in past:
            basket = sessions[customer][day]
            mean = sum(replenish[article] for article in basket) / len(basket)
            affinity = signal.affinity_decay * affinity + (1 - signal.affinity_decay) * mean
        survival = 1.0
        for day in range(cutoff, cutoff + HORIZON_DAYS):
            recency = math.exp(-(day - past[-1]) / signal.recency_scale_days) if past else 0.0
            recent = sum(1 for d in past if day - FREQUENCY_WINDOW_DAYS <= d)
            logit = (
                signal.intercept
                + log_rate(customer, day)
                + signal.recency_weight * recency
                + signal.frequency_weight * math.log1p(recent)
                + signal.affinity_weight * affinity
            )
            survival *= 1 - 1 / (1 + math.exp(-logit))
        assert math.isclose(recorded, 1 - survival, rel_tol=1e-9, abs_tol=1e-12), (
            customer,
            cutoff,
        )
        checked += 1
    assert checked == xs.metadata.sizes["anchors"]


def test_ceiling_is_the_auroc_of_the_recorded_probability(s: HMDataset) -> None:
    rows = anchors_with_oracle(s)
    for split in ("train", "val", "test"):
        labels = np.array([label for _, _, sp, label, _ in rows if sp == split])
        probs = np.array([prob for _, _, sp, _, prob in rows if sp == split])
        assert s.metadata.ceiling_auroc[split] == auroc(labels, probs)
        positives = int(labels.sum())
        assert s.metadata.ceiling_auroc_se[split] == auroc_standard_error(
            s.metadata.ceiling_auroc[split], positives, labels.size - positives
        )


def test_labels_are_drawn_from_the_recorded_probability(s: HMDataset) -> None:
    rows = anchors_with_oracle(s)
    labels = np.array([label for *_, label, _ in rows], dtype=np.float64)
    probs = np.array([prob for *_, prob in rows])

    # Windows never overlap, so label - prob are martingale differences: their
    # sum has variance sum(p(1 - p)), and 4 standard errors bound a sound generator.
    def bound(chosen: npt.NDArray[np.bool_]) -> float:
        return 4 * math.sqrt(float((probs[chosen] * (1 - probs[chosen])).sum())) / chosen.sum()

    everything = np.ones(labels.size, dtype=np.bool_)
    assert abs(labels.mean() - probs.mean()) < bound(everything)
    edges = np.quantile(probs, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, 9)
    for decile in range(10):
        chosen = bins == decile
        assert abs(labels[chosen].mean() - probs[chosen].mean()) < bound(chosen), decile


def test_ceiling_separates_past_only_scores_from_leaky_ones(s: HMDataset) -> None:
    """A past-only score stays under the ceiling; one that reads the label window beats it.

    Shorter leaks (two or three days) stay under the band: the ceiling catches gross
    leaks, and the e2e flows' future-row check catches the rest.
    """
    sessions = sessions_by_customer(s)
    test = [row for row in anchors_with_oracle(s) if row[2] == "test"]
    labels = np.array([label for *_, label, _ in test])

    def sessions_in(customer: int, start: int, stop: int) -> int:
        return sum(1 for day in sessions[customer] if start <= day < stop)

    past = np.array([sessions_in(c, t - 30, t) for c, t, *_ in test], dtype=np.float64)
    leaky = np.array(
        [sessions_in(c, t - 30, t + HORIZON_DAYS) for c, t, *_ in test], dtype=np.float64
    )
    ceiling = s.metadata.ceiling_auroc["test"]
    band = 3 * s.metadata.ceiling_auroc_se["test"]
    assert auroc(labels, past) < ceiling
    assert auroc(labels, leaky) > ceiling + band


def test_point_lookups_are_extreme_in_both_directions(s: HMDataset) -> None:
    lookups = s.metadata.point_lookups
    assert {lookup.kind for lookup in lookups} == {"active", "dormant"}
    sessions = sessions_by_customer(s)
    first_test = SCALES["S"].test_cutoffs[0]
    for lookup in lookups:
        if lookup.expected == "high":
            assert min(lookup.test_true_prob) > 0.9
        else:
            assert max(lookup.test_true_prob) < 0.05
            history = sessions[lookup.customer_id]
            assert history, "a dormant customer was active before churning"
            assert max(history) < first_test


def test_recorded_baseline_is_for_this_dataset(s: HMDataset) -> None:
    baseline = json.loads((REPOSITORY / "e2e" / "baselines" / "hm-s.json").read_text())
    dataset = baseline["dataset"]
    assert (dataset["scale"], dataset["seed"]) == ("S", 0)
    assert dataset["generator_version"] == s.metadata.generator_version
    mismatched = fingerprint_mismatches(s.metadata, dataset["table_sha256"], dataset["float_sums"])
    assert not mismatched, f"re-record the baseline: {mismatched} differ"


def test_an_anchor_sample_is_fixed_and_keeps_the_unnamed_splits(
    xs: HMDataset, tmp_path: Path
) -> None:
    fractions = {"train": 0.25, "val": 0.5}
    sampled = sample_anchors(xs, fractions, 3, tmp_path / "a")
    again = sample_anchors(xs, fractions, 3, tmp_path / "b")
    full = pq.read_table(xs.anchors)
    kept = pq.read_table(sampled.anchors)
    assert kept.equals(pq.read_table(again.anchors))
    assert set(kept["sample_id"].to_pylist()) <= set(full["sample_id"].to_pylist())
    for split, fraction in {**fractions, "test": 1.0}.items():
        total = full["split"].to_pylist().count(split)
        assert kept["split"].to_pylist().count(split) == round(fraction * total)
    assert sampled.metadata == xs.metadata
    assert pq.read_table(sampled.table("transactions")).equals(
        pq.read_table(xs.table("transactions"))
    )
    with pytest.raises(ValueError, match="no 'valid' anchors"):
        sample_anchors(xs, {"valid": 0.5}, 3, tmp_path / "d")
    other = sample_anchors(xs, fractions, 4, tmp_path / "c")
    assert not kept.equals(pq.read_table(other.anchors))
