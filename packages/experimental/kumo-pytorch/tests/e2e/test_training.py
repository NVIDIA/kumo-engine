# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""H&M training checks over shared saved runs."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("diskgraph")
pytest.importorskip("torch_geometric")

from e2e.datasets.hm import HMDataset, fingerprint_mismatches  # noqa: E402
from e2e.flows.common import oracle_auroc  # noqa: E402
from tests.e2e.hm_support import (  # noqa: E402
    REPOSITORY,
    Report,
    Run,
    _recency_auroc,
)
from tests.e2e.tiers import Tier  # noqa: E402


def test_the_baseline_was_recorded_on_this_dataset(hm: HMDataset, tier: Tier) -> None:
    baseline = json.loads((REPOSITORY / "e2e" / "baselines" / tier.baseline).read_text())
    recorded = baseline["dataset"]
    assert (recorded["scale"], recorded["seed"]) == (tier.scale, 0)
    assert recorded["generator_version"] == hm.metadata.generator_version
    mismatched = fingerprint_mismatches(
        hm.metadata, recorded["table_sha256"], recorded["float_sums"]
    )
    assert not mismatched, f"re-record the baseline: {mismatched} differ"


def test_test_auroc_is_within_the_band(
    runs: dict[str, list[Run]],
    flow: str,
    hm: HMDataset,
    baseline_auroc: float,
    report: Report,
    tier: Tier,
) -> None:
    ceiling = hm.metadata.ceiling_auroc["test"]
    upper = ceiling + tier.leak_standard_errors * hm.metadata.ceiling_auroc_se["test"]
    recency = _recency_auroc(hm)
    lower = recency + tier.above_recency
    areas = [oracle_auroc(hm.root, run.predictions) for run in runs[flow]]
    report.record(f"{flow}.test_auroc", areas)
    report.record(f"{flow}.band", {"lower": lower, "upper": upper, "ceiling": ceiling})
    report.record("recency_only.test_auroc", recency)
    report.record(
        f"{flow}.vs_kumo_engine",
        {
            "kumo_engine_lowest": baseline_auroc,
            "difference": [area - baseline_auroc for area in areas],
        },
    )
    for run, area in zip(runs[flow], areas, strict=True):
        assert area >= lower, f"seed {run.seed}: test AUROC {area:.4f} below {lower:.4f}"
        assert area <= upper, f"seed {run.seed}: {area:.4f} beats the ceiling; a leak?"
    assert max(areas) - min(areas) <= tier.seed_spread, areas


def test_point_lookups_rank_active_customers_above_dormant_ones(
    runs: dict[str, list[Run]], flow: str, hm: HMDataset, report: Report
) -> None:
    anchors = pq.read_table(hm.anchors)
    test = anchors.filter(pc.equal(anchors["split"], "test"))
    samples: dict[int, list[int]] = {}
    for sample_id, customer in zip(
        test["sample_id"].to_pylist(), test["customer_id"].to_pylist(), strict=True
    ):
        samples.setdefault(customer, []).append(sample_id)
    for run in runs[flow]:
        predictions = run.predictions
        mean = float(np.mean(list(predictions.values())))
        by_kind: dict[str, list[float]] = {"active": [], "dormant": []}
        for lookup in hm.metadata.point_lookups:
            by_kind[lookup.kind] += [predictions[s] for s in samples[lookup.customer_id]]
        report.record(
            f"{flow}.seed{run.seed}.point_lookups",
            {kind: [min(values), max(values)] for kind, values in by_kind.items()},
        )
        assert min(by_kind["active"]) > max(by_kind["dormant"])
        assert min(by_kind["active"]) > mean


def test_predictions_have_a_sane_distribution(
    runs: dict[str, list[Run]], flow: str, hm: HMDataset, report: Report, tier: Tier
) -> None:
    positive_rate = hm.metadata.positive_rate["test"]
    for run in runs[flow]:
        table = pq.read_table(run.work / "predictions.parquet")
        scores = table["score"].to_numpy()
        quantiles = np.quantile(scores, [0.1, 0.5, 0.9]).tolist()
        report.record(
            f"{flow}.seed{run.seed}.predictions",
            {"mean": float(scores.mean()), "std": float(scores.std()), "q10_50_90": quantiles},
        )
        assert not np.isnan(scores).any()
        assert scores.std() > 0.01, "a constant model"
        ratio = scores.mean() / positive_rate
        factor = tier.calibration_factor
        assert 1 / factor <= ratio <= factor, (scores.mean(), ratio)
