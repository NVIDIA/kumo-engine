# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register shared H&M fixtures once; native helpers load only when requested."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.e2e import runs as runs_module
from tests.e2e.runs import FLOWS, REPOSITORY
from tests.e2e.tiers import TIERS, Tier

if TYPE_CHECKING:
    from e2e.datasets.hm import HMDataset

    from kumo.pytorch.api import Batch
    from tests.e2e.hm_support import FutureScores, Report, Run


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(tier, id=tier.name, marks=getattr(pytest.mark, tier.name)) for tier in TIERS
    ],
)
def tier(request: pytest.FixtureRequest) -> Tier:
    return request.param


@pytest.fixture(scope="session")
def root(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, tier: Tier
) -> Path:
    chosen: Path | None = request.config.getoption("--e2e-dir")
    if chosen is None:
        return tmp_path_factory.mktemp(tier.name)
    (chosen / tier.name).mkdir(parents=True, exist_ok=True)
    return chosen / tier.name


@pytest.fixture(scope="session")
def report(root: Path) -> Report:
    from tests.e2e.hm_support import Report

    return Report(root / "metrics.json")


@pytest.fixture(scope="session")
def hm(root: Path, tier: Tier) -> HMDataset:
    return runs_module.dataset(root, tier)


@pytest.fixture(scope="session")
def data(hm: HMDataset, root: Path, tier: Tier) -> HMDataset:
    return runs_module.training_data(hm, root, tier)


@pytest.fixture(scope="session")
def baseline_auroc(tier: Tier) -> float:
    """kumo-engine's lowest test AUROC over its recorded runs."""
    baseline = json.loads((REPOSITORY / "e2e" / "baselines" / tier.baseline).read_text())
    return float(baseline["summary"]["test_auroc"]["min"])


@pytest.fixture(scope="session")
def runs(data: HMDataset, root: Path, report: Report, tier: Tier) -> dict[str, list[Run]]:
    """Every flow and seed of the tier, trained here or by a CI job before (``runs.py``)."""
    from tests.e2e.hm_support import Run

    done: dict[str, list[Run]] = {}
    for flow in FLOWS:
        done[flow] = []
        for seed in tier.seeds[flow]:
            work, seconds = runs_module.run(data, root, flow, seed)
            done[flow].append(Run(flow, seed, work, seconds))
            report.record(f"{flow}.seed{seed}.wall_seconds", round(seconds, 1))
    return done


@pytest.fixture(params=sorted(FLOWS))
def flow(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def future_scores(runs: dict[str, list[Run]], hm: HMDataset, root: Path) -> FutureScores:
    from e2e.flows.common import read_predictions

    from tests.e2e.hm_support import _dataset_at

    scores: FutureScores = {}
    for flow, module in FLOWS.items():
        scores[flow] = {}
        for cutoff in hm.metadata.cutoffs["test"]:
            at = datetime.fromisoformat(cutoff)
            base = root / f"{flow}-future-{at.date().isoformat()}"
            pair = []
            for name, future in (("plain", False), ("future", True)):
                data = _dataset_at(hm, at, base / name, future=future)
                out = base / f"{name}-predictions"
                subprocess.run(
                    [sys.executable, "-m", module, "infer", "--data", str(data)]
                    + ["--work", str(runs[flow][0].work), "--out", str(out)],
                    cwd=REPOSITORY,
                    check=True,
                )
                pair.append(read_predictions(out / "predictions.parquet"))
            scores[flow][cutoff] = (pair[0], pair[1])
    return scores


@pytest.fixture(scope="session")
def reference(runs: dict[str, list[Run]], hm: HMDataset) -> dict[str, dict[int, Batch]]:
    """Native batch inference for the served requests, over every customer
    and article and each served customer's purchases at any time (both contracts reach
    purchases only through their customer; a missing row would fail the checks)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from e2e.flows.common import CustomerRows, serving_requests

    from kumo.pytorch.api import Artifact, Dataset, InMemoryRows
    from tests.e2e.hm_support import _reference_batches

    store = CustomerRows.open(hm.root)
    requests = serving_requests(hm.root, store, count=16)
    tables = dict(store.tables)
    served = pa.array([request.entity_id for request in requests])
    purchases = tables["transactions"]
    tables["transactions"] = purchases.filter(pc.is_in(purchases["customer_id"], served))
    rows = Dataset({name: InMemoryRows(table.to_pylist()) for name, table in tables.items()})
    return {
        flow: _reference_batches(Artifact.load(flow_runs[0].work / "artifact"), rows, requests)
        for flow, flow_runs in runs.items()
    }
