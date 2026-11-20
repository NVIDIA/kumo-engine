# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The end-to-end tiers: which data, how many seeds, and the model bands at each.

T2a runs after every merge to ``main`` at S; T2b runs nightly at M. Both run the
same flows and the same checks (``test_lifecycle.py``); only what is here
differs. Every band was set after measuring, never to make a bad model pass;
its reasoning sits next to its value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    scale: str
    #: Training seeds per flow.
    seeds: Mapping[str, tuple[int, ...]]
    #: Both flows must beat recency alone, the strongest single signal, by this much.
    above_recency: float
    #: Seeds of one flow may spread by this much in test AUROC.
    seed_spread: float
    #: The kumo-engine baseline on the GNN's neighborhood, under ``e2e/baselines/``.
    #: The GNN's difference to its lowest run is reported, not gated: kumo-engine is
    #: nondeterministic on CPU and moves per pin, and a band wide enough for that sits
    #: below the recency floor.
    baseline: str
    #: The fraction of each named split's anchors the flows train and validate on,
    #: drawn once with ``sample_seed``; the test split is always scored in full.
    sample: Mapping[str, float] = field(default_factory=dict)
    sample_seed: int = 0
    #: Past-only models cannot beat the ceiling except by sampling noise
    #: (e2e/README.md); more than this many standard errors above it is a leak.
    leak_standard_errors: float = 3.0
    #: The mean prediction is within this factor of the test positive rate. A wrong
    #: label threshold, a reweighted loss or a train/inference mismatch moves it by
    #: 2x or more; small models stopped early on AUROC measured 0.96x to 1.27x at S.
    calibration_factor: float = 1.5


T2A = Tier(
    name="t2a",
    scale="S",
    # Each flow and seed trains in its own CI job (runs.py), so seeds add no wall time.
    seeds={"gnn": (0, 1, 2), "pragma": (0, 1)},
    # Bands compare models scored on the same 8,000 test anchors, so differences are
    # paired: far less noisy than the test AUROC's own standard error (0.0076).
    # Measured margins over recency alone: GNN 0.013 to 0.015, PRAGMA 0.016 and 0.018.
    above_recency=0.005,
    # Measured spreads were 0.0024 (GNN) and 0.0020 (PRAGMA); kumo-engine's 0.0021.
    seed_spread=0.005,
    # kumo-engine's lowest run is 0.8250; the GNN measured -0.0001 to +0.0023 from it.
    baseline="hm-s.json",
)

T2B = Tier(
    name="t2b",
    scale="M",
    # Two seeds per flow, each its own CI job: about 1.6 min per GNN seed and 5 min per
    # PRAGMA seed on an Apple M5, about 3x and 6x that on a 2-vCPU GitHub runner.
    seeds={"gnn": (0, 1), "pragma": (0, 1)},
    # Paired on the same 200,000 test anchors; the test AUROC's own standard error is
    # 0.0015, a fifth of S's. Measured margins over recency alone (0.8103): GNN 0.023
    # and 0.024, PRAGMA 0.025 and 0.025; a model that adds less than 0.01 fails.
    above_recency=0.01,
    # Measured spreads 0.0003 (GNN) and 0.0004 (PRAGMA); kumo-engine's four runs 0.0006.
    seed_spread=0.003,
    # kumo-engine's lowest run is 0.8316; the GNN measured +0.0021 and +0.0024 from it.
    baseline="hm-m.json",
    # A quarter of train and val (100,000 and 25,000 anchors) keeps PRAGMA's fine-tune
    # within the nightly budget; the GNN stops early either way.
    sample={"train": 0.25, "val": 0.25},
)

TIERS = (T2A, T2B)
BY_NAME = {tier.name: tier for tier in TIERS}


def matrix(tier: Tier) -> dict[str, list[dict[str, object]]]:
    """One CI job per flow and seed."""

    return {
        "include": [
            {"flow": flow, "seed": seed} for flow, seeds in tier.seeds.items() for seed in seeds
        ]
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(matrix(BY_NAME[sys.argv[1]])))
