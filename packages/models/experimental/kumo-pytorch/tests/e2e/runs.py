# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one flow and seed of a tier, so CI can train each in its own job.

``python -m tests.e2e.runs TIER FLOW SEED --e2e-dir DIR`` prepares the tier's data
under ``DIR/<tier>`` and trains into ``DIR/<tier>/<flow>-seed<seed>``, the layout
``test_lifecycle.py`` reads. A finished run (``run.json``) on the same tables and
training anchors is reused, not retrained: after changing a flow, use a fresh
``DIR``. With ``KUMO_PYTORCH_E2E_REUSE_ONLY=1`` (the CI checks job) a missing run
is an error instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e.tiers import BY_NAME, Tier

REPOSITORY = Path(__file__).resolve().parents[2]
FLOWS = {"gnn": "e2e.flows.hm_gnn", "pragma": "e2e.flows.hm_pragma"}
MARKER = "run.json"


def dataset(root: Path, tier: Tier) -> HMDataset:
    return generate(tier.scale, 0, root / f"hm-{tier.scale.lower()}")


def training_data(hm: HMDataset, root: Path, tier: Tier) -> HMDataset:
    """What the flows train on: the dataset, or the tier's sample of its anchors."""

    if not tier.sample:
        return hm
    return sample_anchors(hm, tier.sample, tier.sample_seed, root / "sampled")


def run(data: HMDataset, root: Path, flow: str, seed: int) -> tuple[Path, float]:
    """Train ``flow`` with ``seed`` unless a finished run on the same data is already there.

    The same data means the same table hashes and the same ``anchors.parquet`` bytes, so a
    run on another sample of the anchors is not reused.
    """

    work = root / f"{flow}-seed{seed}"
    marker = work / MARKER
    identity = {
        "flow": flow,
        "seed": seed,
        "data": dict(data.metadata.table_sha256),
        "anchors": hashlib.sha256(data.anchors.read_bytes()).hexdigest(),
    }
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if {key: recorded.get(key) for key in identity} == identity:
            return work, float(recorded["seconds"])
    if os.environ.get("KUMO_PYTORCH_E2E_REUSE_ONLY") == "1":
        raise RuntimeError(f"no finished run on this data at {work}")
    shutil.rmtree(work, ignore_errors=True)
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", FLOWS[flow], "--data", str(data.root), "--work", str(work)]
        + ["--seed", str(seed)],
        cwd=REPOSITORY,
        check=True,
    )
    seconds = time.perf_counter() - started
    marker.write_text(json.dumps({**identity, "seconds": seconds}) + "\n")
    return work, seconds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tier", choices=sorted(BY_NAME))
    parser.add_argument("flow", choices=sorted(FLOWS))
    parser.add_argument("seed", type=int)
    parser.add_argument("--e2e-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    tier = BY_NAME[args.tier]
    root = args.e2e_dir / tier.name
    root.mkdir(parents=True, exist_ok=True)
    work, seconds = run(training_data(dataset(root, tier), root, tier), root, args.flow, args.seed)
    print(f"{work}: {seconds:.0f}s")


if __name__ == "__main__":
    main()
