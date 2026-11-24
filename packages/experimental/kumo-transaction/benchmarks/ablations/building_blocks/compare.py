# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluate every variant in variants.yaml, then write one table.

uv run python benchmarks/ablations/building_blocks/compare.py
uv run python benchmarks/ablations/building_blocks/compare.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfm.config import load_recipe_config
from tfm.runtime.common import default_run_dir
from tfm.runtime.compare import (
    comparison_markdown,
    comparison_record,
    load_variants,
    run_comparison,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap one building-block slot at a time.")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--variants", type=Path, default=HERE / "variants.yaml")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Run only these variant names.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    base = load_recipe_config(args.config)
    variants = load_variants(args.variants)
    if args.only:
        unknown = set(args.only) - {v.name for v in variants}
        if unknown:
            raise SystemExit(f"Unknown variants: {sorted(unknown)}")
        variants = [v for v in variants if v.name in args.only]
    run_dir = args.run_dir or default_run_dir(recipe="ablation_building_blocks")
    results = run_comparison(
        base, variants, run_dir, seeds=args.seeds or [base.seed], device=args.device
    )
    table = comparison_markdown(results)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison.md").write_text(table, encoding="utf-8")
    (run_dir / "comparison.json").write_text(
        json.dumps(comparison_record(results), indent=2), encoding="utf-8"
    )
    print(table)
    print(f"Wrote {run_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
