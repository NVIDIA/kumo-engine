# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe synthetic smoke run: every variant, two training steps, one seed."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tfm.config import load_recipe_config
from tfm.runtime.compare import (
    VariantResult,
    apply_overrides,
    comparison_markdown,
    load_variants,
    run_comparison,
)

HERE = Path(__file__).resolve().parent


def main(run_dir: Path | None = None) -> list[VariantResult]:
    base = apply_overrides(load_recipe_config(HERE / "config.yaml"), {"training.max_steps": 2})
    root = run_dir or Path(tempfile.mkdtemp(prefix="ablation_building_blocks_"))
    results = run_comparison(base, load_variants(HERE / "variants.yaml"), root, seeds=[base.seed])
    print(comparison_markdown(results))
    return results


if __name__ == "__main__":
    main()
