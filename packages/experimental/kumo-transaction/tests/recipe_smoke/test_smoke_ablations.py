# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

from tfm.config import load_recipe_config
from tfm.runtime.compare import SLOTS, Variant, apply_overrides, comparison_markdown, load_variants

RECIPE = Path("benchmarks/ablations/building_blocks")


def _smoke_module():
    spec = importlib.util.spec_from_file_location("bbc_smoke", RECIPE / "smoke_test.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_variant_is_a_valid_config() -> None:
    base = load_recipe_config(RECIPE / "config.yaml")
    variants = load_variants(RECIPE / "variants.yaml")
    assert {v.slot for v in variants} == set(SLOTS)
    for variant in variants:
        config = apply_overrides(base, variant.overrides)
        assert config.data.temporal_split == base.data.temporal_split
        assert config.seed == base.seed


def test_unknown_slot_and_section_are_refused() -> None:
    with pytest.raises(ValueError, match="slot"):
        Variant(name="x", slot="tokenizer")
    with pytest.raises(KeyError, match="no section"):
        apply_overrides(load_recipe_config(RECIPE / "config.yaml"), {"modle.readout": "mean"})


def test_comparison_smoke_shares_one_probe_split(tmp_path: Path) -> None:
    results = _smoke_module().main(run_dir=tmp_path)
    table = comparison_markdown(results)
    assert table.count("\n| `") == len(results)
    splits = set()
    for result in results:
        run = tmp_path / result.variant.name / f"seed_{result.seeds[0]}"
        manifest = json.loads((run / "probe_manifest.json").read_text(encoding="utf-8"))
        assert manifest["fraud_split"]["shared_entities"] == []
        splits.add(json.dumps([manifest["fraud_split"], manifest["tasks"]["fraud"]]))
        assert "roc_auc" in result.metrics[0]["embedding"]
    # Same book, cutoffs, windows, and seed: every variant is scored on the same users.
    assert len(splits) == 1
