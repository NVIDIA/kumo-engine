# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`tfm migrate-config`: 2.0.0a1 configs rewritten into the current keys."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tfm.cli.main import main
from tfm.config import RecipeConfig, load_recipe_config
from tfm.config_migration import MOVES, migrate_config
from tfm.contracts.errors import SchemaError

FIXTURES = Path("tests/fixtures/configs")
LAB_CHECKPOINT = FIXTURES / "v2.0.0a1_blueprint_v2_lab_tuned.yaml"
CANDIDATE_D = Path("benchmarks/ablations/blueprint_v2_final_d.yaml")


def _flat() -> dict[str, Any]:
    return yaml.safe_load((FIXTURES / "flat_decoder.yaml").read_text(encoding="utf-8"))


def _with(raw: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    node = raw
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    return raw


def _lookup(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        raw = raw[key]
    return raw


def test_the_lab_checkpoint_config_is_candidate_d() -> None:
    raw = yaml.safe_load(LAB_CHECKPOINT.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="was removed"):
        RecipeConfig.model_validate(raw)

    lab = migrate_config(raw).validated().model_dump(mode="json")
    candidate = load_recipe_config(CANDIDATE_D).model_dump(mode="json")

    # 2.0.0a1 scored the fixed XGBoost setting and the surprise arm from the lab
    # notebooks and benchmarks/final_eval.py; candidate D's config declares them.
    for section in (lab, candidate):
        tabular = section["evaluation"]["tabular_baselines"]
        del section["evaluation"]["surprise"], tabular["xgboost_params"], tabular["xgboost_trials"]
    assert lab == candidate


@pytest.mark.parametrize(("old", "new"), list(MOVES.items()), ids=".".join)
def test_each_moved_key_keeps_its_value(old: tuple[str, ...], new: tuple[str, ...]) -> None:
    values: dict[str, Any] = {
        "pretrain_window_events": 4,
        "pretrain_max_tokens": 64,
        "muon_learning_rate": 0.01,
        "muon_momentum": 0.9,
        "muon_adjust_lr": "match_rms_adamw",
        "history_window_probe": True,
        "history_window_max_rows": 50,
        "history_window_train_negatives_per_positive": 5.0,
        "train_head": True,
        "head_loss_weight": 0.5,
        "head_probe_only": True,
    }
    raw = _with(_flat(), old, values[old[-1]])
    if old[-1] == "pretrain_max_tokens":
        raw["data"]["pretrain_window_events"] = 4

    migration = migrate_config(raw)

    assert _lookup(migration.config, new) == values[old[-1]]
    assert f"{'.'.join(old)} -> {'.'.join(new)}" in migration.notes
    migration.validated()


def test_the_input_is_left_unchanged() -> None:
    raw = _with(_flat(), ("data", "pretrain_window_events"), 4)
    before = copy.deepcopy(raw)
    migrate_config(raw)
    assert raw == before


def test_a_config_with_current_keys_passes_through() -> None:
    raw = _flat()
    migration = migrate_config(raw)
    assert migration.config == raw
    assert migration.notes == []


def test_non_overlapping_stride_is_dropped_and_overlap_refused() -> None:
    raw = _with(_flat(), ("data", "pretrain_window_events"), 4)
    raw["data"]["pretrain_window_stride"] = 4
    assert "pretrain_window_stride" not in migrate_config(raw).config["data"]

    raw["data"]["pretrain_window_stride"] = 2
    with pytest.raises(SchemaError, match="overlaps windows of 4 events"):
        migrate_config(raw)


def test_a_row_width_without_windows_is_dropped() -> None:
    raw = _with(_flat(), ("data", "pretrain_max_tokens"), 64)
    migration = migrate_config(raw)
    assert "pretrain_windows" not in migration.config["data"]
    assert any("pretrain_max_tokens" in note for note in migration.notes)
    migration.validated()


def test_packing_length_must_be_the_decoder_context() -> None:
    raw = _flat()
    context = raw["model"]["composition"]["granularity"]["decoder"]["max_seq_len"]
    raw["training"] = {"packing": True, "packing_seq_length": context}
    assert "packing_seq_length" not in migrate_config(raw).config["training"]

    raw["training"]["packing_seq_length"] = context + 1
    with pytest.raises(SchemaError, match="decoder.max_seq_len"):
        migrate_config(raw)


def test_joint_fusion_defaults_are_dropped_and_others_refused() -> None:
    raw = _with(_flat(), ("fraud_task", "finetune", "cross_layers"), 2)
    raw["fraud_task"]["finetune"]["plr_frequencies"] = 8
    assert migrate_config(raw).config["fraud_task"]["finetune"] == {}

    raw["fraud_task"]["finetune"]["cross_layers"] = 3
    with pytest.raises(SchemaError, match="cross_layers=2"):
        migrate_config(raw)


def test_nemo_becomes_fsdp2() -> None:
    raw = _with(_flat(), ("training", "backend"), "nemo")
    assert migrate_config(raw).config["training"]["backend"] == "fsdp2"


def test_old_and_new_key_together_are_refused() -> None:
    raw = _with(_flat(), ("evaluation", "history_window_probe"), True)
    raw["evaluation"]["history_window"] = {"enabled": False}
    with pytest.raises(SchemaError, match="keep one"):
        migrate_config(raw)


def test_cli_writes_a_config_that_loads(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "migrated.yaml"

    assert main(["migrate-config", str(LAB_CHECKPOINT), "--out", str(out)]) == 0

    config = load_recipe_config(out)
    assert config.data.pretrain_windows is not None
    assert config.data.pretrain_windows.events == 128
    assert "data.pretrain_window_events -> data.pretrain_windows.events" in capsys.readouterr().err
