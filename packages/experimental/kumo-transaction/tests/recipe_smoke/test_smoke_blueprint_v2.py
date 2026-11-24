# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

REQUIRED = (
    "resolved_config.yaml",
    "data_validation_report.json",
    "split_manifest.json",
    "tokenizer_manifest.json",
    "checkpoint_manifest.json",
    "embeddings_manifest.json",
    "metrics.json",
    "metrics.md",
    "model_card.md",
    "environment.json",
)


def test_blueprint_v2_smoke(tmp_path: Path) -> None:
    from tfm.runtime import run_smoke

    out = run_smoke(
        run_dir=tmp_path / "run",
        config_path=Path("recipes/blueprint_v2/config.smoke.yaml"),
        device="cpu",
    )
    for name in REQUIRED:
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for baseline in ("raw", "embedding", "combined"):
        for key in ("roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "calibration"):
            assert key in metrics[baseline]
    checkpoint = json.loads((out / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert checkpoint["architecture"] == "profile_event_decoder"
    tok = json.loads((out / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    assert tok["layout"] == "key_value_time"
