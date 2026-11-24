# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_pragma_like_smoke(tmp_path: Path) -> None:
    from tfm.runtime import run_smoke

    out = run_smoke(
        run_dir=tmp_path / "run",
        config_path=Path("recipes/pragma_like/config.smoke.yaml"),
        device="cpu",
    )
    for name in REQUIRED:
        assert (out / name).exists(), name
    import json

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for baseline in ("raw", "embedding", "combined"):
        assert baseline in metrics
        for key in ("roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "calibration"):
            assert key in metrics[baseline]
    for baseline in ("default_12m_raw", "default_12m_embedding", "default_12m_combined"):
        assert baseline in metrics
        for key in ("roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "calibration"):
            assert key in metrics[baseline]
    tok = json.loads((out / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    assert tok["layout"] == "key_value_time"
    assert tok["support_tier"] == "experimental"
    split = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["n_train_sequences"] > 0
    embed = json.loads((out / "embeddings_manifest.json").read_text(encoding="utf-8"))
    assert embed["n_train"] == split["n_train_sequences"]
    assert embed["n_test"] == split["n_test_sequences"]
    import numpy as np

    described = embed["files"]["embeddings.npz"]
    with np.load(out / "embeddings.npz") as arrays:
        expected = set(described["vectors"]) | set(described["labels"])
        expected |= {f"{side}_{name}" for side in described["vectors"] for name in described["ids"]}
        assert set(arrays.files) == expected
    train_log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    import yaml

    resolved = yaml.safe_load((out / "resolved_config.yaml").read_text(encoding="utf-8"))
    max_steps = int(resolved["training"]["max_steps"])
    assert [row["step"] for row in train_log["steps"]] == [
        float(s) for s in range(1, max_steps + 1)
    ]
