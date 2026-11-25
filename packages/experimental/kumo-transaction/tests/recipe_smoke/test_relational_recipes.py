# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end shared dataset recipes; synthetic data only, CPU-safe."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tfm.cli.recipe import REQUIRED_ARTIFACTS, run_command


@pytest.mark.parametrize("variant", ["smoke", "graph_smoke", "hybrid_smoke"])
@pytest.mark.parametrize("loader", ["python", "kumo_pytorch"])
def test_relational_recipe_and_checkpoint_reuse(variant, loader, tmp_path):
    if loader == "kumo_pytorch":
        pytest.importorskip("kumo.pytorch.api", reason="requires optional native extra")
    if variant != "smoke":
        pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    config = yaml.safe_load(
        (Path("recipes/relational_inputs") / f"config.{variant}.yaml").read_text()
    )
    config["data"]["loader"] = loader
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    assert all((out / name).exists() for name in REQUIRED_ARTIFACTS)
    assert {"raw", "embedding", "combined", "task_head"} <= json.loads(
        (out / "metrics.json").read_text()
    ).keys()
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["schema"]["target_level"] == "entity"
    split = json.loads((out / "split_manifest.json").read_text())
    assert len(split["anchors"]) == 8
    assert split["loader"]["name"] == loader
    before = dict(np.load(out / "embeddings.npz"))
    # The embedded identity is authoritative and repairs a corrupt sidecar.
    (out / "model.manifest.json").write_text("broken")
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
    after = dict(np.load(out / "embeddings.npz"))
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert json.loads((out / "model.manifest.json").read_text()) == metadata


def test_sequence_mode_never_builds_or_transfers_a_graph(monkeypatch, tmp_path):
    from tfm.data.relational_batches import RelationalBatches

    def forbidden(*args, **kwargs):
        raise AssertionError("sequence mode must not construct a graph")

    monkeypatch.setattr(RelationalBatches, "graph_sampler", forbidden)
    run_command(
        "smoke",
        config_path=Path("recipes/relational_inputs/config.smoke.yaml"),
        run_dir=tmp_path,
        device="cpu",
    )


@pytest.mark.parametrize("variant", ["graph_smoke", "hybrid_smoke"])
def test_relational_future_pretraining_and_full_downstream(variant, tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    config = yaml.safe_load(
        (Path("recipes/relational_inputs") / f"config.{variant}.yaml").read_text()
    )
    config["training"]["objective"] = "future_subgraph"
    if variant == "graph_smoke":
        config["model"]["composition"]["objective"] = "future_subgraph"
    path = tmp_path / "future.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert "future_subgraph" in metadata["stages"]
    assert metadata["target_transform"]["selected"]
    assert (out / "encoder.pt").exists()
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)


def test_label_free_encoder_export_and_graph_downstream_fit(tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    pretrain = Path("recipes/relational_inputs/config.encoder_pretrain_smoke.yaml")
    out = run_command("train", config_path=pretrain, run_dir=tmp_path / "pretrain", device="cpu")
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["tokenizer"] is None
    assert metadata["target_transform"] is not None
    config = yaml.safe_load(Path("recipes/relational_inputs/config.graph_smoke.yaml").read_text())
    config["runtime"].update(stage="downstream", encoder_checkpoint=str(out / "encoder.pt"))
    path = tmp_path / "downstream.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_command("smoke", config_path=path, run_dir=tmp_path / "downstream", device="cpu")
    loaded = json.loads((result / "model.manifest.json").read_text())
    assert "graph_binary" in loaded["stages"]  # The downstream head cannot remain random.


def test_reuse_rejects_same_width_feature_reordering_before_state_installation(
    monkeypatch, tmp_path
):
    from dataclasses import replace

    import torch

    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    from tfm.runtime import relational

    original = relational.load_adapter_events
    reverse = False

    def adapter(*args, **kwargs):
        loaded = original(*args, **kwargs)
        dataset = loaded.relational
        fields = [("amount", "value"), ("other", "other_value")]
        if reverse:
            fields.reverse()
        tables = tuple(
            replace(table, numeric=dict(fields)) if table.name == "activity" else table
            for table in dataset.tables
        )
        rows = dict(dataset.rows)
        rows["activity"] = tuple(
            {**row, "other_value": row["value"] + 100} for row in rows["activity"]
        )
        return replace(loaded, relational=replace(dataset, tables=tables, rows=rows))

    monkeypatch.setattr(relational, "load_adapter_events", adapter)
    config = Path("recipes/relational_inputs/config.graph_smoke.yaml")
    out = run_command("train", config_path=config, run_dir=tmp_path / "run", device="cpu")
    reverse = True

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint identity must be checked before state installation")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    with pytest.raises(ValueError, match="schema differs"):
        run_command(
            "evaluate", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True
        )


def test_explicit_all_negative_entity_labels_remain_valid_for_evaluation(monkeypatch, tmp_path):
    from dataclasses import replace

    from tfm.runtime import relational

    original = relational.load_adapter_events

    def adapter(*args, **kwargs):
        loaded = original(*args, **kwargs)
        rows = dict(loaded.relational.rows)
        rows["outcomes"] = tuple({**row, "outcome": 0.0} for row in rows["outcomes"])
        return replace(loaded, relational=replace(loaded.relational, rows=rows))

    monkeypatch.setattr(relational, "load_adapter_events", adapter)
    out = run_command(
        "evaluate",
        config_path=Path("recipes/relational_inputs/config.smoke.yaml"),
        run_dir=tmp_path,
        device="cpu",
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["raw"]["n_eval_positive"] == 0
    assert metrics["embedding"]["n_eval_positive"] == 0


def test_shared_future_batches_group_different_input_cutoffs(monkeypatch, tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    from tfm.data.graph import GraphSampler
    from tfm.runtime import relational

    original, sample = relational.load_adapter_events, GraphSampler.sample
    observed = []

    def adapter(*args, **kwargs):
        loaded = original(*args, **kwargs)
        rows = dict(loaded.relational.rows)
        rows["outcomes"] = tuple(
            {**row, "cutoff": row["cutoff"] - timedelta(days=1)} if i < 2 else dict(row)
            for i, row in enumerate(rows["outcomes"])
        )
        return replace(loaded, relational=replace(loaded.relational, rows=rows))

    def sample_at_one_clock(self, anchors, root_type, **kwargs):
        if kwargs.get("sharing") == "shared":
            assert len({(anchor.cutoff, anchor.observation_start) for anchor in anchors}) == 1
            observed.append(anchors[0].cutoff)
        return sample(self, anchors, root_type, **kwargs)

    monkeypatch.setattr(relational, "load_adapter_events", adapter)
    monkeypatch.setattr(GraphSampler, "sample", sample_at_one_clock)
    run_command(
        "smoke",
        config_path=Path("recipes/relational_inputs/config.future_smoke.yaml"),
        run_dir=tmp_path,
        device="cpu",
    )
    assert len(set(observed)) == 2


def test_partial_inference_labels_keep_sample_alignment(monkeypatch, tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    import pyarrow as pa
    import pyarrow.parquet as pq

    from tfm.config import load_recipe_config
    from tfm.runtime import relational

    source = Path("recipes/relational_inputs/config.smoke.yaml")
    config = load_recipe_config(source)
    loaded = relational.load_adapter_events(config, flat_fields=False)
    rows = dict(loaded.relational.rows)
    prediction = {
        **rows["outcomes"][0],
        "id": "prediction",
        "split": "predict",
        "cutoff": rows["outcomes"][0]["cutoff"] + timedelta(days=3),
        "available": rows["outcomes"][0]["available"] + timedelta(days=3),
    }
    rows["outcomes"] = (*rows["outcomes"], prediction)
    dataset = replace(loaded.relational, rows=rows)
    monkeypatch.setattr(
        relational,
        "load_adapter_events",
        lambda *args, **kwargs: replace(loaded, relational=dataset),
    )
    anchors = [
        {
            "sample_id": f"{row['split']}-{row['owner']}",
            "id": row["owner"],
            "cutoff": row["cutoff"],
            "split": row["split"],
        }
        for row in rows["outcomes"]
    ]
    anchors.append(
        {
            "sample_id": "predict-entity1",
            "id": "entity1",
            "cutoff": prediction["cutoff"],
            "split": "predict",
        }
    )
    replay = tmp_path / "anchors.parquet"
    pq.write_table(pa.Table.from_pylist(anchors), replay)
    values = yaml.safe_load(source.read_text())
    values["data"]["relational"]["anchors"] = str(replay)
    values["data"]["relational"]["label_cutoffs"]["predict"] = "2024-01-21"
    path = tmp_path / "infer.yaml"
    path.write_text(yaml.safe_dump(values))
    out = run_command("infer", config_path=path, run_dir=tmp_path / "run", device="cpu")
    arrays = np.load(out / "embeddings.npz")
    assert arrays["predict_sample_ids"].tolist() == ["predict-entity0", "predict-entity1"]
    assert arrays["predict_labels"].shape == (2,)
    assert arrays["predict_labels_valid"].tolist() == [True, False]
    assert arrays["predict_labels"][0] == 0.0
    assert np.isnan(arrays["predict_labels"][1])
