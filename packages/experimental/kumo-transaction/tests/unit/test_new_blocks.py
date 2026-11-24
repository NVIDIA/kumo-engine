# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental blocks: GRU, mean readout, joint fusion, fine-tuning arms, auxiliaries."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from tfm.adaptation.batching import minibatches, select_rows, shuffled
from tfm.adaptation.finetune import FineTune, finetune_binary
from tfm.adaptation.fusion import JointFusionHead, PeriodicEmbedding
from tfm.adaptation.probe import probe_arms
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.layers.decoder import Decoder
from tfm.layers.gru import GRUBackbone
from tfm.pooling.mean import MeanPoolModule
from tfm.runtime.data import prepare_data
from tfm.runtime.pipeline import build_model
from tfm.training.lora import LoRA

FLAT = Path("tests/fixtures/configs/flat_decoder.yaml")
V2 = Path("recipes/blueprint_v2/config.smoke.yaml")


def _config(path: Path, **sections: dict[str, object]) -> RecipeConfig:
    raw = load_recipe_config(path).model_dump(mode="json")
    for section, values in sections.items():
        if section == "composition":
            raw["model"]["composition"].update(values)
        else:
            raw[section].update(values)
    return RecipeConfig.model_validate(raw)


def test_mean_pool_ignores_padding() -> None:
    states = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert MeanPoolModule()(states, mask).item() == pytest.approx(2.0)


def test_gru_is_causal() -> None:
    torch.manual_seed(0)
    gru = GRUBackbone(Decoder(depth=2, hidden_size=16, num_heads=4, vocab_size=32, dropout=0.0))
    ids = torch.randint(5, 32, (1, 6))
    mask = torch.ones_like(ids)
    changed = ids.clone()
    changed[0, 4:] = 7
    a = gru(ids, mask)
    b = gru(changed, mask)
    assert torch.allclose(a[:, :4], b[:, :4])
    with pytest.raises(IncompatibleCompositionError):
        gru(ids, mask, causal=False)


def test_gru_backbone_trains_through_the_flat_stack(tmp_path: Path) -> None:
    config = _config(FLAT, model={"backbone": "gru"})
    data = prepare_data(config)
    model = build_model(config, data.tokenizer.vocab_size)
    out = model(data.batches[0])
    assert out.loss is not None and torch.isfinite(out.loss)


def test_gru_refuses_packed_rows() -> None:
    with pytest.raises(ValueError, match="packs flat token streams|backbone"):
        _config(V2, model={"backbone": "gru"})


def test_periodic_embedding_standardizes_with_training_statistics() -> None:
    plr = PeriodicEmbedding(3, frequencies=4, dim=5)
    train = torch.tensor([[0.0, 10.0, 100.0], [2.0, 30.0, 300.0]])
    plr.fit_statistics(train)
    assert torch.allclose(plr.mean, torch.tensor([1.0, 20.0, 200.0]))
    assert plr(train).shape == (2, 15)


def test_joint_fusion_head_uses_both_inputs() -> None:
    torch.manual_seed(0)
    head = JointFusionHead(8, 3, cross_layers=2, frequencies=4)
    emb = torch.randn(4, 8)
    raw = torch.randn(4, 3)
    base = head(emb, raw)
    assert base.shape == (4,)
    assert not torch.allclose(base, head(emb, raw + 1.0))
    assert not torch.allclose(base, head(torch.randn(4, 8), raw))


def test_probe_arms_follow_the_config_flags() -> None:
    assert probe_arms(use_raw_features=True, use_tfm_embeddings=True) == (
        "raw",
        "embedding",
        "combined",
    )
    assert probe_arms(use_raw_features=False, use_tfm_embeddings=True) == ("embedding",)
    with pytest.raises(ValueError):
        probe_arms(use_raw_features=False, use_tfm_embeddings=False)


def test_batching_keeps_rows_aligned() -> None:
    config = load_recipe_config(FLAT)
    batch = prepare_data(config).batches[0]
    picked = select_rows(batch, [2, 0])
    assert picked.entity_ids == (batch.entity_ids[2], batch.entity_ids[0])
    assert torch.equal(picked.input_ids[1], batch.input_ids[0])
    assert sum(c.batch_size for c in minibatches(batch, 5)) == batch.batch_size
    assert sorted(shuffled(batch, 1).entity_ids) == sorted(batch.entity_ids)


@pytest.mark.parametrize(
    ("path", "spec"),
    [
        (FLAT, FineTune(method="lora", max_steps=3, batch_size=4)),
        (FLAT, FineTune(method="full", head="joint_fusion", max_steps=3, batch_size=4)),
        (V2, FineTune(method="lora", head="joint_fusion", max_steps=3, batch_size=4)),
    ],
)
def test_finetune_arms_run_on_every_stack(path: Path, spec: FineTune, tmp_path: Path) -> None:
    config = load_recipe_config(path)
    data = prepare_data(config, fixture_dir=tmp_path / "book")
    model = build_model(config, data.tokenizer.vocab_size)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    train, test = data.batches[0], data.batches[2]
    y_tr = np.array([i % 2 for i in range(train.batch_size)], dtype=np.float32)
    y_ev = np.array([i % 2 for i in range(test.batch_size)], dtype=np.float32)
    metrics, manifest = finetune_binary(
        model, train, test, y_tr, y_ev, spec, device=torch.device("cpu"), seed=0, k=3
    )
    assert "roc_auc" in metrics
    assert manifest["steps"] == 3
    # The pretrained model is copied; the frozen baseline is untouched.
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    if spec.method == "lora":
        assert manifest["n_adapted_modules"] > 0


@dataclasses.dataclass
class _Rows:
    features: torch.Tensor
    numeric_features: torch.Tensor | None = None
    entity_ids: tuple[str, ...] = ()

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device) -> _Rows:
        return _Rows(self.features.to(device))


class _Encoder(torch.nn.Module):
    hidden_size = 4
    # Class-level so the deep copy that fine-tuning trains records here too.
    seen: ClassVar[list[torch.Tensor]] = []

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 4)

    def forward(self, batch: _Rows, compute_loss: bool = False) -> SimpleNamespace:
        if self.training:
            type(self).seen.append(batch.features.detach().clone())
        return SimpleNamespace(embeddings=self.proj(batch.features))


def test_finetune_mixes_classes_when_rows_arrive_grouped() -> None:
    """`split_entities` returns every negative, then every positive.

    Stepping through those rows in order gives one-class minibatches, and a run
    that ends on a long one-class stretch scores every row the same.
    """
    torch.manual_seed(0)
    y = np.array([0.0] * 60 + [1.0] * 60, dtype=np.float32)
    rows = _Rows(torch.tensor(y).unsqueeze(1) * 2.0 - 1.0 + 0.3 * torch.randn(120, 1))
    _Encoder.seen.clear()
    metrics, _manifest = finetune_binary(
        _Encoder(),
        rows,
        rows,
        y,
        y,
        FineTune(method="full", max_steps=30, batch_size=8, learning_rate=1e-2),
        device=torch.device("cpu"),
        seed=0,
        k=10,
    )
    seen = [b for b in _Encoder.seen if b.shape[0] == 8]
    assert seen, "the copy that trains should record its minibatches"
    mixed = sum(bool((b > 0).any() and (b < 0).any()) for b in seen)
    assert mixed >= len(seen) // 2
    assert metrics["roc_auc"] > 0.95


def test_finetune_early_stopping_holds_out_whole_entities() -> None:
    torch.manual_seed(0)
    y = np.array([0.0] * 60 + [1.0] * 60, dtype=np.float32)
    rows = _Rows(
        torch.tensor(y).unsqueeze(1) * 2.0 - 1.0 + 0.3 * torch.randn(120, 1),
        entity_ids=tuple(f"u{i // 4}" for i in range(120)),
    )
    spec = FineTune(
        method="full",
        max_steps=500,
        batch_size=8,
        learning_rate=1e-2,
        validation_fraction=0.25,
        eval_every=5,
        patience=2,
    )
    metrics, manifest = finetune_binary(
        _Encoder(), rows, rows, y, y, spec, device=torch.device("cpu"), seed=0, k=10
    )
    # 8 of the 30 entities, 4 rows each.
    assert manifest["n_validation_rows"] == 32
    assert manifest["steps"] < 500
    assert manifest["best_step"] % 5 == 0 and manifest["best_step"] <= manifest["steps"]
    curve = dict(manifest["validation_curve"])
    assert min(curve, key=lambda step: curve[step]) == manifest["best_step"]
    assert curve[manifest["best_step"]] == manifest["best_validation_loss"]
    assert metrics["roc_auc"] > 0.95


def test_finetune_early_stopping_needs_entity_ids() -> None:
    y = np.array([0.0, 1.0] * 10, dtype=np.float32)
    rows = _Rows(torch.randn(20, 1))
    with pytest.raises(IncompatibleCompositionError, match="entity_ids"):
        finetune_binary(
            _Encoder(),
            rows,
            rows,
            y,
            y,
            FineTune(method="full", max_steps=5, validation_fraction=0.2),
            device=torch.device("cpu"),
            seed=0,
            k=3,
        )
    with pytest.raises(IncompatibleCompositionError, match="validation_fraction"):
        FineTune(validation_fraction=1.0)


def test_finetune_arm_names() -> None:
    assert FineTune().arm == "lora"
    assert FineTune(method="full").arm == "full"
    assert FineTune(head="joint_fusion", lora=LoRA()).arm == "lora_joint_fusion"


@pytest.mark.parametrize("auxiliary", ["past_reconstruction", "subsequence_contrastive"])
def test_v2_auxiliary_objectives_add_a_loss(auxiliary: str, tmp_path: Path) -> None:
    config = _config(V2, objective={"auxiliary": [auxiliary]})
    data = prepare_data(config, fixture_dir=tmp_path / "book")
    model = build_model(config, data.tokenizer.vocab_size)
    out = model(data.batches[0])
    assert out.losses is not None and auxiliary in out.losses
    assert torch.isfinite(out.losses[auxiliary])
    out.loss.backward()  # type: ignore[union-attr]


def test_mean_readout_on_v2(tmp_path: Path) -> None:
    config = _config(V2, composition={"readout": "mean"})
    data = prepare_data(config, fixture_dir=tmp_path / "book")
    model = build_model(config, data.tokenizer.vocab_size)
    out = model(data.batches[0], compute_loss=False)
    assert out.embeddings is not None and out.embeddings.shape[0] == data.batches[0].batch_size
