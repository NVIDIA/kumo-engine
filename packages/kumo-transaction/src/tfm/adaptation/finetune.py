# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune a pretrained model with a binary head, for any stack.

The pretrained model is deep-copied, so the frozen-embedding probe it feeds
stays the untouched baseline. `method` decides which encoder weights move:

- `lora`: every pretrained weight is frozen and low-rank adapters train on
  the targeted linear layers (PRAGMA, nuFormer).
- `full`: every weight trains.

`head` decides what reads the embedding:

- `linear`: a `Binary` head on the embedding.
- `joint_fusion`: `JointFusionHead` over the embedding and the batch's raw
  features, trained end to end (nuFormer).

With `validation_fraction > 0`, that share of training entities is held out,
the trained weights with the lowest validation log-loss are kept, and training
stops early once it stops improving. Log-loss rather than PR-AUC: a ranking
metric can sit flat while the scores are still moving toward a better order.

The model only needs `forward(batch, compute_loss=False).embeddings` and a
`hidden_size`, so the same loop adapts the flat decoder, the GRU, and both
hierarchical stacks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tfm.adaptation.batching import minibatches, select_rows
from tfm.adaptation.fusion import JointFusionHead
from tfm.contracts.errors import IncompatibleCompositionError, InsufficientLabelSupportError
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.heads.binary import BinaryHead
from tfm.training.lora import LoRA, apply_lora
from tfm.training.reproducibility import seed_everything
from tfm.training.trainer import AUTOCAST_DTYPES


@dataclass(frozen=True)
class FineTune:
    """One fine-tuning arm. Reported as `<arm>_<task>` beside the frozen probe."""

    method: Literal["lora", "full"] = "lora"
    head: Literal["linear", "joint_fusion"] = "linear"
    lora: LoRA = field(default_factory=LoRA)
    max_steps: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 8
    cross_layers: int = 2
    plr_frequencies: int = 8
    validation_fraction: float = 0.0
    """Share of training entities held out for early stopping; 0 trains `max_steps`."""
    eval_every: int = 50
    patience: int = 3
    """Validation checks without a lower log-loss before stopping."""
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise IncompatibleCompositionError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}."
            )
        if self.eval_every < 1 or self.patience < 1:
            raise IncompatibleCompositionError("eval_every and patience must be >= 1.")

    @property
    def arm(self) -> str:
        """Metric prefix: `lora` and `full` for a linear head, else `<method>_<head>`."""
        return self.method if self.head == "linear" else f"{self.method}_{self.head}"


class _Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, *, fused: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.fused = fused

    def forward(self, batch: Any) -> Tensor:
        output = self.encoder(batch, compute_loss=False)
        if output.embeddings is None:
            raise IncompatibleCompositionError("Fine-tuning needs a model that returns embeddings.")
        if not self.fused:
            return self.head(output.embeddings)
        if batch.numeric_features is None:
            raise IncompatibleCompositionError(
                "head: joint_fusion needs the batch's raw numeric_features."
            )
        return self.head(output.embeddings, batch.numeric_features)


def _head(spec: FineTune, hidden_size: int, train_batch: Any) -> tuple[nn.Module, bool]:
    if spec.head == "linear":
        return BinaryHead(hidden_size), False
    features = getattr(train_batch, "numeric_features", None)
    if features is None or features.dim() != 2 or features.shape[1] == 0:
        raise IncompatibleCompositionError(
            "head: joint_fusion needs raw numeric_features on the training batch."
        )
    head = JointFusionHead(
        hidden_size,
        int(features.shape[1]),
        cross_layers=spec.cross_layers,
        frequencies=spec.plr_frequencies,
    )
    head.numeric.fit_statistics(features)
    return head, True


def _validation_rows(
    train_batch: Any, y_train: NDArray[np.floating], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split training rows by entity, so no entity is on both sides."""
    entities: list[str] = [str(e) for e in getattr(train_batch, "entity_ids", ()) or ()]
    if len(entities) != len(y_train):
        raise IncompatibleCompositionError(
            "validation_fraction needs entity_ids on every training row."
        )
    unique = sorted(set(entities))
    order = np.random.default_rng(seed).permutation(len(unique)).tolist()
    held = {unique[i] for i in order[: max(1, round(fraction * len(unique)))]}
    fit = [i for i, e in enumerate(entities) if e not in held]
    val = [i for i, e in enumerate(entities) if e in held]
    labels = np.asarray(y_train)
    if len(set(labels[val].tolist())) < 2 or len(set(labels[fit].tolist())) < 2:
        raise InsufficientLabelSupportError(
            "The early-stopping split left one class on a side; lower validation_fraction."
        )
    return fit, val


def finetune_binary(
    model: nn.Module,
    train_batch: Any,
    eval_batch: Any,
    y_train: NDArray[np.floating],
    y_eval: NDArray[np.floating],
    spec: FineTune,
    *,
    device: torch.device,
    seed: int,
    k: int,
    precision: str = "fp32",
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fine-tune a copy of `model` on `(train_batch, y_train)`; score `eval_batch`.

    Returns `(metrics, manifest)`. The manifest records what trained: adapted
    modules and trainable fraction for LoRA, the parameter counts otherwise.
    """
    if train_batch.batch_size != len(y_train) or eval_batch.batch_size != len(y_eval):
        raise IncompatibleCompositionError("Fine-tuning labels must align with batch rows.")
    compute_dtype = AUTOCAST_DTYPES[precision]
    seed_everything(seed)
    manifest: dict[str, Any] = {"method": spec.method, "head": spec.head}
    y_all = np.asarray(y_train)
    val_batch: Any = None
    y_val = y_all[:0]
    if spec.validation_fraction > 0.0:
        fit_rows, val_rows = _validation_rows(train_batch, y_all, spec.validation_fraction, seed)
        val_batch, y_val = select_rows(train_batch, val_rows), y_all[val_rows]
        train_batch, y_all = select_rows(train_batch, fit_rows), y_all[fit_rows]
        manifest["n_validation_rows"] = len(val_rows)
    encoder: Any = copy.deepcopy(model)
    if spec.method == "lora":
        manifest.update(apply_lora(encoder, spec.lora).as_dict())
    else:
        for param in encoder.parameters():
            param.requires_grad_(True)
    hidden = int(encoder.hidden_size)
    head, fused = _head(spec, hidden, train_batch)
    classifier = _Classifier(encoder, head, fused=fused).to(device)
    params = [p for p in classifier.parameters() if p.requires_grad]
    if spec.method == "full":
        manifest["n_trainable_params"] = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=spec.learning_rate)
    crit = nn.BCEWithLogitsLoss()
    n_rows = len(y_all)
    labels = torch.as_tensor(y_all, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    def epoch() -> list[tuple[Any, Tensor]]:
        # Callers may pass rows grouped by class (split_entities does); unshuffled,
        # every minibatch is one class and the head ends on a constant score.
        order = torch.randperm(n_rows, generator=generator).tolist()
        return [
            (select_rows(train_batch, rows), labels[rows])
            for rows in (order[s : s + spec.batch_size] for s in range(0, n_rows, spec.batch_size))
        ]

    def autocast() -> torch.autocast:
        return torch.autocast(
            device_type=device.type, dtype=compute_dtype, enabled=compute_dtype is not None
        )

    def score(batch: Any) -> NDArray[np.float64]:
        classifier.eval()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for chunk in minibatches(batch, spec.batch_size):
                with autocast():
                    logits = classifier(chunk.to(device))
                scores.append(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64))
        classifier.train()
        return np.concatenate(scores, axis=0) if scores else np.zeros((0,), np.float64)

    classifier.train()
    losses: list[float] = []
    pending: list[tuple[Any, Tensor]] = []
    best: tuple[float, int, dict[str, Tensor]] | None = None
    checks_without_gain = 0
    for step in range(1, spec.max_steps + 1):
        if not pending:
            pending = epoch()
        chunk, target = pending.pop(0)
        chunk, target = chunk.to(device), target.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = classifier(chunk)
        loss = crit(logits.float(), target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if val_batch is not None and (step % spec.eval_every == 0 or step == spec.max_steps):
            p = np.clip(score(val_batch), 1e-7, 1 - 1e-7)
            val_loss = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
            manifest.setdefault("validation_curve", []).append([step, val_loss])
            if best is None or val_loss < best[0]:
                trained = {
                    name: w.detach().clone()
                    for name, w in classifier.named_parameters()
                    if w.requires_grad
                }
                best, checks_without_gain = (val_loss, step, trained), 0
                manifest["best_validation_pr_auc"] = classification_metrics(y_val, p, k=k)["pr_auc"]
            else:
                checks_without_gain += 1
                if checks_without_gain >= spec.patience:
                    break
    if best is not None:
        classifier.load_state_dict(best[2], strict=False)
        manifest["best_step"] = best[1]
        manifest["best_validation_loss"] = best[0]

    predicted = score(eval_batch)
    metrics = classification_metrics(np.asarray(y_eval), predicted, k=k)
    manifest["final_loss"] = losses[-1] if losses else float("nan")
    manifest["steps"] = len(losses)
    manifest["precision"] = precision
    return metrics, manifest
