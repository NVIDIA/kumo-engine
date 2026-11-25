# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract every training backend must pass (ADR 0011).

A backend executes the model's semantics; it does not define them. So each
registered backend is held to eager training on the same tiny model and batch:
matching loss, no learned copying, untouched masks, a checkpoint that
round-trips, and determinism under a seed. A backend that cannot run here
(no CUDA, missing extra) is skipped by its declared capabilities, never by
catching its failure.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from tfm import Decoder, Flat, TrxFM
from tfm.config import OptimizerConfig, RecipeConfig
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.capabilities import BackendCapabilities
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.extras.capability import extra_available
from tfm.training.backends import (
    advertised_backends,
    backend_capabilities,
    build_trainer,
    check_backend_supports,
)
from tfm.training.diagnostics import label_alignment_losses

VOCAB = 32
FIRST_TOKEN = 3
LEARNING_RATE = 1e-2
PARITY_STEPS = 20
ANTI_COPY_STEPS = 150


def _row(start: int, length: int, width: int) -> tuple[list[int], list[int]]:
    """Tokens that count upward: the next token is always this one plus one."""
    span = VOCAB - FIRST_TOKEN
    ids = [FIRST_TOKEN + (start + i) % span for i in range(length)]
    return ids + [0] * (width - length), [1] * length + [0] * (width - length)


def _batch(rows: list[tuple[int, int]], width: int = 12) -> TransactionBatch:
    built = [_row(start, length, width) for start, length in rows]
    input_ids = torch.tensor([ids for ids, _ in built], dtype=torch.long)
    attention_mask = torch.tensor([mask for _, mask in built], dtype=torch.long)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return TransactionBatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def _batches() -> list[TransactionBatch]:
    """Two batches with right padding, so masks are exercised on every step."""
    return [
        _batch([(0, 12), (5, 9), (11, 12), (17, 6)]),
        _batch([(2, 12), (8, 12), (14, 10), (23, 7)]),
    ]


def _hf_model() -> TrxFM:
    from tfm.runtime import build_model

    config = RecipeConfig.model_validate(
        {
            "seed": 0,
            "data": {
                "adapter": "synthetic",
                "max_tokens": 12,
                "temporal_split": {
                    "train_end": "2024-01-04T00:00:00Z",
                    "validation_end": "2024-01-07T00:00:00Z",
                },
            },
            "model": {
                "backbone": "huggingface",
                "composition": {
                    "position": "rope",
                    "granularity": {
                        "decoder": {
                            "depth": 1,
                            "hidden_size": 32,
                            "num_heads": 4,
                            "max_seq_len": 12,
                            "dropout": 0.0,
                            "mask": "causal",
                        }
                    },
                },
            },
        }
    )
    torch.manual_seed(0)
    return build_model(config, VOCAB)


def _eager_model() -> TrxFM:
    torch.manual_seed(0)
    return TrxFM(
        granularity=Flat(
            Decoder(depth=1, hidden_size=32, num_heads=4, vocab_size=VOCAB, dropout=0.0)
        )
    )


def _runnable(name: str) -> tuple[BackendCapabilities, str, torch.device]:
    """Capabilities, the backbone to test with, and the device, or skip."""
    capabilities = backend_capabilities(name)
    if capabilities.requires_cuda and not torch.cuda.is_available():
        pytest.skip(f"{name} declares requires_cuda and no CUDA device is visible")
    backbone = "eager" if "eager" in capabilities.backbones else "huggingface"
    if backbone == "huggingface" and not extra_available("transformers"):
        pytest.skip(f"{name} trains only the huggingface backbone; transformers is missing")
    device = torch.device("cuda" if capabilities.requires_cuda else "cpu")
    return capabilities, backbone, device


def _model(backbone: str) -> TrxFM:
    return _hf_model() if backbone == "huggingface" else _eager_model()


def _fit(
    name: str,
    model: nn.Module,
    *,
    steps: int,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    batches: list[TransactionBatch] | None = None,
    precision: str | None = None,
) -> dict[str, float | str]:
    trainer = build_trainer(name, checkpoint_dir=checkpoint_dir)
    # Omitted, each backend runs its own default precision.
    extra = {} if precision is None else {"precision": precision}
    return trainer.fit(
        model,
        batches if batches is not None else _batches(),
        max_steps=steps,
        learning_rate=LEARNING_RATE,
        seed=0,
        device=device,
        optimizer=OptimizerConfig(),
        **extra,
    )


def _logits(model: nn.Module, batch: TransactionBatch, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(batch.to(device), compute_loss=False)
    assert output.logits is not None
    return output.logits.float().cpu()


BACKENDS = advertised_backends()


def test_every_backend_declares_its_own_capabilities() -> None:
    for name in BACKENDS:
        capabilities = backend_capabilities(name)
        assert capabilities.name == name
        if "huggingface" in capabilities.backbones:
            assert capabilities.consolidated_hf_export, f"{name} must export HF checkpoints"
        assert capabilities.precisions, f"{name} must declare a precision"
        assert capabilities.precisions <= {"fp32", "bf16"}


def test_eager_is_the_reference_baseline() -> None:
    capabilities = backend_capabilities("eager")
    assert capabilities.support_tier == "reference"
    assert not capabilities.requires_cuda


@pytest.mark.parametrize("name", [n for n in BACKENDS if n != "eager"])
def test_loss_matches_eager(name: str) -> None:
    _, backbone, device = _runnable(name)
    reference = _model(backbone)
    candidate = copy.deepcopy(reference)

    expected = _fit("eager", reference, steps=PARITY_STEPS, device=device)
    actual = _fit(name, candidate, steps=PARITY_STEPS, device=device)

    # Non-eager backends may compute in bf16; eager runs fp32.
    assert actual["first_loss"] == pytest.approx(expected["first_loss"], rel=2e-2)
    assert actual["final_loss"] == pytest.approx(expected["final_loss"], rel=5e-2)


@pytest.mark.parametrize(
    "name", [n for n in BACKENDS if "bf16" in backend_capabilities(n).precisions]
)
def test_bf16_matches_fp32_eager_and_keeps_fp32_weights(name: str) -> None:
    _, backbone, device = _runnable(name)
    reference = _model(backbone)
    candidate = copy.deepcopy(reference)

    expected = _fit("eager", reference, steps=PARITY_STEPS, device=device, precision="fp32")
    actual = _fit(name, candidate, steps=PARITY_STEPS, device=device, precision="bf16")

    assert actual["precision"] == "bf16"
    assert actual["first_loss"] == pytest.approx(expected["first_loss"], rel=2e-2)
    assert actual["final_loss"] == pytest.approx(expected["final_loss"], rel=5e-2)
    assert actual["final_loss"] != expected["final_loss"], "bf16 compute left fp32 losses unchanged"
    assert all(p.dtype == torch.float32 for p in candidate.parameters())


@pytest.mark.parametrize("name", BACKENDS)
def test_model_predicts_the_next_token_not_its_input(name: str) -> None:
    """The check that would have caught v1: a copying model scores same-token lower."""
    _, backbone, device = _runnable(name)
    model = _model(backbone)

    _fit(name, model, steps=ANTI_COPY_STEPS, device=device)

    batch = _batches()[0]
    losses = label_alignment_losses(
        _logits(model, batch, device), batch.input_ids, batch.attention_mask
    )
    assert not losses.learned_to_copy, losses
    assert losses.next_token < 0.5 * losses.same_token, losses


def _with_garbage_padding(batch: TransactionBatch, seed: int) -> TransactionBatch:
    """The same batch with random tokens where the mask is zero."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randint(FIRST_TOKEN, VOCAB, batch.input_ids.shape, generator=generator)
    padded = batch.attention_mask == 0
    return TransactionBatch(
        input_ids=torch.where(padded, noise, batch.input_ids),
        attention_mask=batch.attention_mask.clone(),
        labels=None if batch.labels is None else batch.labels.clone(),
    )


@pytest.mark.parametrize("name", BACKENDS)
def test_backend_does_not_mutate_its_batches(name: str) -> None:
    _, backbone, device = _runnable(name)
    batches = _batches()
    before = [(b.input_ids.clone(), b.attention_mask.clone(), b.labels.clone()) for b in batches]  # type: ignore[union-attr]

    _fit(name, _model(backbone), steps=PARITY_STEPS, device=device, batches=batches)

    for batch, (ids, mask, labels) in zip(batches, before, strict=True):
        assert torch.equal(batch.input_ids, ids)
        assert torch.equal(batch.attention_mask, mask)
        assert batch.labels is not None and torch.equal(batch.labels, labels)


@pytest.mark.parametrize("name", BACKENDS)
def test_training_ignores_masked_positions(name: str) -> None:
    """Whatever sits under a zero mask must not change a single step."""
    _, backbone, device = _runnable(name)
    clean = _model(backbone)
    noisy = copy.deepcopy(clean)

    a = _fit(name, clean, steps=PARITY_STEPS, device=device)
    b = _fit(
        name,
        noisy,
        steps=PARITY_STEPS,
        device=device,
        batches=[_with_garbage_padding(batch, seed=i) for i, batch in enumerate(_batches())],
    )

    tolerance = 1e-6 if device.type == "cpu" else 1e-4
    assert b["first_loss"] == pytest.approx(a["first_loss"], rel=tolerance, abs=tolerance)
    assert b["final_loss"] == pytest.approx(a["final_loss"], rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("name", BACKENDS)
def test_consolidated_hf_checkpoint_round_trips(name: str, tmp_path: Path) -> None:
    capabilities = backend_capabilities(name)
    if "huggingface" not in capabilities.backbones:
        pytest.skip(f"{name} does not train the huggingface backbone")
    if not extra_available("transformers"):
        pytest.skip("transformers is not installed")
    _, _, device = _runnable(name)
    model = _hf_model()

    stats = _fit(name, model, steps=PARITY_STEPS, device=device, checkpoint_dir=tmp_path)

    assert stats["hf_checkpoint"] == str(tmp_path)
    from transformers import AutoModelForCausalLM

    restored = AutoModelForCausalLM.from_pretrained(str(tmp_path)).to(device).eval()
    batch = _batches()[0]
    with torch.no_grad():
        reloaded = restored(
            input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device)
        ).logits.float()
    valid = batch.attention_mask.bool()
    torch.testing.assert_close(
        reloaded.cpu()[valid], _logits(model, batch, device)[valid], atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("name", BACKENDS)
def test_training_is_deterministic_under_a_seed(name: str) -> None:
    _, backbone, device = _runnable(name)
    first = _model(backbone)
    second = copy.deepcopy(first)

    a = _fit(name, first, steps=PARITY_STEPS, device=device)
    b = _fit(name, second, steps=PARITY_STEPS, device=device)

    tolerance = 0.0 if device.type == "cpu" else 1e-4
    assert b["final_loss"] == pytest.approx(a["final_loss"], rel=tolerance, abs=tolerance)


def test_alignment_check_flags_a_copying_model() -> None:
    """Negative control: logits that reproduce the input are caught."""
    batch = _batches()[0]
    copying = torch.nn.functional.one_hot(batch.input_ids, VOCAB).float() * 20.0

    losses = label_alignment_losses(copying, batch.input_ids, batch.attention_mask)

    assert losses.learned_to_copy
    assert losses.same_token_accuracy == 1.0


def test_runtime_refuses_packing_on_a_backend_without_segment_masks() -> None:
    capabilities = backend_capabilities("fsdp2")
    with pytest.raises(UnsupportedCombinationError, match="segment masks"):
        check_backend_supports(
            capabilities, backbone="huggingface", packing=True, device_type="cuda", world_size=1
        )


def test_runtime_refuses_multi_process_on_a_single_process_backend() -> None:
    with pytest.raises(UnsupportedCombinationError, match="single-process"):
        check_backend_supports(
            backend_capabilities("eager"),
            backbone="eager",
            packing=False,
            device_type="cpu",
            world_size=8,
        )


def test_runtime_refuses_an_undeclared_precision() -> None:
    with pytest.raises(UnsupportedCombinationError, match="precision: fp32"):
        check_backend_supports(
            backend_capabilities("fsdp2"),
            backbone="huggingface",
            packing=False,
            device_type="cuda",
            world_size=1,
            precision="fp32",
        )


def test_fsdp2_refuses_fp32_rather_than_computing_bf16() -> None:
    with pytest.raises(UnsupportedCombinationError, match="training.precision: bf16"):
        build_trainer("fsdp2").fit(
            _eager_model(),
            _batches(),
            max_steps=1,
            learning_rate=LEARNING_RATE,
            seed=0,
            device=torch.device("cpu"),
            precision="fp32",
        )


def test_runtime_refuses_an_unsupported_backbone_and_layout() -> None:
    with pytest.raises(UnsupportedCombinationError, match="backbone: eager.*layout"):
        check_backend_supports(
            backend_capabilities("fsdp2"),
            backbone="eager",
            packing=False,
            device_type="cuda",
            world_size=1,
            layout="key_value_time",
        )
