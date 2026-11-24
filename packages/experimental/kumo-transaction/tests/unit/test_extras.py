# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest
import torch

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import ExtraUnavailableError, SchemaError, UnsupportedCombinationError
from tfm.extras.capability import extra_available, require_extra
from tfm.training.trainer import build_trainer


def test_import_tfm_does_not_load_optional_backends() -> None:
    script = (
        "import sys, tfm; "
        "assert 'tfm.extras.huggingface' not in sys.modules; "
        "assert 'tfm.extras.rapids' not in sys.modules; "
        "assert 'tfm.extras.fsdp2' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_fsdp2_trainer_never_falls_back_to_eager() -> None:
    from tfm import Decoder, Flat, TrxFM

    model = TrxFM(granularity=Flat(Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=20)))
    before = [p.detach().clone() for p in model.parameters()]
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )
    with pytest.raises((ExtraUnavailableError, UnsupportedCombinationError)):
        build_trainer("fsdp2").fit(
            model,
            [batch],
            max_steps=1,
            learning_rate=1e-3,
            seed=0,
            device=torch.device("cpu"),
        )
    for old, new in zip(before, model.parameters(), strict=True):
        assert torch.equal(old, new)


def test_require_extra_missing_module() -> None:
    with pytest.raises(ExtraUnavailableError, match="not-a-real-extra"):
        require_extra("not-a-real-extra", "tfm_missing_backend_xyz")


def test_build_trainer_eager() -> None:
    trainer = build_trainer("eager")
    assert trainer.__class__.__name__ == "EagerTrainer"


def test_build_trainer_unknown() -> None:
    with pytest.raises(UnsupportedCombinationError, match="training.backend"):
        build_trainer("horovod")


def test_transaction_batch_dataset() -> None:
    from tfm.extras.fsdp2 import TransactionBatchDataset

    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        labels=torch.tensor([[2, -100, -100]]),
    )
    dataset = TransactionBatchDataset([batch])
    assert len(dataset) == 1
    item = dataset[0]
    assert item["input_ids"].shape == (1, 3)
    assert "labels" in item


def _batches(n: int) -> list[TransactionBatch]:
    return [
        TransactionBatch(
            input_ids=torch.full((1, 3), i, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )
        for i in range(n)
    ]


def test_shard_batches_is_a_disjoint_cover() -> None:
    from tfm.extras.fsdp2 import shard_batches

    batches = _batches(8)
    shards = [shard_batches(batches, world_size=4, rank=r) for r in range(4)]

    assert [len(s) for s in shards] == [2, 2, 2, 2]
    seen = [int(b.input_ids[0, 0]) for shard in shards for b in shard]
    assert sorted(seen) == list(range(8))


def test_shard_batches_gives_every_rank_equal_work() -> None:
    """Unequal step counts deadlock: FSDP2 collectives run inside forward."""
    from tfm.extras.fsdp2 import shard_batches

    shards = [shard_batches(_batches(10), world_size=4, rank=r) for r in range(4)]

    assert [len(s) for s in shards] == [2, 2, 2, 2]


def test_shard_batches_rejects_a_world_larger_than_the_corpus() -> None:
    from tfm.extras.fsdp2 import shard_batches

    with pytest.raises(UnsupportedCombinationError, match="cannot be split"):
        shard_batches(_batches(3), world_size=4, rank=0)


def test_shard_batches_is_identity_single_process() -> None:
    from tfm.extras.fsdp2 import shard_batches

    batches = _batches(3)
    assert shard_batches(batches, world_size=1, rank=0) == batches


def test_join_distributed_group_is_noop_without_a_launcher(monkeypatch) -> None:
    from tfm.extras.fsdp2 import join_distributed_group

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert join_distributed_group() == (1, 0, 0)


def test_huggingface_requires_transformers() -> None:
    if extra_available("transformers"):
        pytest.skip("transformers is installed")
    from tfm.config import RecipeConfig
    from tfm.extras.huggingface import build_huggingface_backbone

    config = RecipeConfig.model_validate(
        {
            "data": {
                "temporal_split": {
                    "train_end": "2024-01-04T00:00:00Z",
                    "validation_end": "2024-01-07T00:00:00Z",
                }
            }
        }
    )
    with pytest.raises(ExtraUnavailableError, match="huggingface"):
        build_huggingface_backbone(config, vocab_size=32)


def test_rapids_tokenizer_requires_cudf() -> None:
    if extra_available("cudf"):
        pytest.skip("cudf is installed")
    from tfm.features.tokenizers.registry import advertised_tokenizers, build_tokenizer

    with pytest.raises(ExtraUnavailableError):
        build_tokenizer("rapids_field_tokenizer")
    assert "rapids_field_tokenizer" not in advertised_tokenizers()


@pytest.mark.skipif(not extra_available("cudf"), reason="cudf not installed")
def test_rapids_tokenizer_hashes_each_batch_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
    from tfm.data.synthetic import generate_synthetic_events
    from tfm.extras import rapids
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer

    sequences = build_entity_sequences(
        generate_synthetic_events(n_entities=6, seed=0),
        datetime(2030, 1, 1, tzinfo=UTC),
        max_events=16,
    )
    calls: list[int] = []
    real = rapids.cudf_hash_buckets

    def counted(values, buckets):  # type: ignore[no-untyped-def]
        calls.append(len(values))
        return real(values, buckets)

    monkeypatch.setattr(rapids, "cudf_hash_buckets", counted)
    gpu = rapids.RapidsFieldTokenizer()
    batch = gpu.collate(sequences, max_length=256)
    assert len(calls) == 1
    gpu_again = rapids.RapidsFieldTokenizer()
    one_by_one = [gpu_again.encode(seq).token_ids for seq in sequences]
    assert [gpu.encode(seq).token_ids for seq in sequences] == one_by_one
    cpu = TransactionFieldTokenizer().collate(sequences, max_length=256)
    assert batch.input_ids.shape == cpu.input_ids.shape
    assert gpu.vocab_size == TransactionFieldTokenizer().vocab_size


def test_rapids_field_order_matches_cpu_tokenizer() -> None:
    from tfm.extras.rapids import advertised_field_order
    from tfm.features.tokenizers.financial import FIELD_ORDER

    assert advertised_field_order() == FIELD_ORDER


@pytest.mark.skipif(not extra_available("transformers"), reason="transformers extra not installed")
def test_huggingface_tiny_backbone_and_train() -> None:
    from tfm import Binary, Decoder, EagerAttention, Flat, LastTokenPool, NextEvent, TrxFM
    from tfm.config import RecipeConfig
    from tfm.extras.huggingface import HuggingFaceCausalBackbone, build_huggingface_backbone
    from tfm.position.rope import RoPE
    from tfm.training.trainer import train_model

    config = RecipeConfig.model_validate(
        {
            "seed": 0,
            "data": {
                "adapter": "synthetic",
                "max_tokens": 16,
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
                            "max_seq_len": 16,
                            "dropout": 0.0,
                            "mask": "causal",
                        }
                    },
                },
            },
        }
    )
    backbone = build_huggingface_backbone(config, vocab_size=64)
    assert isinstance(backbone, HuggingFaceCausalBackbone)
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=32, num_heads=4, vocab_size=64, dropout=0.0)),
        position=RoPE(),
        attention=EagerAttention(),
        objective=NextEvent(),
        readout=LastTokenPool(),
        heads={"fraud": Binary()},
        backbone=backbone,
    )
    batch = TransactionBatch(
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        labels=torch.tensor([[2, 3, -100, -100], [5, -100, -100, -100]], dtype=torch.long),
    )
    stats = train_model(
        model, [batch], max_steps=1, learning_rate=1e-3, seed=0, device=torch.device("cpu")
    )
    assert stats["steps"] == 1.0
    out = model(batch, compute_loss=True)
    assert out.embeddings is not None
    assert out.embeddings.shape == (2, 32)
    assert out.logits is not None and out.logits.shape[-1] == 64


@pytest.mark.skipif(not extra_available("transformers"), reason="transformers extra not installed")
def test_huggingface_to_config_describes_the_llama_not_the_decoder_spec() -> None:
    from tfm import Decoder, Flat, TrxFM
    from tfm.config import RecipeConfig
    from tfm.diagnose.cost import cost_from_composition
    from tfm.extras.huggingface import build_huggingface_backbone
    from tfm.position.rope import RoPE

    config = RecipeConfig.model_validate(
        {
            "seed": 0,
            "data": {
                "adapter": "synthetic",
                "max_tokens": 16,
                "temporal_split": {
                    "train_end": "2024-01-04T00:00:00Z",
                    "validation_end": "2024-01-07T00:00:00Z",
                },
            },
            "model": {
                "backbone": "huggingface",
                "huggingface": {"num_key_value_heads": 2, "tie_word_embeddings": False},
                "composition": {
                    "position": "rope",
                    "granularity": {
                        "decoder": {
                            "depth": 3,
                            "hidden_size": 32,
                            "num_heads": 4,
                            "max_seq_len": 24,
                            "ffn_size": 48,
                            "dropout": 0.0,
                            "mask": "causal",
                        }
                    },
                },
            },
        }
    )
    backbone = build_huggingface_backbone(config, vocab_size=64)
    # The Decoder spec differs from the Llama on purpose: with a backbone it is never built.
    model = TrxFM(
        granularity=Flat(Decoder(depth=1, hidden_size=8, num_heads=2, vocab_size=64)),
        position=RoPE(),
        backbone=backbone,
    )
    composition = model.to_config()
    assert composition["position"] == "rope"
    assert composition["backbone"] == "HuggingFaceCausalBackbone"
    assert composition["granularity"] == {
        "type": "flat",
        "decoder": {
            "depth": 3,
            "mask": "causal",
            "hidden_size": 32,
            "num_heads": 4,
            "num_key_value_heads": 2,
            "dropout": 0.0,
            "vocab_size": 64,
            "max_seq_len": 24,
            "ffn_size": 48,
            "tie_word_embeddings": False,
        },
    }
    with pytest.raises(SchemaError, match="HuggingFaceCausalBackbone"):
        cost_from_composition(composition)


@pytest.mark.skipif(not extra_available("transformers"), reason="transformers extra not installed")
def test_huggingface_vocab_mismatch(tmp_path) -> None:
    from tfm.config import RecipeConfig
    from tfm.contracts.errors import IncompatibleCompositionError
    from tfm.extras.huggingface import build_huggingface_backbone

    config = RecipeConfig.model_validate(
        {
            "data": {
                "max_tokens": 16,
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
                            "max_seq_len": 16,
                            "mask": "causal",
                        }
                    },
                },
            },
        }
    )
    backbone = build_huggingface_backbone(config, vocab_size=64)
    ckpt = tmp_path / "tiny-llama"
    backbone.model.save_pretrained(ckpt)
    mismatched = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "huggingface": config.model.huggingface.model_copy(
                        update={"pretrained_path": str(ckpt)}
                    )
                }
            )
        }
    )
    with pytest.raises(IncompatibleCompositionError, match="vocab_size"):
        build_huggingface_backbone(mismatched, vocab_size=99)


def test_xgboost_probe_optional() -> None:
    import numpy as np

    from tfm.evaluation.fraud.baselines import evaluate_binary_probes
    from tfm.inference.embedding import EmbeddingTable

    rng = np.random.default_rng(0)
    table = EmbeddingTable(
        vectors=rng.normal(size=(8, 4)).astype(np.float32),
        entity_ids=tuple(f"e{i}" for i in range(8)),
        sequence_ids=tuple(f"s{i}" for i in range(8)),
        event_ids=tuple("" for _ in range(8)),
        fraud_labels=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32),
        numeric_features=rng.normal(size=(8, 3)).astype(np.float32),
    )
    if extra_available("xgboost"):
        metrics = evaluate_binary_probes(table, table, seed=0, k=3, head="xgboost")
        assert "raw" in metrics
        return
    with pytest.raises(ExtraUnavailableError, match="xgboost"):
        evaluate_binary_probes(table, table, seed=0, k=3, head="xgboost")
