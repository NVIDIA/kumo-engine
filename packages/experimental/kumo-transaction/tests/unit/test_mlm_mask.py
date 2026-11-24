# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import UTC, datetime

import pytest
import torch

from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture, load_pragmatiq_dataset
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
from tfm.objectives.mlm import TokenEventKeyMLM, apply_mlm_mask


@pytest.fixture(scope="module")
def batch_and_vocab(tmp_path_factory: pytest.TempPathFactory) -> tuple[KeyValueTimeBatch, int]:
    root = generate_pragmatiq_fixture(
        tmp_path_factory.mktemp("mlm") / "book", n_users=8, events_per_user=12, seed=3
    )
    dataset = load_pragmatiq_dataset(root)
    seqs = build_entity_sequences(
        list(dataset.events), datetime(2024, 12, 31, tzinfo=UTC), max_events=12
    )
    tok = KeyValueTimeTokenizer()
    tok.set_profiles({p.entity_id: p for p in dataset.profiles})
    tok.fit(seqs)
    return tok.collate(seqs, max_length=24), tok.vocab_size


def _selected(masked: KeyValueTimeBatch, spec: TokenEventKeyMLM) -> torch.Tensor:
    """Tokens chosen for prediction or [UNK] dropout (`p_unk` 0 leaves only the former)."""
    assert masked.mlm_labels is not None
    return masked.mlm_labels != spec.ignore_index


def test_a_drawn_key_masks_every_token_of_it_in_its_row(
    batch_and_vocab: tuple[KeyValueTimeBatch, int],
) -> None:
    batch, _ = batch_and_vocab
    spec = TokenEventKeyMLM(p_token=0.0, p_event=0.0, p_key=0.5, p_unk=0.0)
    valid = batch.token_mask.bool()
    hit = groups = 0
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        selected = _selected(apply_mlm_mask(batch, spec, generator=g), spec)
        assert not (selected & ~valid).any()
        for row in range(batch.key_ids.shape[0]):
            for key in batch.key_ids[row][valid[row]].unique():
                where = valid[row] & (batch.key_ids[row] == key)
                chosen = selected[row][where]
                assert chosen.all() or not chosen.any()
                hit += int(chosen.all())
                groups += 1
    rate = hit / groups
    assert abs(rate - 0.5) < 4 * (0.25 / groups) ** 0.5, rate


def test_rows_draw_independently_for_any_key_id(
    batch_and_vocab: tuple[KeyValueTimeBatch, int],
) -> None:
    """4,096 copies of one row, every key id 999,999: each row decides alone, at `p_key`."""
    batch, _ = batch_and_vocab
    n = 4096
    names = ("key_ids", "value_ids", "token_mask", "event_mask")
    rows = {k: getattr(batch, k)[:1].expand(n, *getattr(batch, k).shape[1:]) for k in names}
    rows["key_ids"] = torch.full(rows["key_ids"].shape, 999_999)
    spec = TokenEventKeyMLM(p_token=0.0, p_event=0.0, p_key=0.5, p_unk=0.0)
    g = torch.Generator().manual_seed(0)
    out = apply_mlm_mask(dataclasses.replace(batch, **rows), spec, generator=g)
    rate = _selected(out, spec).flatten(1).any(1).float().mean().item()
    assert abs(rate - 0.5) < 4 * (0.25 / n) ** 0.5, rate


@pytest.mark.parametrize(
    "spec",
    [
        TokenEventKeyMLM(p_token=0.0, p_event=0.0, p_key=0.0, p_unk=0.0),
        TokenEventKeyMLM(p_token=1.0, p_event=0.0, p_key=0.0, p_unk=1.0),
    ],
    ids=["nothing-drawn", "all-dropped-to-unk"],
)
def test_with_no_supervised_token_the_first_valid_token_is_masked(
    batch_and_vocab: tuple[KeyValueTimeBatch, int], spec: TokenEventKeyMLM
) -> None:
    batch, _ = batch_and_vocab
    selected = _selected(apply_mlm_mask(batch, spec), spec)
    first = batch.token_mask.reshape(-1).nonzero()[0, 0]
    assert selected.reshape(-1).nonzero().flatten().tolist() == [int(first)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_masking_does_not_sync_the_host(batch_and_vocab: tuple[KeyValueTimeBatch, int]) -> None:
    batch, _ = batch_and_vocab
    on_gpu = batch.to(torch.device("cuda"))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        apply_mlm_mask(on_gpu, TokenEventKeyMLM())
    finally:
        torch.cuda.set_sync_debug_mode("default")
