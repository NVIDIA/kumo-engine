# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta

import torch

from tfm import Decoder, EagerAttention, Flat, LastTokenPool, NextEvent, NoPE, TrxFM
from tfm.data.collate import collate_sequences, sequence_label
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.data.splits.temporal import TemporalSplitSpec, assign_temporal_split
from tfm.data.synthetic.generator import generate_synthetic_events
from tfm.features.tokenizers.financial import TransactionFieldTokenizer
from tfm.inference.embedding import extract_embeddings


def test_embedding_rows_align_to_labels() -> None:
    events = generate_synthetic_events(seed=3, n_entities=8, events_per_entity=16)
    spec = TemporalSplitSpec(
        train_end=datetime(2024, 1, 4, tzinfo=UTC),
        validation_end=datetime(2024, 1, 7, tzinfo=UTC),
        label_delay=timedelta(hours=12),
    )
    split = assign_temporal_split(events, spec)
    seqs = build_entity_sequences(split.train, spec.train_end, max_events=8)
    tok = TransactionFieldTokenizer()
    batch = collate_sequences(seqs, tok, max_tokens=96)
    model = TrxFM(
        granularity=Flat(
            Decoder(depth=1, hidden_size=16, num_heads=4, vocab_size=tok.vocab_size, dropout=0.0)
        ),
        position=NoPE(),
        attention=EagerAttention(),
        objective=NextEvent(),
        readout=LastTokenPool(),
    )
    table = extract_embeddings(model, batch, device=torch.device("cpu"))
    assert len(table.entity_ids) == batch.batch_size
    assert list(table.entity_ids) == list(batch.entity_ids)
    for seq, label in zip(seqs, table.fraud_labels, strict=True):
        assert float(label) == sequence_label(seq)
    assert all(seq.events[-1].event_timestamp < spec.train_end for seq in seqs)
